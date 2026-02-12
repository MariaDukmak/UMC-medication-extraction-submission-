import warnings
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from typing import List, Optional, Union

# -------------------------
# Warnings
# -------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Class weights helper
# -------------------------
def get_class_weights_for_column(df, column: str):
    """
    Compute balanced class weights for a classification column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the column
    column : str
        Name of the target column

    Returns
    -------
    torch.Tensor
        Class weights
    int
        Number of classes
    """
    y = df[column].dropna().astype(int)
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float), len(classes)


# -------------------------
# Dataset
# -------------------------
class GenericTargetDataset(Dataset):
    """
    Generic dataset for a single target.
    """

    def __init__(self, df, tokenizer, text_cols: List[str], label_col: str,
                 task_type: str = "classification", max_len: int = 32):
        self.tokenizer = tokenizer
        self.texts = df[text_cols].fillna('').agg(' '.join, axis=1).tolist()
        self.labels = df[label_col].tolist()
        self.task_type = task_type
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long if self.task_type == "classification" else torch.float
        )

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': label
        }


# -------------------------
# Model
# -------------------------
class RobBERTTargetModel(nn.Module):
    """
    RobBERT-based model for a single target (classification or regression).
    """

    def __init__(self, roberta, num_outputs: int, task_type: str = "classification"):
        super().__init__()
        self.roberta = roberta  # shared RobBERT model
        self.dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(self.roberta.config.hidden_size, num_outputs)
        self.task_type = task_type

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(out.last_hidden_state[:, 0, :])
        return self.output_layer(cls_output)


# -------------------------
# Training & Evaluation
# -------------------------
scaler = GradScaler()


def train_model(model: nn.Module, dataloader: DataLoader, optimizer, loss_fn: nn.Module, device: torch.device) -> float:
    """
    Train a single-target model for one epoch using mixed precision.

    Returns average loss.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids, attention_mask)
            out_tensor = outputs.squeeze() if outputs.ndim == 2 and outputs.shape[1] == 1 else outputs
            loss = loss_fn(out_tensor, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_model(model: nn.Module, dataloader: DataLoader, task_type: str, device: torch.device):
    """
    Evaluate a single-target model and return predictions and labels.
    """
    model.eval()
    preds, labels_all = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask)
                if task_type == "classification":
                    predictions = torch.argmax(outputs, dim=1)
                else:
                    predictions = outputs.squeeze()

            preds.append(predictions.cpu())
            labels_all.append(labels.cpu())

    return torch.cat(preds).numpy(), torch.cat(labels_all).numpy()


def train_target(shared_roberta, label_col: str, task_type: str, encoder,
                 train_dataset: Dataset, device: torch.device,
                 strict_weights: torch.Tensor, unit_weights: torch.Tensor, epochs: int = 10):
    """
    Train a RobBERT model for a single target.

    Parameters
    ----------
    shared_roberta : nn.Module
        Shared RobBERT backbone
    label_col : str
        Target column name
    task_type : str
        "classification" or "regression"
    encoder : sklearn LabelEncoder or None
        Encoder for classification labels
    train_dataset : Dataset
        Training dataset
    device : torch.device
        Device for training
    strict_weights : torch.Tensor
        Class weights for strict classification targets
    unit_weights : torch.Tensor
        Class weights for other classification targets
    epochs : int
        Number of training epochs
    """
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    num_outputs = len(encoder.classes_) if encoder else 1
    model = RobBERTTargetModel(shared_roberta, num_outputs, task_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Select loss function
    if task_type == "classification":
        loss_fn = torch.nn.CrossEntropyLoss(weight=(strict_weights.to(device) if label_col == "strict_label_enc" else unit_weights.to(device)))
    else:
        loss_fn = torch.nn.HuberLoss(delta=10.0)

    for epoch in range(epochs):
        print(f"\n[Target: {label_col}] Epoch {epoch + 1}")
        train_loss = train_model(model, train_loader, optimizer, loss_fn, device)

        train_preds, train_true = eval_model(model, train_loader, task_type, device)

        if task_type == "classification":
            train_score = accuracy_score(train_true, train_preds)
            metric_name = "Accuracy"
        else:
            # Transform back from log-space if needed
            train_preds_real = np.expm1(train_preds)
            train_true_real = np.expm1(train_true)
            train_score = mean_squared_error(train_true_real, train_preds_real)
            train_score_mae = mean_absolute_error(train_true_real, train_preds_real)
            metric_name = "RMSE"
            print(f"MAE: {train_score_mae:.4f}")

        print(f"{metric_name}: {train_score:.4f} | Loss: {train_loss:.4f}")

    return model

[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

# 🏥 UMC Medication Extraction

Prototype for extracting medication information, developed at the Julius Center, UMC.  
This repository contains code, data preprocessing scripts, and models for both rule-based and transformer-based approaches.

THIS REPO DOES NOT CONTAIN ALL CODE YET, WITH THE FINAL SUBMISSION ALL CODE WILL BE ADDED. 

---

## Project Structure

- **`data/`** – Place your raw dataset here.  
- **`data_prep/`** – Notebooks and scripts for data cleaning, preprocessing, and splitting.  
- **`main/`** – Model code and methods:
  - **`rule-based/`** – Rule-based extraction methods.  
  - **`medroberta/`** – MedRoBERTa model code.  
  - **`robbert/`** – RobBERT model code.  

Each subdirectory may contain its own README with specific usage instructions.

---

##  Installation

1. **Clone the repository**  
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

3. **Install the required packages**  
   ```bash
   pip install -r requirements.txt
   ```

---

##  Usage

1. Place your dataset in the **`data/`** directory.  
2. Run the preprocessing notebooks in **`data_prep/`** to clean, prepare, and split the data.  
3. Download and upload the required Hugging Face models to their respective folders in **`main/`**.  
   > Without these models, running the code on DRE will not be possible.  
4. Run the notebooks in the subfolders of **`main/`** to train, test, and evaluate models:  
   - `main/rule-based/`  
   - `main/medroberta/`  
   - `main/robbert/`  

Each notebook contains step-by-step instructions.

---

##  Models

- Transformer models are saved after running the notebooks.  
- Saved weights can later be used for testing and deployment.  

---

##  Notes

- Ensure all dependencies and model files are available before running the notebooks.  
- Refer to the README in each subfolder for model-specific instructions.  

---

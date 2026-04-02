# Automated ICD Code Prediction using Baselines, LLMs, and Structured Prompting

This repository contains an end-to-end project for **automated ICD code prediction from discharge summaries**. The project compares:

- **Classical ML baseline**: TF-IDF + One-vs-Rest Logistic Regression
- **Domain-specific clinical baseline**: BioBERT with chunking and attention
- **Prompt-based generative models**: LLMs and SLMs via Ollama
- **Structured prompting strategies**:
  - Direct prompting
  - Chain-of-Thought (CoT)
  - Multi-role prompting (Clinician -> Coder -> Auditor)

The central question of the project is not just whether larger models perform better, but whether **structured prompting improves structured clinical prediction**, and whether that improvement depends on **model capacity**.

---

## 1. Project Objective

Given a long clinical discharge summary, predict the most relevant ICD diagnosis codes from a fixed label space (Top-50 ICD codes).

This is a challenging task because:
- discharge summaries are long and unstructured,
- ICD prediction is a **multi-label classification** problem,
- and generative language models are not naturally aligned with fixed-label prediction.

---

## 2. Repository Structure


├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_llm_slm_prompting.ipynb
├── src/
│   ├── data_utils.py
│   ├── eval_utils.py
│   ├── prompt_utils.py
│   ├── ollama_utils.py
│   └── config.py
├── artifacts/
│   ├── train_df.pkl
│   ├── val_df.pkl
│   ├── test_df.pkl
│   ├── mlb.pkl
│   ├── Y_train.pkl
│   ├── Y_val.pkl
│   ├── Y_test.pkl
│   └── label_list.pkl
├── results/
│   ├── tfidf_results.csv
│   ├── direct_prompt_results.csv
│   ├── three_role_results.csv
│   └── cot_results.csv
└── presentation/
    └── final_presentation.pdf


## 3. Environment Setup

### Option A: Google Colab

Install the required dependencies:

!pip install -q pandas numpy scikit-learn requests joblib tqdm
!pip install -q transformers sentencepiece accelerate
!pip install -q google-cloud-bigquery db-dtypes python-dotenv

If you are using Google Drive for persistence:

from google.colab import drive
drive.mount('/content/drive')

Then set your working directory, for example:

import os
BASE_DIR = '/content/drive/MyDrive/icd_project'
os.makedirs(BASE_DIR, exist_ok=True)


### Option B: Local setup

pip install -r requirements.txt


## 4. Data Pipeline Overview

### Step 1: Data extraction
The project uses MIMIC-style clinical notes and diagnosis mappings extracted through BigQuery-based SQL queries.

### Step 2: Preprocessing
- cleaned discharge summaries
- truncated long notes for modeling efficiency
- grouped ICD labels per admission
- selected top-50 most frequent ICD codes

### Step 3: Train/Validation/Test split
- patient-aware splitting using grouped splitting logic
- multi-label binarization using `MultiLabelBinarizer`

Saved artifacts include:
- `train_df.pkl`, `val_df.pkl`, `test_df.pkl`
- `Y_train.pkl`, `Y_val.pkl`, `Y_test.pkl`
- `mlb.pkl`
- `label_list.pkl`

---

## 5. Models Implemented

### A. TF-IDF + Logistic Regression
A classical, interpretable baseline using:
- TF-IDF vectorization
- One-vs-Rest Logistic Regression

### B. BioBERT + Chunking + Attention
A domain-specific clinical baseline using:
- BioBERT tokenizer and encoder
- chunking for long notes
- attention aggregation across chunks
- multi-label output layer

### C. Prompt-based LLM/SLM models via Ollama
Prompted models are tested using:
- direct prompting
- chain-of-thought prompting
- multi-role prompting

Example models used during experimentation:
- `gpt-oss:20b`
- `gemma3:27b`
- `llama3.3:70b`
- `deepseek-r1:32b`
- `deepseek-r1:7b`
- `codellama:latest`


## 6. Prompting Strategies

### 6.1 Direct Prompting
The model directly predicts ICD codes from the discharge summary.

### 6.2 Chain-of-Thought Prompting
A two-stage structured prompting method:
1. extract clinical diagnoses and findings,
2. map those findings to ICD codes.

### 6.3 Multi-role Prompting
A structured workflow inspired by real coding practice:
- **Clinician**: identifies candidate diagnoses
- **Coder**: maps diagnoses to ICD codes
- **Auditor**: verifies whether predicted codes are supported by the text


## 7. Evaluation Metrics

The following metrics are used:
- **Macro-F1**
- **Micro-F1**
- **Precision@5**
- **Recall@5**

These allow comparison across:
- classical baselines,
- domain-specific models,
- direct prompting,
- chain-of-thought prompting,
- and multi-role prompting.


## 8. Main Findings

Current project findings:
- Direct prompting performs poorly for structured ICD prediction.
- Structured prompting improves performance for larger models.
- Multi-role prompting performs better than direct prompting for LLMs.
- Smaller models do not consistently benefit from more complex prompting.
- Prompting effectiveness depends on model capacity.


## 9. How to Run the Project

### A. Run preprocessing and artifact creation
If your notebook still contains BigQuery extraction, run the data preparation section first and save all artifacts.

### B. Run baselines
Run:
- TF-IDF baseline
- BioBERT baseline

Save results to the `results/` folder.

### C. Run prompt-based experiments
Set Ollama host
Then run:
- direct prompting
- CoT prompting
- multi-role prompting

### D. Save outputs
Save CSVs for each method, for example:
- `direct_prompt_results.csv`
- `cot_results.csv`
- `three_role_results.csv`


## 10. Limitations

- Access to some large models may depend on hardware or external Ollama servers.
- Structured prompting increases inference cost and latency.
- SLMs may degrade with complex prompts.
- Clinical coding remains difficult without label-constrained decoding or fine-tuning.


## 11. Future Work

- Label-constrained prediction
- Retrieval-augmented ICD candidate selection
- Hybrid LLM + classifier pipelines
- Fine-tuning on clinical datasets
- Adaptive prompting based on model capacity

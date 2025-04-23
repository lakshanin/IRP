# Multi-Task Conformer: Assessment of English Pronunciation Errors In Non-Native Speakers

**Multi-Task Conformer model** for assessing English pronunciation at the **phoneme level**, designed for non-native speakers. It performs:

- **CTC-based phoneme recognition**
- **Cross-attention-based error classification** (Correct, Substituted, Deleted)

Includes preprocessing, training, evaluation, and a user-friendly Streamlit UI.

## Usage

**Train:**
```bash
python backend/scripts/train.py
```

**Evaluate:**
```bash
python backend/scripts/evaluate.py
```

**Inference:**
```bash
python backend/scripts/inference.py 
```

**Launch Web UI:**
```bash
streamlit run frontend/app.py
```

---

## Project Structure

```
IRP/
├── backend/
│   ├── benchmarking/   
│   ├── data/  
│   ├── exps/                   
│   ├── scripts/
│   │   ├── dataloader.py      
│   │   ├── EDA.ipynb            
│   │   ├── evaluate.py          
│   │   ├── inference.py          
│   │   ├── model.py            
│   │   ├── phoneme_mapping.py   
│   │   ├── preprocess.py         
│   │   └── train.py              
│   └── train_plots/           
│
├── frontend/
│   ├── app.py                  
│   ├── inference.py             
│   └── model.py                 

```

---

## Model Overview

- **Encoder:** Conformer Blocks
- **Branch 1:** CTC head (Phoneme Recognition)
- **Branch 2:** Cross-attention + Focal Loss head (Error Classification)
- **Training:** Multi-task loss with λ-weighting
- **Input Features:** 40-dim log-Mel spectrograms

---

## Metrics

- Phone Error Rate (PER)
- F1 Score for error classification
- Confusion Matrix

---

## Dataset

The dataset required for this project can be accessed via the following link:

[Dataset Link](https://psi.engr.tamu.edu/l2-arctic-corpus/)


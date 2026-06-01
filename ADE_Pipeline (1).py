#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# all Dependencies
get_ipython().system('apt-get update > /dev/null')
get_ipython().system('apt-get install -y poppler-utils tesseract-ocr libtesseract-dev > /dev/null')
get_ipython().system('pip install seqeval datasets pdf2image transformers tqdm scikit-learn pytesseract presidio-analyzer statsmodels requests spacy -q')
get_ipython().system('python -m spacy download en_core_web_lg -q')

import os
import json
import requests
import torch
import pandas as pd
import numpy as np
import re
import warnings
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from transformers import pipeline
from presidio_analyzer import AnalyzerEngine
from seqeval.metrics import classification_report
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# Replace with the actual API key
API_KEY = "API_KEY" 

URL_PARSE = "https://api.va.eu-west-1.landing.ai/v1/ade/parse"
URL_EXTRACT = "https://api.va.eu-west-1.landing.ai/v1/ade/extract"

PATH_METADATA = "metadata.csv"

# Sample size for validation
SAMPLE_SIZE = 200 

TARGET_LABELS = ["IBAN", "PERSON", "PERSON_NAME", "COMPANY", "COMPANY_NAME", "LOCATION"]
CATEGORIES = ["IBAN", "PERSON_NAME", "COMPANY_NAME", "LOCATION"] 

# Initialize NLP pipelines and analyzers
presidio_analyzer = AnalyzerEngine()
bert_ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
IBAN_PATTERN = r'[A-Z]{2}\d{2}[A-Z0-9]{11,31}'

# Define schema for the extraction API
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "iban": {"type": "array", "items": {"type": "string"}},
        "person_name": {"type": "array", "items": {"type": "string"}},
        "company_name": {"type": "array", "items": {"type": "string"}},
        "location": {"type": "array", "items": {"type": "string"}}
    }
}


# Helper Functions for Ensemble Logic
# Parses the ADE API JSON output into a list of tuples (Category, Entity)

def extract_ade_entities(ade_json):

    entities = []
    if not ade_json: return entities
    
    mapping = {"iban": "IBAN", "person_name": "PERSON_NAME", "company_name": "COMPANY_NAME", "location": "LOCATION"}
    
    for key, val_list in ade_json.items():
        cat = mapping.get(key.lower())
        if cat and isinstance(val_list, list):
            for text in val_list:
                if text and str(text).strip():
                    entities.append((cat, str(text).strip()))
    return list(set(entities))


#Runs regex, Presidio, and BERT to extract PII entities.
def extract_expert_entities(full_text):
    
    entities = {"R": [], "P": [], "B": []}
    if not full_text: return entities
    
    # Regex for IBANs
    for match in re.finditer(IBAN_PATTERN, full_text):
        entities["R"].append(("IBAN", match.group().strip()))
        
        
    # Presidio PII analysis
    p_hits = presidio_analyzer.analyze(text=full_text, entities=["PERSON", "LOCATION", "ORGANIZATION"], language='en')
    for h in p_hits:
        span = full_text[h.start:h.end].strip()
        if not span: continue
        if h.entity_type == "PERSON": entities["P"].append(("PERSON_NAME", span))
        elif h.entity_type == "LOCATION": entities["P"].append(("LOCATION", span))
        elif h.entity_type == "ORGANIZATION": entities["P"].append(("COMPANY_NAME", span))
        
        
    # BERT-based entity recognition
    b_hits = bert_ner(full_text)
    for h in b_hits:
        span = h['word'].strip()
        if not span: continue
        if h['entity_group'] == "PER": entities["B"].append(("PERSON_NAME", span))
        elif h['entity_group'] == "LOC": entities["B"].append(("LOCATION", span))
        elif h['entity_group'] == "ORG": entities["B"].append(("COMPANY_NAME", span))
        
    return {k: list(set(v)) for k, v in entities.items()}


def build_aligned_bio(true_ents, pred_ents):
    """Aligns true and predicted entities into BIO tags for seqeval compatibility."""
    aligned_true, aligned_pred = [], []
    pred_pool = pred_ents.copy()
    
    
    for t_label, t_text in true_ents:
        words = str(t_text).split()
        if not words: continue 
        
        
        bio = [f"B-{t_label}"] + [f"I-{t_label}"] * (len(words)-1)
        if (t_label, t_text) in pred_pool:
            aligned_true.extend(bio); aligned_pred.extend(bio)
            pred_pool.remove((t_label, t_text))
        else:
            aligned_true.extend(bio); aligned_pred.extend(["O"] * len(words))
            
    for p_label, p_text in pred_pool:
        words = str(p_text).split()
        if not words: continue 
        
        bio = [f"B-{p_label}"] + [f"I-{p_label}"] * (len(words)-1)
        aligned_true.extend(["O"] * len(words)); aligned_pred.extend(bio)
        
    if not aligned_true: 
        aligned_true, aligned_pred = ["O"], ["O"]
        
    return aligned_true, aligned_pred



# API Extraction Pipeline

if not os.path.exists(PATH_METADATA):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "metadata.csv" in files: 
            PATH_METADATA = os.path.join(root, "metadata.csv")
            break

            
full_df = pd.read_csv(PATH_METADATA).dropna(subset=['ground_truth']).reset_index(drop=True)
test_df = full_df.head(SAMPLE_SIZE)

ade_dataset = []
headers = {"Authorization": f"Bearer {API_KEY}"}


for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Extracting documents via API"):
    pdf_name = os.path.basename(row['pdf_path'])
    
    pdf_path = None
    for root, _, files in os.walk("/kaggle/input"):
        if pdf_name in files:
            pdf_path = os.path.join(root, pdf_name)
            break
            
    if not pdf_path: continue

        
    try:
        with open(pdf_path, "rb") as f:
            res_p = requests.post(URL_PARSE, headers=headers, files={"document": f})
        
        if res_p.status_code != 200: continue
        
        parse_json = res_p.json()
        markdown_text = parse_json.get("markdown", parse_json.get("data", {}).get("markdown", ""))
        
        res_e = requests.post(URL_EXTRACT, headers=headers, 
                              files={'markdown': ('doc.md', markdown_text)},
                              data={'schema': json.dumps(EXTRACTION_SCHEMA), 'model': 'extract-latest'})
        
        if res_e.status_code != 200: continue
            
            
        extract_json = res_e.json()
        ade_json = extract_json.get("data", {}).get("extraction", extract_json.get("extraction", {}))

        gt_entries = json.loads(row['ground_truth'])
        true_ents = []
        for gt in gt_entries:
            label = gt['label'].upper()
            if label in TARGET_LABELS:
                if label == "PERSON": label = "PERSON_NAME"
                if label == "COMPANY": label = "COMPANY_NAME"
                true_ents.append((label, gt['text']))
        
        ade_dataset.append({
            "doc_id": pdf_name,
            "markdown": markdown_text,
            "ade_json": ade_json,
            "true_ents": list(set(true_ents))
        })
    except Exception:
        continue

ade_df = pd.DataFrame(ade_dataset)

# Evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
master_results = []
fold_analysis_data = []

combos = {
    "ADE_Solo": ("SOLO", ["A"]),
    "Regex_Solo": ("SOLO", ["R"]),
    "Presidio_Solo": ("SOLO", ["P"]),
    "BERT_Solo": ("SOLO", ["B"]),
    "A+R_Union (OR)": ("OR", ["A", "R"]),
    "A+P_Union (OR)": ("OR", ["A", "P"]),
    "ALL_Majority": ("MAJORITY", ["A", "R", "P", "B"])
}

for fold, (_, test_idx) in enumerate(kf.split(ade_df)):
    fold_df = ade_df.iloc[test_idx]
    fold_outputs = {k: {"true": [], "pred": []} for k in combos.keys()}
    
    for i, row in fold_df.iterrows():
        
        true_ents = row["true_ents"]
        ade_ents = extract_ade_entities(row["ade_json"])
        expert_ents = extract_expert_entities(row["markdown"])
        expert_ents["A"] = ade_ents
        
        
        # Build logic for ensemble methods
        for name, (logic, keys) in combos.items():
            if logic == "SOLO":
                combined_ents = expert_ents[keys[0]]
            elif logic == "OR":
                all_cands = []
                for k in keys: all_cands.extend(expert_ents[k])
                combined_ents = list(set(all_cands))
            elif logic == "MAJORITY":
                all_cands = []
                for k in keys: all_cands.extend(expert_ents[k])
                counts = {e: all_cands.count(e) for e in set(all_cands)}
                threshold = max(1, len(keys) / 2)
                combined_ents = [ent for ent, c in counts.items() if c >= threshold]
            
            t_bio, p_bio = build_aligned_bio(true_ents, combined_ents)
            fold_outputs[name]["true"].append(t_bio)
            fold_outputs[name]["pred"].append(p_bio)

    fold_results = {k: classification_report(fold_outputs[k]["true"], fold_outputs[k]["pred"], output_dict=True, zero_division=0) for k in combos.keys()}
    master_results.append(fold_results)

# Output Results (Report)
report_path = os.path.join(WORKING_DIR, "ADE_FULL_STATISTICS.txt")
with open(report_path, "w") as f:
    f.write("ADE (VLM) FINAL STATS REPORT\n" + "="*65 + "\n\n")
    
    methods = list(combos.keys())
    for m in methods:
        
        f.write(f"{'Kategorie':<18} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10}\n")
        
        
        
        for cat in CATEGORIES + ["macro avg"]:
            label = "OVERALL (Macro)" if cat == "macro avg" else cat
            f1_vals = [fold[m][cat]["f1-score"] for fold in master_results if cat in fold[m]]
            pr_vals = [fold[m][cat]["precision"] for fold in master_results if cat in fold[m]]
            rc_vals = [fold[m][cat]["recall"] for fold in master_results if cat in fold[m]]
            
            
            if not f1_vals: continue 
            f.write(f"{label:<18} | {np.mean(f1_vals):.4f}     | {np.mean(pr_vals):.4f}      | {np.mean(rc_vals):.4f}\n")


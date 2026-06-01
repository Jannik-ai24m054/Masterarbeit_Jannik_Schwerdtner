#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('apt-get update > /dev/null')
get_ipython().system('apt-get install -y poppler-utils tesseract-ocr libtesseract-dev > /dev/null')
get_ipython().system('pip install seqeval datasets pdf2image transformers tqdm scikit-learn pytesseract -q')




import os
import gc
import json
import torch
import torch.nn.functional as F
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from sklearn.model_selection import KFold, train_test_split
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)


from seqeval.metrics import classification_report, accuracy_score
import warnings


DATASET_PATH = "pii-dataset"
os.environ["WANDB_DISABLED"] = "true"


# Define PII categories and than map them to label IDs
CATEGORIES = ['COMPANY_NAME', 'IBAN', 'LOCATION', 'PERSON_NAME']
LABEL_LIST = ["O"]
for cat in CATEGORIES:
    LABEL_LIST.append(f"B-{cat}")
    LABEL_LIST.append(f"I-{cat}")

    
    
id2label = {k: v for k, v in enumerate(LABEL_LIST)}
label2id = {v: k for k, v in enumerate(LABEL_LIST)}


# Helper Functions to calculate the Intersection over Union (IoU) for two bounding boxes

def get_iou(boxA, boxB):
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if inter_area == 0: return 0
    
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    return inter_area / float(boxA_area + boxB_area - inter_area)

# Get Dataset for Layout
class LayoutDataset(Dataset):
    def __init__(self, df, processor, desc="Processing"):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.cached_data = []

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc=desc):
            filename = os.path.basename(row['pdf_path'])
            pdf_path = None
            
            # TO locate the file in the directory structure
            for root, _, files in os.walk(DATASET_PATH):
                if filename in files:
                    pdf_path = os.path.join(root, filename)
                    break
            
            if not pdf_path: continue

            try: 
                image = convert_from_path(pdf_path)[0].convert("RGB")
                
                ocr_df = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
                ocr_df = ocr_df.dropna().reset_index(drop=True)
                
            except: continue

                
            try: gt_data = json.loads(row['ground_truth'])
            except: gt_data = []
                
            
            words, boxes, labels = [], [], []
            width, height = image.size
            

            for _, ocr_row in ocr_df.iterrows():
                if str(ocr_row['text']).strip() == "": continue
                
                
                # Normalize bounding boxes for LayoutLMv3
                x0, y0, w, h = ocr_row['left'], ocr_row['top'], ocr_row['width'], ocr_row['height']
                norm_box = [int(1000*x0/width), int(1000*y0/height), int(1000*(x0+w)/width), int(1000*(y0+h)/height)]
                
                assigned_label = "O"
                for gt_item in gt_data:
                    if get_iou(norm_box, gt_item['bbox']) > 0.3:
                        assigned_label = f"B-{gt_item['label'].upper()}"
                        break
                
                words.append(str(ocr_row['text']))
                boxes.append(norm_box)
                labels.append(label2id.get(assigned_label, 0))

            if not words: continue

            encoding = self.processor(
                image, words, boxes=boxes, word_labels=labels,
                truncation=True, padding="max_length", max_length=512, return_tensors="pt"
            )

            
            self.cached_data.append({
                "input_ids": encoding.input_ids.squeeze(),
                "attention_mask": encoding.attention_mask.squeeze(),
                "bbox": encoding.bbox.squeeze(),
                "pixel_values": encoding.pixel_values.squeeze(),
                "labels": encoding.labels.squeeze()
            })

    def __len__(self): return len(self.cached_data)
    def __getitem__(self, idx): return self.cached_data[idx]

# Functino to calculate the precision, recall, f1 and the accuracy
def compute_metrics(p):
    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    tp = [[LABEL_LIST[p] for (p, l) in zip(pr, la) if l != -100] for pr, la in zip(predictions, labels)]
    tl = [[LABEL_LIST[l] for (p, l) in zip(pr, la) if l != -100] for pr, la in zip(predictions, labels)]
    
    res = classification_report(tl, tp, output_dict=True)
    return {
        "precision": res["macro avg"]["precision"], 
        "recall": res["macro avg"]["recall"], 
        "f1": res["macro avg"]["f1-score"], 
        "accuracy": accuracy_score(tl, tp)
    }


# Analysis & Post-Processing

# to get the location of the PIIs
def get_location_tag(bbox):
    
    if isinstance(bbox, torch.Tensor): bbox = bbox.tolist()
    y_min, y_max = bbox[1], bbox[3]
    
    if y_min < 150: return "Header"
    elif y_max > 850: return "Footer"
    elif bbox[0] < 200: return "Left-Margin"
    else: return "Body"

# Extracts the  predictions and confidence scores 
def extract_analysis_data(trainer, dataset, fold, id2label):
    
    predictions_output = trainer.predict(dataset)
    logits = torch.tensor(predictions_output.predictions)
    
    probs = F.softmax(logits, dim=-1)
    confidences, preds = torch.max(probs, dim=-1)
    labels = predictions_output.label_ids
    
    results = []
    for i in range(len(dataset)):
        item = dataset[i]
        bbox_list = item['bbox']
        
        for j in range(len(labels[i])):
            true_id, pred_id = labels[i][j].item(), preds[i][j].item()
            if true_id == -100: continue
            
            true_label, pred_label = id2label[true_id], id2label[pred_id]
            
            # Filter for PII entities or misclassifications
            if true_label == "O" and pred_label == "O": continue
            
            results.append({
                "Document_ID": f"Fold_{fold}_Doc_{i}",
                "PII_Type": true_label.replace("B-", "").replace("I-", "") if true_label != "O" else pred_label.replace("B-", "").replace("I-", ""),
                "Location_Tag": get_location_tag(bbox_list[j]),
                "Confidence_Score": round(confidences[i][j].item(), 4),
                "Prediction_Correctness": (true_id == pred_id),
                "True_Label": true_label,
                "Pred_Label": pred_label
            })
    return pd.DataFrame(results)

# Training 
def main():
    
    # Load and prepare metadata
    all_dfs = [pd.read_csv(os.path.join(DATASET_PATH, f"{s}/metadata.csv")) 
               for s in ['train', 'val', 'test'] if os.path.exists(os.path.join(DATASET_PATH, f"{s}/metadata.csv"))]
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    
    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    final_stats = []

    
    for fold, (train_idx, test_idx) in enumerate(kf.split(full_df)):
        
        df_train, df_val = train_test_split(full_df.iloc[train_idx], test_size=0.15, random_state=42)
        
        
        ds_train = LayoutDataset(df_train, processor, f"Train F{fold+1}")
        ds_val = LayoutDataset(df_val, processor, f"Val F{fold+1}")
        ds_test = LayoutDataset(full_df.iloc[test_idx], processor, f"Test F{fold+1}")

        
        model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", id2label=id2label, label2id=label2id)
        args = TrainingArguments(
            output_dir=os.path.join(WORKING_DIR, f"fold_{fold+1}"), num_train_epochs=12,
            per_device_train_batch_size=4, learning_rate=2e-5, fp16=True,
            eval_strategy="epoch", save_strategy="no"
        )
        
        trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val, compute_metrics=compute_metrics)
        trainer.train()
        
        # Export metrics and analysis data
        final_stats.append(trainer.evaluate(ds_test))
        df_analysis = extract_analysis_data(trainer, ds_test, fold+1, id2label)
        df_analysis.to_csv(os.path.join(WORKING_DIR, f"analysis_fold_{fold+1}.csv"), index=False)
        
        del model, trainer; torch.cuda.empty_cache(); gc.collect()

    #print("Done")

if __name__ == "__main__":
    main()


# In[ ]:


# Hyperparameter Tuning for Learning Rate 
  
    tune_df, _ = train_test_split(full_df, train_size=0.2, random_state=42)
    t_train, t_val = train_test_split(tune_df, test_size=0.2, random_state=42)
    
    ds_t_train = LayoutDataset(t_train, processor, "Tuning Train")
    ds_t_val = LayoutDataset(t_val, processor, "Tuning Val")

    best_lr, best_f1 = 2e-5, 0.0
    for lr in [2e-5, 5e-5]:
        print(f"Testing learning rate: {lr}")
        model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", id2label=id2label, label2id=label2id)
        
        args = TrainingArguments(
            output_dir=f"tuning_lr_{lr}", num_train_epochs=3, per_device_train_batch_size=4, 
            gradient_accumulation_steps=2, learning_rate=lr, fp16=True, 
            eval_strategy="epoch", save_strategy="no"
        )
        
        trainer = Trainer(model=model, args=args, train_dataset=ds_t_train, eval_dataset=ds_t_val, compute_metrics=compute_metrics)
        trainer.train()
        
        metrics = trainer.evaluate()
        if metrics['eval_f1'] > best_f1:
            best_f1, best_lr = metrics['eval_f1'], lr
            
        del model, trainer
        torch.cuda.empty_cache()
        gc.collect()
        
    print(f"Optimal learning rate identified: {best_lr} (F1: {best_f1:.4f})")


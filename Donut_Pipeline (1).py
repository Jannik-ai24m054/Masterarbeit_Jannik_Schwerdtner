
# install 
    !apt-get update > /dev/null
    !apt-get install -y poppler-utils tesseract-ocr libtesseract-dev > /dev/null
    !pip install seqeval datasets pdf2image transformers tqdm scikit-learn pytesseract presidio-analyzer statsmodels -q

    import os, gc, json, torch, pytesseract, re, warnings
    import numpy as np
    import pandas as pd
    from PIL import Image
    from pdf2image import convert_from_path
    from torch.utils.data import Dataset
    from tqdm.auto import tqdm
    from sklearn.model_selection import KFold, train_test_split
    from transformers import (
        LayoutLMv3Processor, LayoutLMv3ForTokenClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback, pipeline
    )
    from presidio_analyzer import AnalyzerEngine
    from seqeval.metrics import classification_report
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    # Link to the dataset
    KAGGLE_INPUT_DIR = "/pii-dataset"

    warnings.filterwarnings("ignore")
    os.environ["WANDB_DISABLED"] = "true"


    CATEGORIES = ['COMPANY_NAME', 'IBAN', 'LOCATION', 'PERSON_NAME']

    LABEL_LIST = ["O"]
    for cat in CATEGORIES:
        LABEL_LIST.append(f"B-{cat}"); LABEL_LIST.append(f"I-{cat}")


    id2label = {k: v for k, v in enumerate(LABEL_LIST)}
    label2id = {v: k for k, v in enumerate(LABEL_LIST)}


    # Presidio analyzer 
    presidio_analyzer = AnalyzerEngine()
    bert_ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
    IBAN_PATTERN = r'[A-Z]{2}\d{2}[A-Z0-9]{11,31}'

    # multiple models based on voting strategy
    def combine_votes(preds_list, mode="union"):

        final = []
        n_models = len(preds_list)

        for i in range(len(preds_list[0])):

            votes = [p[i] for p in preds_list if p[i] != 0]

            if not votes:
                final.append(0)

            elif mode == "union":
                final.append(votes[0])

            elif mode == "intersection":
                final.append(votes[0] if len(votes) == n_models else 0)

            elif mode == "majority":
                final.append(max(set(votes), key=votes.count) if len(votes) >= (n_models/2) else 0)
        return final

    # predictions from models.
    def get_expert_predictions(words, layout_preds):

        full_text = " ".join(words)

        # Regex (IBAN)
        regex = [label2id["O"]] * len(words)
        for i, w in enumerate(words):
            if re.fullmatch(IBAN_PATTERN, w): regex[i] = label2id["B-IBAN"]

        # Presidio (PII)
        presidio = [label2id["O"]] * len(words)
        p_res = presidio_analyzer.analyze(text=full_text, entities=["PERSON", "LOCATION", "ORGANIZATION"], language='en')

        for res in p_res:

            start_char = res.start
            curr = 0

            for i, w in enumerate(words):
                if curr <= start_char < curr + len(w) + 1:
                    lbl = "PERSON_NAME" if res.entity_type == "PERSON" else "LOCATION" if res.entity_type == "LOCATION" else "COMPANY_NAME"
                    presidio[i] = label2id.get(f"B-{lbl}", 0); break
                curr += len(w) + 1

        # BERT (Named Entity Recognition)
        bert = [label2id["O"]] * len(words)
        b_res = bert_ner(full_text)

        for res in b_res:
            start_char = res['start']
            curr = 0
            for i, w in enumerate(words):
                if curr <= start_char < curr + len(w) + 1:
                    lbl = "PERSON_NAME" if res['entity_group'] == "PER" else "LOCATION" if res['entity_group'] == "LOC" else "COMPANY_NAME"
                    bert[i] = label2id.get(f"B-{lbl}", 0); break
                curr += len(w) + 1

        return {"L": layout_preds, "R": regex, "P": presidio, "B": bert}

    # The Pipeline Logik
    def main():

        # Load the dataset
        all_dfs = [pd.read_csv(os.path.join(KAGGLE_INPUT_DIR, f"{s}/metadata.csv")) for s in ['train', 'val', 'test'] if os.path.exists(os.path.join(KAGGLE_INPUT_DIR, f"{s}/metadata.csv"))]
        df = pd.concat(all_dfs, ignore_index=True)

        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        master_results = []


        # Define expert combinations
        # Here we use the first Letter of the Model -> Regex -> R 
        solo_experts = ["L", "R", "P", "B"]

        combos = {
            "L+R": ["L", "R"], "L+P": ["L", "P"], "L+B": ["L", "B"],
            "L+R+B": ["L", "R", "B"], "L+R+P": ["L", "R", "P"], "L+B+P": ["L", "B", "P"],
            "ALL": ["L", "R", "P", "B"]
        }

        logics = ["union", "intersection", "majority"]

        # Training & Evaluation Loop
        for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
                print(f"\n Fold {fold+1}/5")

                # Partition data into training, validation, and test sets for the current fold
                df_train, df_val = train_test_split(df.iloc[train_idx], test_size=0.15, random_state=42)
                ds_train, ds_val, ds_test = LayoutDataset(df_train, processor, "Train"), LayoutDataset(df_val, processor, "Val"), LayoutDataset(df.iloc[test_idx], processor, "Test")

                # Initializing the model 
                model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", id2label=id2label, label2id=label2id)

                # Configure training parameters
                # here the learnig rate is fixed (the code für the parameter finetunig is at the end)
                # the winning lr is then set here
                args = TrainingArguments(
                    output_dir=f"fold_{fold}", 
                    num_train_epochs=8, 
                    per_device_train_batch_size=4, 
                    gradient_accumulation_steps=2, 
                    learning_rate=5e-05, 
                    fp16=True, 
                    eval_strategy="epoch", 
                    save_strategy="epoch", 
                    metric_for_best_model="f1",
                    load_best_model_at_end=True,
                    save_total_limit=1,
                    report_to="none"
                )

                # Configure training
                trainer = Trainer(
                    model=model, args=args, 
                    train_dataset=ds_train, 
                    eval_dataset=ds_val, 
                    compute_metrics=compute_metrics,
                    callbacks=[EarlyStoppingCallback(2)]
                )

                trainer.train()

                raw_preds = trainer.predict(ds_test)
                pred_ids = np.argmax(raw_preds.predictions, axis=2)

                fold_outputs = {}
                all_true = []


                # Set up the trainer with early stopping to prevent overfitting
                for i in range(len(ds_test)):
                    true = [LABEL_LIST[l] for l in raw_preds.label_ids[i] if l != -100]
                    all_true.append(true)
                    l_preds = [p for (p, l) in zip(pred_ids[i], raw_preds.label_ids[i]) if l != -100]


                    valid_words = ds_test.raw_words[i][:len(l_preds)]
                    exp = get_expert_predictions(valid_words, l_preds)

                    for s in solo_experts:
                        if s not in fold_outputs: fold_outputs[s] = []
                        fold_outputs[s].append([LABEL_LIST[x] for x in exp[s]])

                    for c_name, c_models in combos.items():
                        for l in logics:
                            key = f"{c_name}_{l}"
                            if key not in fold_outputs: fold_outputs[key] = []
                            voted = combine_votes([exp[m] for m in c_models], mode=l)
                            fold_outputs[key].append([LABEL_LIST[x] for x in voted])

                master_results.append({k: classification_report(all_true, v, output_dict=True) for k, v in fold_outputs.items()})
                del model, trainer; torch.cuda.empty_cache(); gc.collect()

            # Statistical Evaluation & Report Generation
            res_path = os.path.join(WORKING_DIR, "MASTER_THESIS_FINAL_REPORT.txt")
            with open(res_path, "w") as f:
                f.write("MASTER THESIS: HYBRID ENSEMBLE BENCHMARK REPORT\n" + "="*60 + "\n")

                methods = list(master_results[0].keys())
                baseline_f1s = [fold["L"]["macro avg"]["f1-score"] for fold in master_results]

                # Perform Wilcoxon signed-rank test to determine statistical significance
                p_values = []
                method_names_for_test = [m for m in methods if m != "L"]


                for method in method_names_for_test:
                    current_f1s = [fold[method]["macro avg"]["f1-score"] for fold in master_results]
                    _, p = wilcoxon(current_f1s, baseline_f1s)
                    p_values.append(p)

                _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
                p_map = dict(zip(method_names_for_test, p_adj))

                for m in sorted(methods):

                    if m != "L": f.write(f"Wilcoxon p-adj (vs Layout Solo): {p_map[m]:.6f}\n")

                    for metric in ["precision", "recall", "f1-score"]:
                        vals = [fold[m]["macro avg"][metric] for fold in master_results]
                        f.write(f"Overall {metric.capitalize()}: {np.mean(vals):.4f} (+/- {np.std(vals):.4f})\n")

                    f.write("Per Category Stats (Mean F1 / Rec / Prec):\n")
                    for cat in CATEGORIES:
                        cat_f1s = [fold[m][cat]["f1-score"] for fold in master_results]
                        cat_rec = [fold[m][cat]["recall"] for fold in master_results]
                        cat_pre = [fold[m][cat]["precision"] for fold in master_results]
                        f.write(f"  - {cat:<15}: F1: {np.mean(cat_f1s):.4f} | Rec: {np.mean(cat_rec):.4f} | Prec: {np.mean(cat_pre):.4f}\n")
                    f.write("-" * 40 + "\n")

            print(f"Link here: {res_path}")

if __name__ == "__main__":
    main()


    tune_df, _ = train_test_split(df, train_size=0.1, random_state=42) # Kleiner Split
    t_train, t_val = train_test_split(tune_df, test_size=0.2, random_state=42)
    
    ds_t_train = DonutDataset(t_train, processor, split="tuning_train")
    ds_t_val = DonutDataset(t_val, processor, split="tuning_val")

    best_lr, best_score = 2e-5, 0.0
    
    for lr in [2e-5, 5e-5]:
        print(f"Testing learning rate: {lr}")
        
        # Donut-Modell initialisieren
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        
        args = Seq2SeqTrainingArguments(
            output_dir=f"tuning_donut_lr_{lr}",
            num_train_epochs=3, 
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=lr,
            fp16=True,
            eval_strategy="epoch",
            save_strategy="no",
            predict_with_generate=True, 
            remove_unused_columns=False
        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=args,
            train_dataset=ds_t_train,
            eval_dataset=ds_t_val,
            data_collator=lambda data: {'pixel_values': torch.stack([d['pixel_values'] for d in data]), 
                                        'labels': torch.stack([d['labels'] for d in data])}
        )
        
        trainer.train()
        
        metrics = trainer.evaluate()
        current_score = metrics.get('eval_f1', metrics.get('eval_loss', 0)) # Je nachdem was compute_metrics liefert
        
        if current_score > best_score:
            best_score, best_lr = current_score, lr
            
        del model, trainer
        torch.cuda.empty_cache(); gc.collect()

    print(f"Winner: {best_lr} (Score: {best_score:.4f})")
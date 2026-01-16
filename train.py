import pandas as pd
import os
import torch
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

# ==========================================
# 🔧 PATH SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input Files
TRANS_FILE = os.path.join(BASE_DIR, "eng_to_ben_trans_qa_dataset.csv")
RTI_FILE = os.path.join(BASE_DIR, "synthetic_rti_data.csv")

# Output Folder (Where the trained brain will be saved)
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "rti_model_final")

# ==========================================
# 1. PREPARE DATA
# ==========================================
def prepare_training_data():
    print("⏳ Loading Datasets...")
    all_data = []

    # --- A. Load Kaggle Translation Data ---
    if os.path.exists(TRANS_FILE):
        df = pd.read_csv(TRANS_FILE).head(2000) # Sample size
        print(f"   Found Translation Data: {len(df)} rows")
        
        for _, row in df.iterrows():
            # CAREFUL HERE: The dataset is Eng->Ben. 
            # We want Ben->Eng. So we swap the columns.
            # 'output' column contains Bengali. 'input' contains English.
            ben_text = str(row.get('output', '')).strip()
            eng_text = str(row.get('input', '')).strip()

            if ben_text and eng_text:
                all_data.append({
                    "input_text": f"Translate to English: {ben_text}",
                    "target_text": eng_text
                })
    else:
        print(f"❌ Error: Translation file not found at {TRANS_FILE}")

    # --- B. Load Synthetic RTI Data ---
    if os.path.exists(RTI_FILE):
        df = pd.read_csv(RTI_FILE)
        print(f"   Found RTI Data: {len(df)} rows")
        
        for _, row in df.iterrows():
            # Prompt-based Classification
            all_data.append({
                "input_text": f"Classify Query: {row['query_text']}",
                "target_text": row['ministry_label']
            })
    else:
        print(f"❌ Error: Synthetic RTI file not found at {RTI_FILE}")

    print(f"✅ Total Training Samples: {len(all_data)}")
    return pd.DataFrame(all_data)

# ==========================================
# 2. TRAIN & SAVE
# ==========================================
def train():
    df = prepare_training_data()
    if df.empty:
        print("❌ No data found. Exiting.")
        return

    print("🚀 Initializing Model (mT5-small)...")
    model_name = "google/mt5-small"
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1)

    def preprocess(examples):
        model_inputs = tokenizer(examples['input_text'], max_length=128, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("⚙️ Tokenizing data...")
    tokenized_ds = dataset.map(preprocess, batched=True)

    args = Seq2SeqTrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=5e-4,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_total_limit=1,
        predict_with_generate=True,
        logging_dir=f"{MODEL_OUTPUT_DIR}/logs"
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"], tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    print("🔥 STARTING TRAINING (This may take 5-10 mins)...")
    trainer.train()
    
    print("💾 SAVING MODEL...")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"✅ Model saved successfully to: {MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    train()
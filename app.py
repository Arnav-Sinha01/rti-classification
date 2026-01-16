import streamlit as st
import torch
import os
import sys
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# ==========================================
# 🔧 PATH CONFIGURATION
# ==========================================
# Use absolute paths to avoid "File not found" errors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "rti_model_final")

# ==========================================
# 1. LOAD MODEL (Cached for Speed)
# ==========================================
@st.cache_resource
def load_model():
    # Check if model folder exists
    if not os.path.exists(MODEL_DIR):
        return None, None
    
    try:
        device = "cpu"  # Force CPU for stability unless you are sure about CUDA
        # Load Tokenizer (requires sentencepiece)
        tokenizer = MT5Tokenizer.from_pretrained(MODEL_DIR)
        # Load Model
        model = MT5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# ==========================================
# 2. PREDICTION LOGIC
# ==========================================
def solve_query(tokenizer, model, ben_text):
    device = model.device
    
    # --- Step A: Translate (Bengali -> English) ---
    # We must match the prefix used in training!
    input_text_trans = f"Translate to English: {ben_text}"
    input_ids = tokenizer(input_text_trans, return_tensors="pt").input_ids.to(device)
    
    # Generate English text
    eng_out = model.generate(input_ids, max_new_tokens=60)
    eng_text = tokenizer.decode(eng_out[0], skip_special_tokens=True)

    # --- Step B: Classify (English -> Ministry) ---
    input_text_class = f"Classify Query: {eng_text}"
    input_ids = tokenizer(input_text_class, return_tensors="pt").input_ids.to(device)
    
    # Generate Ministry Name
    class_out = model.generate(input_ids, max_new_tokens=30)
    ministry = tokenizer.decode(class_out[0], skip_special_tokens=True)

    return eng_text, ministry

# ==========================================
# 3. STREAMLIT UI
# ==========================================
def main():
    st.set_page_config(page_title="RTI AI Portal", layout="centered")

    st.title("🇮🇳 RTI Query Classification System")
    st.caption("Powered by Fine-Tuned mT5 Transformer Model")

    # Load Model
    with st.spinner("Loading AI Brain... (this may take a moment)"):
        tokenizer, model = load_model()

    # Check if model loaded successfully
    if model is None:
        st.error("🚨 Model NOT Found or Failed to Load!")
        st.warning(f"1. Make sure you ran 'train.py' successfully.\n"
                   f"2. Check if this folder exists: {MODEL_DIR}")
        if st.button("Re-check for Model"):
            st.rerun()
        return

    st.success("✅ AI Model Loaded & Ready")

    # User Input
    st.subheader("Citizen Grievance Interface")
    query = st.text_area("Enter your query in Bengali:", height=100, 
                         placeholder="Example: আমার আধার কার্ডে নামের বানান ভুল আছে।")

    if st.button("🚀 Process Query"):
        if not query.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("AI is analyzing keywords..."):
                try:
                    eng_text, ministry = solve_query(tokenizer, model, query)
                    
                    st.divider()
                    
                    # Result 1: Classification
                    st.markdown("### 🏛️ Ministry Routing")
                    st.success(f"**Forwarded To:** {ministry}")
                    
                    st.caption(f"Reasoning: Detected keywords in translated text mapped to {ministry}.")
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
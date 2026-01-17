-> 🇮🇳 AI-Powered RTI Query Classification 

This NLP project automates the processing of Right to Information (RTI) queries. It uses a fine-tuned **Google mT5 (multilingual T5)** transformer to:
1. **Translate** queries from Bengali to English.
2. **Classify** the query to route it to the correct Indian Ministry (e.g., Ministry of Railways, UIDAI).

-> Project Overview
* **`train.py`**: The training script that fine-tunes the mT5 model.
* **`app.py`**: A Streamlit web interface for users to test the model.
* **`generate_dataset.py`**: Generates synthetic training data.

-> Setup & Installation

**1. Clone the repository**
git clone [https://github.com/Arnav-Sinha01/rti-classification.git](https://github.com/Arnav-Sinha01/rti-classification.git)
cd rti-classification

**2. Install dependencies**
pip install -r requirements.txt

-> **How to Run**
Note: The trained model is too large (>1GB) to host on GitHub. You must train it locally first.

Step 1: Train the Model Run the training script. This will generate the rti_model_final folder on your machine.
python train.py

(Note: Training may take 10-15 minutes on a standard CPU).

Step 2: Launch the App Once the model folder exists, start the interface:
streamlit run app.py

-> **Model Details**
Architecture: Google mT5-Small
Input: Bengali Text
Output: Ministry Label (Classification)

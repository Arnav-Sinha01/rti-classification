import pandas as pd
import random
import re
import os

# ==========================================
# 🔧 PATH FIXER (Ensures files are found)
# ==========================================
# Get the folder where THIS script is currently running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define exact paths relative to this script
INPUT_CSV = os.path.join(BASE_DIR, "datafile.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "synthetic_rti_data.csv")

# ==========================================
# 1. SMART KEYWORD MAPPING
# ==========================================
KEYWORD_MAP = {
    "Central Board of Direct Taxes (Income Tax)": ["Income Tax Return", "Tax Refund", "PAN Card Correction", "TDS Deductions"],
    "Unique Identification Authority of India": ["Aadhaar Card Update", "Biometric Issue", "UIDAI Status", "Link Aadhaar to Mobile"],
    "Ministry of Railways ( Railway Board)": ["Train Ticket Refund", "PNR Status Invalid", "Station Cleanliness", "Train Delay Complaint"],
    "Department of Agriculture, Cooperation and Farmers Welfare": ["Crop Insurance Claim", "Kisan Credit Card", "MSP Rates", "Farmers Subsidy"],
    "Department of Posts": ["Speed Post Delivery", "Postal Savings Account", "Money Order Delay", "Post Office Passport Seva"],
    "Department of Telecommunications": ["Mobile Network Issue", "Broadband Speed", "5G Rollout Plan", "Sim Card Deactivation"],
    "Ministry of Road Transport and Highways": ["National Highway Condition", "Toll Plaza Fees", "Driving License Status", "Road Repair"],
    "Ministry of External Affairs": ["Passport Application", "Visa Delay", "Embassy Assistance", "OCI Card Status"],
    "Department of Higher Education": ["Scholarship Delay", "University Grant", "PhD Admission", "College Fee Refund"],
    "Department of Financial Services (Banking Division)": ["Bank Loan Rejection", "ATM Fraud", "Savings Account Issue", "Bank Manager Complaint"]
}

# Generic templates for organizations NOT in the map above
GENERIC_TEMPLATES = [
    "I want to file an RTI regarding {keyword}.",
    "Please provide details about funds allocated to {keyword}.",
    "What is the status of my application with {keyword}?",
    "Complaint regarding delay in service by {keyword}.",
    "Requesting information on new schemes under {keyword}.",
    "Why is there no response from the local office of {keyword}?",
    "I need the contact details of the nodal officer for {keyword}."
]

def clean_org_name(name):
    # Removes fancy prefixes to get the core name
    name = re.sub(r'Department of |Ministry of |Government of |Central Board of ', '', str(name))
    return name.strip()

def generate_synthetic_data():
    print(f"📍 Script Location: {BASE_DIR}")
    print(f"📂 Reading input:   {INPUT_CSV}")
    
    if not os.path.exists(INPUT_CSV):
        print("❌ Error: 'datafile.csv' not found! Please verify it is in this folder.")
        return

    try:
        df = pd.read_csv(INPUT_CSV)
        unique_orgs = df['org_name'].unique()
        print(f"✅ Found {len(unique_orgs)} unique Ministries/Departments.")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    synthetic_data = []

    print("⚙️ Generating synthetic data (High Volume Mode)...")
    
    for org in unique_orgs:
        # Determine keywords to use
        if org in KEYWORD_MAP:
            keywords = KEYWORD_MAP[org]
        else:
            keywords = [clean_org_name(org)]

        # ======================================================
        # 🔥 SETTING: GENERATE 20 QUERIES PER MINISTRY
        # ======================================================
        for _ in range(20): 
            chosen_kw = random.choice(keywords)
            chosen_tmpl = random.choice(GENERIC_TEMPLATES)
            query_text = chosen_tmpl.format(keyword=chosen_kw)
            
            synthetic_data.append({
                "query_text": query_text,
                "ministry_label": org
            })

    # Save to CSV
    out_df = pd.DataFrame(synthetic_data)
    out_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n🎉 SUCCESS! Generated {len(out_df)} queries.")
    print(f"💾 Saved to: {OUTPUT_CSV}")
    print(f"👉 You can now run 'streamlit run rti_project.py' and click 'Retrain Model'")

if __name__ == "__main__":
    generate_synthetic_data()
import streamlit as st
import sys
import os
import pandas as pd
import glob

# 1. Page Config MUST be the absolute first Streamlit command
st.set_page_config(page_title="FairMind AI Dashboard", layout="wide")

# 2. System Path and Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from framework.fair_ai_framework import FairAIFramework

# 3. Main UI Headers
st.title("FairMind AI: Universal Bias Auditor")
st.markdown("_Advanced Multi-Agent Fairness Evaluation & Auto-Mitigation Framework_")

# 4. Sidebar Controls
with st.sidebar:
    st.header("Dataset Configuration")
    data_source = st.radio("Select Data Source:", ["Default (Adult Income)", "Upload Custom CSV"])
    
    uploaded_file = None
    default_sensitive_attr = "sex"
    
    if data_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
    else:
        default_sensitive_attr = st.selectbox("Target Sensitive Attribute", ["sex", "race"])
        
    st.markdown("---")
    run_btn = st.button("▶ Run Full AI Analysis", type="primary", use_container_width=True)

# 5. Dynamic File Processing & UI
custom_target = None
custom_sensitive = None
temp_df = None

if data_source == "Upload Custom CSV" and uploaded_file is not None:
    st.write("### Uploaded Dataset Preview")
    
    has_header = st.checkbox("My CSV file has a header row with column names", value=True)
    uploaded_file.seek(0)
    
    # Read file safely based on header status
    if has_header:
        temp_df = pd.read_csv(uploaded_file)
    else:
        temp_df = pd.read_csv(uploaded_file, header=None)
        temp_df.columns = [f"Column_{i}" for i in range(len(temp_df.columns))]
        
    st.dataframe(temp_df.head())
    
    st.write("### Map Your Columns")
    col1, col2 = st.columns(2)
    with col1:
        custom_target = st.selectbox("Select Target Column (What to predict)", temp_df.columns)
    with col2:
        custom_sensitive = st.selectbox("Select Sensitive Column (Demographics)", temp_df.columns)

# 6. Execution Logic
if run_btn:
    if data_source == "Upload Custom CSV" and uploaded_file is None:
        st.error("Please upload a CSV file before running the analysis.")
    else:
        with st.spinner(f"Initializing Framework & Deploying Models..."):
            
            # Route A: Custom CSV Execution
            if data_source == "Upload Custom CSV":
                fw = FairAIFramework(sensitive_attr=custom_sensitive)
                # Pass a pristine copy of our dataframe so we don't need to read the file again
                fw.load_data_dynamically(temp_df.copy(), custom_target, custom_sensitive)
            
            # Route B: Default Adult Dataset Execution
            else:
                fw = FairAIFramework(sensitive_attr=default_sensitive_attr)
            
            # Execute the AI Pipeline
            fw.run() 
            
        st.success("Analysis Complete!")
        
        # --- Rendering the Results Table ---
        st.subheader("Model Performance & Fairness Comparison")
        rows = []
        for name, r in fw.results.items():
            row = {"Model Type": name.replace("_", " ").upper()}
            acc = r['performance']['accuracy'] if isinstance(r['performance'], dict) else getattr(r['performance'], 'accuracy', 0)
            row.update({"Accuracy": f"{acc:.4f}"})
            row.update({k.replace("_", " ").title(): f"{v:.4f}" for k, v in r["fairness"].items()})
            rows.append(row)
        st.table(pd.DataFrame(rows))

        # --- Rendering the Graphs ---
        st.subheader("The Fairness-Accuracy Trade-off")
        
        # Force Streamlit to use strict absolute paths
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        outputs_dir = os.path.join(base_dir, "outputs")
        tradeoff_path = os.path.join(outputs_dir, "accuracy_vs_fairness.png")
        
        if os.path.exists(tradeoff_path):
            st.image(tradeoff_path, use_container_width=True)
        else:
            st.warning(f"Graph not found at: {tradeoff_path}")

        st.subheader("Bias Profiles by Model")
        search_pattern = os.path.join(outputs_dir, "*.png")
        bias_imgs = [img for img in glob.glob(search_pattern) if "accuracy_vs_fairness" not in img]
        
        if not bias_imgs:
            st.info("No individual bias profile graphs found in the outputs folder.")
            
        cols = st.columns(3)
        for idx, img_path in enumerate(bias_imgs):
            with cols[idx % 3]:
                m_name = os.path.basename(img_path).split('_')[0].upper()
                st.image(img_path, caption=f"{m_name} Group Disparity", use_container_width=True)
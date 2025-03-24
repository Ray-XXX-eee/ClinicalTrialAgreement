import streamlit as st
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from jinja2 import Template
from docx import Document
import base64
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load Sentence Transformer Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FAISS Setup (Load or Create)
index_file = "faiss_index.pkl"
metadata_file = "metadata.pkl"

if os.path.exists(index_file) and os.path.exists(metadata_file):
    # Load FAISS Index and Metadata
    with open(index_file, "rb") as f:
        index = pickle.load(f)
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
else:
    # Initialize FAISS Index
    index = faiss.IndexFlatL2(384)  # 384-dim embeddings
    metadata = []
    with open(index_file, "wb") as f:
        pickle.dump(index, f)
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)

# Function to add clauses to FAISS Index
def add_clause_to_faiss(text, clause_type):
    vector = embedding_model.encode(text).reshape(1, -1)
    index.add(vector)
    metadata.append({"clause_type": clause_type, "text": text})

    # Save updated index
    with open(index_file, "wb") as f:
        pickle.dump(index, f)
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)

# Sample clauses (only if the FAISS index is empty)
if index.ntotal == 0:
    sample_clauses = [
        ("The Sponsor agrees to indemnify the Institution from any liability...", "Indemnification"),
        ("The Institution must keep all trial data confidential...", "Confidentiality"),
        ("The trial budget shall be disbursed in accordance with the terms outlined...", "Financial Terms"),
        ("The trial must comply with FDA regulations and EU Directives...", "Regulatory Compliance"),
    ]
    for text, clause_type in sample_clauses:
        add_clause_to_faiss(text, clause_type)

# Function to retrieve clauses using FAISS
def retrieve_clauses(query, top_k=3):
    query_vector = embedding_model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    return [metadata[i]["text"] for i in indices[0] if i < len(metadata)]

# Function to generate agreement text using a template
def generate_contract(sponsor, jurisdiction, trial_phase, clauses):
    contract_template = """
    Clinical Trial Agreement
    ========================
    Sponsor: {{ sponsor }}
    Jurisdiction: {{ jurisdiction }}
    Trial Phase: {{ trial_phase }}

    1. **Indemnification**  
       {{ clauses[0] }}

    2. **Confidentiality**  
       {{ clauses[1] }}

    3. **Financial Terms**  
       {{ clauses[2] }}

    4. **Regulatory Compliance**  
       {{ clauses[3] }}
    """
    template = Template(contract_template)
    return template.render(
        sponsor=sponsor, jurisdiction=jurisdiction, trial_phase=trial_phase, clauses=clauses
    )

# Function to save contract as PDF

def save_as_pdf(contract_text, filename="clinical_trial_agreement.pdf"):
    pdf_path = filename
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    
    # Split text into lines and add to PDF
    y_position = 750  # Start from the top
    for line in contract_text.split("\n"):
        c.drawString(50, y_position, line)
        y_position -= 20  # Move to next line

    c.save()
    return pdf_path




# Function to display PDF in Streamlit
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500px"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit UI
st.title("📝 Clinical Trial Agreement Generator (FAISS-Based)")
st.sidebar.header("Input Parameters")


# User Inputs
sponsor = st.sidebar.text_input("Sponsor Name")
trial_phase = st.sidebar.selectbox("Trial Phase", ["Phase 1", "Phase 2", "Phase 3", "Phase 4"])
jurisdiction = st.sidebar.selectbox("Jurisdiction", ["US", "EU", "Global"])
custom_clause = st.sidebar.text_area("Custom Clause Request (Optional)")

# Generate Agreement
if st.sidebar.button("Generate Agreement"):
    st.sidebar.success("Generating Agreement...")

    # Retrieve relevant clauses
    indemnification_clause = retrieve_clauses("Indemnification clause for " + trial_phase + " in " + jurisdiction)[0]
    confidentiality_clause = retrieve_clauses("Confidentiality clause for " + trial_phase + " in " + jurisdiction)[0]
    financial_clause = retrieve_clauses("Financial terms for " + trial_phase + " in " + jurisdiction)[0]
    compliance_clause = retrieve_clauses("Regulatory compliance for " + trial_phase + " in " + jurisdiction)[0]

    clauses = [indemnification_clause, confidentiality_clause, financial_clause, compliance_clause]
    agreement_text = generate_contract(sponsor, jurisdiction, trial_phase, clauses)

    # Save and display PDF
    DATA_DIR = "data"
    os.makedirs(DATA_DIR, exist_ok=True)
    pdf_path = save_as_pdf(agreement_text, filename=os.path.join(DATA_DIR, "clinical_trial_agreement.pdf"))
    st.success("Agreement Generated Successfully! ✅")
    
    # Display Agreement Preview
    st.subheader("📄 Agreement Preview")
    st.text_area("Generated Agreement", agreement_text, height=400)
    display_pdf(pdf_path)

    # Human in the Loop (HITL) Review
    st.subheader("👨‍⚖️ Human Review & Approval")
    approval = st.radio("Approve Agreement?", ["Pending", "Approve", "Request Changes"])

    if approval == "Approve":
        st.success("✅ Agreement Approved!")
    elif approval == "Request Changes":
        change_request = st.text_area("Specify Required Changes")
        if st.button("Submit Change Request"):
            st.warning(f"Changes Requested: {change_request}")


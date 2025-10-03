# CAD Prediction Streamlit App - written with love (and a bit of caffeine)

import streamlit as st
import pandas as pd
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# --- XGBoost check ---
try:
    import xgboost
except ImportError:
    st.error("❌ XGBoost isn't installed. Try running: pip install xgboost")
    st.stop()

# --- Streamlit config ---
st.set_page_config(page_title="CAD Prediction App", page_icon="🫀", layout="wide")

# --- Some light styling for polish ---
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button {
            background-color: #007bff; color: white;
            border-radius: 8px; padding: 0.6em 1.2em; font-weight: bold;
        }
        .stButton>button:hover { background-color: #0056b3; color: white; }
        h1, h2, h3 { color: #2c3e50; }
        .reportview-container .markdown-text-container { font-size: 1.1em; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar: logo, title, navigation ---
if os.path.exists("Main.jpg"):
    st.sidebar.image("Main.jpg", caption="Gut Microbiome & CAD", use_container_width=True)

st.sidebar.title("🫀 CAD Prediction App")
page = st.sidebar.radio("Navigate to:", ["Home", "CAD Prediction Tool", "Datasets", "About"])

# ----------------------------
# Page 1: Home
# ----------------------------
if page == "Home":
    st.markdown("<h1 style='text-align:center;'>🫀 CAD Prediction using Gut Microbiome</h1>", unsafe_allow_html=True)
    st.write("""
    <div style="text-align: center; font-size: 18px;">
        Coronary Artery Disease (CAD) is one of the top killers globally.  
        This tool explores how your <b>gut bacteria</b> could help predict CAD using machine learning.  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        ### Why does this matter?
        - CAD is super common — but hard to catch early.
        - Your gut microbiome might offer new clues.
        - We're using a combo of **biology + AI** to do something useful here.

        💡 With this app you can:
        - Upload your own gut microbiome data  
        - Get predictions in seconds  
        - Download performance reports  
        """)
    with col2:
        if os.path.exists("Main.jpg"):
            st.image("Main.jpg", caption="Gut Microbiome & CAD", use_container_width=True)

# ----------------------------
# Page 2: CAD Prediction Tool
# ----------------------------
elif page == "CAD Prediction Tool":
    st.markdown("## 🔬 CAD Prediction Tool")
    uploaded_file = st.file_uploader("Upload your microbiome dataset (CSV)", type="csv")

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview")
            st.dataframe(data.head())

            if "Status" not in data.columns:
                st.error("⚠️ Your dataset needs a 'Status' column — values should be 'cad' or 'control'")
            else:
                model = pickle.load(open("cad_model.pkl", "rb"))
                with open("cad_features.json", "r") as f:
                    expected_features = json.load(f)

                X = data.drop(columns=["Status"], errors="ignore")
                y_true = data["Status"].str.lower()

                # Encode strings to numbers (hacky, but works)
                for col in X.select_dtypes(include="object").columns:
                    X[col] = pd.factorize(X[col])[0]

                # Fill in any missing columns with 0
                for c in expected_features:
                    if c not in X.columns:
                        X[c] = 0
                X = X[expected_features]

                if st.button("🚀 Run CAD Prediction"):
                    preds = model.predict(X)
                    mapping = {0: "cad", 1: "control"}
                    data["Predicted_Status"] = [mapping.get(p, p) for p in preds]
                    data["Actual_Status"] = y_true

                    st.success("✅ Done! Here's what we found:")
                    st.dataframe(data.head())

                    # Prediction bar chart
                    pred_counts = data["Predicted_Status"].value_counts()
                    fig1, ax1 = plt.subplots()
                    sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax1, palette="Set2")
                    ax1.set_title("CAD vs Control Predictions")
                    st.pyplot(fig1)

                    # Confusion matrix
                    cm = confusion_matrix(y_true, data["Predicted_Status"], labels=["cad", "control"])
                    fig2, ax2 = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=["cad", "control"],
                                yticklabels=["cad", "control"],
                                ax=ax2)
                    ax2.set_xlabel("Predicted")
                    ax2.set_ylabel("Actual")
                    st.pyplot(fig2)

                    # Accuracy
                    acc = accuracy_score(y_true, data["Predicted_Status"])
                    st.markdown(f"### 🎯 Model Accuracy: **{acc:.2f}**")

                    # Storing stuff for PDF later
                    st.session_state["data"] = data.copy()
                    st.session_state["fig1"] = fig1
                    st.session_state["fig2"] = fig2
                    st.session_state["acc"] = acc

                if st.button("📄 Generate PDF Report"):
                    if "data" not in st.session_state:
                        st.error("⚠️ You need to run predictions before generating a report.")
                    else:
                        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        pdf_path = tmp_pdf.name
                        tmp_pdf.close()

                        doc = SimpleDocTemplate(pdf_path)
                        styles = getSampleStyleSheet()
                        story = []

                        story.append(Paragraph("CAD Prediction Report", styles["Title"]))
                        story.append(Spacer(1, 20))

                        if "acc" in st.session_state:
                            story.append(Paragraph(f"Accuracy: {st.session_state['acc']:.2f}", styles["Normal"]))
                            story.append(Spacer(1, 10))

                        report = classification_report(
                            st.session_state["data"]["Actual_Status"],
                            st.session_state["data"]["Predicted_Status"],
                            output_dict=True
                        )
                        report_df = pd.DataFrame(report).transpose().round(2)

                        table_data = [["Class", "Precision", "Recall", "F1-Score", "Support"]]
                        for cls in report_df.index:
                            if cls == "accuracy":
                                continue
                            row = [cls,
                                   report_df.loc[cls, "precision"],
                                   report_df.loc[cls, "recall"],
                                   report_df.loc[cls, "f1-score"],
                                   int(report_df.loc[cls, "support"])]
                            table_data.append(row)

                        tbl = Table(table_data, hAlign="LEFT")
                        tbl.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), colors.grey),
                            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                            ('ALIGN',(1,1),(-1,-1),'CENTER'),
                            ('GRID', (0,0), (-1,-1), 1, colors.black),
                            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ]))
                        story.append(Paragraph("Classification Report", styles["Heading2"]))
                        story.append(tbl)
                        story.append(Spacer(1, 20))

                        for fig, title in zip([st.session_state["fig1"], st.session_state["fig2"]],
                                              ["Prediction Ratio", "Confusion Matrix"]):
                            if fig:
                                tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                                fig.savefig(tmp_img.name, bbox_inches="tight", facecolor="white")
                                story.append(Paragraph(title, styles["Heading2"]))
                                story.append(Image(tmp_img.name, width=400, height=250))
                                story.append(Spacer(1, 20))
                                tmp_img.close()

                        doc.build(story)

                        with open(pdf_path, "rb") as f:
                            st.download_button("📥 Download PDF", data=f,
                                               file_name="CAD_Prediction_Report.pdf",
                                               mime="application/pdf")

        except Exception as e:
            st.error(f"Something went wrong while processing your file: {e}")

# ----------------------------
# Page 3: Datasets
# ----------------------------
elif page == "Datasets":
    st.markdown("## 📂 Sample Datasets")
    st.write("You can use one of the example datasets below, or bring your own from the Prediction Tool tab.")

    sample_files = {
        "🧬 American Gut Microbiome": "dataset_filtered.csv",
        "🍣 Japan Gut Microbiome": "PRJDB6472.csv",
    }

    for label, filename in sample_files.items():
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                st.download_button(f"⬇️ Download {label}", data=f, file_name=filename, mime="text/csv")
        else:
            st.warning(f"{filename} not found. Add it to the app directory to enable this download.")

# ----------------------------
# Page 4: About
# ----------------------------
elif page == "About":
    st.markdown("## ℹ️ About this Project")
    st.write("""
    This project explores the link between your gut microbes and the risk of developing Coronary Artery Disease (CAD).  

    🛠️ **Built with**  
    - XGBoost for ML  
    - Python + Pandas + Streamlit  
    - ReportLab for printable reports  

    🔍 **Features**  
    - Upload your own data  
    - Get real-time predictions  
    - Visualize & download performance reports  

    👩‍🔬 **Goal**: Build a practical AI tool to support CAD research using microbiome data.
    """)

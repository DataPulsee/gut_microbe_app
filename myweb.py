# --- Imports & setup ---
import os
import json
import pickle
import tempfile
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Optional libs
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

# Make ReportLab optional (for PDF export)
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors as RLcolors
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False


# --- Helper: find model files ---
def find_models():
    """
    Scan the current dir for best_model_*.pkl and optional sibling .json files.
    Returns:
        dict like:
        {
          "usa": {"pkl": "best_model_usa.pkl", "json": "best_model_usa.json" or None},
          "use": {"pkl": "best_model_use.pkl", "json": None},
          ...
        }
    """
    mapping = {}
    for fname in os.listdir("."):
        if fname.startswith("best_model_") and fname.endswith(".pkl"):
            base = fname[len("best_model_"):-4]  # between prefix and ".pkl"
            key = base.lower().strip()
            json_name = f"best_model_{base}.json"
            mapping[key] = {
                "pkl": fname,
                "json": json_name if os.path.exists(json_name) else None
            }
    return mapping


# --- UI header and layout ---
st.markdown("---")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.write("""
    ### Why does this matter?
    - CAD is tricky — often caught too late.
    - Your gut microbiome might have some answers.
    - We’re combining **biology + machine learning** — hoping to help.

    💡 This app lets you:
    - Upload your microbiome CSV  
    - Get predictions in seconds  
    - Download results + reports  
    """)

with col_right:
    if os.path.exists("Main.jpg"):
        st.image("Main.jpg", caption="Gut Microbiome & CAD", use_container_width=True)

# --- Models available? ---
available = find_models()
if not available:
    st.warning("No `best_model_*.pkl` files found. Drop them in the folder and reload the app.")
else:
    selected_model_label = st.selectbox(
        "Choose which trained model to use",
        options=[k.upper() for k in available.keys()],
        index=0
    )
    selected_model_key = selected_model_label.lower()
    model_files = available[selected_model_key]

    uploaded_file = st.file_uploader("Upload microbiome dataset (.csv)", type="csv")

    model_obj = None
    feature_list = None

    # Load model early to surface any issues
    try:
        with open(model_files["pkl"], "rb") as f:
            model_obj = pickle.load(f)
    except Exception as err:
        st.error(f"Couldn’t load model: {model_files['pkl']}\n\nError: {err}")
        st.stop()

    # Optional: load expected features from JSON
    feature_json = model_files.get("json")
    if feature_json and os.path.exists(feature_json):
        try:
            with open(feature_json, "r") as f:
                feature_list = json.load(f)
                if isinstance(feature_list, dict):
                    feature_list = (feature_list.get("features")
                                    or feature_list.get("expected_features")
                                    or list(feature_list.keys()))
                if not isinstance(feature_list, list):
                    feature_list = None
        except Exception as e:
            st.warning(f"Couldn’t parse features from `{feature_json}`: {e}")
            feature_list = None
    else:
        st.info("Feature list missing — I’ll align based on whatever columns are in your CSV.")

    # --- CSV upload & prediction flow ---
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Preview of Uploaded Data")
            st.dataframe(df.head())

            if "Status" not in df.columns:
                st.error("⚠️ Your file needs a 'Status' column — expected values: 'cad' or 'control'")
            else:
                X = df.drop(columns=["Status"], errors="ignore")
                y_actual = df["Status"].astype(str).str.lower()

                # Encode object columns naively
                for c in X.select_dtypes(include="object").columns:
                    X[c] = pd.factorize(X[c])[0]

                # Align to expected features if provided
                if feature_list:
                    for feat in feature_list:
                        if feat not in X.columns:
                            X[feat] = 0
                    X = X.reindex(columns=feature_list, fill_value=0)

                if st.button("🚀 Run CAD Prediction"):
                    preds = model_obj.predict(X)
                    label_map = {0: "cad", 1: "control"}  # adjust if your model uses the opposite mapping
                    df["Predicted_Status"] = [label_map.get(int(p), str(p)) for p in preds]
                    df["Actual_Status"] = y_actual

                    st.success("✅ Prediction complete!")
                    st.dataframe(df.head())

                    # --- Prediction bar chart ---
                    pred_summary = df["Predicted_Status"].value_counts()
                    fig_pred, ax_pred = plt.subplots()
                    if HAS_SEABORN:
                        sns.barplot(x=pred_summary.index, y=pred_summary.values, ax=ax_pred)
                    else:
                        ax_pred.bar(pred_summary.index, pred_summary.values)
                    ax_pred.set_title(f"Predictions ({selected_model_label} model)")
                    ax_pred.set_xlabel("Label")
                    ax_pred.set_ylabel("Count")
                    st.pyplot(fig_pred)

                    # --- Confusion Matrix ---
                    cm = confusion_matrix(y_actual, df["Predicted_Status"], labels=["cad", "control"])
                    fig_cm, ax_cm = plt.subplots()
                    if HAS_SEABORN:
                        sns.heatmap(cm, annot=True, fmt="d",
                                    xticklabels=["cad", "control"],
                                    yticklabels=["cad", "control"],
                                    ax=ax_cm)
                    else:
                        im = ax_cm.imshow(cm, aspect="auto")
                        # annotations
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center")
                        ax_cm.set_xticks([0, 1])
                        ax_cm.set_xticklabels(["cad", "control"])
                        ax_cm.set_yticks([0, 1])
                        ax_cm.set_yticklabels(["cad", "control"])
                        ax_cm.figure.colorbar(im, ax=ax_cm)
                    ax_cm.set_title("Confusion Matrix")
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    st.pyplot(fig_cm)

                    acc = accuracy_score(y_actual, df["Predicted_Status"])
                    st.markdown(f"### 🎯 Accuracy: **{acc:.2f}**")

                    # Cache for report
                    st.session_state["data"] = df.copy()
                    st.session_state["fig1"] = fig_pred
                    st.session_state["fig2"] = fig_cm
                    st.session_state["acc"] = acc
                    st.session_state["model_label"] = selected_model_label

                # --- PDF generation (optional) ---
                if st.button("📄 Generate PDF Report"):
                    if "data" not in st.session_state:
                        st.error("Run predictions first to enable report download.")
                    elif not HAS_REPORTLAB:
                        st.info("PDF export requires ReportLab. Add `reportlab` to requirements.txt to enable this.")
                    else:
                        # Build a simple PDF using ReportLab
                        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        doc = SimpleDocTemplate(tmp_pdf.name)
                        styles = getSampleStyleSheet()
                        content = []

                        content.append(Paragraph("CAD Prediction Report", styles["Title"]))
                        content.append(Spacer(1, 20))
                        content.append(Paragraph(f"Model Used: {st.session_state.get('model_label')}", styles["Normal"]))
                        content.append(Paragraph(f"Accuracy: {st.session_state['acc']:.2f}", styles["Normal"]))
                        content.append(Spacer(1, 12))

                        report_dict = classification_report(
                            st.session_state["data"]["Actual_Status"],
                            st.session_state["data"]["Predicted_Status"],
                            output_dict=True
                        )
                        report_df = pd.DataFrame(report_dict).transpose().round(2)

                        table_data = [["Class", "Precision", "Recall", "F1", "Support"]]
                        for cls in report_df.index:
                            if cls == "accuracy":
                                continue
                            row = [
                                cls,
                                report_df.loc[cls, "precision"],
                                report_df.loc[cls, "recall"],
                                report_df.loc[cls, "f1-score"],
                                int(report_df.loc[cls, "support"]),
                            ]
                            table_data.append(row)

                        tbl = Table(table_data, hAlign="LEFT")
                        tbl.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), RLcolors.grey),
                            ('TEXTCOLOR', (0,0), (-1,0), RLcolors.whitesmoke),
                            ('ALIGN', (1,1), (-1,-1), 'CENTER'),
                            ('GRID', (0,0), (-1,-1), 1, RLcolors.black),
                            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ]))
                        content.append(Paragraph("Classification Report", styles["Heading2"]))
                        content.append(tbl)
                        content.append(Spacer(1, 20))

                        # Add figures as images
                        for fig, label in zip([st.session_state["fig1"], st.session_state["fig2"]],
                                              ["Prediction Breakdown", "Confusion Matrix"]):
                            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            fig.savefig(tmp_img.name, bbox_inches="tight", facecolor="white")
                            content.append(Paragraph(label, styles["Heading2"]))
                            content.append(RLImage(tmp_img.name, width=400, height=250))
                            content.append(Spacer(1, 20))

                        doc.build(content)

                        with open(tmp_pdf.name, "rb") as f:
                            st.download_button(
                                "📥 Download Report PDF",
                                data=f.read(),
                                file_name=f"CAD_Report_{selected_model_key}.pdf",
                                mime="application/pdf"
                            )

        except Exception as e:
            st.error(f"Something went wrong while reading your file: {e}")

# --- Optional: Offer any local CSVs for download ---
csv_files = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
if not csv_files:
    st.info("No CSV files found in directory.")
else:
    for fname in csv_files:
        with open(fname, "rb") as f:
            st.download_button(f"⬇️ Download {fname}", data=f.read(), file_name=fname, mime="text/csv")

# --- Debugging aid: directory listing ---
with st.expander("🔎 Full directory listing"):
    filelist = sorted(os.listdir("."))
    st.write("\n".join(filelist))

# --- Footer ---
st.markdown("""
🛠️ **Built using**
- Python, Streamlit, Pandas, XGBoost
- ReportLab for PDF generation *(optional)*

🔍 **Features**
- Upload and analyze your microbiome data
- Visual model selection
- Downloadable visual + tabular reports

⚙️ **File expectations**
- Model: best_model_*.pkl
- Optional features file: best_model_*.json
""")

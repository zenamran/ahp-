import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib import colors
from reportlab.lib.units import cm
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image
from reportlab.platypus import Image
from reportlab.lib.units import cm
elements = []

logo = Image("logo_sonatrach.png", width=4*cm, height=2*cm)
elements.append(logo)
pdfmetrics.registerFont(TTFont("DejaVu", "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", "DejaVuSans-Bold.ttf"))

# ===== GREEN & ORANGE PROFESSIONAL THEME =====

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

st.sidebar.divider()
st.sidebar.subheader("🎨 Appearance")
st.session_state.dark_mode = st.sidebar.toggle("Dark Mode", value=st.session_state.dark_mode)

if st.session_state.dark_mode:
    st.markdown("""
    <style>
    .stApp { background-color: #0F172A; color: #E2E8F0; }

    h1, h2, h3, h4 { color: #F8FAFC; }

    .stSidebar { background-color: #064E3B; }

    .stButton>button {
        background-color: #16A34A;
        color: white;
        border-radius: 12px;
        font-weight: 600;
        padding: 8px 18px;
    }

    .stButton>button:hover {
        background-color: #F97316;
        color: white;
    }

    .stDataFrame, .stTable {
        background-color: #1E293B;
        border-radius: 12px;
    }

    .stMetric {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 14px;
        border-left: 6px solid #16A34A;
    }

    div[data-testid="stExpander"] {
        background-color: #1E293B;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; color: #0F172A; }

    h1, h2, h3, h4 { color: #064E3B; }

    .stSidebar { background-color: #DCFCE7; }

    .stButton>button {
        background-color: #16A34A;
        color: white;
        border-radius: 12px;
        font-weight: 600;
        padding: 8px 18px;
    }

    .stButton>button:hover {
        background-color: #F97316;
        color: white;
    }

    .stDataFrame, .stTable {
        background-color: white;
        border-radius: 12px;
    }

    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 14px;
        border-left: 6px solid #F97316;
    }

    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)



st.set_page_config(page_title="Vendor Selection Tool", layout="wide")

st.title("🚀 Decision Support System (DSS)")
st.subheader("Integrated AHP & Weighted Scoring Model")

# --- 1. SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ General Configuration")
    n_suppliers = st.number_input("Number of Suppliers", min_value=2, max_value=10, value=2)
    n_criteria = st.number_input("Number of Criteria", min_value=2, max_value=10, value=2)
    
# --- 2. NAMES INPUT ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("👥 Supplier Names")
    supplier_names = [st.text_input(f"Supplier {i+1}", f"Supplier {chr(65+i)}") for i in range(n_suppliers)]

with col2:
    st.subheader("📋 Criteria Names")
    criteria_names = [st.text_input(f"Criterion {j+1}", f"Criterion {j+1}") for j in range(n_criteria)]

# --- 3. AHP COMPARISON MATRIX ---
st.divider()
st.subheader("⚖️ AHP Pairwise Comparison Matrix (Saaty Scale)")
st.write("Compare the relative importance of criteria (1: Equal, 3: Moderate, 5: Strong, 7: Very Strong, 9: Extreme)")

A = np.eye(n_criteria)
for i in range(n_criteria):
    for j in range(i + 1, n_criteria):
        val = st.number_input(f"How important is {criteria_names[i]} vs {criteria_names[j]}?", 
                               min_value=0.11, max_value=9.0, value=1.0, step=0.1, key=f"A_{i}_{j}")
        A[i, j] = val
        A[j, i] = 1 / val

# --- 4. PERFORMANCE MATRIX (SCORES) ---
st.divider()
st.subheader("⭐ Performance Scoring (Scale 0-10)")
st.write("Enter the raw performance scores for each supplier per criterion.")

scores_data = np.zeros((n_suppliers, n_criteria))
for i in range(n_suppliers):
    with st.expander(f"Scores for {supplier_names[i]}"):
        cols = st.columns(n_criteria)
        for j in range(n_criteria):
            scores_data[i, j] = cols[j].number_input(f"{criteria_names[j]}", 0.0, 10.0, 0.0, key=f"S_{i}_{j}")

# --- 5. MATHEMATICAL CALCULATIONS ---
# AHP Weights & Consistency
eig_vals, eig_vecs = np.linalg.eig(A)
max_eig = np.real(eig_vals.max())
w_ahp = np.real(eig_vecs[:, eig_vals.argmax()])
w_ahp = w_ahp / w_ahp.sum()

# RI Table for Consistency Ratio
RI_table = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
CI = (max_eig - n_criteria) / (n_criteria - 1)
CR = CI / RI_table[n_criteria] if n_criteria > 2 else 0

# AHP Method Score
score_ahp_final = np.dot(scores_data, w_ahp)
# --- 6. RESULTS & OUTPUT ---
st.divider()
st.header("📊 Final Results & Ranking")

res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    st.subheader("Consistency Check")
    st.metric("Consistency Ratio (CR)", f"{CR:.2%}")
    if CR < 0.1:
        st.success("Consistent Matrix ✅")
    else:
        st.error("Inconsistent Matrix ❌ (Please revise AHP values)")

st.subheader("📋 Criteria Weight AHP Results")

df_scoring = pd.DataFrame({
    "Criteria": criteria_names,
    "Weight": w_ahp
}).sort_values(by="Weight", ascending=False)

st.dataframe(df_scoring, use_container_width=True)
st.subheader("🏆 AHP-Based Weighted Scoring Results")

df_ahp = pd.DataFrame({
    "Supplier": supplier_names,
    "Score": score_ahp_final
}).sort_values(by="Score", ascending=False)

st.dataframe(df_ahp, use_container_width=True)

# --- 7. SENSITIVITY ANALYSIS CHART ---
st.divider()
st.subheader("📈 Sensitivity Analysis")

selected_criterion = st.selectbox(
    "Select the criterion to analyze",
    criteria_names
)

crit_index = criteria_names.index(selected_criterion)

st.write(f"Effect of varying the weight of '{selected_criterion}' on final scores.")

variation = np.linspace(0.05, 0.95, 20)
sens_results = []

for v in variation:
    temp_w = np.copy(w_ahp)
    temp_w[crit_index] = v

    # إعادة توزيع الأوزان على باقي المعايير
    others = [i for i in range(n_criteria) if i != crit_index]
    sum_others = temp_w[others].sum()

    if sum_others > 0:
        temp_w[others] = (temp_w[others] / sum_others) * (1 - v)

    # نحفظ نتيجة كل الموردين عند هذا الوزن
    sens_results.append(np.dot(scores_data, temp_w))

sens_results = np.array(sens_results)  # الشكل يصبح (20, n_suppliers)

# الرسم البياني
fig, ax = plt.subplots(figsize=(10, 5))

for i in range(n_suppliers):
    ax.plot(variation * 100, sens_results[:, i], label=supplier_names[i], linewidth=2)

ax.set_xlabel(f"Weight of {selected_criterion} (%)")
ax.set_ylabel("Score")
ax.set_title("Sensitivity Analysis Graph")
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(True, linestyle='--', alpha=0.7)

st.pyplot(fig)


st.write("---")

st.caption("Developed for Strategic Sourcing and Procurement Analysis.")
st.caption("Developed by Zennani Amran / Zerguine Moussa.")

#.......PV.........
def generate_pv():
    pdf_path = "PV_Sonatrach.pdf"
    # هنا كود reportlab لإنشاء الملف

    elements.build(elements)
    return pdf_path


    elements = []

    # ===== Styles =====
    header_style = ParagraphStyle(fontSize=10, spaceAfter=4)
    title_style = ParagraphStyle(fontSize=14, alignment=1, spaceAfter=15)
    section_style = ParagraphStyle(fontSize=12, spaceBefore=10, spaceAfter=6)
    normal_style = ParagraphStyle(fontSize=11, spaceAfter=6)

    # ===== Logo =====
    logo = Image("logo_sonatrach.png", width=4*cm, height=2*cm)
    elements.append(logo)

    # ===== Header =====
    elements.append(Paragraph("<b>SONATRACH</b>", header_style))
    elements.append(Paragraph(f"Direction : {data['direction']}", header_style))
    elements.append(Paragraph(f"Département : {data['departement']}", header_style))
    elements.append(Paragraph(f"Service : {data['service']}", header_style))

    elements.append(Spacer(1, 12))

    # ===== Title =====
    elements.append(Paragraph(
        "<b>PROCÈS-VERBAL D'ÉVALUATION DES OFFRES</b>",
        title_style
    ))

    # ===== Info Bloc =====
    elements.append(Paragraph(f"<b>Référence :</b> {data['ref']}", normal_style))
    elements.append(Paragraph(f"<b>AO N° :</b> {data['ao']}", normal_style))
    elements.append(Paragraph(f"<b>Objet :</b> {data['objet']}", normal_style))
    elements.append(Paragraph(f"<b>Date :</b> {data['date']}", normal_style))
    elements.append(Paragraph(f"<b>Lieu :</b> {data['lieu']}", normal_style))

    elements.append(Spacer(1, 10))

    # ===== Function to add styled table =====
    def add_table(title, df):
        elements.append(Paragraph(f"<b>{title}</b>", section_style))
        table_data = [df.columns.tolist()] + df.values.tolist()
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.7, None),
            ('BACKGROUND', (0,0), (-1,0), None),
            ('ALIGN', (1,1), (-1,-1), 'CENTER'),
            ('FONT', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 8))

    # ===== Tables =====
    add_table("1. Résultats des poids des critères (AHP)", df_weights)
    add_table("2. Tableau d’évaluation des fournisseurs", df_scores)
    add_table("3. Résultats finaux et classement", df_results)
    add_table("4. Membres de la commission présents", df_members)

    # ===== Conclusion =====
    elements.append(Paragraph("<b>Conclusion :</b>", section_style))
    elements.append(Paragraph(
        "Après analyse détaillée des offres selon la méthodologie adoptée, "
        "la commission recommande l’attribution provisoire du marché au soumissionnaire "
        "ayant obtenu la meilleure note globale.",
        normal_style
    ))

    elements.append(Spacer(1, 15))

    # ===== Signatures =====
    elements.append(Paragraph("<b>Signatures :</b>", section_style))
    elements.append(Paragraph("Président de la commission : ____________________________", normal_style))
    elements.append(Paragraph("Membre 1 : ____________________________", normal_style))
    elements.append(Paragraph("Membre 2 : ____________________________", normal_style))

    doc.build(elements)

    return file_path

df_weights = df_scoring

df_scores = pd.DataFrame(
    scores_data,
    columns=criteria_names,
    index=supplier_names
).reset_index().rename(columns={"index": "Supplier"})

df_results = df_ahp

df_members = pd.DataFrame({
    "Nom": ["Zennani Amran", "Zerguine Moussa", "Membre 3"],
    "Fonction": ["Président", "Membre", "Membre"]
})
if st.button("📄 Generate PV"):
    pdf_path = generate_pv()

    with open(pdf_path = "PV_Sonatrach.pdf") as f:
        st.download_button(
            "⬇️ Download Official PV",
            data=f,
            file_name="PV_Sonatrach.pdf",
            mime="application/pdf"
        )


























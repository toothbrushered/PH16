"""
Automated Dental Screening — Minimal Demo
===========================================
3 image inputs -> referral note output. Nothing else.

RUN:
    pip install streamlit pillow --break-system-packages
    streamlit run app_minimal.py

WHERE TO PLUG IN YOUR CODE: search "# >>> PLUG IN"
"""

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Dental Screening", page_icon="🦷")

st.title("Dental Screening")

# ── MODEL LOADING ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """
    # >>> PLUG IN: load your 3 YOLOv8 models once here.
        from ultralytics import YOLO
        return YOLO("malocclusion.pt"), YOLO("gingivitis.pt"), YOLO("caries.pt")
    """
    return None, None, None

malocclusion_model, gingivitis_model, caries_model = load_models()

# ── PIPELINE FUNCTION ─────────────────────────────────────────────────
def generate_referral_note(malocclusion_img, gingivitis_img, caries_img):
    """
    # >>> PLUG IN: run your 3 models + NDCS scoring + RAG/phi-3 note
    generation here, and return the final note text.

    e.g.
        m_result = malocclusion_model(malocclusion_img)
        g_result = gingivitis_model(gingivitis_img)
        c_result = caries_model(caries_img)
        note = your_existing_pipeline(m_result, g_result, c_result)
        return note
    """
    return (
        "PLACEHOLDER NOTE — plug in your pipeline in generate_referral_note().\n\n"
        "Mild tooth torsion noted. Moderate gingival inflammation observed in "
        "the anterior maxillary region. Two teeth show signs consistent with "
        "caries. Referral to a general dentist recommended within 4-6 weeks."
    )

# ── UI ───────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    malocclusion_file = st.file_uploader("Malocclusion", type=["jpg", "jpeg", "png"])
with c2:
    gingivitis_file = st.file_uploader("Gingivitis", type=["jpg", "jpeg", "png"])
with c3:
    caries_file = st.file_uploader("Caries", type=["jpg", "jpeg", "png"])

ready = malocclusion_file and gingivitis_file and caries_file

if st.button("Generate Referral Note", disabled=not ready):
    with st.spinner("Analysing..."):
        note = generate_referral_note(
            Image.open(malocclusion_file).convert("RGB"),
            Image.open(gingivitis_file).convert("RGB"),
            Image.open(caries_file).convert("RGB"),
        )
    st.divider()
    st.subheader("Referral Note")
    st.write(note)
    st.download_button("Download", note, file_name="referral_note.txt")
elif not ready:
    st.caption("Upload all 3 images to continue.")
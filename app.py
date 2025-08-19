import streamlit as st
import fitz  # PyMuPDF (to read PDFs)
import docx
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- Helper Functions --------------------
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def get_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    cos_sim = cosine_similarity(vectors)
    return cos_sim[0][1] * 100

def keyword_match(resume_text, jd_text):
    resume_words = set(resume_text.split())
    jd_words = set(jd_text.split())
    matched = resume_words.intersection(jd_words)
    missing = jd_words - resume_words
    coverage = (len(matched) / len(jd_words)) * 100 if jd_words else 0
    return matched, missing, coverage

def analyze_resume_vs_jd(resume_clean, jd_clean):
    semantic_similarity = get_cosine_similarity(resume_clean, jd_clean)
    matched, missing, keyword_coverage = keyword_match(resume_clean, jd_clean)
    overall_match = (semantic_similarity + keyword_coverage) / 2
    return overall_match, semantic_similarity, keyword_coverage, matched, missing

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="AI Resume ↔ JD Matcher", layout="wide")

st.title("📄 AI Resume ↔ Job Description Matcher")

# Upload Resume
st.subheader("1. Upload Your Resume")
resume_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])

# Enter Multiple Job Descriptions
st.subheader("2. Paste Job Descriptions")
jd1 = st.text_area("Job Description 1")
jd2 = st.text_area("Job Description 2 (Optional)", "")
jd3 = st.text_area("Job Description 3 (Optional)", "")

# Analyze Button
if st.button("🔎 Analyze"):
    if resume_file and jd1:
        # Extract Resume Text
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(resume_file)
        else:
            resume_text = resume_file.read().decode("utf-8")

        resume_clean = clean_text(resume_text)

        # Store JDs
        jd_inputs = [("JD #1", jd1), ("JD #2", jd2), ("JD #3", jd3)]
        jd_inputs = [(title, jd) for title, jd in jd_inputs if jd.strip()]

        # Results for Each JD
        for title, jd in jd_inputs:
            st.subheader(f"📌 Results for {title}")

            jd_clean = clean_text(jd)
            overall_match, semantic_similarity, keyword_coverage, matched, missing = analyze_resume_vs_jd(resume_clean, jd_clean)

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Overall Match %", f"{overall_match:.2f}%", "Higher is better")
            with col2:
                st.metric("🔎 Semantic Similarity", f"{semantic_similarity:.2f}%", "Meaning overlap")
            with col3:
                st.metric("📝 Keyword Coverage", f"{keyword_coverage:.2f}%", "Exact skill match")

            # Matched & Missing
            st.write("✅ **Matched Skills/Keywords:**", ", ".join(list(matched)[:15]) or "None")
            st.write("❌ **Missing Skills/Keywords:**", ", ".join(list(missing)[:15]) or "None")

            # Pie Chart (smaller size)
            fig, ax = plt.subplots(figsize=(3,3))  # smaller pie chart
            ax.pie(
                [overall_match, 100 - overall_match],
                labels=["Match", "Gap"],
                autopct='%1.1f%%',
                startangle=90,
                colors=["#4CAF50", "#FF5252"]
            )
            ax.axis('equal')
            st.pyplot(fig)
    else:
        st.warning("⚠️ Please upload a resume and enter at least Job Description 1.")

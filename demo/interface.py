import io
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# ─── CONFIG ─────────────────────────────────────────────────────────────────
DATA_CSV       = "data/main-data/synthetic-resumes.csv"
EMBED_MODEL    = "all-MiniLM-L6-v2"
GEN_MODELS     = ["distilgpt2", "gpt2-medium", "google/flan-t5-large"]
DEFAULT_GEN    = "google/flan-t5-large"
TOP_K          = 5
MAX_NEW_TOKENS = 150
TEMPERATURE    = 0.3
SIM_THRESHOLD  = 0.2  # minimum cosine similarity to accept

# ─── STYLES ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="✨ SmartCandidate Analyzer", layout="wide")
st.markdown("""
    <style>
      .stButton>button { border-radius: 8px; padding: 0.6em 1.2em; }
      .stMetric > div { background: #ffffffcc; border-radius: 10px; }
      .css-1d391kg { padding: 1rem 2rem; }
    </style>
""", unsafe_allow_html=True)

# ─── HEADER & MESSAGES ───────────────────────────────────────────────────────
st.image(
    "https://raw.githubusercontent.com/AbBasitMSU/SmartCandidate-Analyzer-RAG-Based-Resume-Screening/main/logo.png",
    width=120
)
st.title("📄 SmartCandidate Analyzer")
st.caption("RAG‑powered resume screening, now with a shiny new UI!")

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
mode         = st.sidebar.radio("🔍 Retrieval Mode", ["Generic RAG","Fusion RAG"])
model_choice = st.sidebar.selectbox("🤖 Answer Model", GEN_MODELS, index=GEN_MODELS.index(DEFAULT_GEN))
uploaded_pdf = st.sidebar.file_uploader("📄 Upload your resume (PDF/TXT)", type=["pdf","txt"])
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Instructions**  
1. Enter a Job Description.  
2. (Optional) Upload your resume.  
3. Hit **Run** to see match score and top resumes!
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by [AbBasitMSU](https://github.com/AbBasitMSU)")

# ─── CACHES ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_data(path):
    df = pd.read_csv(path)
    df["ID"] = df["ID"].astype(str)
    embedder = SentenceTransformer(EMBED_MODEL)
    embs = embedder.encode(df["Resume"].tolist(), convert_to_numpy=True, show_progress_bar=False)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return df, embedder, index

@st.cache_resource
def get_generator(model_name):
    task = "text2text-generation" if "flan" in model_name else "text-generation"
    gen = pipeline(
        task,
        model=model_name,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        truncation=True,
        pad_token_id=None,
        device=-1
    )
    gen.tokenizer.pad_token_id = gen.tokenizer.eos_token_id
    return gen

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def extract_text(f) -> str:
    if f.type == "application/pdf":
        r = PdfReader(io.BytesIO(f.read()))
        return "\n\n".join(page.extract_text() or "" for page in r.pages)
    return f.read().decode("utf-8")

def compute_match_score(jd, resume, emb):
    vecs = emb.encode([jd, resume], convert_to_numpy=True)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return float(cosine_similarity([vecs[0]], [vecs[1]])[0,0])

def retrieve_results(jd, mode, emb, index):
    # embed + normalize
    qv = emb.encode([jd], convert_to_numpy=True)
    qv /= np.linalg.norm(qv, keepdims=True)
    if mode == "Generic RAG":
        scores, ids = index.search(qv, TOP_K)
        return list(zip(ids[0].tolist(), scores[0].tolist()))
    # Fusion RAG
    parts = [jd] + jd.split('.')[:4]
    agg = {}
    for chunk in parts:
        cv = emb.encode([chunk], convert_to_numpy=True)
        cv /= np.linalg.norm(cv, keepdims=True)
        sc, ids = index.search(cv, TOP_K)
        for rank, idx in enumerate(ids[0]):
            agg[idx] = agg.get(idx, 0.0) + 1.0/(rank+1)
    fused = sorted(agg.items(), key=lambda x: -x[1])[:TOP_K]
    return fused

def generate_recommendation(jd, ids, df, gen):
    context = "\n\n".join(
        f"ID {df.iloc[i]['ID']}:\n{df.iloc[i]['Resume'][:200]}…"
        for i in ids
    )
    prompt = f"""You are a hiring consultant.
Recommend the single best candidate by Applicant ID, with a 2-3 sentence explanation.

Job Description:
{jd}

Resumes:
{context}

Recommendation:"""
    out = gen(prompt)[0]["generated_text"]
    return out.replace(prompt, "").strip()

# ─── LOAD & RUN ──────────────────────────────────────────────────────────────
df, embedder, idx = load_data(DATA_CSV)
generator = get_generator(model_choice)

st.subheader("📝 Job Description")
jd = st.text_area("", height=120)

user_text = None
if uploaded_pdf:
    user_text = extract_text(uploaded_pdf)
    st.subheader("📑 Your Resume Preview")
    st.write(user_text[:200] + "…")

if st.button("🚀 Run"):
    # 1) sanity: JD length
    if len(jd.split()) < 5:
        st.error("Please enter a more detailed job description (at least 5 words).")
        st.stop()

    # 2) match score
    col1, col2, col3 = st.columns(3)
    if user_text:
        score = compute_match_score(jd, user_text, embedder)
        col1.metric("Your Resume Match", f"{score*100:.1f}%")
    col2.metric("Mode", mode)
    col3.metric("Top K", TOP_K)

    # 3) retrieval + threshold check
    results = retrieve_results(jd, mode, embedder, idx)
    if not results or results[0][1] < SIM_THRESHOLD:
        st.warning("No relevant resumes found for that job description.")
        st.stop()

    # 4) display in tabs
    tabs = st.tabs(["🔍 Top Resumes","🤖 Recommendation"])
    with tabs[0]:
        st.subheader("Top Existing Resumes")
        for rank, (i, sc) in enumerate(results, start=1):
            st.markdown(f"**{rank}. Applicant ID {df.iloc[i]['ID']}** — Score {sc:.3f}")
            st.write(df.iloc[i]["Resume"][:200] + "…")

    with tabs[1]:
        rec = generate_recommendation(jd, [i for i,_ in results], df, generator)
        st.subheader("Recommendation")
        st.write(rec)
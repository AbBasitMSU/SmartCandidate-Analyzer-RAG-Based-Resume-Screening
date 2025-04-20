import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_CSV       = "data/main-data/synthetic-resumes.csv"
EMBED_MODEL    = "all-MiniLM-L6-v2"
GEN_MODELS     = ["distilgpt2", "gpt2-medium", "google/flan-t5-large"]
DEFAULT_GEN    = "google/flan-t5-large"
TOP_K          = 5
MAX_NEW_TOKENS = 150
TEMPERATURE    = 0.3
SIM_THRESHOLD  = 0.2  # minimum cosine similarity

# ─── THEME & CSS ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SmartCandidate Analyzer", layout="wide")
st.markdown("""
  <style>
    .main-title { text-align: center; font-size: 2.5rem; margin: 0; }
    .sub-title  { text-align: center; color: #555; margin-top:0.2rem; margin-bottom:1rem; }
    .stButton>button { border-radius: 8px; padding: 0.6em 1.2em; }
    .stMetric > div { background: #ffffffcc; border-radius: 10px; }
    .css-1d391kg { padding: 1rem 2rem; }
  </style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
logo = "assets/logo.png"
if os.path.exists(logo):
    st.image(logo, width=120)
else:
    st.markdown("<h1 class='main-title'>📄</h1>", unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>SmartCandidate Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>RAG‑powered resume screening, now with a shiny new UI!</p>",
            unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
mode         = st.sidebar.radio("Retrieval Mode", ["Generic RAG", "Fusion RAG"])
model_choice = st.sidebar.selectbox("Answer Model", GEN_MODELS, index=GEN_MODELS.index(DEFAULT_GEN))
uploaded_pdf = st.sidebar.file_uploader("Upload your resume (PDF/TXT)", type=["pdf","txt"])

st.sidebar.markdown("---")
st.sidebar.header("Help")
help_option = st.sidebar.radio("", [ "Instructions", "Documentation"], index=0)
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
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return df, embedder, idx

@st.cache_resource
def get_generator(model_name):
    task = "text2text-generation" if "flan" in model_name else "text-generation"
    gen = pipeline(
        task, model=model_name,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        truncation=True,
        pad_token_id=None,
        device=-1
    )
    gen.tokenizer.pad_token_id = gen.tokenizer.eos_token_id
    return gen

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def extract_text(f) -> str:
    if f.type == "application/pdf":
        reader = PdfReader(io.BytesIO(f.read()))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return f.read().decode("utf-8")

def compute_match_score(jd, resume, emb):
    vecs = emb.encode([jd, resume], convert_to_numpy=True)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return float(cosine_similarity([vecs[0]], [vecs[1]])[0,0])

def retrieve_results(jd, mode, emb, idx):
    qv = emb.encode([jd], convert_to_numpy=True)
    qv /= np.linalg.norm(qv, keepdims=True)
    if mode == "Generic RAG":
        sc, ids = idx.search(qv, TOP_K)
        return list(zip(ids[0].tolist(), sc[0].tolist()))
    # Fusion: split into sub-queries and fuse
    parts = [jd] + jd.split('.')[:4]
    agg = {}
    for c in parts:
        cv = emb.encode([c], convert_to_numpy=True)
        cv /= np.linalg.norm(cv, keepdims=True)
        sc, ids = idx.search(cv, TOP_K)
        for rank, i in enumerate(ids[0]):
            agg[i] = agg.get(i, 0.0) + 1.0/(rank+1)
    return sorted(agg.items(), key=lambda x: -x[1])[:TOP_K]

def generate_recommendation(jd, ids, df, gen):
    ctx = "\n\n".join(f"ID {df.iloc[i]['ID']}:\n{df.iloc[i]['Resume'][:200]}…" for i in ids)
    prompt = f"""You are a hiring consultant.
Recommend the single best candidate by Applicant ID, with a 2–3 sentence explanation.

Job Description:
{jd}

Resumes:
{ctx}

Recommendation:"""
    out = gen(prompt)[0]["generated_text"]
    return out.replace(prompt, "").strip()

# ─── LOAD DATA & GENERATOR ────────────────────────────────────────────────────
df, embedder, idx = load_data(DATA_CSV)
generator = get_generator(model_choice)

# ─── HELP / DOC DISPLAY ───────────────────────────────────────────────────────
if help_option == "Instructions":
    st.header("🛠️ Instructions")
    st.markdown("""
1. **Enter** a clear Job Description (≥ 5 words) in the main panel.  
2. **(Optional)** Upload a resume PDF or TXT to compute a match score.  
3. Click **Run** to retrieve, rank, and get a recommendation.  
4. Switch to **Book Interview** to send invites to top candidates.
""")
    st.stop()
elif help_option == "Documentation":
    st.header("📄 Documentation")
    st.markdown("""
**SmartCandidate Analyzer** is a Retrieval‑Augmented Generation demo for resume screening.

- **Embeddings**: sentence-transformers/all‑MiniLM‑L6‑v2  
- **Retriever**: FAISS (Inner Product)  
- **Generator**: Local HF models (e.g. google/flan‑t5‑large)  

Full details and usage examples will be added here soon.
""")
    st.stop()

# ─── RUN & BOOK INTERVIEW TABS ────────────────────────────────────────────────
tab_run, tab_book = st.tabs(["🚀 Run", "📅 Book Interview"])

with tab_run:
    st.subheader("📝 Job Description")
    jd = st.text_area("", height=120)

    user_text = None
    if uploaded_pdf:
        user_text = extract_text(uploaded_pdf)
        st.subheader("📑 Your Resume Preview")
        st.write(user_text[:200] + "…")

    if st.button("Run"):
        if len(jd.split()) < 5:
            st.error("Please enter at least 5 words.")
            st.stop()

        col1, col2, col3 = st.columns(3)
        if user_text:
            score = compute_match_score(jd, user_text, embedder)
            col1.metric("Your Resume Match", f"{score*100:.1f}%")
        col2.metric("Mode", mode)
        col3.metric("Top K", TOP_K)

        results = retrieve_results(jd, mode, embedder, idx)
        if not results or results[0][1] < SIM_THRESHOLD:
            st.warning("No relevant resumes found.")
            st.stop()

        # save for booking
        st.session_state.last_results = results
        st.session_state.last_jd = jd

        st.subheader("🔍 Top Existing Resumes")
        for rank, (i, sc) in enumerate(results, start=1):
            st.markdown(f"**{rank}. Applicant ID {df.iloc[i]['ID']}** — Score {sc:.3f}")
            st.write(df.iloc[i]["Resume"][:200] + "…")

        rec = generate_recommendation(jd, [i for i,_ in results], df, generator)
        st.subheader("🤖 Recommendation")
        st.write(rec)

with tab_book:
    st.subheader("📅 Book Interview")
    if "last_results" not in st.session_state:
        st.info("Run a JD first to select candidates.")
    else:
        ids = [i for i,_ in st.session_state.last_results]
        candidates = [f"Applicant ID {df.iloc[i]['ID']}" for i in ids]
        selected = st.multiselect("Select candidates", candidates)
        interview_date = st.date_input("Interview Date", value=datetime.today())
        interview_time = st.time_input("Interview Time", value=datetime.now().time())
        email_body = st.text_area(
            "Email Body",
            value=f"Dear Candidate,\n\nWe invite you for an interview on {interview_date} at {interview_time}.\n\nBest regards,"
        )
        if st.button("Send Invitations"):
            for cand in selected:
                st.success(f"Invitation sent to {cand}.")
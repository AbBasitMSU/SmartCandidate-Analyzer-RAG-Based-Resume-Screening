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
GEN_MODELS     = ["distilgpt2","gpt2-medium","google/flan-t5-large"]
DEFAULT_GEN    = "google/flan-t5-large"
TOP_K          = 5
MAX_NEW_TOKENS = 150
TEMPERATURE    = 0.3  # lower temp for less repetition

# ─── STYLES ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="✨ SmartCandidate Analyzer", layout="wide")
st.markdown("""
    <style>
      .stButton>button { border-radius: 8px; padding: 0.6em 1.2em; }
      .stMetric > div { background: #ffffffcc; border-radius: 10px; }
      .css-1d391kg { padding: 1rem 2rem; }
    </style>
""", unsafe_allow_html=True)

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.image("https://raw.githubusercontent.com/AbBasitMSU/SmartCandidate-Analyzer-RAG-Based-Resume-Screening/main/logo.png",
         width=120)
st.title("📄 SmartCandidate Analyzer")
st.caption("RAG‑powered resume screening, now with a shiny new UI!")

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
mode         = st.sidebar.radio("🔍 Retrieval Mode", ["Generic RAG","Fusion RAG"])
model_choice = st.sidebar.selectbox("🤖 Answer Model", GEN_MODELS, index=GEN_MODELS.index(DEFAULT_GEN))
uploaded_pdf = st.sidebar.file_uploader("📄 Upload your resume (PDF/TXT)", type=["pdf","txt"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Instructions**:\n\n1. Enter a Job Description.  \n2. (Optional) Upload your resume.  \n3. Hit **Run**!")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by [AbBasitMSU](https://github.com/AbBasitMSU)")

# ─── CACHES ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_data(path):
    df = pd.read_csv(path)
    df["ID"] = df["ID"].astype(str)
    embedder = SentenceTransformer(EMBED_MODEL)
    embs = embedder.encode(df["Resume"].tolist(), convert_to_numpy=True)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    return df, embedder, idx

@st.cache_resource
def get_generator(model_name):
    # use text2text for flan models, text-generation otherwise
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
    if f.type=="application/pdf":
        r = PdfReader(io.BytesIO(f.read()))
        return "\n\n".join(p.extract_text() or "" for p in r.pages)
    return f.read().decode("utf-8")

def match_score(jd, resume, emb):
    v = emb.encode([jd,resume], convert_to_numpy=True)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return float(cosine_similarity([v[0]],[v[1]])[0,0])

def retrieve(jd, mode, emb, idx):
    qv = emb.encode([jd], convert_to_numpy=True)
    qv /= np.linalg.norm(qv, keepdims=True)
    if mode=="Generic RAG":
        _, ids = idx.search(qv, TOP_K)
        return ids[0].tolist()
    # Fusion
    parts = jd.split('.')[:4]
    scores={}
    for chunk in [jd]+parts:
        cv=emb.encode([chunk],convert_to_numpy=True)
        cv/=np.linalg.norm(cv,keepdims=True)
        _, ids = idx.search(cv, TOP_K)
        for r,i in enumerate(ids[0]):
            scores[i]=scores.get(i,0)+1/(r+1)
    top = sorted(scores.items(), key=lambda x:-x[1])[:TOP_K]
    return [i for i,_ in top]

def recommend(jd, ids, df, gen):
    ctx="\n\n".join(f"ID {df.iloc[i]['ID']}:\n{df.iloc[i]['Resume'][:200]}…" for i in ids)
    prompt=f"""You are a hiring consultant.  
Given the job description and these top resumes, **recommend the single best candidate** by ID, and explain in 2–3 sentences only.

Job Description:
{jd}

Resumes:
{ctx}

Recommendation:"""
    out=gen(prompt)[0]["generated_text"]
    return out.replace(prompt,"").strip()

# ─── LOAD & RUN ──────────────────────────────────────────────────────────────
df, embedder, idx = load_data(DATA_CSV)
generator = get_generator(model_choice)

st.subheader("📝 Job Description")
jd = st.text_area("",height=120)

user_text=None
if uploaded_pdf:
    user_text=extract_text(uploaded_pdf)
    st.subheader("📑 Your Resume Preview")
    st.write(user_text[:200]+"…")

if st.button("🚀 Run"):
    if not jd:
        st.error("Please enter a job description.")
        st.stop()

    # two‑column metrics
    col1,col2,col3 = st.columns(3)
    if user_text:
        score=match_score(jd,user_text,embedder)
        col1.metric("Your Resume Match",f"{score*100:.1f}%")
    col2.metric("Mode", mode)
    col3.metric("Top K", TOP_K)

    # show results in tabs
    tabs = st.tabs(["🔍 Top Resumes","🤖 Recommendation"])
    with tabs[0]:
        ids=retrieve(jd,mode,embedder,idx)
        for rank,i in enumerate(ids,1):
            st.markdown(f"**{rank}. Applicant ID {df.iloc[i]['ID']}** — Score")
            st.write(df.iloc[i]["Resume"][:200]+"…")
    with tabs[1]:
        rec = recommend(jd,ids,df,generator)
        st.write(rec)
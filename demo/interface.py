import os, io
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
TOP_K          = 5
MAX_NEW_TOKENS = 150
TEMPERATURE    = 0.3
SIM_THRESHOLD  = 0.2

# ─── PAGE SETUP & CSS ─────────────────────────────────────────────────────────
st.set_page_config(page_title="SmartCandidate Analyzer", layout="wide")
st.markdown("""
  <style>
    .main-title { text-align: center; font-size: 2.5rem; margin: 0; }
    .sub-title  { text-align: center; color: #555; margin-top:0.2rem; margin-bottom:1rem; }
    .stButton>button { border-radius: 8px; padding: 0.6em 1.2em; }
  </style>
""", unsafe_allow_html=True)

# ─── HELPERS & CACHES ──────────────────────────────────────────────────────────
@st.cache_resource
def load_data(path):
    df = pd.read_csv(path); df["ID"] = df["ID"].astype(str)
    embedder = SentenceTransformer(EMBED_MODEL)
    embs = embedder.encode(df["Resume"].tolist(), convert_to_numpy=True)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
    return df, embedder, idx

@st.cache_resource
def get_generator(model_name):
    task = "text2text-generation" if "flan" in model_name else "text-generation"
    gen = pipeline(task, model=model_name,
                   max_new_tokens=MAX_NEW_TOKENS,
                   temperature=TEMPERATURE,
                   truncation=True,
                   pad_token_id=None,
                   device=-1)
    gen.tokenizer.pad_token_id = gen.tokenizer.eos_token_id
    return gen

def extract_text(f):
    if f.type == "application/pdf":
        reader = PdfReader(io.BytesIO(f.read()))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    return f.read().decode("utf-8")

def compute_match_score(jd, resume, emb):
    vecs = emb.encode([jd, resume], convert_to_numpy=True)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return float(cosine_similarity([vecs[0]], [vecs[1]])[0,0])

def retrieve_results(jd, mode, emb, idx):
    qv = emb.encode([jd], convert_to_numpy=True); qv /= np.linalg.norm(qv, keepdims=True)
    if mode=="Generic RAG":
        sc, ids = idx.search(qv, TOP_K)
        return list(zip(ids[0].tolist(), sc[0].tolist()))
    # Fusion
    parts, agg = [jd]+jd.split('.')[:4], {}
    for p in parts:
        cv = emb.encode([p], convert_to_numpy=True); cv /= np.linalg.norm(cv, keepdims=True)
        sc, ids = idx.search(cv, TOP_K)
        for rank,i in enumerate(ids[0]):
            agg[i] = agg.get(i,0) + 1/(rank+1)
    return sorted(agg.items(), key=lambda x:-x[1])[:TOP_K]

def generate_recommendation(jd, ids, df, gen):
    ctx = "\n\n".join(f"ID {df.iloc[i]['ID']}:\n{df.iloc[i]['Resume'][:200]}…"
                      for i in ids)
    prompt = f"""You are a hiring consultant.
Recommend the single best candidate by Applicant ID, with a 2–3 sentence explanation.

Job Description:
{jd}

Resumes:
{ctx}

Recommendation:"""
    out = gen(prompt)[0]["generated_text"]
    return out.replace(prompt, "").strip()

# ─── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select a Section", ["Home", "Instructions", "SmartCandidate tool", "Documentation"])
st.sidebar.markdown("---")
st.sidebar.write("Built by AbBasitMSU")

# ─── HOME ──────────────────────────────────────────────────────────────────────
if section == "Home":
    st.markdown("<h1 class='main-title'>SmartCandidate Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>RAG‑powered resume screening, now with a shiny new UI!</p>", unsafe_allow_html=True)

# ─── INSTRUCTIONS ─────────────────────────────────────────────────────────────
elif section == "Instructions":
    st.header("🛠 How to use")
    st.markdown("""
1. Enter a clear JD (≥5 words).  
2. (Optional) Upload your resume to see match %.  
3. Click **Run** to retrieve + rank top candidates.  
4. Go to **Book Interview** to schedule invites.
""")

# ─── SMARTCANDIDATE TOOL ──────────────────────────────────────────────────────
elif section == "SmartCandidate tool":
    # only now do we load data & LLM
    df, embedder, idx = load_data(DATA_CSV)

    # move RAG info + controls into sidebar
    st.sidebar.markdown("### Retrieval Mode")
    mode = st.sidebar.radio("", ["Generic RAG", "Fusion RAG"])
    st.sidebar.markdown("### Answer Model")
    model_choice = st.sidebar.selectbox("", GEN_MODELS, index=GEN_MODELS.index("google/flan-t5-large"))
    uploaded = st.sidebar.file_uploader("Upload your resume (PDF/TXT)", type=["pdf","txt"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("Fusion RAG splits the JD into sub‑queries and fuses results.")

    # main panel: JD, Run, Results, Book Interview
    st.header("🚀 SmartCandidate Tool")
    jd = st.text_area("📄 Job Description", height=150)

    if st.button("Run"):
        if len(jd.split()) < 5:
            st.error("Please enter at least 5 words.")
            st.stop()

        user_text = extract_text(uploaded) if uploaded else None
        col1, col2, col3 = st.columns(3)
        if user_text:
            score = compute_match_score(jd, user_text, embedder)
            col1.metric("Your Resume Match", f"{score*100:.1f}%")
        col2.metric("Mode", mode); col3.metric("Top K", TOP_K)

        results = retrieve_results(jd, mode, embedder, idx)
        if not results or results[0][1] < SIM_THRESHOLD:
            st.warning("No relevant resumes found."); st.stop()

        st.session_state.last_results = results
        st.session_state.last_jd      = jd

        st.subheader("🔍 Top Candidates")
        for r,(i,sc) in enumerate(results,1):
            st.markdown(f"**{r}. Applicant ID {df.iloc[i]['ID']}**  —  Score {sc:.3f}")
            st.write(df.iloc[i]["Resume"][:200] + "…")

        gen = get_generator(model_choice)
        rec = generate_recommendation(jd, [i for i,_ in results], df, gen)
        st.subheader("🤖 Recommendation"); st.write(rec)

    st.markdown("---")
    st.subheader("📅 Book Interview")
    if "last_results" not in st.session_state:
        st.info("Run a JD first to select candidates.")
    else:
        ids = [i for i,_ in st.session_state.last_results]
        labels = [f"Applicant ID {df.iloc[i]['ID']}" for i in ids]
        chosen = st.multiselect("Select candidates", labels)
        d = st.date_input("Interview Date", datetime.today())
        t = st.time_input("Interview Time", datetime.now().time())
        body = st.text_area("Email Body",
            value=f"Dear Candidate,\n\nWe invite you on {d} at {t}.\n")
        if st.button("Send Invitations"):
            for c in chosen:
                st.success(f"Invitation sent to {c}.")

# ─── DOCUMENTATION ────────────────────────────────────────────────────────────
elif section == "Documentation":
    st.header("📄 Documentation")
    st.markdown("""
*(Paste your full project documentation here…)*
    """)
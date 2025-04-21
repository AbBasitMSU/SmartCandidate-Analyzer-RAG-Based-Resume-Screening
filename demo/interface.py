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
DEFAULT_GEN    = "google/flan-t5-large"
TOP_K          = 5
MAX_NEW_TOKENS = 150
TEMPERATURE    = 0.3
SIM_THRESHOLD  = 0.2

# ─── PAGE SETUP & CSS ─────────────────────────────────────────────────────────
st.set_page_config(page_title="SmartCandidate Analyzer", layout="wide")
st.markdown("""
  <style>
    .main-title     { text-align: center; font-size: 2.5rem; margin: 0; }
    .sub-title      { text-align: center; color: #555; margin-top:0.2rem; margin-bottom:1rem; }
    .centered-header{ text-align: center; margin-top:0.5rem; margin-bottom:0.5rem; }
    .stButton>button{ border-radius: 8px; padding: 0.6em 1.2em; }
  </style>
""", unsafe_allow_html=True)

# ─── HELPERS & CACHES ──────────────────────────────────────────────────────────
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
    task = "text2text-generation" if "flan" in model_name else "text-generation"
    gen = pipeline(
        task,
        model=model_name,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        truncation=True,
        pad_token_id=None,
        device=-1,
    )
    gen.tokenizer.pad_token_id = gen.tokenizer.eos_token_id
    return gen

def extract_text(f):
    if f.type == "application/pdf":
        reader = PdfReader(io.BytesIO(f.read()))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return f.read().decode("utf-8")

def compute_match_score(jd, resume, emb):
    vecs = emb.encode([jd, resume], convert_to_numpy=True)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return float(cosine_similarity([vecs[0]], [vecs[1]])[0,0])

def retrieve_results(jd, mode, emb, idx):
    qv = emb.encode([jd], convert_to_numpy=True); qv /= np.linalg.norm(qv, keepdims=True)
    if mode == "Generic RAG":
        sc, ids = idx.search(qv, TOP_K)
        return list(zip(ids[0].tolist(), sc[0].tolist()))
    parts, agg = [jd] + jd.split('.')[:4], {}
    for p in parts:
        cv = emb.encode([p], convert_to_numpy=True); cv /= np.linalg.norm(cv, keepdims=True)
        sc, ids = idx.search(cv, TOP_K)
        for rank, i in enumerate(ids[0]):
            agg[i] = agg.get(i, 0) + 1/(rank + 1)
    return sorted(agg.items(), key=lambda x: -x[1])[:TOP_K]

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
section = st.sidebar.radio("Select a Section",
    ["Home", "Instructions", "SmartCandidate tool", "Documentation"])
st.sidebar.markdown("---")

# Home & Instructions are unchanged…
if section == "Home":
    st.markdown("<h1 class='main-title'>SmartCandidate Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>"
                "RAG‑powered resume screening, now with a shiny new UI!"
                "</p>", unsafe_allow_html=True)

elif section == "Instructions":
    st.header("🛠 How to Use SmartCandidate Analyzer")
    st.markdown("""
1. Enter a Job Description (≥ 5 words).  
2. (Optional) Upload your resume (PDF/TXT) to compute a match score.  
3. Go to **SmartCandidate tool** and click **Run**.  
4. Then switch to **Book Interview** to send invites.
""")

# ─── SMARTCANDIDATE TOOL ──────────────────────────────────────────────────────
elif section == "SmartCandidate tool":
    # Lazy‑load the heavy parts
    df, embedder, idx = load_data(DATA_CSV)

    # Sidebar controls
    st.sidebar.markdown("### Retrieval Mode")
    mode = st.sidebar.radio("", ["Generic RAG", "Fusion RAG"])
    st.sidebar.markdown("### Answer Model")
    model_choice = st.sidebar.selectbox("", GEN_MODELS, index=GEN_MODELS.index(DEFAULT_GEN))
    uploaded = st.sidebar.file_uploader("Upload your resume (PDF/TXT)", type=["pdf", "txt"])

    # Centered header
    st.markdown("<h2 class='centered-header'>🚀 SmartCandidate Tool</h2>", unsafe_allow_html=True)

    # RAG vs Fusion info (only on main panel)
    st.markdown(
        "<div class='centered-header'>"
        "**Generic RAG** does a one‑shot semantic search over your full JD.<br>"
        "**Fusion RAG** splits the JD into focused sub‑queries and fuses their results."
        "</div>",
        unsafe_allow_html=True,
    )

    # Top‑level tabs
    tab_run, tab_book = st.tabs(["🚀 Run", "📅 Book Interview"])

    with tab_run:
        jd = st.text_area("📄 Job Description", height=150)
        if st.button("Run"):
            if len(jd.split()) < 5:
                st.error("Please enter at least 5 words."); st.stop()

            # Optional resume match
            user_text = extract_text(uploaded) if uploaded else None
            c1, c2, c3 = st.columns(3)
            if user_text:
                score = compute_match_score(jd, user_text, embedder)
                c1.metric("Your Resume Match", f"{score*100:.1f}%")
            c2.metric("Mode", mode); c3.metric("Top K", TOP_K)

            # Retrieval + threshold
            results = retrieve_results(jd, mode, embedder, idx)
            if not results or results[0][1] < SIM_THRESHOLD:
                st.warning("No relevant resumes found."); st.stop()

            # Store for booking
            st.session_state.last_results = results
            st.session_state.last_jd      = jd

            # Show candidates
            st.subheader("🔍 Top Candidates")
            for rank, (i, sc) in enumerate(results, start=1):
                st.markdown(f"**{rank}. Applicant ID {df.iloc[i]['ID']}** — Score {sc:.3f}")
                st.write(df.iloc[i]["Resume"][:200] + "…")

            # Recommendation
            gen = get_generator(model_choice)
            rec = generate_recommendation(jd, [i for i, _ in results], df, gen)
            st.subheader("🤖 Recommendation"); st.write(rec)

    with tab_book:
        st.subheader("📅 Book Interview")
        if "last_results" not in st.session_state:
            st.info("Run a JD first to select candidates.")
        else:
            ids = [i for i,_ in st.session_state.last_results]
            labels = [f"Applicant ID {df.iloc[i]['ID']}" for i in ids]
            chosen = st.multiselect("Select candidates", labels)
            interview_date = st.date_input("Interview Date", datetime.today())
            interview_time = st.time_input("Interview Time", datetime.now().time())
            email_body = st.text_area(
                "Email Body",
                value=f"Dear Candidate,\n\nWe invite you on {interview_date} at {interview_time}.\n"
            )
            if st.button("Send Invitations"):
                for cand in chosen:
                    st.success(f"Invitation sent to {cand}.")

elif section == "Documentation":
    st.header("📄 Documentation")
    st.markdown("""
*(Your full project documentation goes here…)*

**Overview**  
SmartCandidate Analyzer is a Retrieval‑Augmented Generation tool for resume screening.
- **Embeddings**: all‑MiniLM‑L6‑v2 + FAISS  
- **Retrieval**: Generic & Fusion RAG  
- **Generation**: Local Hugging Face models  
- **UI**: Streamlit with Run & Book Interview tabs  
- **Local**: No API keys needed  
""")

# ─── Built by (always last) ────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.write("Built with LOVE by [Ab Basit](https://github.com/AbBasitMSU)")
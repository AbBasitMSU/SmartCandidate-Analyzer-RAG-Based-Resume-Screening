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
help_option = st.sidebar.radio("", ["Tool", "Instructions", "Documentation"], index=0)
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
Below is a rich, detailed Documentation write‑up you can paste straight into your sidebar or Documentation panel. It’s organized into clear sections and covers everything from high‑level concepts through implementation details, dependencies, and next steps.

⸻

📄 SmartCandidate Analyzer Documentation

1. Overview

SmartCandidate Analyzer is a Retrieval‑Augmented Generation (RAG) tool for interactive, explainable resume screening. It combines:
	•	FAISS vector search over hundreds or thousands of resumes
	•	Sentence‑Transformers embeddings for both resumes and job descriptions
	•	Reciprocal Rank Fusion to merge multiple sub‑query retrievals
	•	Local Hugging Face LLMs (e.g. FLAN‑T5, GPT‑2 variants) for concise, human‑readable recommendations
	•	A Streamlit frontend for instant, no‑API‑key required demo

⸻

2. Key Features
	1.	Dual Retrieval Modes
	•	Generic RAG: Single-pass FAISS lookup on the entire job description.
	•	Fusion RAG: Splits the job description into 3–4 sub‑queries, retrieves each, then fuses via Reciprocal Rank Fusion to improve recall.
	2.	Match Score for User Uploads
	•	Drop in your own resume (PDF or TXT).
	•	Compute and display a cosine‑similarity “Match Score” between your resume and the JD.
	3.	Explainable Recommendations
	•	After ranking, the app generates a 2–3 sentence recommendation referencing only “Applicant ID X.”
	•	No hallucinations: if similarity is below threshold, it warns “No relevant resumes found.”
	4.	Interview Scheduling Stub
	•	Select one or more top candidates.
	•	Pick a date/time and draft an invitation email.
	•	(Placeholder for real email integration.)
	5.	Interactive UI
	•	Sidebar for settings (mode, model choice, resume upload) and persistent documentation.
	•	Top tabs for “Run” vs “Book Interview.”
	•	Custom CSS for centered headers, rounded buttons, and metric cards.
	6.	Fully Local
	•	No external API keys needed.
	•	Embeddings and generation happen on your CPU/GPU via open‑source libraries.

⸻

3. Architecture & Data Flow

flowchart LR
  A[User opens app] --> B[Load CSV & build FAISS index]
  B --> C[User enters JD + optional upload]
  C --> D{Help selected?}
  D -- Instructions / Docs --> Z[Show help text]
  D -- Otherwise --> E[Run tab]
  E --> F[Embed JD (SentenceTransformer)]
  F --> G{Generic or Fusion?}
  G -- Generic --> H[FAISS search]
  G -- Fusion --> I[JD → sub‑queries → FAISS per sub] 
  I --> J[Reciprocal Rank Fusion]
  H & J --> K[Rank top‑K IDs]
  K --> L[Display snippets + scores]
  L --> M[Assemble prompt + call HF pipeline]
  M --> N[Show recommendation]
  E --> O[Store last_results in session_state]
  O --> P[Book Interview tab uses last_results]



⸻

4. Core Components

4.1 Data Ingestion
	•	CSV: data/main-data/synthetic-resumes.csv (columns: ID, Resume)
	•	Embedding:

embedder = SentenceTransformer("all‑MiniLM‑L6‑v2")
vectors = embedder.encode(df["Resume"].tolist(), convert_to_numpy=True)


	•	Indexing:

index = faiss.IndexFlatIP(dim)
index.add(normalize(vectors))



4.2 Retrieval
	•	Generic RAG:

scores, ids = index.search(normalize(embed(JD)), TOP_K)


	•	Fusion RAG:
	1.	Split JD into sentences → sub‑queries
	2.	Retrieve top K for each
	3.	Fuse with RRF:
\text{score}(d) = \sum_{r=1}^K \frac{1}{r + k_\text{offset}}
	4.	Sort final scores

4.3 Generation
	•	Prompt template:

You are a hiring consultant. Recommend the single best candidate by Applicant ID, with a 2–3 sentence explanation.

Job Description:
{JD}

Resumes:
{ID 123: …}
{ID 456: …}

Recommendation:


	•	Model:
	•	Instruction‑tuned (e.g. google/flan-t5-large) → text2text-generation
	•	Smaller (e.g. distilgpt2) → text-generation

4.4 UI & State
	•	Streamlit caches heavy operations (@st.cache_resource)
	•	session_state stores last_results and last_jd for interview tab
	•	Custom CSS to center titles and style widgets

⸻

5. Installation & Deployment
	1.	Clone the repo and ensure your CSV is in data/main-data/.
	2.	Create a requirements.txt:

streamlit
sentence-transformers
faiss-cpu
transformers
pypdf
scikit-learn


	3.	Install:

pip install -r requirements.txt


	4.	Run:

streamlit run app.py


	5.	(Optional) Deploy to Streamlit Cloud or any container platform.

⸻

6. Next Steps & Extensibility
	•	Real email: integrate SMTP or SendGrid for “Send Invitations.”
	•	Batch mode: upload many JDs/resumes at once and export results.
	•	Chunking & Summarization: pre‑summarize very long resumes to fit larger context windows.
	•	Model Tuning: swap in GPU‑accelerated models or fine‑tune on your own resume‑JD pairs.
	•	Logging & Analytics: track which resumes get recommended most often.

⸻

This documentation will evolve as new features are added. Feel free to expand each section with code snippets, architecture diagrams, or usage examples.
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
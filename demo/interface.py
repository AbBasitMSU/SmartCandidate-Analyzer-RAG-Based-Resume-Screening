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

  /* Full-page background with a 60% white overlay */
  [data-testid="stAppViewContainer"] {
    background:
      linear-gradient(rgba(255,255,255,0.6), rgba(255,255,255,0.6)),
      url("https://raw.githubusercontent.com/AbBasitMSU/SmartCandidate-Analyzer-RAG-Based-Resume-Screening/main/AC68D056-255F-46BC-86F4-BC3666BC9FBA.png")
      center/cover no-repeat;
  }
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
    st.markdown("<p class='sub-title'>"
                "RAG‑powered resume screening, now with a shiny new UI!"
                "</p>", unsafe_allow_html=True)

    # RAG vs Fusion info (only on main panel)
    st.markdown(
        "<div class='centered-header'>"
        "Please Choose 1 of 2 RAG on Side based below info<br>"
        "Generic RAG is like running one big search on your entire job description at once. It’s fast and gives you solid matches when the JD is straightforward.<br>"
        "Fusion RAG is like breaking your JD into several smaller, focused questions (e.g. “experience,” “skills,” “education”), searching for each of those, and then combining the best results. It takes a bit longer but catches more nuanced fits—great for complex roles with multiple requirements."
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
Below is a rich, detailed Documentation write‑up you can paste straight into your sidebar or Documentation panel. It’s organized into clear sections and covers everything from high‑level concepts through implementation details, dependencies, and next steps.

⸻

Introduction

“SmartCandidate Analyzer” is a tool that helps you find the best job applicants from a big pile of resumes. Instead of just matching keywords (which can miss important details), it uses a two‑step “retrieve & generate” approach:
                
	1.	Retrieve the most relevant resumes by turning both job descriptions and resumes into math‑friendly vectors and comparing them.
                
	2.	Generate a short, human‑readable recommendation explaining which resume seems best and why.

All of this runs right in your browser with Streamlit—no API keys or fancy setup needed.

⸻

Key Features
                
	1.	Two Retrieval Modes
                
	•	Generic RAG: Looks at your entire job description in one go and fetches the top matches.       
	•	Fusion RAG: Breaks the description into smaller sub‑queries, finds matches for each piece, then blends those results together for better coverage.
                
	2.	Personal Resume Match
                
	•	You can upload your own resume (PDF or TXT).          
	•	The app will compute and show you a “match score” (like 87%) between your resume and the job description.
                
	3.	Explainable Recommendations
                
	•	After ranking, it writes a 2–3 sentence summary saying something like, “Applicant ID 123 is the best fit because…”            
	•	If nothing matches well, it warns you instead of guessing.
                
	4.	Interview Scheduling (Stub)
                
	•	Pick one or more top candidates from the results.
	•	Choose a date/time and draft an email invitation.
	•	(Email integration is a placeholder for now.)
                
	5.	Interactive and Local
                
	•	All settings live in a sidebar: choose your RAG mode, pick an answer model (e.g. Flan‑T5), upload resumes, and view documentation.
	•	The main area has two tabs: Run (fetch and judge resumes) and Book Interview (schedule invites).
	•	Everything runs on your machine—no external API calls.

⸻

How It Works (High‑Level Flow)
                
	1.	User opens the app.
                
	2.	Data loads from a CSV of resumes and an embedding model builds an index in FAISS.
                
	3.	User enters a job description (and optional resume).
                
	4.	If they asked for help (Instructions or Documentation), that appears. Otherwise we go into the Run tab:
                
	•	The text is embedded via Sentence‑Transformers.
	•	Generic vs Fusion logic picks top “K” candidates.
	•	We show snippets of each resume and their similarity scores.
	•	We build a final prompt combining the job description + those snippets
	•	We feed that to a Hugging Face model (e.g. Flan‑T5) that writes the recommendation.
                
	5.	Results (top candidates + a short recommendation) appear on screen.
                
	6.	Book Interview tab can then use those stored results to let you pick candidates and send invites.

⸻

The Pieces Behind the Scenes
                
	•	Data Ingestion
	•	Resumes are kept in a simple CSV file with two columns: ID and Resume.
	•	We load that into a pandas DataFrame.
	•	Embeddings & Indexing
	•	We use the all‑MiniLM‑L6‑v2 Sentence‑Transformers model to turn each resume into a numeric vector.
	•	Those vectors go into a FAISS index for super-fast similarity lookups.
	•	Retrieval
	•	Generic RAG: one vector lookup on the full job description.
	•	Fusion RAG: split the job description into 3–4 chunks, do separate lookups, then combine their ranks (via “Reciprocal Rank Fusion”) into a final list.
	•	Generation                
	•	We craft a prompt like:                
You are a hiring consultant. Recommend the single best candidate by Applicant ID…
Job Description: …
Resumes: ID 123: …
                
Recommendation:
                
	•	Then we call a local Hugging Face model (text‑generation or text2text‐generation, depending on whether it’s Flan‑T5 or GPT‑2) to write that final explanation.
	•	User Interface
	•	Streamlit’s @st.cache_resource ensures we only build the FAISS index once per session.
	•	st.session_state keeps the last results around so the Book Interview tab can use them.
	•	A bit of custom CSS centers titles, rounds button corners, and generally polishes the look.

⸻

Installation & Deployment
                
	1.	Clone the repo, and make sure your CSV of resumes lives at data/main-data/synthetic-resumes.csv.
	2.	Create a requirements.txt containing:

            streamlit
            sentence-transformers
            faiss-cpu
            transformers
            pypdf
            scikit-learn


	3.	Install with:

pip install -r requirements.txt


	4.	Run your app:

streamlit run interface.py


	5.	(Optional) Deploy on Streamlit Cloud or any container platform.

⸻

What’s Next & How You Can Extend It
                
	•	Real email integration: hook up SMTP or SendGrid so invites actually go out.
	•	Batch mode: upload multiple JDs or resumes at once and export a report.
	•	Resume Summaries: pre-summarize very long resumes so they fit easily.
	•	Model Tuning: swap in GPU‑accelerated models or fine‑tune on your own data.
	•	Analytics & Logging: track which candidates get chosen most often, measure pipeline performance, and so on.
""")

# ─── Built by (always last) ────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ by [Ab Basit](https://github.com/AbBasitMSU)")
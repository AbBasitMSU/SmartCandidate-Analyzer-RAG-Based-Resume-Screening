import sys, os
sys.dont_write_bytecode = True

import time
from dotenv import load_dotenv

import pandas as pd
import streamlit as st

from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy

import chatbot_verbosity as chatbot_verbosity
from ingest_data import ingest  # unchanged
# we no longer import llm_agent or openai
# from llm_agent import ChatBot
# import openai

load_dotenv()

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH       = os.getenv("DATA_PATH")
FAISS_PATH      = os.getenv("FAISS_PATH")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Local HF model settings – no API key needed
HF_MODEL_NAME   = os.getenv("HF_MODEL_NAME", "gpt2-medium")  
MAX_NEW_TOKENS  = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE     = float(os.getenv("GEN_TEMPERATURE", "0.7"))

# RAG parameters
RAG_TOP_K       = 5      # same as your old “top‑K”
MAX_SUBQUERIES  = 4      # split into this many sub‑queries


# ─── PAGE SETUP ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Resume Screening GPT")
st.title("Resume Screening (Local HF Edition)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(DATA_PATH)

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

if "vectordb" not in st.session_state:
    st.session_state.vectordb = FAISS.load_local(
        FAISS_PATH,
        st.session_state.embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
        allow_dangerous_deserialization=True
    )


# ─── LLM INITIALIZATION ───────────────────────────────────────────────────────
# Create a Hugging Face text‑generation pipeline
generator = pipeline(
    "text-generation",
    model=HF_MODEL_NAME,
    max_new_tokens=MAX_NEW_TOKENS,
    truncation=True,
    pad_token_id=None,      # we’ll set it after init
    temperature=TEMPERATURE,
    # device=0              # uncomment if you have a GPU
)
# ensure we have a padding token
generator.tokenizer.pad_token_id = generator.tokenizer.eos_token_id

# Wrap it for LangChain‐style calls
llm = HuggingFacePipeline(pipeline=generator)


# ─── RAG HELPER FUNCTIONS ────────────────────────────────────────────────────
def generate_subquestions(question: str) -> list[str]:
    prompt = f"""
You are an expert in talent acquisition. Split this job description into {MAX_SUBQUERIES} focused sub-queries,
one per line. Use only information from the original text.

Job Description:
{question}

Sub-queries:
"""
    out = llm(prompt).strip()
    return [line for line in out.splitlines() if line.strip()]


def reciprocal_rank_fusion(ranklists: list[dict[str, float]], k=50) -> dict[str, float]:
    fused = {}
    for ranks in ranklists:
        for idx, (doc_id, _) in enumerate(ranks.items()):
            fused.setdefault(doc_id, 0.0)
            fused[doc_id] += 1.0 / (idx + k)
    return dict(sorted(fused.items(), key=lambda x: x[1], reverse=True))


def retrieve_id_and_rerank(queries: list[str]) -> dict[str, float]:
    all_ranks = []
    for q in queries:
        hits = st.session_state.vectordb.similarity_search_with_score(q, k=RAG_TOP_K)
        ranks = { str(doc.metadata["ID"]): score for doc, score in hits }
        all_ranks.append(ranks)
    return reciprocal_rank_fusion(all_ranks)


def retrieve_documents_with_id(id_scores: dict[str, float], threshold=5) -> list[str]:
    top_ids = sorted(id_scores, key=id_scores.get, reverse=True)[:threshold]
    docs = []
    for doc_id in top_ids:
        text = st.session_state.df.loc[
            st.session_state.df["ID"].astype(str)==doc_id, "Resume"
        ].values[0]
        docs.append(f"Applicant ID {doc_id}\n{text}")
    return docs


def generate_final_answer(question: str, docs: list[str]) -> str:
    context = "\n\n".join(docs)
    prompt = f"""
You are an expert in talent acquisition. Using ONLY Applicant IDs, pick the best resume and explain why.

Context:
{context}

Question:
{question}

Answer:
"""
    return llm(prompt).strip()


# ─── FILE UPLOAD / RE‑INGEST ───────────────────────────────────────────────────
def upload_file():
    uploaded = st.session_state.uploaded_file
    if uploaded:
        try:
            df_new = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return
        if not {"Resume","ID"}.issubset(df_new.columns):
            st.error("CSV must have columns `Resume` and `ID`.")
            return
        st.session_state.df = df_new
        with st.spinner("Re‑indexing… this can take a while"):
            vectordb = ingest(df_new, "Resume", st.session_state.embedding_model)
            st.session_state.vectordb = vectordb
    else:
        st.session_state.df = pd.read_csv(DATA_PATH)
        st.session_state.vectordb = FAISS.load_local(
            FAISS_PATH,
            st.session_state.embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True
        )


# ─── SIDEBAR CONTROLS ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    st.file_uploader("Upload resumes CSV", type="csv", key="uploaded_file", on_change=upload_file)
    st.button("Reset to default resumes", on_click=upload_file)
    st.markdown(
        """
        **Note:**  
        • No API key is required—this uses a local HuggingFace model.  
        • You can modify HF model & gen‐tokens via `.env` if you like.
        """
    )


# ─── CHAT UI ──────────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input("Type your job description here…")
if user_query:
    # show human message
    st.chat_message("user").write(user_query)
    st.session_state.chat_history.append({"role":"user","content":user_query})

    # RAG + Generation
    with st.spinner("Thinking…"):
        start = time.time()
        subqs   = generate_subquestions(user_query)
        id_scores = retrieve_id_and_rerank([user_query] + subqs)
        docs    = retrieve_documents_with_id(id_scores)
        answer  = generate_final_answer(user_query, docs)
        elapsed = time.time() - start

    # show AI reply
    st.chat_message("assistant").write(answer)
    st.session_state.chat_history.append({"role":"assistant","content":answer})

    # verbosity panel
    chatbot_verbosity.render(docs, {"query_type":"retrieve_applicant_jd","rag_mode":"Generic RAG"}, elapsed)
import sys, os
sys.dont_write_bytecode = True

from dotenv import load_dotenv
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

load_dotenv()

class ChatBot:
    def __init__(self, api_key: str, model: str):
        # api_key is ignored; `model` is now the name of a local HF model
        self.pipe = pipeline(
            "text-generation",
            model=model,
            max_new_tokens=256,
            truncation=True,
            pad_token_id=None,
            temperature=0.1,
            # device=0  # uncomment if you have a GPU
        )
        # ensure padding token is set
        self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

    def generate_subquestions(self, question: str):
        prompt = f"""You are an expert in talent acquisition. Separate this job description into 3‑4 more focused aspects for efficient resume retrieval. Make sure every single relevant aspect is covered in at least one query. Remove any irrelevant details. Output one sub‑query per line.

Job Description:
{question}

Sub‑queries:"""
        output = self.llm(prompt)
        # split on blank lines to mirror original behavior
        chunks = output.split("\n\n")
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def generate_message_stream(self, question: str, docs: list[str], history: list[str], prompt_cls: str):
        context = "\n\n".join(docs)

        if prompt_cls == "retrieve_applicant_jd":
            prompt = f"""You are an expert in talent acquisition helping determine the best candidate among multiple suitable resumes. Use only Applicant IDs when referring to resumes and provide detailed reasoning.

Chat history:
{history}

Context:
{context}

Question:
{question}

Answer:"""
        else:
            prompt = f"""You are an expert in talent acquisition helping analyze resumes. Use the provided context and chat history to answer. Do not mention that you have chat history.

Chat history:
{history}

Context:
{context}

Question:
{question}

Answer:"""

        answer = self.llm(prompt)
        # yield a single‑chunk stream for compatibility with interface.py
        yield answer
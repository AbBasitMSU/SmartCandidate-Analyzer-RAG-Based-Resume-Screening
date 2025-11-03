import sys
sys.dont_write_bytecode = True

from typing import List
from langchain_community.vectorstores import FAISS

RAG_K_THRESHOLD = 5

class RAGRetriever:
    def __init__(self, vectorstore_db: FAISS, df):
        self.vectorstore = vectorstore_db
        self.df = df

    def __reciprocal_rank_fusion__(self, document_rank_list: List[dict], k: int = 50) -> dict:
        fused_scores = {}
        for doc_list in document_rank_list:
            for rank, (doc_id, _) in enumerate(doc_list.items()):
                fused_scores.setdefault(doc_id, 0.0)
                fused_scores[doc_id] += 1.0 / (rank + k)
        return dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))

    def __retrieve_docs_id__(self, query: str, k: int) -> dict:
        hits = self.vectorstore.similarity_search_with_score(query, k=k)
        return { str(doc.metadata["ID"]): score for doc, score in hits }

    def retrieve_id_and_rerank(self, subquestions: List[str]) -> dict:
        rank_lists = [
            self.__retrieve_docs_id__(q, RAG_K_THRESHOLD)
            for q in subquestions
        ]
        return self.__reciprocal_rank_fusion__(rank_lists)

    def retrieve_documents_with_id(self, id_scores: dict, threshold: int = 5) -> List[str]:
        id_to_resume = dict(zip(
            self.df["ID"].astype(str), self.df["Resume"]
        ))
        top_ids = sorted(id_scores, key=id_scores.get, reverse=True)[:threshold]
        return [f"Applicant ID {aid}\n\n{id_to_resume.get(aid, '')}" for aid in top_ids]


class SelfQueryRetriever(RAGRetriever):
    def __init__(self, vectorstore_db: FAISS, df):
        super().__init__(vectorstore_db, df)
        self.meta_data = {
            "rag_mode": "",
            "query_type": "no_retrieve",
            "subquestion_list": [],
            "retrieved_docs_with_scores": {}
        }

    def retrieve_docs(self, question: str, llm, rag_mode: str) -> List[str]:
        self.meta_data["rag_mode"] = rag_mode
        self.meta_data["query_type"] = "retrieve_applicant_jd"

        if rag_mode == "Generic RAG":
            id_scores = self.__retrieve_docs_id__(question, RAG_K_THRESHOLD)
            self.meta_data["subquestion_list"] = [question]
            self.meta_data["retrieved_docs_with_scores"] = id_scores
        else:
            subqs = llm.generate_subquestions(question)
            queries = [question] + subqs
            self.meta_data["subquestion_list"] = queries
            id_scores = self.retrieve_id_and_rerank(queries)
            self.meta_data["retrieved_docs_with_scores"] = id_scores

        return self.retrieve_documents_with_id(id_scores)
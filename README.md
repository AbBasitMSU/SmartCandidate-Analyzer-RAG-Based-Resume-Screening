# Resume Screening RAG Pipeline

## Introduction

The Project is part of the author's graduating degree, which aims to present a POC of an LLM chatbot that can assist hiring managers in the resume screening process. The assistant is a cost-efficient, user-friendly, and more effective alternative to the conventional keyword-based screening methods. Powered by state-of-the-art LLMs, it can handle unstructured and complex natural language data in job descriptions/resumes while performing high-level tasks as effectively as a human recruiter.  

The core design of the assistant involves the use of hybrid retrieval methods to augment the LLM agent with suitable resumes as context:

1. Adaptive Retrieval:
   - Similarity-based retrieval: When a job description is provided, the retriever utilizes RAG/RAG Fusion to search for similar resumes to narrow the pool of applicants to the most relevant profiles.
   - Keyword-based retrieval: When applicant information is provided (IDs), the retriever can also retrieve additional information about specified candidates.
3. Generation: The retrieved resumes are then used to augment the LLM generator so it is conditioned on the data of the retrieved applicants. The generator can then be used for further downstream tasks like cross-comparisons, analysis, summarization, or decision-making.

#### Why resume screening?

Despite the increasingly large volume of applicants each year, there are limited tools that can assist the screening process effectively and reliably. Existing methods often revolve around keyword-based approaches, which cannot accurately handle the complexity of natural language in human-written documents. Because of this, there is a clear opportunity to integrate LLM-based methods into this domain, which the project aims to address. 

#### Why RAG/RAG Fusion? 

RAG-like frameworks are great tools to enhance the reliability of chatbots. Overall, RAG aims to provide an external knowledge base for LLM agents, allowing them to receive additional context relevant to user queries. This increases the relevance and accuracy of the generated answers, which is especially important in data-intensive environments such as the recruitment domain.

On the other hand, RAG Fusion is effective in addressing complex and ambiguous human-written prompts. While the LLM generator can handle this problem effectively, the retriever may struggle to find relevant documents when presented with multifaceted queries. Therefore, this technique can be used to improve resume retrieval quality when the system receives complex job descriptions (which are quite common in hiring).



## Demo

The demo interface of the chatbot can be found here: [Streamlit](https://smartcandidate-analyzer.streamlit.app)

Default synthetic resume data set used in the demo: [GitHub](https://github.com/AbBasitMSU/SmartCandidate-Analyzer-RAG-Based-Resume-Screening/blob/59c64f32b480bf6ae4ea066c8f0b486ca193d117/data/main-data/synthetic-resumes.csv)

Source job description dataset: [Kaggle](https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description)

> [!WARNING]
> The file uploader is still quite unstable in Streamlit deployment. I do not recommend using it.

**Starting screen:**
![FEFDF847-B1ED-46B9-860D-4448DC0582FF](https://github.com/user-attachments/assets/71c504f8-5e62-44a8-ba3e-d639bda0c07f)




**The system's example response when receiving a job description:**
![3B9C70B7-CF02-4FBD-8DF3-AFE2E4A1D3F8](https://github.com/user-attachments/assets/eec1fb59-e77b-4492-976c-882c6e4528fd)




**The system's example response when receiving specific applicant IDs**
![EE3C978E-A301-4452-B8CF-A93B17D88148](https://github.com/user-attachments/assets/6ce63659-7855-45e5-950e-8351fd3f3533)




## System Description

### 1. Chatbot Structure

![chatbot_structure](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/assets/46376260/dc97c06c-ca5d-4882-8e78-9101d528ee75)

The deployed chatbot utilizes certain techniques to be more suitable for real-world use cases:

- Chat history access: The LLM is fed with the entire conversation and the (latest) retrieved documents for every message, allowing it to perform follow-up tasks. 
- Query classification: Utilizing function-calling and an adaptive approach, the LLM extracts the necessary information to decide whether to toggle the retrieval process on/off. In other words, the system only performs document retrieval when a suitable input query is provided; otherwise, it will only utilize the chat history to answer.
- Small-to-Big retrieval: The retrieval process is performed using text chunks for efficiency. The retrieved chunks are then traced back to their original full-text documents to augment the LLM generator, allowing the generator to receive the complete context of the resumes. 

**Tech stacks:** 
- `langchain`, `openai`, `huggingface`: RAG pipeline and chatbot construction.
- `faiss`: Vector indexing and similarity retrieval.
- `streamlit`: User interface development.

### 2. Under-the-hood RAG Pipeline 

![Encoder (1)](https://github.com/Hungreeee/Resume-Screening-RAG-Pipeline/assets/46376260/4259837e-9e2c-40f8-8276-e9469667b98b)

The pipeline begins by processing resumes into a vector storage. Upon receiving the input job descriptions query, the LLM agent is prompted to generate sub-queries. The vector storage then performs a retrieval process for each given query to return the top-K most similar documents. The document list for each sub-query is then combined and re-ranked into a new list, representing the most similar documents to the original job description. The LLM then utilizes the retrieved applicants' information as context to form accurate, relevant, and informative responses to assist hiring managers in matching resumes with job descriptions.

## Installation and Setup

To set up the project locally:
```
# Clone the project
git clone (https://github.com/AbBasitMSU/SmartCandidate-Analyzer-RAG-Based-Resume-Screening)

# Install dependencies
pip install requirements.txt
```

To run the Streamlit demo locally:
```
streamlit run demo/interface.py
```

## Contributions

The design of the demo chatbot is relatively simple because it only serves to show the bigger picture of the potential of RAG-like systems in the recruitment domain. As such, the system is still very much a work in progress and any suggestion, feedback, or contribution is highly appreciated! 

## Acknowledgement

- Inspired by [RAG Fusion](https://github.com/Raudaschl/rag-fusion) 

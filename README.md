# Officer Support Assistant

An AI-powered Retrieval-Augmented Generation (RAG) system designed to assist law enforcement personnel in accessing departmental policies, Nebraska statutes, standard operating procedures (SOPs), and training materials through natural language queries.

**Author:** Sravan Kumar Veerannagari  
**University:** University of Nebraska at Omaha  
**Program:** M.S. Data Science  
**Project Type:** Master's Research Project

---

## Overview

Officer Support Assistant is a Retrieval-Augmented Generation (RAG) application developed to help law enforcement officers quickly access relevant legal and policy information through natural language questions.

The system combines semantic search, FAISS vector retrieval, Sentence Transformer embeddings, and OpenAI GPT models to provide accurate, source-grounded responses based on departmental policies, Nebraska statutes, standard operating procedures (SOPs), and training materials.

By retrieving information directly from trusted documents before generating responses, the system improves answer accuracy and reduces hallucinations commonly associated with standalone language models.

---

## Features

- Natural Language Question Answering
- Retrieval-Augmented Generation (RAG)
- FAISS Vector Search
- Nebraska Statute Lookup
- Department Policy Search
- SOP and Training Material Retrieval
- Citation-Based Responses
- Streamlit Web Interface
- OpenAI GPT Integration
- Source-Grounded Decision Support

---

## Technology Stack

| Component | Technology |
|------------|------------|
| Frontend | Streamlit |
| Backend | Python |
| Vector Database | FAISS |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| LLM | OpenAI GPT |
| Deployment | Streamlit Community Cloud |

---

## System Architecture

```text
User Query
    │
    ▼
Streamlit Interface
    │
    ▼
Sentence Transformer Embeddings
    │
    ▼
FAISS Vector Index
    │
    ▼
Relevant Document Retrieval
    │
    ▼
OpenAI GPT Model
    │
    ▼
Grounded Response + Citations
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sravan12521/officer-support-assistant.git
cd officer-support-assistant
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
```

---

## Running the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Example Questions

### Nebraska Statutes

- Can an officer conduct a welfare check without a warrant?
- What are the legal requirements for probable cause?

### Department Policies

- What is the use-of-force reporting policy?
- What documentation is required after an arrest?

### Standard Operating Procedures

- What steps should be followed during a DUI stop?
- What procedures apply during a traffic stop?

### Training Materials

- What information should be included in an incident report?
- What is the protocol for handling evidence?

---

## RAG Workflow

1. Legal and policy documents are collected and processed.
2. Documents are split into smaller text chunks.
3. Sentence Transformer embeddings are generated for each chunk.
4. Embeddings are stored in a FAISS vector index.
5. User questions are converted into embeddings.
6. Similarity search retrieves the most relevant document chunks.
7. Retrieved context is passed to the OpenAI GPT model.
8. The model generates a response grounded in the retrieved information.
9. Citations are included to provide transparency and traceability.

---

## Research Significance

This project demonstrates the practical application of Retrieval-Augmented Generation (RAG) in legal and policy information retrieval. The system helps improve accessibility to critical information while maintaining transparency through source citations.

Potential benefits include:

- Faster information retrieval
- Reduced time spent searching policy manuals
- Improved policy compliance
- Enhanced decision support
- Increased trust through citation-backed responses

---

## Future Enhancements

- Real-time legal and policy updates
- Multi-agency policy support
- Voice-enabled assistant
- Mobile application deployment
- Incident report drafting assistance
- Case law integration
- Azure cloud deployment
- Role-based access control

---

## Academic Project Information

**Project Title:**  
Officer Support Assistant: A Legal-Policy Retrieval-Augmented Generation System for Law Enforcement Decision Support

**Author:**  
Sravan Kumar Veerannagari

**Program:**  
M.S. Data Science

**Institution:**  
University of Nebraska at Omaha

**Faculty Mentor:**  
Dr. Kerry Ward

**External Mentor:**  
Dr. Chris Street


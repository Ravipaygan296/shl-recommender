# shl-recommender
SHL Assessment Recommendation System (Generative AI Internship Task) hiring manager  often struggle to identify the most appropriate assessments for their job roles due to inefficient keyword filtering. This project aims to solve that by using semantic search to recommend SHL assessment tests based on a natural language query or job description.
# SHL Assessment Recommendation System

This project is a semantic search engine designed to recommend SHL assessments based on natural language job descriptions.

## Features
- Streamlit frontend for user-friendly query input.
- FastAPI backend serving recommendations.
- Embedding-based semantic similarity using Sentence Transformers.
- Evaluation module using Recall@3 and MAP@3.

## Project Structure
```
shl-recommender/
├── app.py                 # Streamlit frontend
├── api.py                 # FastAPI backend API
├── data/
│   └── shl_mock_catalog.csv  # Assessment metadata
├── utils/
│   └── search_engine.py      # Embedding logic and search function
├── evaluation/
│   └── evaluate.py           # Evaluation using Recall@K and MAP@K
├── .streamlit/
│   └── config.toml           # Streamlit config
├── requirements.txt      # Dependencies
├── README.md             # Project overview
```

## Setup
1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the FastAPI server:
```bash
uvicorn api:app --reload
```

3. Run the Streamlit frontend:
```bash
streamlit run app.py
```

4. (Optional) Evaluate performance:
```bash
python evaluation/evaluate.py
```

## License
MIT

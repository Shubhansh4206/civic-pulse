# CivicPulse 🏙️

Smart Department Routing setup, migration, training/evaluation, API additions and verification results are documented in [DEPARTMENT_ROUTING.md](DEPARTMENT_ROUTING.md).

## Semantic duplicate detection

Reports within 5 metres are compared with CPU `all-MiniLM-L6-v2` embeddings and cosine similarity (default threshold 0.80). Every submission now retains its own row, evidence, routing and priority; the old spatial merge is no longer used by POST. Existing historical records/counts are unchanged. Matching OPEN reports, regardless of visual category, share `duplicate_group_id`; `duplicate_similarity` records a matching pair's score and `related_report_count` counts other saved rows in the group. Group membership is advisory and can be transitive, not a guarantee that every pair matches. No report is deleted or automatically resolved.

Before starting the updated API, install the additional dependency, provision the model once, and apply the additive migration to your actual database (Feature 1 migration must already be applied):

```powershell
python -m pip install sentence-transformers==5.1.2
python scripts/prepare_semantic_model.py
psql -h localhost -U postgres -d civicpulse_db -v ON_ERROR_STOP=1 -f backend/database/migrations/002_semantic_duplicates.sql
```

Configure `SEMANTIC_DUPLICATES_ENABLED`, `SEMANTIC_DUPLICATE_RADIUS_METERS=5.0`, and `SEMANTIC_DUPLICATE_THRESHOLD=0.80` in `.env`. Migration is required even when disabled. The model loads once per process from the provisioned local folder; restart after fixing a failed model load. A bounded in-memory embedding cache avoids repeat inference. Missing models or annotation errors leave the original report saved normally. Structured events reuse request IDs without logging text or embeddings.

Smoke check: `python -m pytest tests/test_routing_integration.py -q`. It exercises real Sentence-BERT with isolated SQLite and stubbed YOLO/DistilBERT outputs. Submit similar complaints at the same coordinates and an unrelated complaint: GET and dashboard should show separate rows with a “Possible Duplicate” badge only for matches. Similar text more than 5m away should remain ungrouped. The citizen modal still uses fixed coordinates, so use the API to check different locations. Long texts follow MiniLM's tokenizer truncation; similarity is not a probability.

CivicPulse is an AI-powered municipal hazard reporting and dispatch dashboard. It utilizes a Multi-Modal Decision Fusion System, combining Deep Learning (YOLOv8) for image-based visual severity scoring and Natural Language Processing (DistilBERT) for text-based urgency assessment.

## 💻 Tech Stack

- **Backend**: Python, FastAPI, PostgreSQL, SQLAlchemy
- **AI/ML**: YOLOv8 (Computer Vision), DistilBERT (NLP)
- **Frontend**: React, Vite, Tailwind CSS (v4), React-Leaflet

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- [Python 3.8+](https://www.python.org/downloads/)
- [Node.js (v18+)](https://nodejs.org/)
- [PostgreSQL](https://www.postgresql.org/download/)

---

## 🚀 Getting Started

### 1. Database Setup
1. Open PostgreSQL (pgAdmin or psql).
2. Create a new database named `civic_pulse` (or your preferred name).
3. In the root directory of this project, copy the `.env.example` file and rename it to `.env`.
4. Update your database URL in the `.env` file:
   ```env
   DATABASE_URL=postgresql://<username>:<password>@localhost:5432/civic_pulse
   ```

### 2. Backend Setup (FastAPI)
1. Open a terminal in the root folder of the project.
2. Create and activate a virtual environment:

   **Windows:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

   **Mac/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies and run the server:
   ```bash
   pip install -r requirements.txt
   uvicorn backend.main:app --reload
   ```

The backend will now be running at [http://127.0.0.1:8000](http://127.0.0.1:8000). You can view the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 3. Frontend Setup (React/Vite)
1. Open a new (second) terminal window and navigate to the frontend folder:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

The frontend will now be running at [http://localhost:5173](http://localhost:5173).

---

## 🧠 Architecture Note: The Fusion Engine

When a citizen submits a ticket, the backend does not rely on a single data point:
- **YOLOv8** extracts bounding boxes from the uploaded image to calculate a `visual_severity` score.
- **DistilBERT** tokenizes the text description to calculate semantic `text_urgency`.
- The custom **Late Fusion Engine** mathematically combines these multi-modal probability outputs to assign a definitive `CRITICAL`, `HIGH`, or `LOW` priority for the dispatcher.

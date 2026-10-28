# CivicPulse 🏙️
### AI-Powered Multi-Modal Municipal Hazard Triage & Dispatch Platform

CivicPulse is an advanced, AI-driven municipal incident reporting, triage, and dispatch dashboard. It combines Deep Learning Computer Vision, Natural Language Processing, Cross-Modal Vision-Language Verification, and PostGIS Geospatial Analysis into a **Multi-Modal Decision Fusion System** designed for municipal authorities and dispatchers.

---

## 🌟 Key Features & AI Pipeline

CivicPulse does not rely on a single data point when triaging municipal defects. Every incoming report passes through an ensemble of dedicated AI/ML services:

- **📸 Computer Vision Defect Detection (YOLOv8)**:
  - Custom-trained YOLOv8 weights (`models/yolov8_civic.pt`) detect and categorize municipal hazards (e.g., `pothole`, `garbage_pile`).
  - Automatically isolates the highest-confidence defect and outputs a normalized `visual_severity` score ($0.0 - 1.0$).
- **✍️ Deep Contextual NLP Urgency Scoring (DistilBERT)**:
  - Leverages local CPU DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) to evaluate citizen distress and sentiment.
  - Modulated by a domain-specific municipal hazard keyword lexicon with graded adjustments for critical hazard keywords (e.g., *sinkhole, explosion, live wire, gas leak*), medium concerns (*deep, broken, flooding*), and routine maintenance (*cosmetic, paint*).
- **🔀 Multi-Modal Late Fusion Engine**:
  - Dynamically weights visual severity ($0.60$) and textual urgency ($0.40$).
  - **Critical Synergy Escalation**: When both independent modalities confirm critical hazard levels ($\ge 0.80$), an automatic $+0.15$ escalation boost is applied.
  - Quantizes the fused priority score into four operational triage tiers: `CRITICAL` ($\ge 0.80$), `HIGH` ($\ge 0.60$), `MEDIUM` ($\ge 0.40$), and `LOW` ($< 0.40$).
- **🏛️ Smart Department Routing (WordNet Lesk WSD + Naive Bayes)**:
  - Word Sense Disambiguation (WSD) disambiguates civic homonyms (*bank, drain, main, line*) using NLTK WordNet synsets.
  - A Multinomial Naive Bayes classifier automatically routes tickets to the appropriate municipal authority (`ROADS`, `DRAINAGE`, `WATER_SUPPLY`, `SOLID_WASTE`, `STREET_LIGHTING`), falling back to `MANUAL_REVIEW` when confidence is below $0.60$.
- **🔍 Spatial & Semantic Deduplication (Sentence-BERT + PostGIS)**:
  - Combines PostGIS spatial radius filtering ($\le 5\text{m}$) with CPU-based Sentence-BERT (`all-MiniLM-L6-v2`) dense 384-d semantic embeddings.
  - Matches with cosine similarity $\ge 0.80$ share an advisory `duplicate_group_id` with real-time UI badge counters.
  - **Non-Destructive**: Every submission retains its own database row, original image evidence, and independent priority without loss of citizen input.
- **🛡️ Advisory Image-Text Consistency Verification (OpenAI CLIP)**:
  - Employs zero-shot OpenAI CLIP (`clip-vit-base-patch32`) to compute cross-modal cosine similarity between the uploaded photo and text complaint.
  - Flags tickets as `CONSISTENT`, `SUSPICIOUS`, or `UNVERIFIED` in the dispatcher dashboard. Strictly advisory—never rejects citizen reports or lowers priority.
- **🗺️ Real-Time Interactive Dispatcher Console**:
  - Interactive OpenStreetMap (Leaflet) canvas with color-coded priority markers, smooth fly-to focus, expandable diagnostic accordions, copyable GPS coordinates, and one-click ticket resolution.

---

## 💻 Tech Stack

| Domain | Technologies |
| :--- | :--- |
| **Backend Framework** | Python 3.8+, FastAPI, Uvicorn, Pydantic (v1/v2 compatible) |
| **Database & GIS** | PostgreSQL, PostGIS, SQLAlchemy, GeoAlchemy2, Psycopg2 |
| **Computer Vision** | Ultralytics YOLOv8, Pillow |
| **NLP & Vision-Language** | Hugging Face Transformers (`DistilBERT`, `CLIP ViT-B/32`), Sentence-Transformers (`all-MiniLM-L6-v2`), NLTK (`WordNet`), Scikit-Learn |
| **Frontend Framework** | React 19, Vite, Tailwind CSS v4 |
| **Mapping & Icons** | React-Leaflet, Leaflet, OpenStreetMap, Lucide React, Axios |
| **Testing** | Pytest, Pytest-Asyncio |

---

## 📋 Prerequisites

Before starting, ensure you have the following installed:
- [Python 3.8+](https://www.python.org/downloads/)
- [Node.js (v18+) & npm](https://nodejs.org/)
- [PostgreSQL](https://www.postgresql.org/download/) (with PostGIS extension support)

---

## 🚀 Quick Start Guide

### 1. Repository Setup & Environment Configuration

Clone the repository and copy the example environment configuration:

```powershell
# Windows (PowerShell)
Copy-Item .env.example .env
```
```bash
# Linux / macOS
cp .env.example .env
```

Open `.env` and configure your PostgreSQL database credentials and model parameters:
```env
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/civicpulse_db
YOLO_MODEL_PATH=models/yolov8_civic.pt
YOLO_CONFIDENCE_THRESHOLD=0.25
ROUTING_ENABLED=True
ROUTING_MODEL_PATH=models/department_router/router.joblib
SEMANTIC_DUPLICATES_ENABLED=True
SEMANTIC_DUPLICATE_RADIUS_METERS=5.0
SEMANTIC_DUPLICATE_THRESHOLD=0.80
```

---

### 2. Backend Setup (FastAPI & AI Models)

1. **Activate Python Virtual Environment**:
   - **Windows:**
     ```powershell
     .\venv\Scripts\activate
     ```
   - **Linux / macOS:**
     ```bash
     source venv/bin/activate
     ```

2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Provision Pretrained AI Models (One-Time Setup)**:
   Download and cache the required offline transformer models and NLTK corpora:
   ```bash
   # Download Sentence-BERT for semantic duplicate detection
   python scripts/download_semantic_model.py

   # Download OpenAI CLIP for image-text consistency verification
   python scripts/download_clip_model.py

   # Train and export the Department Router artifact
   python scripts/train_department_router.py
   ```

4. **Initialize Database & Apply Additive Migrations**:
   ```powershell
   # Create database if it doesn't already exist
   python create_db.py

   # Initialize tables and enable PostGIS
   python backend/init_db.py

   # Apply additive migrations
   psql -h localhost -U postgres -d civicpulse_db -v ON_ERROR_STOP=1 -f backend/database/migrations/001_department_routing.sql
   psql -h localhost -U postgres -d civicpulse_db -v ON_ERROR_STOP=1 -f backend/database/migrations/002_semantic_duplicates.sql
   psql -h localhost -U postgres -d civicpulse_db -v ON_ERROR_STOP=1 -f backend/database/migrations/003_clip_consistency.sql
   ```

5. **Start the FastAPI Backend Server**:
   ```bash
   uvicorn backend.main:app --reload
   ```
   - **Backend API**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
   - **Interactive API Documentation (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - **Alternative ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### 3. Frontend Setup (React & Vite)

Open a **separate terminal window** for the frontend application:

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install frontend dependencies**:
   ```bash
   npm install
   ```

3. **Launch the Vite development server**:
   ```bash
   npm run dev
   ```
   - **CivicPulse Web Console**: [http://localhost:5173](http://localhost:5173)

---

## 📡 REST API Reference

| Endpoint | Method | Description | Key Request / Response Parameters |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | Health check probe | `{"status": "CivicPulse API is running"}` |
| `/api/v1/report` | `GET` | List all tickets ordered by priority score | Returns ticket list with multimodal scores, department, duplicate stats, and CLIP status |
| `/api/v1/report` | `POST` | Ingest and triage a new citizen report | **Multipart Form**: `title`, `description`, `latitude`, `longitude`, `image` (file). Runs full AI pipeline and returns `ReportResponse` |
| `/api/v1/report/{ticket_id}` | `PATCH` | Update incident resolution status | **JSON Body**: `{"status": "RESOLVED"}` |

*(Aliases `/api/v1/tickets` for GET and PATCH are also available for backward compatibility).*

---

## 🧪 Testing & Verification

CivicPulse includes comprehensive unit and integration test suites covering the late fusion engine, department routing, semantic duplicate detection, and CLIP consistency checking.

Run the test suite:
```bash
python -m pytest tests/ -v
```

---

## 📂 Project Architecture

```
civic-pulse/
├── backend/
│   ├── api/
│   │   └── endpoints.py               # REST API route handlers
│   ├── database/
│   │   ├── migrations/                # SQL schema migrations (001, 002, 003)
│   │   ├── models.py                  # SQLAlchemy MasterTicket ORM model
│   │   └── session.py                 # Engine & SessionLocal lifecycle
│   ├── services/
│   │   ├── consistency_service.py     # OpenAI CLIP image-text consistency
│   │   ├── cv_service.py              # YOLOv8 object detection service
│   │   ├── deduplication.py           # PostGIS ST_DWithin & Haversine proximity
│   │   ├── department_service.py      # Naive Bayes department router
│   │   ├── fusion_service.py          # Multimodal Late Fusion decision engine
│   │   ├── nlp_service.py             # DistilBERT urgency & civic lexicon scoring
│   │   ├── routing_features.py        # NLTK WordNet Lesk WSD feature extractor
│   │   └── semantic_duplicate_service.py # Sentence-BERT embedding similarity
│   ├── config.py                      # Pydantic environment configuration
│   ├── init_db.py                     # Database initialization & PostGIS check
│   ├── main.py                        # FastAPI application entry point
│   └── routing_logging.py             # Structured audit and event logger
├── frontend/
│   ├── src/
│   │   ├── App.jsx                    # Primary React application component
│   │   ├── main.jsx                   # Vite application mount point
│   │   └── index.css                  # Tailwind CSS root styling
│   ├── package.json                   # React 19 dependencies & scripts
│   └── vite.config.js                 # Vite configuration
├── models/
│   ├── clip-vit-base-patch32/         # Pretrained CLIP model weights
│   ├── department_router/             # Serialized Naive Bayes joblib artifact
│   ├── all-MiniLM-L6-v2/              # Sentence-BERT embedding model
│   └── yolov8_civic.pt                # Custom YOLOv8 defect detector weights
├── scripts/
│   ├── download_clip_model.py         # Offline CLIP provisioning script
│   ├── download_semantic_model.py     # Offline Sentence-BERT provisioning
│   ├── train_department_router.py     # Department routing classifier trainer
│   └── train_yolo.py                  # YOLOv8 training & dataset merge CLI
├── tests/                             # Comprehensive test suite
├── create_db.py                       # PostgreSQL automated database creator
├── requirements.txt                   # Production Python dependencies
└── README.md                          # Project documentation
```

---

## 📄 License
This project is licensed under the Apache 2.0 License.

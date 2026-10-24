# CivicPulse 🏙️

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

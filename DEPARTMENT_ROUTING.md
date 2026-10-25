# Smart Department Routing

Implemented 26 October 2026 as an independent add-on inside CivicPulse. Feature 2 is now implemented as described in the README: new submissions are saved separately, so the historical duplicate-merge behavior described below no longer applies. Feature 3 (CLIP) remains unimplemented.

## Behavior

The existing `POST /api/v1/report` now runs:

`upload → existing YOLO/category fallback → WordNet/Lesk features → trained MultinomialNB → existing DistilBERT → unchanged fusion → existing spatial merge/create → dashboard`

Routing uses title plus description. The original description is passed unchanged to DistilBERT. The feature extractor retains the original lexical tokens, adds contextual WordNet noun synset/lemma features for bank, drain, main and line, and abstains from sense selection when no gloss overlaps the context. NLTK's original Lesk selects the sense; department labels come from a fitted scikit-learn MultinomialNB over unigram/bigram counts, not a keyword-to-department lookup. Both training and inference use the same feature extractor.

Supported academic labels are `ROADS`, `DRAINAGE`, `WATER_SUPPLY`, `SOLID_WASTE`, and `STREET_LIGHTING`. These are demonstration labels, not a verified official BMC department directory. They extend routing vocabulary without adding YOLO object classes or changing legacy hazard categories.

CPU inference runs in the existing Starlette thread pool. The classifier and fitted vectorizer are loaded together once per configured artifact path/process, with a lock and cached failures. WordNet's lazy corpus initialization is also locked. There is no request-time training or resource download. A restart is required after replacing a cached artifact or fixing a previously failed artifact load. Multiple server workers each hold their own small NB artifact and WordNet corpus in memory.

## Install and provision

Use the project's backend environment with Python 3.10+ and its existing dependencies. The new direct dependencies are NLTK 3.9.2, scikit-learn 1.7.2 and joblib 1.5.3. The saved model was trained with those versions; retrain when upgrading scikit-learn. Joblib artifacts must come from trusted local training, since deserialization can execute code.

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m nltk.downloader -d .nltk_data wordnet
python scripts/train_department_router.py
python scripts/evaluate_department_router.py
```

`.nltk_data/` is ignored by Git. The feature extractor checks this project-local corpus directory first, then normal NLTK paths (including `NLTK_DATA`). Provision WordNet before serving traffic; a missing corpus yields MANUAL_REVIEW. An offline deployment can copy a provisioned NLTK data directory instead of downloading it.

`models/department_router/router.joblib` contains both the fitted CountVectorizer and MultinomialNB, with model/preprocessing version metadata. `router.json` contains readable provenance. The artifact and metadata are included in this change. Training reads only the fixed TRAIN split and can be repeated without a database or YOLO weights. `--output` selects another artifact location. Evaluation reads only TEST for predictions; `--model`, `--threshold`, and `--output` are available. Neither script starts the server.

The existing requirements file contains other exact pins that were not installed or upgraded during this feature's verification. Verification used a project-local `.venv-routing` inheriting installed backend packages, with the missing routing/API packages installed locally. It is not proof that the entire existing requirements manifest resolves on every platform.

## Required migration before deployment

Stop API writers and back up the target database according to your normal process. Apply the versioned SQL migration to the **existing** database before starting this backend version:

```powershell
psql -h localhost -U postgres -d civicpulse_db -v ON_ERROR_STOP=1 -f backend/database/migrations/001_department_routing.sql
```

Use your actual database name/user/host; allow the PostgreSQL client to prompt for credentials rather than putting passwords in command text. The migration is transactional and repeatable. It adds nullable columns `predicted_department`, `department_confidence`, `routing_method`, `routing_model_version` and records `001_department_routing` in `civicpulse_schema_migrations`. It does not recreate tables, change spatial storage or update existing ticket values. Historical routing fields remain NULL. `create_all` alone will not migrate an existing table.

Verify the four columns and the migration ledger entry, then restart the API. Fresh databases created using the existing initializer include the ORM columns, but applying the migration also records the schema version. To roll application code back, leave the additive columns and ledger in place; do not drop data. `ROUTING_ENABLED=False` disables inference but **does not remove the need for migration**, since the ORM still selects the new columns.

No live PostgreSQL database was configured or modified during implementation; the SQL migration has not been executed against your database. API integration tests used an isolated SQLite database and the actual ORM.

## Configuration and fallback semantics

```dotenv
ROUTING_ENABLED=True
ROUTING_MODEL_PATH=models/department_router/router.joblib
ROUTING_CONFIDENCE_THRESHOLD=0.60
ROUTING_FALLBACK_DEPARTMENT=MANUAL_REVIEW
```

Relative artifact paths resolve against the repository root. Confidence thresholds are validated in [0,1]. The 0.60 default is a demonstration threshold, not a calibrated operational guarantee.

Fewer than three alphabetic tokens across title/description yields `insufficient_text`. A detailed title can still be routed with an empty description. Fewer than two distinct known original unigram features yields `unsupported_vocabulary`. Disabled routing, missing/corrupt/incompatible artifacts, unavailable WordNet and inference errors return the configured fallback independently of legacy scoring. Below-threshold predictions return the fallback with the best candidate's probability retained. The method suffix distinguishes `disabled`, `insufficient_text`, `unsupported_vocabulary`, `low_confidence` and `unavailable`. Confidence is NULL if no model prediction was made; it is not a fabricated zero.

For accepted predictions, `routing_method` is `wordnet_lesk_multinomial_nb`; model version is currently `synthetic-nb-v1-2edda6d04834`. Confidence is a Naive Bayes posterior, not calibrated real-world correctness.

## Persistence, responses and UI

The four nullable routing fields are additive to POST, GET, PATCH and the existing GET/PATCH aliases. Existing keys, status codes, paths, multipart inputs and score formulas are retained. POST additionally returns `request_id`; HTTP responses handled by the request middleware include `X-Request-ID`, exposed to the browser through CORS. Routing does not replace hazard category.

New master tickets store the routing result. On a spatial duplicate, the original master's routing is preserved along with its original evidence; POST returns that stored routing. An incoming routing prediction is logged with the current request ID but does not overwrite an existing assignment. Historical masters with NULL routing remain NULL on merges; there is no speculative backfill. This is deliberate and avoids changing pre-existing records or rerouting an issue based on a possibly unrelated duplicate. Existing report-count and max-priority behavior is untouched.

`TicketCard` shows a compact department/confidence indicator using the existing Tailwind palette. It hides the indicator for historical NULL values and labels low-confidence fallback probabilities as “best candidate confidence.” Map behavior, sorting, polling, modal submission and resolution remain unchanged.

Direct inference without FastAPI/Yolo:

```powershell
python -c "from backend.services.department_service import predict_department; print(predict_department('Blocked drain', 'Sewage overflows from the sewer and drainage channel').to_dict())"
```

Example API routing fragment (other existing fields remain present):

```json
{
  "predicted_department": "DRAINAGE",
  "department_confidence": 0.999943,
  "routing_method": "wordnet_lesk_multinomial_nb",
  "routing_model_version": "synthetic-nb-v1-2edda6d04834"
}
```

The fragment above is the measured output of the direct inference command with the saved artifact. Exact held-out probabilities are recorded in `models/department_router/evaluation.json`.

## Structured routing diagnostics

`backend/routing_logging.py` emits allowlisted JSON to stderr, independently of Uvicorn log formatting. Events are `routing_started`, `wsd_completed`, `department_predicted`, `routing_fallback`, and `routing_failed`. They contain the server-generated request ID, durations in milliseconds, model version, prediction/confidence when available, and safe reason/error-type codes. WSD reports a sense count, not original tokens or text. The prediction event precedes a low-confidence fallback, so diagnostics retain the candidate without mislabeling manual review as a learned class.

The middleware ignores untrusted client request IDs and generates UUIDs. The model stage's request ID is the same as the POST response/header. No complaint body, raw image, filename, credentials or raw exception message is added to routing logs. SQLAlchemy now hides bound parameters even in DEBUG mode; database initialization/session/persistence errors log exception types rather than connection URLs or exception messages containing report data. Persistence failures retain HTTP 500 but return a generic detail instead of database internals. The existing general application logging is not otherwise replaced by this feature; broader legacy logging hardening remains in the extension plan.

## Dataset and evaluation

`scripts/routing_dataset.py` is the reproducible synthetic dataset: 20 training and 6 independently authored held-out scenarios per class (100 train / 30 test). It uses fixed separate scenario lists, not random splitting of location-swapped templates. No test examples are fitted or used to choose the model parameters. Evaluation screens cross-split token-set Jaccard overlap and fails at 0.65; observed maximum was 0.2857. This is a lexical leakage screen, not a proof of semantic independence. Shared domain vocabulary is intentional.

Measured synthetic test results:

- Raw top-class accuracy: 29/30 (96.67%); macro F1: 0.9664.
- At threshold 0.60: 28/30 accepted (93.33% coverage), 2 manual-review fallbacks.
- Accepted predictions: 27/28 correct (96.43%); one Drainage scenario was classified as Water Supply.
- Full per-class precision/recall, confusion matrix and per-example confidence are saved in `evaluation.json`.

**Synthetic evaluation does not establish real-world accuracy.** A small, clean, English dataset can yield overconfident posteriors. WordNet/Lesk depends on definition overlap and cannot resolve every synonym/context. No operational department mapping, Marathi/Hindi support or human-reviewed real complaint evaluation is claimed. Do not tune the threshold repeatedly on this test set; obtain a separate validation corpus for calibration and a new held-out real test corpus for deployment assessment.

## Verification

```powershell
python -m pytest tests -q
cd frontend
npm ci --ignore-scripts
npm run build
npm run lint
```

Observed on 26 October 2026: **26 tests passed**; frontend production build passed. Frontend lint reports one pre-existing `react-hooks/set-state-in-effect` error at the initial `fetchTickets()` call; running ESLint against the Git HEAD version reproduces the same error. It was not changed as part of the department indicator.

Tests cover real WordNet/Lesk distinctions, no-overlap abstention, five normal department predictions, short/empty text, unsupported vocabulary, low-confidence/configurable fallback, missing model/corpus, cached concurrent loading, private log fields, fixed splits, and API persistence/aliases/merges/null historical values. The integration test runs the actual YOLO result-processing wrapper, urgency function and fusion function with deterministic neural outputs, comparing enabled vs disabled routing. Both produce visual severity 0.84, urgency 0.975 and priority 1.0, and DistilBERT receives identical original input.

Custom YOLO weights are absent, so neural forward-pass accuracy and a full live deployment were not validated. The API tests stub only neural model outputs and use real routing/ORM/API logic. In this Windows sandbox, asyncio's local socketpair needed execution outside the network sandbox; tests then completed normally. No external database or inference service is contacted by these tests.

## Changed files

- Runtime: `backend/services/department_service.py`, `routing_features.py`, service package lazy exports; `backend/routing_logging.py`, `request_context.py`.
- Integration/configuration: `backend/api/endpoints.py`, `backend/main.py`, `backend/config.py`, `.env.example`.
- Storage: `backend/database/models.py`, `backend/database/session.py`, `backend/database/migrations/001_department_routing.sql`.
- Training/evaluation/data: `scripts/routing_dataset.py`, `train_department_router.py`, `evaluate_department_router.py`; `models/department_router/router.joblib`, `router.json`, `evaluation.json`.
- UI: `frontend/src/App.jsx`.
- Verification/dependencies: `tests/conftest.py`, `test_department_routing.py`, `test_routing_integration.py`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`.
- Documentation: this file and the README link. The earlier extension plan remains a record of pre-implementation findings.

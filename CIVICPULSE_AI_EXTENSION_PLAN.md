# CivicPulse AI Extension Plan

Repository analysis: 25 October 2026; repository and integration points rechecked 26 October 2026. **Analysis and proposal only; no feature implementation.**

## 1. Existing architecture and actual pipeline

### Scope and evidence

Inspected all application Python and React source, package initializers, configuration, database setup scripts, training/download scripts, dependency manifests, frontend lockfile metadata, documentation, and the training run configuration. Inventoried static assets and training JPG artifacts; these are not runtime inference implementations. No repository `AGENTS.md` was found. Initial Git working tree was clean.

This checkout contains no `models/`, `data/`, `.env`, installed frontend dependencies, test suite, CI configuration, container setup, or migration directory. In particular, `models/yolov8_civic.pt` is absent (weights are gitignored). No database connection, provisioning script, training job, model download, or application startup was executed. Database statements below describe ORM/code behavior, not a verified live schema or existing records. Training images and arguments alone do not establish model accuracy or a completed training run.

### Source map: existing components

| Area | Exact source and symbols | Actual responsibility |
|---|---|---|
| API bootstrap | `backend/main.py`: `app`, `root_health_check` | FastAPI, permissive CORS, router prefix, `/` liveness response; no lifespan or model readiness check |
| Configuration | `backend/config.py`: `Settings`, `get_settings`, `settings` | Pydantic settings with v2/v1 branches and cached singleton; DB, YOLO and fusion settings |
| Ingestion and responses | `backend/api/endpoints.py`: `report_issue`, `ReportResponse` | Multipart upload, sequential AI calls, deduplication, merge/create, POST serialization |
| Reads and updates | Same file: `list_reports`, `StatusUpdate`, `update_report_status` | Priority-ordered list; uppercase status update |
| Vision | `backend/services/cv_service.py`: global `model`, `AnalysisResult`, `analyze_image`, `process_image_with_yolo`, `load_yolo_model` | Import-time custom YOLO loading, image verification, top-confidence detection |
| Text | `backend/services/nlp_service.py`: `get_nlp_pipeline`, `compute_text_urgency`, `_calculate_urgency_delta`, `_heuristic_urgency_score` | Lazy CPU sentiment pipeline plus civic keyword urgency rules |
| Fusion | `backend/services/fusion_service.py`: `PriorityLevel`, `_validate_and_clamp_score`, `fuse_scores`, `classify_priority`, `fuse_multimodal_ticket` | Existing numerical priority calculation; detailed helper is not used by endpoint |
| Spatial duplicates | `backend/services/deduplication.py`: `find_duplicate_ticket`, `_find_duplicate_fallback`, `_haversine_distance_meters` | Same-category OPEN tickets within 50 metres |
| Persistence | `backend/database/models.py`: `MasterTicket`, `check_postgis_support`, `USE_POSTGIS` | Only discovered application table: `master_tickets`; import-time PostGIS capability probe |
| DB lifecycle | `backend/database/session.py`: `engine`, `SessionLocal`, `Base`, `get_db` | Synchronous SQLAlchemy sessions, rollback on propagated exceptions, close on completion |
| Provisioning | `create_db.py`: `create_database`; `backend/init_db.py`: `init_database` | Create database; enable available PostGIS, create tables, ad hoc column/type upgrades |
| UI entry and build | `frontend/src/main.jsx`, `index.html`, `vite.config.js`, `src/index.css`, `eslint.config.js` | React StrictMode, Vite React/Tailwind plugins, Tailwind v4 import, ESLint |
| Citizen UI | `frontend/src/App.jsx`: `ReportModal`, `handleImageChange`, `removeImage`, `handleSubmit` | Required title/description/image, local preview, multipart submission |
| Dispatcher UI | Same file: `App`, `fetchTickets`, `resolveTicket`, `TicketCard`, `ScoreBar`, `PriorityBadge`, `StatusPill`, `FlyToActive`, `getMarkerIcon` | Polling, score-sorted cards and Leaflet markers, selection, optimistic resolution |
| Data preparation | `scripts/download_data.py`: `setup_kaggle_credentials`, `download_kaggle_potholes`, `download_roboflow_garbage`, `parse_args`, `main` | Explicit Kaggle/Roboflow dataset download CLI, not report processing |
| Training | `scripts/train_yolo.py`: `ensure_extracted`, `process_dataset_split`, `merge_datasets`, `train_yolo`, `parse_args`, `main` | Merge/remap datasets, train YOLOv8n, copy best/last weights to runtime path |

Package initializers re-export services and database symbols. In particular, importing a service submodule can execute `backend/services/__init__.py`, which imports CV and thus requires YOLO weights even for otherwise isolated utility tests. Database package imports likewise trigger the model capability probe.

`runs/train/civicpulse_yolov8/args.yaml` records CPU training, 10 epochs, batch 16, image size 640 and historical absolute paths on another drive. Training maps class 1 to `garbage_dumpster`; runtime explicitly maps class 1 to `garbage`, then API category `garbage_pile`. Do not silently alter this mapping. Static frontend assets comprise SVG branding/icons and `hero.png`; the application uses Lucide icons and Leaflet rather than a separate frontend service layer.

### Actual submission sequence

1. `ReportModal.handleSubmit` validates nonblank title/description and image presence. It sends `FormData` via Axios to `http://127.0.0.1:8000/api/v1/report`. **Latitude and longitude are hardcoded to 19.076 and 72.8777**, despite the “Auto-Tagged” label. There is no browser geolocation or map-based input. The UI advertises 10 MB and drag/drop, but no size validation or drop handler exists.
2. `app.include_router(..., prefix=settings.API_V1_PREFIX)` gives the default paths `POST /api/v1/report`, `GET /api/v1/report`, and `PATCH /api/v1/report/{ticket_id}`. Hidden GET/PATCH `/api/v1/tickets` aliases also exist. There is no POST alias. Preserve all of these and the configurable prefix.
3. `report_issue` receives required `title`, `description`, bounded numeric coordinates, and `UploadFile image`. It saves under working-directory-relative `data/raw/<uuid>_<basename>`, closes the upload, and then invokes vision. The filename UUID is not a request correlation ID. There is no server-side file-size limit in this handler.
4. `process_image_with_yolo` calls `analyze_image`: Pillow verifies readability, the existing global YOLO instance runs with configured threshold/device, and the highest-confidence box is selected. **Visual severity equals that confidence rounded to four decimals**, not bounding-box area or a weighted severity estimate. `TARGET_CLASS_WEIGHTS` is unused. No detections returns label `none`, score 0.0 and category `unclassified`.
5. The endpoint converts `unclassified` to `garbage_pile` if title plus description contains garbage/trash/waste, otherwise `pothole`. This fallback remains part of legacy category behavior.
6. `compute_text_urgency(description)` uses `distilbert/distilbert-base-uncased-finetuned-sst-2-english`, a sentiment classifier, **not a custom urgency-trained DistilBERT**. Input under five stripped characters returns 0.20. Otherwise keyword adjustments use all description tokens; transformer input is the first 512 **characters**. NEGATIVE gives base `0.35 + 0.15 * confidence`; other labels give `0.25 - 0.10 * confidence`; malformed non-list/empty predictions use 0.45. Keyword delta is `0.25 * min(2, high_hits) + 0.12 * min(2, medium_hits) - 0.15 * min(2, low_hits)`. Result is clamped to [0.10, 1.0] and rounded to four decimals. Regex tokens are individual English alphabetic words, so multiword keywords such as `gas leak` and `live wire` do not match as phrases.
7. Missing/failed NLP initialization or inference uses heuristic base 0.40 plus the same delta and clamp. An exception escaping the service is separately caught by the endpoint and becomes 0.30. Preserve both fallback layers.
8. `fuse_scores` validates/clamps inputs, normalizes nonnegative weights (default 0.50/0.50; both zero also becomes 0.50/0.50), adds 0.10 when both inputs are at least 0.80, clamps to [0,1], and rounds to four decimals. `classify_priority` uses CRITICAL >=0.80, HIGH >=0.60, MEDIUM >=0.40, otherwise LOW. The helper supports a missing visual modality but the actual API requires an image.
9. `find_duplicate_ticket(..., radius_meters=50)` chooses an existing match or the endpoint creates a new `MasterTicket`.
10. Duplicate: increment `report_count`, update timestamp, and raise stored final score/tier only when the incoming fused score is higher. Original title, description, image, coordinates, category, text urgency and visual severity remain unchanged. POST returns **incoming component scores but the master's potentially higher stored final score**. New ticket: store all original fields and scores, commit and refresh. Both branches return HTTP 201.
11. `ReportModal` ignores the POST body, clears the form, calls `onSuccess={fetchTickets}`, and closes. `fetchTickets` performs GET immediately; `App` also polls every 30 seconds. Server and frontend sort by final score. `TicketCard` displays original text/image, three score bars, count and coordinates. `resolveTicket` optimistically PATCHes RESOLVED and restores state on failure.

### Spatial duplicate behavior

`check_postgis_support` checks the installed extension once at module import; exceptions silently select the fallback. With PostGIS, `location` is `Geometry(POINT,4326, spatial_index=True)`; otherwise it is a WKT string column. Points use longitude before latitude.

PostGIS query: `status == OPEN`, case-insensitive category match via `ilike`, non-null geometry, `ST_DWithin(location::geography, reference::geography, 50)`. Distances are in metres. It selects the oldest `created_at`, **not the nearest ticket**. IN_PROGRESS and RESOLVED tickets are excluded. No time window or textual similarity exists.

Without PostGIS, the fallback fetches every OPEN same-category row with coordinates and checks Haversine distance in Python; the first match within the radius wins, with no explicit ordering. PostGIS query failure rolls back before fallback; fallback failure rolls back and returns None, allowing a new ticket. The endpoint also catches lookup exceptions and proceeds as new. No locking, idempotency protection or atomic increment is present, so concurrent submissions can create separate masters or lose count increments.

`init_database` uses `create_all`, adds missing `report_count`, and may convert VARCHAR WKT location to geometry. It is not a versioned migration system. The capability flag/ORM type is established before this script enables PostGIS, so a restart/reinspection is needed after extension changes. The configured geometry index should not be assumed to accelerate the geography cast: inspect live indexes and EXPLAIN before adding a matching geography expression index.

### Existing storage and API contract

`master_tickets` columns: integer primary key `id`; required `title` (255), `category` (100), `status` (50); nullable `description` Text, `location` geometry/string, latitude/longitude Float, `image_url` (1024), text/visual/final Float scores, `priority_level` (20); required `report_count` default 1; created/updated timezone-aware DateTime columns with server defaults. Index declarations cover id, category, status, final score, priority level and PostGIS geometry when enabled.

POST `ReportResponse` fields: `ticket_id`, `is_duplicate`, `message`, `category`, `final_priority_score`, `priority_level`, `text_urgency`, `visual_severity`, `report_count`, `status`. GET returns a list of dictionaries containing ticket_id, title, description, category, status, coordinates, image_url, the three scores, priority_level and report_count. GET does not include timestamps or `is_duplicate`. PATCH accepts arbitrary status strings, uppercases them, and returns ticket_id/status/message; unknown IDs return 404.

YOLO returns detected object, confidence, synthetic single-detection details, zero spatial coverage and metadata, but the endpoint persists only category/severity. NLP sentiment label/confidence are local variables and are neither returned nor stored. No embeddings, department, consistency result, inference duration, model revision, request ID or per-submission record exists. Duplicate uploads remain on disk without a corresponding new record. The API does not mount `StaticFiles` for `data/raw`, although the UI constructs backend image URLs and hides failed images.

### Initialization, logging, errors and execution

YOLO loads once per process at import from a hardcoded project-relative path; missing weights abort startup. `settings.YOLO_MODEL_PATH`, `custom_model_path`, and loader custom path/device arguments do not select another model. NLP lazily caches `_NLP_PIPELINE` on CPU (`device=-1`); `_PIPELINE_INIT_FAILED` suppresses subsequent initialization attempts for that process. Neither pattern has an explicit initialization lock. Cached settings and models are process-local, not shared across server workers.

The async POST performs synchronous disk, model and database work sequentially on its event loop. There is no BackgroundTasks usage, queue, worker, websocket, scheduler or durable job processing. GET/PATCH are synchronous handlers. SQLAlchemy uses pre-ping, a 10-connection pool plus 20 overflow for non-SQLite connections, and request-scoped session cleanup.

Module loggers exist, but `main.py` does not configure structured application logging. Provisioning/training/download scripts use plain-text `basicConfig`. Existing events cover initialization, some fallbacks, merge/create and fusion synergy; there are no timings or correlation IDs and no full prediction lifecycle.

Privacy gaps: engine initialization errors interpolate the full database URL; SQL echo follows DEBUG (the example environment enables it); SQLAlchemy exception strings can contain bound report parameters; commit error details are exposed to clients. Spatial logs contain coordinates and image errors can contain user-derived filenames. These need sanitization when logging is introduced, without changing score behavior.

Upload write failures return 500; corrupt/missing images return 400 and are deleted on that path; other CV errors return 500. Partial uploads and later failures can leave files. New-ticket persistence has explicit rollback/error wrapping; duplicate/PATCH failures rely on `get_db` rollback and generic framework handling. Validation failures occur before handler execution. No custom global exception handler exists.

## 2. Exact integration points for the three features

All additions belong inside the existing FastAPI application and service directory. Keep YOLO, DistilBERT, fusion formulas, category fallback and legacy response fields intact. New scores must never feed `fuse_scores`, alter its arguments, or replace existing component scores.

### Smart Department Routing — proposed

Call `predict_department(title, description, detected_category)` in `report_issue` after current fusion and before persistence. Separate preprocessing tokenizes the combined text, uses WordNet and Lesk for contextual senses, and supplies bounded lemma/sense features alongside ordinary token features to a trained Multinomial Naive Bayes classifier. Use identical preprocessing during training/inference; do not modify urgency tokens. Lesk is a sense selector, not the department classifier, and its overlap result is not a probability. [NLTK Lesk](https://www.nltk.org/api/nltk.wsd.html), [WordNet](https://www.nltk.org/howto/wordnet.html), [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html).

Reuse scikit-learn count-vectorization/pipeline conventions within the new module. Train offline on labeled civic reports and a versioned department code mapping. **No routing dataset, department directory, official BMC assignment mapping or trained classifier exists here.** Establish these before claiming automatic assignment. Do not infer official department names from model classes. Start with advisory predictions, a calibrated confidence/abstention threshold and `needs_review`; absent resources return unavailable/manual review, not a fabricated confidence. A configured category fallback, if later approved, must identify itself as a rule fallback.

For duplicate submissions, retain the master's department recommendation; record the new suggestion in submission diagnostics without rerouting an existing issue automatically.

### Semantic Duplicate Detection — proposed

Insert between current fusion and the merge/new-ticket branch. Extend the existing spatial service with `find_duplicate_candidates` while keeping `find_duplicate_ticket` available and behavior-compatible. Retrieve **all** OPEN, same-legacy-category candidates within the existing 50m radius; do not semantically score only the current `.first()` result.

New `semantic_duplicate_service.py` encodes title plus description once using a cached CPU SentenceTransformer, compares normalized vectors with cosine similarity, ranks candidates, and returns best candidate ID, score, threshold, model revision and decision reason. A compact checkpoint such as `sentence-transformers/all-MiniLM-L6-v2` is a candidate to benchmark, not a verified project dependency. The library supports CPU device selection and normalized encoding. [SentenceTransformer API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).

Use three modes: `off` (exact legacy lookup), `shadow` (measure semantic decision but retain legacy selection), and `enforce` (spatial candidate plus calibrated semantic threshold required). Default off; deploy shadow before enforcement. Enforce is an intentional duplicate-policy change and must be evaluated independently of scoring compatibility. If embeddings/resources are unavailable or candidate coverage is incomplete, record `degraded` and use the legacy lookup; do not interpret missing vectors as similarity zero. Resolve semantic ties deterministically by creation time then ID. No geographic expansion, cross-category auto-merge or status expansion in the first version.

Persist a normalized embedding for each master's original title/description; preserve that meaning after merges. Backfill legacy rows offline in batches without changing text, scores, count or status. Compare only matching model/preprocessing revisions and dimensions; invalidate and backfill after a model change. Do not quietly truncate candidate sets: page through them or explicitly fall back when a configured work bound is exceeded. No vector database/pgvector is needed for the initial geographically constrained workload.

### Image–Text Consistency Verification — proposed

Call `verify_image_text(saved_image_path, title, description)` after legacy inference/fusion and before persistence, using the already-saved image. Cache pretrained `CLIPModel` and `CLIPProcessor` on CPU, evaluation mode with inference-only execution. A candidate is `openai/clip-vit-base-patch32`; load the model's own processor and enforce its token limit explicitly. Reuse Pillow and transformers. [Hugging Face CLIP documentation](https://huggingface.co/docs/transformers/model_doc/clip).

Compute cosine similarity of normalized image/text embeddings. Store the raw similarity, calibrated threshold, model revision and `consistent`, `needs_review` or `unavailable` status. Similarity is not a probability; single-pair softmax is not a useful confidence. Preserve uncertainty for ambiguous/multihazard images, multilingual text, truncation and missing resources. Initially advisory only: never reject a report, reduce urgency, change category, gate duplicate merging or rewrite priority based on CLIP. Low consistency does not establish fraud. For a duplicate, its CLIP result describes the incoming submission, not the original master's photo.

## 3. Proposed module/file structure and CPU lifecycle

All paths below are additions unless marked existing:

```text
backend/services/department_service.py       predict_department; cached routing resources
backend/services/semantic_duplicate_service.py  encode_report; evaluate_candidates
backend/services/consistency_service.py      verify_image_text; cached CLIP resources
backend/logging_config.py                    JSON formatter, filter, timing helpers
backend/middleware/request_context.py        request ID middleware and ContextVar
backend/middleware/__init__.py
backend/database/migrations/001_ai_extensions.sql
scripts/train_department_router.py           offline training/evaluation/export
scripts/backfill_report_embeddings.py        resumable revision-aware backfill
models/department_router/                    trusted model/vectorizer/label metadata
tests/                                      regression, services, API, DB integration
```

Extend existing `endpoints.py`, `models.py`, `config.py`, `main.py`, `deduplication.py`, `.env.example`, and requirements when implementing. Update `App.jsx` only for optional advisory fields after backend compatibility is established. A new app, general-purpose orchestration framework or replacement service layer is unnecessary.

Reuse settings singleton, `get_db`, existing service wrappers, Haversine calculation, category output and priority helpers. `load_yolo_model` already returns a cached object; do not instantiate another YOLO for CLIP. `fuse_multimodal_ticket` can inform diagnostics, but its `weights_used` reports supplied/configured weights rather than always the actual normalized values: do not treat that metadata as authoritative without accounting for normalization.

New model getters should use a per-process lock, cached success/failure state, explicit CPU settings, model/preprocessing revisions, and initialization timing. Warm enabled new services once in a FastAPI lifespan hook after logging setup; keep failure optional/degraded. Pre-provision weights and WordNet corpus outside requests, so offline startup never tries repeated downloads. Do not refactor existing model loaders as part of numerical feature work without regression evidence.

To avoid making event-loop blocking worse, later offload inference stages through existing AnyIO/Starlette thread utilities with bounded concurrency. Keep database sessions out of concurrently running model tasks; serialize access to shared model instances until thread safety is demonstrated. Keep POST synchronous from the client's perspective, returning a complete committed result; no 202/job protocol or request-scoped DB session in background work. Thread cancellation does not stop a running CPU inference, so concurrency bounds matter. Benchmark CPU threads, memory and worker count; every process duplicates model memory. Start sequentially, not with unbounded parallel inference.

Proposed configuration includes feature flags, semantic off/shadow/enforce mode, resource paths/model revisions, validation-derived thresholds, CPU concurrency limits and log level. Add settings to both existing Pydantic branches. Thresholds remain unset/advisory until measured; do not publish invented accuracy or latency targets.

## 4. Required dependency changes

Existing backend declares FastAPI, SQLAlchemy, GeoAlchemy2, psycopg2, Pillow, NumPy, torch/torchvision, Ultralytics, transformers, Pydantic settings, multipart and AnyIO. Requirements mix exact pins with open-ended lower bounds; no Python lockfile is present. `opentelemetry-api` is listed but unused; it is not a configured telemetry pipeline. Frontend already has Axios, React, Leaflet and Tailwind; no new frontend package is needed.

Add direct runtime requirements for `nltk`, `scikit-learn`, and `sentence-transformers`, selecting tested compatible versions in a clean environment. If importing `joblib` for trusted routing artifact loading, declare it directly rather than relying on a transitive dependency. CLIP reuses transformers/torch/Pillow; JSON logging reuses the standard library. No additional CLIP implementation, logging vendor SDK, Celery, Redis, pgvector or external inference API is required. WordNet corpus and pretrained weights are provisioned artifacts, not pip dependencies. Add pytest as a development dependency; httpx is already declared for API tests.

Resolve new requirements without casually upgrading the existing numerical stack. Record CPU wheel selection and artifact hashes; test both existing AI outputs and new APIs with the resolved versions. No dependency installation or compatibility claim was made during this audit. README's Python 3.8+/Node 18+ instructions are not an established supported matrix: the checked-in lockfile resolves Vite 8.3.4 with Node `^20.19.0 || >=22.12.0`, and React 19.3.1. Validate actual Python requirements from the chosen resolved stack before documenting a supported version.

## 5. Required database migrations

Proposal: additive versioned SQL migrations run explicitly before enabling features, reusing SQLAlchemy/PostgreSQL; no destructive `create_all` workaround. Do not run provisioning at application startup. Inspect a read-only schema snapshot and back up the actual database before migration work. Rehearse on a copy with and without PostGIS and confirm existing row values and IDs are unchanged.

| Proposed schema | Purpose and constraints |
|---|---|
| Nullable `master_tickets.department_code` String(100), `department_confidence` Float, `department_status` String(32), `department_model_revision` String(255) | Original/master advisory routing, null for historical rows; confidence [0,1] when supplied |
| New `ticket_embeddings` | `ticket_id` FK to master_tickets plus `model_revision` and `preprocessing_version` composite key; `embedding` JSON, `dimension` integer, `created_at` timestamp; internal normalized vectors only |
| New `report_ai_assessments` | Integer PK; non-null unique request UUID; `ticket_id` FK; `created_at`; `is_duplicate`; `analysis_version`; JSON diagnostics for incoming routing, semantic match, CLIP, legacy component/final scores, stage timings and fallback statuses |

These are proposed tables; only `master_tickets` exists in source today. The assessment JSON should have typed application validation and a versioned shape. It contains no copy of raw image/text, file names or secrets. It preserves per-submission AI provenance that the existing aggregate loses. Do not pretend old individual submissions can be reconstructed from report_count. CLIP is stored per assessment, not misleadingly overwriting master evidence. Embeddings are sensitive derived data: restrict access, exclude from public responses/logs, and define retention with report retention.

Persist master merge/create and its assessment in one transaction, obtaining the new ticket ID with flush before final commit. Embedding rows are optional until ready. Validate optional outputs before writing; if their persistence fails, roll back the whole attempt and retry the legacy transaction at most once with the same request identity, never commit an increment twice. Return success only after committed persistence. Use an explicit atomic increment/row-lock strategy to avoid losing concurrent duplicate updates; this alone does not solve the race to create two new masters. Add concurrency regression coverage before enabling changed merge logic.

Use nullable additions for old rows; absence means not evaluated, not a numeric zero. Backfill only embeddings by default. Do not recompute historic urgency, severity or priority. Index ticket_id/created_at for assessment lookup and unique embedding identity; add geography indexing only after live query-plan evidence. Disabling feature flags must work with the new schema left in place; application rollback should not drop tables or delete records. Existing databases need an explicit migration, since `create_all` does not update existing tables.

## 6. Proposed API response additions

Keep multipart inputs, endpoint paths/aliases, status codes, list shape, existing field meanings and PATCH behavior. Extend POST's `ReportResponse` explicitly; otherwise Pydantic response filtering would discard additions. Add equivalent optional nested data in GET's manual dictionary serializer. Suggested additions:

```json
{
  "request_id": "server-generated UUID for this POST",
  "ai_analysis": {
    "version": 1,
    "scope": "submission",
    "department": {"code": null, "confidence": null, "status": "unavailable", "model_revision": null},
    "semantic_duplicate": {"mode": "shadow", "status": "not_evaluated", "candidate_ticket_id": null, "similarity": null, "threshold": null, "decision": null, "model_revision": null},
    "image_text_consistency": {"status": "unavailable", "similarity": null, "threshold": null, "model_revision": null}
  }
}
```

This is an illustrative extension fragment, not a replacement response or measured prediction. Use explicit unavailable/disabled/not_evaluated statuses rather than fake confidence zeros. Similarities are cosine values, distinct from [0,1] confidence; validate finite values and valid ranges. Do not expose embeddings, paths, exception text or raw model prompts.

POST analysis describes the incoming submission, including on merges. GET should expose optional master `department` and `latest_ai_assessment` with assessment ID/request ID/time and submission scope; label these as latest-submission evidence so they are not mistaken for original-photo analysis. Historical tickets return null additions. The current GET request has its own response header ID, distinct from a stored assessment ID. Leave legacy score inconsistencies on merges documented rather than silently redefining fields.

Return `X-Request-ID` on success, validation failure and handled server errors, and expose it in CORS if the frontend reads it. Do not accept arbitrary client text as a log identifier. Initially no frontend change is required because current code ignores new keys; later add nullable department/consistency badges to `TicketCard`, preserving map, polling, sorting, submission and resolution. Test other consumers for strict schema assumptions before rollout.

## 7. Logging and diagnostics design

Configure standard-library JSON logging before importing backend routers/services so import-time model/DB events are captured. Use an allowlisted formatter and `ContextVar` request context. Generate a fresh server UUID for every request; if client correlation is useful, accept a separate validated bounded UUID and never use it as a uniqueness/idempotency guarantee. Reset context in `finally`, propagate it into inference thread work, and attach ticket ID only after selection/flush. Startup events carry startup context rather than a fabricated report ID.

Common fields: UTC timestamp, level, event, logger, request_id, ticket_id when known, stage, outcome, duration_ms, model_id/revision, device, fallback reason code, and sanitized error type. Measure elapsed durations with `time.perf_counter`, including failure paths; distinguish model initialization, queue wait, inference and DB commit latency. Never use ticket/request IDs as metric labels.

| Event family | Safe diagnostic fields |
|---|---|
| request.received / request.completed / request.failed | route template, method, status code, total duration, request ID; covers validation failures |
| upload.saved / upload.failed | size if available, outcome; no content or original name |
| model.initialization.started / completed / failed | model identifier/revision, CPU device, duration, cached failure state |
| inference.started / completed / fallback / failed | stage, safe label/category code, score, confidence when actually available, duration, reason code |
| fusion.completed | existing inputs/output, normalized effective weights and synergy decision; no rescoring |
| duplicate.evaluated | spatial method, radius, candidate count, mode, selected ticket ID, cosine/threshold, shadow vs actual decision, fallback status |
| database.persist.started / committed / failed | create/merge, ticket ID, duration, rollback outcome |

Instrument inside `compute_text_urgency` around the existing sentiment result to observe label/confidence without changing its float return. Log keyword hit counts only, not matched user tokens. CV diagnostics can use its existing result dictionary after excluding metadata paths. Log both incoming fused score and retained master score on merge, naming them distinctly. A service's own fallback must be distinguishable from successful model inference.

Never log raw uploads/base64, title, description, tokens, embeddings, exact coordinates, original filenames, multipart bodies, authorization headers, environment settings or connection URLs. Replace existing raw exception interpolation with stable error codes/types; disable SQL echo and hide SQL parameter values. Avoid `logger.exception` unless exception messages/trace output are sanitized, since database exceptions embed SQL parameters. Safe client errors should return a generic message and request ID. Apply filters to third-party/model/SQL logs as well, and verify framework validation/access logs do not echo bodies or sensitive query strings. Configure log retention/access outside the application; optional aggregate latency/fallback counters can be derived from structured logs without adding a monitoring stack.

## 8. Testing strategy

No automated tests are present. This documentation-only audit did not run the application, inference, build or database mutations. Implementation verification should include:

1. **Legacy characterization first:** fixed YOLO outputs (top-confidence selection, no detections, invalid image), exact DistilBERT label/confidence mapping, keyword cap behavior, short text, all existing fallbacks, fusion weight normalization/synergy/rounding and tier boundaries. Snapshot current behavior rather than correcting it incidentally. Service package import side effects require pre-import mocks for isolated tests or a separately reviewed lazy-export adjustment.
2. **API contract tests:** required multipart fields, coordinate validation, preserved 201/404/error behavior, aliases, list shape, arbitrary existing status strings, merge count and max-priority semantics, incoming-vs-master scores, and old clients ignoring nullable additions. Features disabled must preserve legacy fields and decisions exactly.
3. **Spatial integration:** isolated PostgreSQL/PostGIS and coordinate fallback fixtures; 50m boundary, category mismatch, OPEN-only filtering, oldest-vs-fallback ordering, null geometry/coordinates, database error rollback, multiple candidates and concurrency. SQLite mocks cannot validate geography semantics.
4. **Routing evaluation:** held-out department labels, class imbalance, ambiguous words, empty/unseen words, missing WordNet/artifact, abstention and confidence calibration. Split related incidents across train/test carefully to prevent leakage. Report per-class precision/recall and confusion matrix; no accuracy claims without data.
5. **Semantic evaluation:** paraphrase duplicates vs nearby distinct hazards, high text similarity far away, category/status mismatch, missing/stale/wrong-dimension/zero-norm vectors, threshold boundaries and ties. Evaluate false merges separately from missed duplicates on labeled local incident pairs. Verify shadow mode never changes the legacy chosen ticket.
6. **CLIP evaluation:** matched/mismatched evidence, unrelated but visually similar photos, mixed hazards, invalid files, tokenizer truncation, English/Hindi/Marathi/transliterated reports and unavailable weights. Calibrate on held-out pairs; verify CLIP never changes score, category or report acceptance.
7. **Lifecycle/performance:** cold vs warm timings, exactly one initialization per process under concurrent requests, offline model failure, bounded CPU work, memory per worker, responsive GET/PATCH during inference and no shared-session concurrency.
8. **Migration/persistence:** populated copy before/after comparison, both location storage forms, resumable backfill, model revision changes, rollback/retry without double count, application downgrade with added schema left intact, and no reconstructed fake history.
9. **Privacy/UI:** log-capture assertions with sentinel secrets/text/filename/coordinates, request IDs on 422/500 and no context bleed between requests; submit, refresh/poll, map selection, resolve/error rollback and null additions. Run existing frontend build/lint after actual UI changes.

## 9. Risks and backward compatibility considerations

- Missing YOLO weights currently block API import; deployment artifacts must be restored through the existing path before real validation. Do not substitute a generic model or retrain as part of extension setup.
- Hardcoded citizen coordinates collapse all same-category UI reports into one spatial area. Semantic validation must use representative coordinates; a future location input fix should be a separate, explicit UI change.
- There is no verified BMC routing taxonomy/training dataset. English WordNet/SST-2 and candidate English embedding models need evaluation for Mumbai languages; unsupported language should lead to uncertainty rather than an assured dispatch.
- Sentiment confidence is not urgency probability, visual confidence is not geometric damage severity, and cosine is not correctness probability. Preserve existing names/contracts but label new diagnostics accurately.
- Semantic enforcement changes aggregation behavior even when scoring is unchanged. Stage it behind a feature mode and measure false merges; preserve legacy fallback and never merge solely because text is similar.
- Merges currently lose submission provenance and may show component scores inconsistent with the retained maximum priority. New assessments clarify this without overwriting historic fields.
- CPU latency and memory increase with two transformer models; initialization downloads and worker multiplication can dominate. Provision offline and benchmark before selecting concurrency limits.
- Import-time PostGIS detection may disagree with the deployed schema after failures or extension upgrades. Inspect actual column types/indexes and restart after provisioning; do not combine a PostGIS conversion with AI migration by default.
- Existing lack of auth, permissive CORS, unlimited uploads, missing static image serving, arbitrary statuses and raw DB error disclosure are documented findings, not implemented fixes. Logging sanitization is needed for the requested privacy guarantee; broader hardening should remain clearly scoped.
- Model version upgrades can change outputs. Freeze the existing scoring stack during extension work, validate new model artifacts, and load serialized routing artifacts only from trusted deployment storage.

## 10. Recommended implementation order

1. Restore/verify existing deployment prerequisites, inspect real schema read-only, and establish baseline API/scoring/spatial tests. Confirm department label ownership and obtain evaluation datasets. Do not alter existing scoring logic.
2. Introduce request IDs, structured sanitized logging and timings; verify privacy and numerical equivalence. Keep synchronous POST semantics and add bounded inference execution only with regression tests.
3. Apply additive schema migration on a tested database copy, then deploy with all new features disabled. Establish optional analysis response schemas and model lifecycle configuration.
4. Train/evaluate and integrate advisory department routing; verify offline cached loading and abstention. No automatic reroute of merged masters.
5. Build spatial candidate retrieval, cached Sentence-BERT and embedding backfill; run semantic mode in shadow and compare with legacy decisions. Enable enforcement only after held-out and concurrency checks pass.
6. Add CPU-cached CLIP as advisory evidence, calibrate thresholds and verify that all original scores remain unchanged.
7. Add optional dispatcher badges with null-safe handling, then run end-to-end, privacy and CPU-load checks. Roll back via feature flags without removing records or columns.

The resulting pipeline stays in `report_issue`: **existing upload → existing YOLO/category fallback → existing DistilBERT urgency → unchanged Late Fusion → proposed routing and CLIP annotations → existing spatial search extended with optional semantic comparison → existing master merge/create plus proposed assessment persistence → existing GET/polling dashboard**. None of the three new modules replaces the current AI models, endpoints, database master records or frontend workflow.

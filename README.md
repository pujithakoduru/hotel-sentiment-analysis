# Hotel Aspect-Based Sentiment Analysis (ABSA)

An end-to-end NLP web application that extracts fine-grained hotel aspects from customer reviews (e.g., room, staff, food, cleanliness, location, price, amenities, noise) and predicts nuanced sentiment (**Positive / Neutral / Negative**) per individual aspect.

Built with **Python**, **Hugging Face Transformers** (`deberta-v3-base-absa-v1.1`), **PyTorch**, **Flask**, **Pandas**, **Bootstrap 5**, and **Chart.js**.

---

## Key Features & Solved Drawbacks

- **Zero-Crash Resilient Architecture**: Non-blocking asynchronous model loading with an intelligent lexicon fallback pipeline so the server starts immediately and stays responsive even if downloading weights or operating offline.
- **Asynchronous Batch Processing**: Background multi-job worker queue with real-time polling progress bars, ensuring long batch inference requests never time out the HTTP server.
- **Multi-Tenant / Concurrency Safety**: Unique job-scoped result storage (`results/analysis_<job_id>.csv`) avoiding file write collisions and race conditions between multiple users.
- **Enhanced Aspect Vocabulary & Disambiguation**: Expanded aspect taxonomy including idiomatic phrase filters (e.g., handles "bar of soap", "room for improvement", "near future" without misclassification).
- **Smart Clause Segmentation**: Splits compound sentences at contrast conjunctions (*but*, *however*, *although*, *though*) and coordinating conjunctions (*and* with predicate clauses) so mixed sentiments don't bleed into adjacent aspects.
- **Executive Analytics Dashboard**: Interactive KPI metrics, Net Sentiment Score (NSS = %Positive - %Negative) by aspect, and dynamic Chart.js visualizations.

---

## Directory Structure

```
├── app.py                      # Flask server, async batch worker queue, REST API
├── nlp_pipeline.py             # Preprocessing, aspect extraction, ABSA inference & fallback
├── requirements.txt            # Project dependencies
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Shared layout, header, footer, status poller
│   ├── index.html              # Hero landing page & capability overview
│   ├── analyze.html            # Interactive single review ABSA tester
│   ├── batch.html              # Drag-and-drop CSV upload, progress bar & filterable table
│   └── dashboard.html          # Executive analytics, KPIs & Chart.js charts
├── static/
│   ├── css/style.css           # Modern design tokens, responsive layout & custom components
│   └── js/main.js              # Fetch client, polling worker & dynamic charts
├── data/
│   └── sample_reviews.csv      # Sample hotel review dataset for instant testing
└── results/                    # Auto-generated CSV outputs for batch runs
```

---

## Installation & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your web browser.

---

## REST API Endpoints

- `GET /api/health`: Health status of server and ABSA model.
- `POST /api/analyze`: Single review analysis. Payload: `{"review": "text..."}`.
- `POST /api/batch`: Multipart CSV file upload (initiates asynchronous background job).
- `GET /api/batch/status/<job_id>`: Check job progress percentage and retrieve completed results.
- `GET /api/dashboard`: Aggregated metrics and aspect breakdowns.
- `GET /download`: Export results as CSV (optional query param: `?job_id=<id>`).
- `GET /api/sample/download`: Download sample CSV for testing.

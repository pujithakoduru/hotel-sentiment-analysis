# Hotel Aspect-Based Sentiment Analysis (ABSA)

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/pujithakoduru/hotel-sentiment-analysis)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/pujithakoduru/hotel-sentiment-analysis)

An end-to-end NLP web application that extracts fine-grained hotel aspects from customer reviews (e.g., room, staff, food, cleanliness, location, price, amenities, noise) and predicts nuanced sentiment (**Positive / Neutral / Negative**) per individual aspect.

Built with **Python**, **Hugging Face Transformers** (`deberta-v3-base-absa-v1.1`), **PyTorch**, **Flask**, **Pandas**, **Bootstrap 5**, and **Chart.js**.

---

## 1-Click Cloud Deployment (Render)

Deploy this project on **Render** (100% Free):

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/pujithakoduru/hotel-sentiment-analysis)

Or manually:
1. Log in to [https://dashboard.render.com](https://dashboard.render.com) with GitHub.
2. Click **New +** ➔ **Web Service**.
3. Select `pujithakoduru/hotel-sentiment-analysis`.
4. Runtime: `Python 3`, Build Command: `pip install -r requirements.txt`, Start Command: `python app.py`.
5. Select the **Free** plan and click **Deploy Web Service**.

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
├── render.yaml                 # 1-Click Render Blueprint configuration
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

## Local Installation & Running

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

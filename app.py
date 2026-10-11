"""
Flask server: web interface routes, REST API, asynchronous batch processing,
and resilient aspect-based sentiment analysis integration.
"""
import os
import uuid
import threading
import time
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename
from nlp_pipeline import (
    build_results,
    extract_aspects,
    get_model_status,
    load_model,
    preprocess_review,
    summarize_by_aspect,
)

# ---------------------------------------------------------------------------
# App initialization & folder management
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit

MAX_REVIEW_CHARS = 4000
MAX_BATCH_ROWS = 1000
DEFAULT_RESULTS_CSV = os.path.join(RESULTS_DIR, "last_analysis.csv")
SAMPLE_CSV_PATH = os.path.join(DATA_DIR, "sample_reviews.csv")
REVIEW_COLUMN_NAMES = ["review", "reviews", "review_text", "text", "content", "comment", "feedback"]

# In-memory background jobs registry
JOBS = {}
JOBS_LOCK = threading.Lock()


def error_response(message, status=400):
    return jsonify({"error": message, "status": "error"}), status


def find_review_column(dataframe):
    """Find the review text column (case-insensitive) among common names."""
    lowered = {str(col).lower().strip(): col for col in dataframe.columns}
    for name in REVIEW_COLUMN_NAMES:
        if name in lowered:
            return lowered[name]
    return None


def run_batch_worker(job_id, reviews, save_file_path):
    """Background worker thread executing ABSA for batch reviews without blocking HTTP workers."""
    try:
        def update_progress(current, total):
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = current
                    JOBS[job_id]["percent"] = int((current / max(total, 1)) * 100)

        rows = build_results(reviews, progress_callback=update_progress)
        
        if not rows:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = "No hotel aspects (room, staff, food, cleanliness, etc.) were found in these reviews."
            return

        df = pd.DataFrame(rows)
        # Save job-specific results
        df.to_csv(save_file_path, index=False)
        # Update latest analysis copy for instant dashboard view
        df.to_csv(DEFAULT_RESULTS_CSV, index=False)

        summary = summarize_by_aspect(rows)

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["progress"] = len(reviews)
            JOBS[job_id]["percent"] = 100
            JOBS[job_id]["results"] = rows
            JOBS[job_id]["summary"] = summary
            JOBS[job_id]["total_mentions"] = len(rows)
            JOBS[job_id]["reviews_count"] = len(reviews)
            JOBS[job_id]["completed_at"] = time.time()

    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(exc)


# ---------------------------------------------------------------------------
# Page Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze")
def analyze_page():
    return render_template("analyze.html")


@app.route("/batch")
def batch_page():
    return render_template("batch.html")


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------
@app.route("/api/health")
def api_health():
    """Health check returning backend and ABSA model statuses."""
    status_info = get_model_status()
    has_results = os.path.exists(DEFAULT_RESULTS_CSV)
    return jsonify({
        "status": "healthy",
        "model_status": status_info,
        "has_cached_analysis": has_results
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Analyze a single review string, extract aspects, and classify sentiment."""
    data = request.get_json(silent=True) or {}
    review = data.get("review")

    if not isinstance(review, str) or not review.strip():
        return error_response("Please enter a review before analyzing.")
    if len(review) > MAX_REVIEW_CHARS:
        return error_response(f"Review is too long. Limit is {MAX_REVIEW_CHARS} characters.")

    rows = build_results([(1, review)])
    summary = summarize_by_aspect(rows)

    return jsonify({
        "status": "success",
        "results": rows,
        "summary": summary,
        "aspect_count": len(rows),
        "clean_text": preprocess_review(review)
    })


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """
    Accepts CSV upload and initiates asynchronous or synchronous processing.
    Returns job_id for tracking and instant progress updates.
    """
    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename == "":
        return error_response("Choose a CSV file to upload.")
    if not uploaded.filename.lower().endswith(".csv"):
        return error_response("Only .csv files are supported.")

    try:
        dataframe = pd.read_csv(uploaded)
    except pd.errors.EmptyDataError:
        return error_response("The CSV file is empty.")
    except Exception as exc:
        return error_response(f"Could not read the file: {str(exc)}. Please upload a valid UTF-8 CSV.")

    review_column = find_review_column(dataframe)
    if review_column is None:
        return error_response(
            "No review column detected. Please name your text column one of: "
            + ", ".join(REVIEW_COLUMN_NAMES)
        )

    # Clean reviews and enforce row limit
    reviews_series = dataframe[review_column].dropna().astype(str)
    reviews_series = reviews_series[reviews_series.str.strip() != ""]
    if reviews_series.empty:
        return error_response("The review column contains no usable text.")

    truncated = len(reviews_series) > MAX_BATCH_ROWS
    reviews_series = reviews_series.head(MAX_BATCH_ROWS)

    reviews = [(int(index) + 1, text[:MAX_REVIEW_CHARS]) for index, text in reviews_series.items()]

    # Create job tracking
    job_id = str(uuid.uuid4())[:8]
    job_file_path = os.path.join(RESULTS_DIR, f"analysis_{job_id}.csv")

    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "progress": 0,
            "percent": 0,
            "total": len(reviews),
            "filename": secure_filename(uploaded.filename),
            "created_at": time.time(),
            "truncated": truncated
        }

    # Launch worker in background thread
    worker = threading.Thread(
        target=run_batch_worker,
        args=(job_id, reviews, job_file_path),
        daemon=True
    )
    worker.start()

    return jsonify({
        "status": "started",
        "job_id": job_id,
        "total_reviews": len(reviews),
        "truncated": truncated,
        "note": f"Analyzing first {MAX_BATCH_ROWS} reviews." if truncated else ""
    })


@app.route("/api/batch/status/<job_id>")
def api_batch_status(job_id):
    """Poll progress status of an active batch job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return error_response(f"Job {job_id} not found.", 404)
        return jsonify(job)


@app.route("/api/dashboard")
def api_dashboard():
    """Retrieve summary metrics and aspect breakdowns from the latest analysis."""
    job_id = request.args.get("job_id")
    target_csv = DEFAULT_RESULTS_CSV
    if job_id:
        custom_csv = os.path.join(RESULTS_DIR, f"analysis_{job_id}.csv")
        if os.path.exists(custom_csv):
            target_csv = custom_csv

    if not os.path.exists(target_csv):
        return error_response("No analysis has been completed yet. Upload a CSV on the Batch page.", 404)

    try:
        df = pd.read_csv(target_csv).fillna("")
        rows = df.to_dict(orient="records")
        summary = summarize_by_aspect(rows)

        # Compute high-level KPI metrics
        total_mentions = len(rows)
        total_pos = sum(r["positive"] for r in summary.values())
        total_neg = sum(r["negative"] for r in summary.values())
        pos_rate = round((total_pos / max(total_mentions, 1)) * 100, 1)

        best_aspect = max(summary.items(), key=lambda x: x[1].get("net_sentiment_score", -100))[0] if summary else "None"
        worst_aspect = min(summary.items(), key=lambda x: x[1].get("net_sentiment_score", 100))[0] if summary else "None"

        return jsonify({
            "status": "success",
            "summary": summary,
            "total_mentions": total_mentions,
            "positive_rate": pos_rate,
            "total_positive": total_pos,
            "total_negative": total_neg,
            "best_aspect": best_aspect,
            "worst_aspect": worst_aspect,
            "recent_mentions": rows[:15]
        })
    except Exception as exc:
        return error_response(f"Error reading dashboard data: {str(exc)}", 500)


@app.route("/download")
def download():
    """Download analysis results as CSV."""
    job_id = request.args.get("job_id")
    target_csv = DEFAULT_RESULTS_CSV
    if job_id:
        custom_csv = os.path.join(RESULTS_DIR, f"analysis_{job_id}.csv")
        if os.path.exists(custom_csv):
            target_csv = custom_csv

    if not os.path.exists(target_csv):
        return error_response("Nothing to download yet. Run a batch analysis first.", 404)

    filename = f"hotel_absa_results_{job_id}.csv" if job_id else "hotel_absa_results.csv"
    return send_file(target_csv, as_attachment=True, download_name=filename)


@app.route("/api/sample/download")
def download_sample():
    """Download the provided sample hotel reviews CSV."""
    if not os.path.exists(SAMPLE_CSV_PATH):
        return error_response("Sample CSV file not found.", 404)
    return send_file(SAMPLE_CSV_PATH, as_attachment=True, download_name="sample_hotel_reviews.csv")


@app.errorhandler(413)
def too_large(_):
    return error_response("File is too large. Limit is 10 MB.", 413)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    # Attempt non-blocking model load in background thread
    threading.Thread(target=load_model, daemon=True).start()
    app.run(host="0.0.0.0", port=port, debug=False)

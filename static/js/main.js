/**
 * Hotel ABSA - Interactive Frontend Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    initHealthCheck();
    initSingleAnalyzePage();
    initBatchPage();
    initDashboardPage();
});

// ---------------------------------------------------------------------------
// Model Health & Status Poller
// ---------------------------------------------------------------------------
function initHealthCheck() {
    const statusText = document.getElementById("modelStatusText");
    const statusBadge = document.getElementById("modelStatusBadge");
    if (!statusText || !statusBadge) return;

    fetch("/api/health")
        .then(res => res.json())
        .then(data => {
            const ms = data.model_status || {};
            if (ms.status === "ready") {
                statusText.textContent = "DeBERTa Model Active";
                statusBadge.className = "badge rounded-pill bg-success-subtle text-success border border-success-subtle d-flex align-items-center gap-1 py-2 px-3";
            } else if (ms.status === "fallback") {
                statusText.textContent = "Fast Lexicon Fallback";
                statusBadge.className = "badge rounded-pill bg-warning-subtle text-warning border border-warning-subtle d-flex align-items-center gap-1 py-2 px-3";
            } else {
                statusText.textContent = "Model Initializing...";
                statusBadge.className = "badge rounded-pill bg-info-subtle text-info border border-info-subtle d-flex align-items-center gap-1 py-2 px-3";
            }
        })
        .catch(() => {
            statusText.textContent = "Server Connected";
        });
}

// ---------------------------------------------------------------------------
// Single Review Analysis
// ---------------------------------------------------------------------------
function initSingleAnalyzePage() {
    const form = document.getElementById("analyzeForm");
    const textarea = document.getElementById("reviewInput");
    const charCount = document.getElementById("charCount");
    const sampleBtns = document.querySelectorAll(".sample-btn");
    const resultsSection = document.getElementById("resultsSection");
    const loadingState = document.getElementById("loadingState");
    const errorAlert = document.getElementById("errorAlert");
    const errorMessage = document.getElementById("errorMessage");

    if (!form || !textarea) return;

    // Character counter
    textarea.addEventListener("input", () => {
        charCount.textContent = `${textarea.value.length} / 4000 characters`;
    });

    // Sample buttons
    sampleBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            textarea.value = btn.getAttribute("data-text");
            charCount.textContent = `${textarea.value.length} / 4000 characters`;
            form.requestSubmit();
        });
    });

    form.addEventListener("submit", (e) => {
        e.preventDefault();
        const reviewText = textarea.value.trim();
        if (!reviewText) return;

        // UI Reset
        errorAlert.classList.add("d-none");
        resultsSection.classList.add("d-none");
        loadingState.classList.remove("d-none");

        fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ review: reviewText })
        })
        .then(res => res.json().then(data => ({ ok: res.ok, data })))
        .then(({ ok, data }) => {
            loadingState.classList.add("d-none");
            if (!ok) {
                errorMessage.textContent = data.error || "An error occurred during analysis.";
                errorAlert.classList.remove("d-none");
                return;
            }

            renderSingleResults(data);
        })
        .catch(err => {
            loadingState.classList.add("d-none");
            errorMessage.textContent = "Network error: " + err.message;
            errorAlert.classList.remove("d-none");
        });
    });
}

function renderSingleResults(data) {
    const resultsSection = document.getElementById("resultsSection");
    const cardsContainer = document.getElementById("aspectCardsContainer");
    const tableBody = document.getElementById("summaryTableBody");
    const aspectCountBadge = document.getElementById("aspectCountBadge");

    cardsContainer.innerHTML = "";
    tableBody.innerHTML = "";

    const rows = data.results || [];
    aspectCountBadge.textContent = `${rows.length} Aspect${rows.length === 1 ? '' : 's'} Found`;

    if (rows.length === 0) {
        cardsContainer.innerHTML = `
            <div class="col-12">
                <div class="alert alert-info rounded-3">
                    <i class="bi bi-info-circle me-2"></i> No specific hotel aspects (room, staff, food, cleanliness, etc.) were found in this text.
                </div>
            </div>`;
        resultsSection.classList.remove("d-none");
        return;
    }

    const aspectIcons = {
        room: "bi-door-closed",
        cleanliness: "bi-stars",
        staff: "bi-people",
        food: "bi-cup-hot",
        location: "bi-geo-alt",
        price: "bi-tag",
        amenities: "bi-wifi",
        noise: "bi-volume-up"
    };

    rows.forEach(r => {
        const sentimentClass = r.sentiment === "positive" ? "success" : (r.sentiment === "negative" ? "danger" : "secondary");
        const icon = aspectIcons[r.aspect.toLowerCase()] || "bi-card-text";
        const confPct = Math.round(r.confidence * 100);

        // Card
        const col = document.createElement("div");
        col.className = "col-md-6 col-lg-4";
        col.innerHTML = `
            <div class="card h-100 border-0 shadow-sm rounded-4 border-start border-4 border-${sentimentClass} p-3">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <div class="d-flex align-items-center gap-2">
                        <i class="bi ${icon} text-${sentimentClass} fs-5"></i>
                        <h6 class="fw-bold mb-0 text-capitalize">${r.aspect}</h6>
                    </div>
                    <span class="badge bg-${sentimentClass}-subtle text-${sentimentClass} border border-${sentimentClass}-subtle text-capitalize px-2 py-1">
                        ${r.sentiment}
                    </span>
                </div>
                <p class="text-secondary small mb-2 fst-italic">"${r.evidence}"</p>
                <div class="mt-auto">
                    <div class="d-flex justify-content-between extra-small text-muted mb-1">
                        <span>Confidence</span>
                        <span>${confPct}%</span>
                    </div>
                    <div class="progress rounded-pill" style="height: 6px;">
                        <div class="progress-bar bg-${sentimentClass}" style="width: ${confPct}%"></div>
                    </div>
                </div>
            </div>`;
        cardsContainer.appendChild(col);

        // Table row
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td class="fw-bold text-capitalize"><i class="bi ${icon} me-1 text-muted"></i> ${r.aspect}</td>
            <td class="small text-secondary">"${r.evidence}"</td>
            <td><span class="badge bg-${sentimentClass}-subtle text-${sentimentClass} text-capitalize">${r.sentiment}</span></td>
            <td>${confPct}%</td>
            <td><span class="badge bg-light text-muted border">${r.model || 'ABSA'}</span></td>
        `;
        tableBody.appendChild(tr);
    });

    resultsSection.classList.remove("d-none");
    resultsSection.scrollIntoView({ behavior: "smooth" });
}

// ---------------------------------------------------------------------------
// Batch CSV Analysis
// ---------------------------------------------------------------------------
let allBatchRows = [];

function initBatchPage() {
    const form = document.getElementById("batchUploadForm");
    const fileInput = document.getElementById("fileInput");
    const dropZone = document.getElementById("dropZone");
    const startBtn = document.getElementById("startBatchBtn");
    const selectedFileName = document.getElementById("selectedFileName");
    const progressContainer = document.getElementById("progressContainer");
    const resultsSection = document.getElementById("batchResultsSection");
    const errorAlert = document.getElementById("batchErrorAlert");
    const errorMessage = document.getElementById("batchErrorMessage");

    if (!form || !fileInput) return;

    // Dropzone interactions
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelected();
        }
    });

    fileInput.addEventListener("change", handleFileSelected);

    function handleFileSelected() {
        if (fileInput.files.length) {
            const file = fileInput.files[0];
            selectedFileName.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            selectedFileName.classList.remove("d-none");
            startBtn.disabled = false;
        }
    }

    form.addEventListener("submit", (e) => {
        e.preventDefault();
        if (!fileInput.files.length) return;

        errorAlert.classList.add("d-none");
        resultsSection.classList.add("d-none");
        progressContainer.classList.remove("d-none");
        startBtn.disabled = true;

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        fetch("/api/batch", {
            method: "POST",
            body: formData
        })
        .then(res => res.json().then(data => ({ ok: res.ok, data })))
        .then(({ ok, data }) => {
            if (!ok) {
                progressContainer.classList.add("d-none");
                errorMessage.textContent = data.error || "Batch processing failed.";
                errorAlert.classList.remove("d-none");
                startBtn.disabled = false;
                return;
            }

            document.getElementById("activeJobId").textContent = data.job_id;
            pollJobProgress(data.job_id);
        })
        .catch(err => {
            progressContainer.classList.add("d-none");
            errorMessage.textContent = "Upload failed: " + err.message;
            errorAlert.classList.remove("d-none");
            startBtn.disabled = false;
        });
    });

    // Filters on results table
    const searchInput = document.getElementById("tableSearch");
    const aspectFilter = document.getElementById("aspectFilter");
    const sentimentFilter = document.getElementById("sentimentFilter");

    if (searchInput) searchInput.addEventListener("input", filterBatchTable);
    if (aspectFilter) aspectFilter.addEventListener("change", filterBatchTable);
    if (sentimentFilter) sentimentFilter.addEventListener("change", filterBatchTable);
}

function pollJobProgress(jobId) {
    const progressBar = document.getElementById("progressBar");
    const progressPercent = document.getElementById("progressPercent");
    const progressCounts = document.getElementById("progressCounts");
    const progressStatusText = document.getElementById("progressStatusText");
    const progressContainer = document.getElementById("progressContainer");
    const startBtn = document.getElementById("startBatchBtn");
    const errorAlert = document.getElementById("batchErrorAlert");
    const errorMessage = document.getElementById("batchErrorMessage");

    const interval = setInterval(() => {
        fetch(`/api/batch/status/${jobId}`)
            .then(res => res.json())
            .then(job => {
                if (job.status === "failed") {
                    clearInterval(interval);
                    progressContainer.classList.add("d-none");
                    errorMessage.textContent = job.error || "Batch processing failed.";
                    errorAlert.classList.remove("d-none");
                    startBtn.disabled = false;
                } else if (job.status === "completed") {
                    clearInterval(interval);
                    progressBar.style.width = "100%";
                    progressPercent.textContent = "100%";
                    progressStatusText.textContent = "Complete!";
                    startBtn.disabled = false;

                    setTimeout(() => {
                        progressContainer.classList.add("d-none");
                        renderBatchResults(job);
                    }, 500);
                } else {
                    const pct = job.percent || 0;
                    progressBar.style.width = `${pct}%`;
                    progressPercent.textContent = `${pct}%`;
                    progressCounts.textContent = `${job.progress || 0} / ${job.total || 0} reviews processed`;
                }
            })
            .catch(() => {
                clearInterval(interval);
            });
    }, 1200);
}

function renderBatchResults(job) {
    const resultsSection = document.getElementById("batchResultsSection");
    const downloadBtn = document.getElementById("downloadResultsBtn");
    const totalCount = document.getElementById("totalMentionsCount");

    allBatchRows = job.results || [];
    totalCount.textContent = allBatchRows.length;
    downloadBtn.href = `/download?job_id=${job.job_id}`;

    filterBatchTable();
    resultsSection.classList.remove("d-none");
    resultsSection.scrollIntoView({ behavior: "smooth" });
}

function filterBatchTable() {
    const tableBody = document.getElementById("batchTableBody");
    const searchVal = (document.getElementById("tableSearch")?.value || "").toLowerCase().trim();
    const aspectVal = document.getElementById("aspectFilter")?.value || "all";
    const sentimentVal = document.getElementById("sentimentFilter")?.value || "all";
    const filteredText = document.getElementById("filteredCountText");

    tableBody.innerHTML = "";

    const filtered = allBatchRows.filter(r => {
        const matchesSearch = !searchVal || r.evidence.toLowerCase().includes(searchVal);
        const matchesAspect = aspectVal === "all" || r.aspect.toLowerCase() === aspectVal;
        const matchesSentiment = sentimentVal === "all" || r.sentiment.toLowerCase() === sentimentVal;
        return matchesSearch && matchesAspect && matchesSentiment;
    });

    if (filteredText) {
        filteredText.textContent = `Showing ${filtered.length} of ${allBatchRows.length} rows`;
    }

    filtered.forEach(r => {
        const sentimentClass = r.sentiment === "positive" ? "success" : (r.sentiment === "negative" ? "danger" : "secondary");
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td class="font-monospace text-muted">#${r.review_id}</td>
            <td class="fw-bold text-capitalize">${r.aspect}</td>
            <td class="small text-secondary">"${r.evidence}"</td>
            <td><span class="badge bg-${sentimentClass}-subtle text-${sentimentClass} text-capitalize">${r.sentiment}</span></td>
            <td>${Math.round(r.confidence * 100)}%</td>
        `;
        tableBody.appendChild(tr);
    });
}

// ---------------------------------------------------------------------------
// Executive Analytics Dashboard
// ---------------------------------------------------------------------------
let aspectChartInstance = null;
let donutChartInstance = null;

function initDashboardPage() {
    const dashboardContent = document.getElementById("dashboardContent");
    const noDataAlert = document.getElementById("noDataAlert");
    if (!dashboardContent) return;

    fetch("/api/dashboard")
        .then(res => res.json().then(data => ({ ok: res.ok, data })))
        .then(({ ok, data }) => {
            if (!ok || !data.summary || Object.keys(data.summary).length === 0) {
                dashboardContent.classList.add("d-none");
                noDataAlert.classList.remove("d-none");
                return;
            }

            // Populate KPIs
            document.getElementById("kpiTotalMentions").textContent = data.total_mentions.toLocaleString();
            document.getElementById("kpiPositiveRate").textContent = `${data.positive_rate}%`;
            document.getElementById("kpiBestAspect").textContent = data.best_aspect;
            document.getElementById("kpiWorstAspect").textContent = data.worst_aspect;

            // Render Charts
            renderAspectBarChart(data.summary);
            renderSentimentDonut(data);

            // Populate Table
            renderDashboardTable(data.summary);
        })
        .catch(() => {
            dashboardContent.classList.add("d-none");
            noDataAlert.classList.remove("d-none");
        });
}

function renderAspectBarChart(summary) {
    const ctx = document.getElementById("aspectBarChart")?.getContext("2d");
    if (!ctx) return;

    const labels = Object.keys(summary).map(k => k.charAt(0).toUpperCase() + k.slice(1));
    const posData = Object.values(summary).map(v => v.positive);
    const neuData = Object.values(summary).map(v => v.neutral);
    const negData = Object.values(summary).map(v => v.negative);

    if (aspectChartInstance) aspectChartInstance.destroy();

    aspectChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                { label: "Positive", data: posData, backgroundColor: "#10b981", borderRadius: 4 },
                { label: "Neutral", data: neuData, backgroundColor: "#94a3b8", borderRadius: 4 },
                { label: "Negative", data: negData, backgroundColor: "#ef4444", borderRadius: 4 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { stacked: true, grid: { display: false } },
                y: { stacked: true, beginAtZero: true, grid: { color: "#f1f5f9" } }
            },
            plugins: {
                legend: { position: "top" }
            }
        }
    });
}

function renderSentimentDonut(data) {
    const ctx = document.getElementById("sentimentDonutChart")?.getContext("2d");
    if (!ctx) return;

    const summary = data.summary;
    const pos = Object.values(summary).reduce((a, b) => a + b.positive, 0);
    const neu = Object.values(summary).reduce((a, b) => a + b.neutral, 0);
    const neg = Object.values(summary).reduce((a, b) => a + b.negative, 0);

    document.getElementById("donutPosCount").textContent = pos;
    document.getElementById("donutNeuCount").textContent = neu;
    document.getElementById("donutNegCount").textContent = neg;

    if (donutChartInstance) donutChartInstance.destroy();

    donutChartInstance = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Positive", "Neutral", "Negative"],
            datasets: [{
                data: [pos, neu, neg],
                backgroundColor: ["#10b981", "#94a3b8", "#ef4444"],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            cutout: "70%"
        }
    });
}

function renderDashboardTable(summary) {
    const tbody = document.getElementById("dashboardTableBody");
    if (!tbody) return;
    tbody.innerHTML = "";

    Object.entries(summary).forEach(([aspect, stats]) => {
        const nss = stats.net_sentiment_score || 0;
        const nssClass = nss > 0 ? "text-success" : (nss < 0 ? "text-danger" : "text-secondary");
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td class="fw-bold text-capitalize">${aspect}</td>
            <td>${stats.total}</td>
            <td class="text-success fw-semibold">${stats.positive}</td>
            <td class="text-secondary">${stats.neutral}</td>
            <td class="text-danger fw-semibold">${stats.negative}</td>
            <td>
                <div class="d-flex align-items-center gap-2">
                    <div class="progress flex-grow-1" style="height: 6px;">
                        <div class="progress-bar bg-success" style="width: ${stats.positive_pct}%"></div>
                    </div>
                    <span class="small">${stats.positive_pct}%</span>
                </div>
            </td>
            <td class="fw-bold ${nssClass}">${nss > 0 ? '+' : ''}${nss}</td>
        `;
        tbody.appendChild(tr);
    });
}

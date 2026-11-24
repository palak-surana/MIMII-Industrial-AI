
document.addEventListener("DOMContentLoaded", () => {

    // ==========================================
    // ELEMENTS
    // ==========================================

    const pages = document.querySelectorAll(".page");
    const navItems = document.querySelectorAll(".nav-item");

    const fileInput = document.getElementById("audio-file");
    const dropZone = document.getElementById("drop-zone");
    const fileName = document.getElementById("file-name");

    const previewWrap = document.getElementById("audio-preview-wrap");
    const audioPreview = document.getElementById("audio-preview");

    const analyzeBtn = document.getElementById("analyze-btn");
    const message = document.getElementById("analysis-message");

    const themeButton = document.getElementById("theme-toggle");

    let selectedFile = null;
    let latestAnalysisId = null;

    let probabilityChart = null;
    let distributionChart = null;

    // ==========================================
    // THEME MANAGEMENT
    // ==========================================

    function getThemeColors() {
        const styles = getComputedStyle(document.body);

        const darkMode = document.body.classList.contains("dark-mode");

        return {
            text: styles.getPropertyValue("--text").trim() || (darkMode ? "#edf4ff" : "#17243a"),
            muted: styles.getPropertyValue("--muted").trim() || (darkMode ? "#8fa8c9" : "#64748b"),
            border: styles.getPropertyValue("--border").trim() || (darkMode ? "#263750" : "#dce4ef"),
            panel: styles.getPropertyValue("--panel").trim() || (darkMode ? "#111d30" : "#ffffff")
        };
    }

    function updateChartTheme() {

        const colors = getThemeColors();

        [probabilityChart, distributionChart].forEach(chart => {

            if (!chart) return;

            if (chart.options.plugins?.legend?.labels) {
                chart.options.plugins.legend.labels.color = colors.muted;
            }

            if (chart.options.scales) {

                Object.values(chart.options.scales).forEach(axis => {

                    if (axis.ticks) {
                        axis.ticks.color = colors.muted;
                    }

                    if (axis.grid) {
                        axis.grid.color = colors.border;
                    }

                    if (axis.title) {
                        axis.title.color = colors.muted;
                    }

                });

            }

            if (chart.data.datasets) {

                chart.data.datasets.forEach(dataset => {

                    if (dataset.label === "Decision Threshold (%)") {
                        dataset.borderColor = "#ef6475";
                    }

                });

            }

            chart.update();

        });

        if (themeButton) {
            themeButton.textContent = document.body.classList.contains("dark-mode")
                ? "☀ Light Mode"
                : "🌙 Dark Mode";
        }
    }

    function applyTheme(theme) {

        document.body.classList.toggle("dark-mode", theme === "dark");

        try {
            localStorage.setItem("mimii-theme", theme);
        } catch (error) {
            console.warn("Theme preference could not be saved.");
        }

        updateChartTheme();
    }

    let savedTheme = "light";

    try {
        savedTheme = localStorage.getItem("mimii-theme") || "light";
    } catch (error) {
        console.warn("Saved theme unavailable.");
    }

    applyTheme(savedTheme);

    if (themeButton) {

        themeButton.addEventListener("click", () => {

            const nextTheme = document.body.classList.contains("dark-mode")
                ? "light"
                : "dark";

            applyTheme(nextTheme);

        });

    }

    // ==========================================
    // PAGE NAVIGATION
    // ==========================================

    function showPage(pageName) {

        const target = document.getElementById(`page-${pageName}`);

        if (!target) return;

        pages.forEach(page => {
            page.classList.remove("active");
        });

        target.classList.add("active");

        navItems.forEach(item => {

            item.classList.toggle(
                "active",
                item.dataset.page === pageName
            );

        });

        if (pageName === "faults" || pageName === "overview") {
            loadHistory();
        }

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    }

    navItems.forEach(item => {

        item.addEventListener("click", () => {
            showPage(item.dataset.page);
        });

    });

    document.querySelectorAll("[data-go]").forEach(button => {

        button.addEventListener("click", () => {
            showPage(button.dataset.go);
        });

    });

    // ==========================================
    // MESSAGE DISPLAY
    // ==========================================

    function showMessage(text, type = "success") {

        if (!message) return;

        message.textContent = text;
        message.className = `message ${type}`;
        message.hidden = false;

    }

    // ==========================================
    // FILE SELECTION
    // ==========================================

    function selectFile(file) {

        if (!file) return;

        if (!file.name.toLowerCase().endsWith(".wav")) {

            showMessage(
                "Please choose a WAV audio file.",
                "error"
            );

            return;
        }

        if (file.size > 25 * 1024 * 1024) {

            showMessage(
                "File size must be below 25 MB.",
                "error"
            );

            return;
        }

        selectedFile = file;

        if (fileName) {
            fileName.textContent = file.name;
        }

        if (previewWrap) {
            previewWrap.hidden = false;
        }

        if (audioPreview) {

            const oldUrl = audioPreview.dataset.objectUrl;

            if (oldUrl) {
                URL.revokeObjectURL(oldUrl);
            }

            const objectUrl = URL.createObjectURL(file);

            audioPreview.src = objectUrl;
            audioPreview.dataset.objectUrl = objectUrl;

        }

        const resultPanel = document.getElementById("analysis-result");

        if (resultPanel) {
            resultPanel.hidden = true;
        }

        showMessage(
            "Audio file selected successfully.",
            "success"
        );
    }

    if (fileInput) {

        fileInput.addEventListener("change", event => {

            selectFile(event.target.files[0]);

        });

    }

    // ==========================================
    // DRAG AND DROP
    // ==========================================

    if (dropZone) {

        dropZone.addEventListener("dragover", event => {

            event.preventDefault();

            dropZone.classList.add("dragover");

        });

        dropZone.addEventListener("dragleave", () => {

            dropZone.classList.remove("dragover");

        });

        dropZone.addEventListener("drop", event => {

            event.preventDefault();

            dropZone.classList.remove("dragover");

            const file = event.dataTransfer.files[0];

            selectFile(file);

        });

    }

    // ==========================================
    // AUDIO ANALYSIS
    // ==========================================

    if (analyzeBtn) {

        analyzeBtn.addEventListener("click", async () => {

            if (!selectedFile) {

                showMessage(
                    "Please select an audio file first.",
                    "error"
                );

                return;
            }

            const formData = new FormData();

            formData.append("audio", selectedFile);

            analyzeBtn.disabled = true;

            analyzeBtn.textContent = "Analyzing audio...";

            showMessage(
                "Processing audio and running CNN-LSTM prediction. Please wait...",
                "success"
            );

            try {

                const response = await fetch("/api/analyze", {

                    method: "POST",

                    body: formData

                });

                const data = await response.json();

                if (!response.ok || !data.success) {

                    throw new Error(
                        data.error || "Audio analysis failed."
                    );

                }

                displayResult(data.result);

                showMessage(
                    "Analysis completed successfully.",
                    "success"
                );

                await loadHistory();

            } catch (error) {

                console.error("Analysis error:", error);

                showMessage(
                    error.message || "Something went wrong.",
                    "error"
                );

            } finally {

                analyzeBtn.disabled = false;

                analyzeBtn.textContent = "✦ Analyze Machine Audio";

            }

        });

    }

    // ==========================================
    // DISPLAY ANALYSIS RESULT
    // ==========================================

    function displayResult(result) {

        const resultPanel = document.getElementById("analysis-result");

        if (!resultPanel || !result) return;

        resultPanel.hidden = false;

        latestAnalysisId = result.analysis_id;

        const isFaulty = result.prediction === "FAULTY";

        const banner = document.getElementById("result-banner");

        if (banner) {

            banner.className = `result-banner ${isFaulty ? "faulty" : "healthy"}`;

            const heading = document.createElement("h2");
            heading.textContent = isFaulty
                ? "⚠ ABNORMAL MACHINE CONDITION"
                : "✓ MACHINE CONDITION HEALTHY";

            const description = document.createElement("p");
            description.textContent = isFaulty
                ? "The majority of analyzed audio windows were classified as faulty. Maintenance inspection is recommended."
                : "The majority of analyzed audio windows were classified as healthy.";

            banner.replaceChildren(heading, description);

        }

        // Recording name

        const resultFile = document.getElementById("result-file");

        if (resultFile) {
            resultFile.textContent = selectedFile ? selectedFile.name : "Recording";
        }

        // Average fault probability

        const average = document.getElementById("result-average");

        if (average) {
            average.textContent = `${Number(result.average_probability).toFixed(2)}%`;
        }

        // Maximum fault probability

        const maximum = document.getElementById("result-maximum");

        if (maximum) {
            maximum.textContent = `${Number(result.maximum_probability).toFixed(2)}%`;
        }

        // Window counts

        const faulty = document.getElementById("result-faulty");
        const healthy = document.getElementById("result-healthy");

        if (faulty) faulty.textContent = result.faulty_windows;
        if (healthy) healthy.textContent = result.healthy_windows;

        // Additional information

        const details = document.getElementById("result-details");

        if (details) {

            const detailItems = [
                ["Recording duration:", `${result.duration} sec`],
                ["Sample rate:", `${result.sample_rate} Hz`],
                ["Total windows:", result.total_windows],
                ["Decision threshold:", `${(result.threshold * 100).toFixed(2)}%`],
                ["Analysis time:", result.analysis_time]
            ];

            details.replaceChildren();

            detailItems.forEach(([label, value]) => {

                const span = document.createElement("span");
                const bold = document.createElement("b");

                span.append(document.createTextNode(`${label} `));

                bold.textContent = value;

                span.appendChild(bold);

                details.appendChild(span);

            });

        }

        drawProbabilityChart(result);

        drawDistributionChart(result);

        updateChartTheme();

        resultPanel.scrollIntoView({

            behavior: "smooth",

            block: "start"

        });

    }

    // ==========================================
    // WINDOW PROBABILITY CHART
    // ==========================================

    function drawProbabilityChart(result) {

        const canvas = document.getElementById("probability-chart");

        if (!canvas || typeof Chart === "undefined") return;

        if (probabilityChart) {
            probabilityChart.destroy();
        }

        const probabilities = result.window_probabilities || [];

        const labels = probabilities.map(
            (_, index) => `Window ${index + 1}`
        );

        const colors = getThemeColors();

        probabilityChart = new Chart(canvas, {

            type: "line",

            data: {

                labels: labels,

                datasets: [

                    {

                        label: "Fault Probability (%)",

                        data: probabilities,

                        borderColor: "#579bff",

                        backgroundColor: "rgba(87,155,255,0.13)",

                        fill: true,

                        tension: 0.3,

                        pointRadius: 3

                    },

                    {

                        label: "Decision Threshold (%)",

                        data: probabilities.map(
                            () => result.threshold * 100
                        ),

                        borderColor: "#ef6475",

                        borderDash: [6, 5],

                        pointRadius: 0,

                        fill: false

                    }

                ]

            },

            options: {

                responsive: true,

                maintainAspectRatio: true,

                plugins: {

                    legend: {

                        labels: {
                            color: colors.muted
                        }

                    }

                },

                scales: {

                    x: {

                        ticks: {
                            color: colors.muted
                        },

                        grid: {
                            color: colors.border
                        }

                    },

                    y: {

                        min: 0,

                        max: 100,

                        ticks: {
                            color: colors.muted
                        },

                        grid: {
                            color: colors.border
                        },

                        title: {

                            display: true,

                            text: "Probability (%)",

                            color: colors.muted

                        }

                    }

                }

            }

        });

    }

    // ==========================================
    // HEALTHY VS FAULTY DOUGHNUT CHART
    // ==========================================

    function drawDistributionChart(result) {

        const canvas = document.getElementById("distribution-chart");

        if (!canvas || typeof Chart === "undefined") return;

        if (distributionChart) {
            distributionChart.destroy();
        }

        const colors = getThemeColors();

        distributionChart = new Chart(canvas, {

            type: "doughnut",

            data: {

                labels: [
                    "Healthy windows",
                    "Faulty windows"
                ],

                datasets: [{

                    data: [

                        result.healthy_windows,

                        result.faulty_windows

                    ],

                    backgroundColor: [

                        "#22c99b",

                        "#ff6877"

                    ],

                    borderColor: colors.panel,

                    borderWidth: 4

                }]

            },

            options: {

                responsive: true,

                plugins: {

                    legend: {

                        position: "bottom",

                        labels: {
                            color: colors.muted
                        }

                    }

                }

            }

        });

    }

    // ==========================================
    // LOAD ANALYSIS HISTORY
    // ==========================================

    async function loadHistory() {

        try {

            const response = await fetch("/api/history");

            const data = await response.json();

            if (!response.ok || !data.success) {

                throw new Error(
                    "Unable to load analysis history."
                );

            }

            renderHistory(data.history || []);

        } catch (error) {

            console.error("History error:", error);

        }

    }

    // ==========================================
    // RENDER HISTORY
    // ==========================================

    function renderHistory(history) {

        const tbody = document.getElementById("history-table");
        const overview = document.getElementById("overview-history");

        // Total analyses

        const overviewCount = document.getElementById("overview-count");
        const historyTotal = document.getElementById("history-total");

        if (overviewCount) overviewCount.textContent = history.length;
        if (historyTotal) historyTotal.textContent = history.length;

        // Faulty and healthy counts

        const faulty = history.filter(
            item => item.prediction === "FAULTY"
        ).length;

        const healthy = history.length - faulty;

        const historyFaulty = document.getElementById("history-faulty");
        const historyHealthy = document.getElementById("history-healthy");

        if (historyFaulty) historyFaulty.textContent = faulty;
        if (historyHealthy) historyHealthy.textContent = healthy;

        // Latest prediction

        const latest = document.getElementById("overview-latest");

        if (latest) {
            latest.textContent = history.length
                ? history[0].prediction
                : "No data";
        }

        // Empty history

        if (history.length === 0) {

            if (tbody) {

                tbody.innerHTML = `
                    <tr>
                        <td colspan="6" class="empty-state">
                            No analysis records available yet.
                        </td>
                    </tr>
                `;

            }

            if (overview) {
                overview.textContent = "No audio has been analyzed yet.";
            }

            return;

        }

        // History table

        if (tbody) {

            tbody.innerHTML = history.map(item => {

                const faultyPrediction = item.prediction === "FAULTY";

                return `
                    <tr>

                        <td>
                            ${escapeHTML(item.filename)}
                        </td>

                        <td>
                            ${escapeHTML(item.analysis_time)}
                        </td>

                        <td class="${faultyPrediction ? "prediction-faulty" : "prediction-healthy"}">
                            ${escapeHTML(item.prediction)}
                        </td>

                        <td>
                            ${Number(item.average_probability).toFixed(2)}%
                        </td>

                        <td>
                            ${Number(item.total_windows)}
                        </td>

                        <td>
                            <a
                                class="report-link"
                                href="/api/report/${encodeURIComponent(item.id)}"
                            >
                                Download
                            </a>
                        </td>

                    </tr>
                `;

            }).join("");

        }

        // Overview recent analyses

        if (overview) {

            overview.innerHTML = history.slice(0, 4).map(item => {

                const faultyPrediction = item.prediction === "FAULTY";

                return `
                    <div class="history-preview">

                        <b>
                            ${escapeHTML(item.filename)}
                        </b>

                        <span class="${faultyPrediction ? "prediction-faulty" : "prediction-healthy"}">
                            ${escapeHTML(item.prediction)}
                        </span>

                        <small>
                            ${escapeHTML(item.analysis_time)}
                            ·
                            ${Number(item.average_probability).toFixed(2)}%
                            average fault probability
                        </small>

                    </div>
                `;

            }).join("");

        }

    }

    // ==========================================
    // HTML ESCAPING
    // ==========================================

    function escapeHTML(value) {

        return String(value).replace(/[&<>"']/g, character => ({

            "&": "&amp;",

            "<": "&lt;",

            ">": "&gt;",

            '"': "&quot;",

            "'": "&#039;"

        })[character]);

    }

    // ==========================================
    // REFRESH HISTORY
    // ==========================================

    const refreshButton = document.getElementById("refresh-history");

    if (refreshButton) {
        refreshButton.addEventListener("click", loadHistory);
    }

    // ==========================================
    // DOWNLOAD ANALYSIS REPORT
    // ==========================================

    const downloadButton = document.getElementById("download-report");

    if (downloadButton) {

        downloadButton.addEventListener("click", () => {

            if (!latestAnalysisId) {

                alert("Please analyze an audio file first.");

                return;

            }

            window.location.href =
                `/api/report/${encodeURIComponent(latestAnalysisId)}`;

        });

    }

    // ==========================================
    // INITIAL LOAD
    // ==========================================

    loadHistory();

    fetch("/api/model-info")

        .then(response => {

            if (!response.ok) {
                throw new Error("Model information unavailable.");
            }

            return response.json();

        })

        .then(data => {

            if (!data.success) {
                console.error("Model information unavailable.");
            }

        })

        .catch(error => {
            console.error("Model API error:", error);
        });

});

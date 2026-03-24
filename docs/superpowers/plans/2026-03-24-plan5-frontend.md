# Plan 5: Frontend (Netlify Web App)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the AnemiaScan web app — a mobile-first static HTML/CSS/JS PWA with two modes (Screening for ASHAs, Advanced/Research for clinicians), deployed to Netlify, calling the HF Space API with fallback to Render.

**Architecture:** Pure HTML/CSS/vanilla JS — no framework. Three JS modules: `app.js` (core orchestration), `screening-mode.js` (ASHA UI), `advanced-mode.js` (research UI). Camera API for capture, file upload fallback. Dual API fallback: HF Space → Render. Medical disclaimer on every result.

**Tech Stack:** HTML5, CSS3, Vanilla JavaScript (ES2022 modules), PWA (Service Worker), Netlify (static hosting)

**Spec:** `docs/superpowers/specs/2026-03-24-anemiascan-ai-pipeline-design.md` §10

**Prereq:** Plan 4 complete (inference API live at HF Space).

---

## File Map

```
frontend/
├── index.html                   ← CREATE: main entry point
├── manifest.json                ← CREATE: PWA manifest
├── service-worker.js            ← CREATE: PWA offline cache
├── css/
│   └── styles.css               ← CREATE: mobile-first styles
├── js/
│   ├── app.js                   ← CREATE: core orchestration
│   ├── screening-mode.js        ← CREATE: ASHA screening UI
│   ├── advanced-mode.js         ← CREATE: research/clinician UI
│   └── api.js                   ← CREATE: API client with fallback
└── assets/
    └── icons/
        ├── icon-192.png          ← CREATE: PWA icon (placeholder)
        └── icon-512.png          ← CREATE: PWA icon (placeholder)
```

---

## Task 1: Write `frontend/index.html`

**Files:**
- Create: `frontend/index.html`

- [ ] **Step 1: Write `index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="theme-color" content="#b91c1c" />
  <meta name="description" content="AnemiaScan — AI-powered non-invasive anemia screening" />

  <title>AnemiaScan</title>

  <!-- PWA -->
  <link rel="manifest" href="manifest.json" />
  <link rel="apple-touch-icon" href="assets/icons/icon-192.png" />

  <link rel="stylesheet" href="css/styles.css" />

  <!-- Env config injected by Netlify build plugin (see netlify.toml) -->
  <!-- In local dev, falls back to defaults in api.js -->
  <script>
    window._ENV = {
      HF_SPACE_URL:   "%%HF_SPACE_URL%%",
      BACKUP_API_URL: "%%BACKUP_API_URL%%"
    };
  </script>
</head>
<body>
  <!-- Header -->
  <header class="app-header">
    <div class="header-inner">
      <h1 class="app-title">🩸 AnemiaScan</h1>
      <button id="mode-toggle" class="btn btn-ghost" aria-label="Switch mode">
        Advanced Mode
      </button>
    </div>
  </header>

  <!-- Medical Disclaimer Banner -->
  <div class="disclaimer-banner" role="alert">
    ⚠️ Research tool only — not a certified diagnostic device. All results require clinical confirmation.
  </div>

  <!-- Screening Mode (default) -->
  <main id="screening-mode" class="mode-panel active">
    <section class="upload-section">
      <h2 class="section-title">Capture Images</h2>
      <div class="capture-grid">
        <div class="capture-card" id="conj-capture">
          <div class="capture-icon">👁️</div>
          <div class="capture-label">Conjunctiva</div>
          <div class="capture-hint">Lower eyelid, everted</div>
          <input type="file" id="conj-input" accept="image/*" capture="environment" hidden />
          <button class="btn btn-capture" data-target="conj-input">
            📷 Capture / Upload
          </button>
          <img id="conj-preview" class="image-preview" alt="Conjunctiva preview" hidden />
        </div>

        <div class="capture-card" id="nail-capture">
          <div class="capture-icon">💅</div>
          <div class="capture-label">Nail Bed</div>
          <div class="capture-hint">Index finger, extended</div>
          <input type="file" id="nail-input" accept="image/*" capture="environment" hidden />
          <button class="btn btn-capture" data-target="nail-input">
            📷 Capture / Upload
          </button>
          <img id="nail-preview" class="image-preview" alt="Nail-bed preview" hidden />
        </div>
      </div>

      <button id="analyze-btn" class="btn btn-primary btn-full" disabled>
        Analyze
      </button>
    </section>

    <!-- Result Card -->
    <section id="result-card" class="result-card" hidden>
      <div class="result-hb">
        <span class="result-label">Estimated Hb</span>
        <span id="result-hb-value" class="result-value">— g/dL</span>
        <span id="result-ci" class="result-ci"></span>
      </div>
      <div id="result-badge" class="result-badge">—</div>
      <div id="result-referral" class="result-referral"></div>
      <div class="result-confidence">
        <span class="result-label">Confidence</span>
        <span id="result-confidence-value">—</span>
      </div>
    </section>

    <!-- Loading State -->
    <div id="loading" class="loading" hidden>
      <div class="spinner"></div>
      <p>Analyzing image...</p>
    </div>

    <!-- Error State -->
    <div id="error-banner" class="error-banner" hidden></div>
  </main>

  <!-- Advanced Mode (hidden by default) -->
  <main id="advanced-mode" class="mode-panel">
    <section class="advanced-controls">
      <h2 class="section-title">Research Mode</h2>
      <div class="advanced-upload-row">
        <label>Conjunctiva: <input type="file" id="adv-conj-input" accept="image/*" /></label>
        <label>Nail Bed: <input type="file" id="adv-nail-input" accept="image/*" /></label>
        <select id="model-select">
          <option value="ensemble">Ensemble (recommended)</option>
          <option value="conjunctiva">Conjunctiva only</option>
          <option value="nailbed">Nail-bed only</option>
        </select>
        <button id="adv-analyze-btn" class="btn btn-primary">Analyze</button>
      </div>
    </section>

    <div id="adv-results" class="adv-results" hidden>
      <div class="adv-grid">
        <div class="adv-panel">
          <h3>Results</h3>
          <table class="results-table" id="adv-results-table"></table>
        </div>
        <div class="adv-panel">
          <h3>Per-Model Breakdown</h3>
          <table class="results-table" id="adv-permodel-table"></table>
        </div>
      </div>
      <div class="adv-actions">
        <button id="export-json-btn" class="btn btn-ghost">Export JSON</button>
      </div>
    </div>

    <div id="adv-loading" class="loading" hidden>
      <div class="spinner"></div><p>Analyzing...</p>
    </div>
    <div id="adv-error" class="error-banner" hidden></div>
  </main>

  <!-- Footer -->
  <footer class="app-footer">
    <p>AnemiaScan · ICMR Extramural Project · SIMSR Tumakuru</p>
    <p>
      <a href="https://github.com/hssling/anemia-detection-ai" target="_blank" rel="noopener">GitHub</a>
      ·
      <a href="https://huggingface.co/spaces/hssling/anemia-screening" target="_blank" rel="noopener">HuggingFace</a>
    </p>
  </footer>

  <script type="module" src="js/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/index.html
git commit -m "feat: add AnemiaScan HTML shell"
```

---

## Task 2: Write `frontend/css/styles.css`

**Files:**
- Create: `frontend/css/styles.css`

- [ ] **Step 1: Write `styles.css`**

```css
/* frontend/css/styles.css — AnemiaScan mobile-first styles */

/* Reset + variables */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --color-primary: #b91c1c;
  --color-primary-dark: #7f1d1d;
  --color-success: #15803d;
  --color-warn: #b45309;
  --color-danger: #dc2626;
  --color-severe: #7c3aed;
  --color-bg: #fafafa;
  --color-surface: #ffffff;
  --color-border: #e5e7eb;
  --color-text: #111827;
  --color-muted: #6b7280;
  --radius: 12px;
  --shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.08);
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

body { font-family: var(--font); background: var(--color-bg); color: var(--color-text); min-height: 100vh; }

/* Header */
.app-header { background: var(--color-primary); color: white; padding: 1rem; box-shadow: var(--shadow); }
.header-inner { display: flex; justify-content: space-between; align-items: center; max-width: 640px; margin: 0 auto; }
.app-title { font-size: 1.25rem; font-weight: 700; }

/* Disclaimer */
.disclaimer-banner { background: #fef3c7; color: #78350f; padding: 0.5rem 1rem; text-align: center; font-size: 0.8rem; border-bottom: 1px solid #fcd34d; }

/* Mode panels */
.mode-panel { display: none; max-width: 640px; margin: 0 auto; padding: 1rem; }
.mode-panel.active { display: block; }

/* Buttons */
.btn { display: inline-flex; align-items: center; justify-content: center; padding: 0.625rem 1.25rem; border-radius: 8px; font-size: 0.9rem; font-weight: 500; cursor: pointer; border: none; transition: background 0.15s, transform 0.1s; }
.btn:active { transform: scale(0.97); }
.btn-primary { background: var(--color-primary); color: white; }
.btn-primary:hover { background: var(--color-primary-dark); }
.btn-primary:disabled { background: #9ca3af; cursor: not-allowed; }
.btn-ghost { background: transparent; color: var(--color-primary); border: 1px solid var(--color-primary); }
.btn-ghost:hover { background: #fef2f2; }
.btn-capture { background: #f3f4f6; color: var(--color-text); width: 100%; margin-top: 0.5rem; }
.btn-full { width: 100%; margin-top: 1rem; padding: 0.875rem; font-size: 1rem; }

/* Capture grid */
.capture-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0; }
.capture-card { background: var(--color-surface); border: 2px dashed var(--color-border); border-radius: var(--radius); padding: 1rem; text-align: center; transition: border-color 0.2s; }
.capture-card.has-image { border-style: solid; border-color: var(--color-primary); }
.capture-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.capture-label { font-weight: 600; font-size: 0.9rem; }
.capture-hint { color: var(--color-muted); font-size: 0.75rem; margin: 0.25rem 0 0.5rem; }
.image-preview { width: 100%; border-radius: 8px; margin-top: 0.5rem; object-fit: cover; aspect-ratio: 1; }

/* Result card */
.result-card { background: var(--color-surface); border-radius: var(--radius); padding: 1.25rem; box-shadow: var(--shadow); margin-top: 1rem; }
.result-hb { display: flex; flex-direction: column; align-items: center; margin-bottom: 1rem; }
.result-label { font-size: 0.8rem; color: var(--color-muted); text-transform: uppercase; letter-spacing: 0.05em; }
.result-value { font-size: 2.5rem; font-weight: 700; color: var(--color-text); }
.result-ci { font-size: 0.85rem; color: var(--color-muted); }
.result-badge { display: inline-block; padding: 0.5rem 1.25rem; border-radius: 999px; font-weight: 700; font-size: 1rem; text-align: center; margin: 0.5rem auto; }
.result-badge.normal   { background: #dcfce7; color: var(--color-success); }
.result-badge.mild     { background: #fef9c3; color: var(--color-warn); }
.result-badge.moderate { background: #fee2e2; color: var(--color-danger); }
.result-badge.severe   { background: #ede9fe; color: var(--color-severe); }
.result-referral { font-size: 0.9rem; color: var(--color-muted); margin-top: 0.5rem; text-align: center; }
.result-confidence { display: flex; justify-content: center; gap: 0.5rem; margin-top: 0.75rem; font-size: 0.85rem; color: var(--color-muted); }

/* Loading */
.loading { display: flex; flex-direction: column; align-items: center; gap: 0.75rem; padding: 2rem; color: var(--color-muted); }
.spinner { width: 36px; height: 36px; border: 3px solid var(--color-border); border-top-color: var(--color-primary); border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Error */
.error-banner { background: #fee2e2; color: var(--color-danger); padding: 0.75rem 1rem; border-radius: 8px; margin-top: 1rem; font-size: 0.9rem; }

/* Advanced mode */
.advanced-controls { margin-bottom: 1rem; }
.advanced-upload-row { display: flex; flex-direction: column; gap: 0.75rem; margin-top: 0.75rem; }
.advanced-upload-row label { font-size: 0.9rem; display: flex; flex-direction: column; gap: 0.25rem; }
.adv-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.adv-panel { background: var(--color-surface); border-radius: var(--radius); padding: 1rem; box-shadow: var(--shadow); }
.adv-panel h3 { font-size: 0.9rem; font-weight: 600; margin-bottom: 0.75rem; }
.results-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.results-table td { padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--color-border); }
.results-table td:first-child { color: var(--color-muted); }
.adv-actions { margin-top: 1rem; display: flex; gap: 0.75rem; }
.section-title { font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem; }

/* Footer */
.app-footer { text-align: center; padding: 2rem 1rem; color: var(--color-muted); font-size: 0.8rem; }
.app-footer a { color: var(--color-primary); text-decoration: none; }

/* Responsive: large screens */
@media (min-width: 640px) {
  .advanced-upload-row { flex-direction: row; flex-wrap: wrap; align-items: flex-end; }
}

@media (max-width: 400px) {
  .capture-grid { grid-template-columns: 1fr; }
  .adv-grid { grid-template-columns: 1fr; }
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/css/styles.css
git commit -m "feat: add AnemiaScan mobile-first CSS"
```

---

## Task 3: Write `frontend/js/api.js`

**Files:**
- Create: `frontend/js/api.js`

- [ ] **Step 1: Write `api.js`**

```javascript
// frontend/js/api.js
// API client with primary (HF Space) + fallback (Render) endpoints.

// API URLs injected at build time via Netlify env vars.
// Netlify replaces process.env references during deploy for static sites via
// a build plugin, but for pure static HTML we use a config endpoint approach:
// netlify.toml injects these as window._ENV via a meta tag substitution.
// Fallback to hardcoded values for local development only.
const PRIMARY_API  = window._ENV?.HF_SPACE_URL
  || "https://hssling-anemia-screening.hf.space/api/predict";
const FALLBACK_API = window._ENV?.BACKUP_API_URL
  || "https://anemiascan-inference-backup.onrender.com/api/predict";
const TIMEOUT_MS    = 45_000;  // HF Spaces can be slow on cold start

/**
 * Send images to the inference API.
 * Tries primary first; falls back to secondary on error or timeout.
 *
 * @param {File|null} conjFile
 * @param {File|null} nailFile
 * @param {string} model  "ensemble" | "conjunctiva" | "nailbed"
 * @returns {Promise<Object>} API response JSON
 */
export async function predict(conjFile, nailFile, model = "ensemble") {
  const formData = new FormData();
  if (conjFile) formData.append("conjunctiva_image", conjFile);
  if (nailFile)  formData.append("nailbed_image",    nailFile);
  formData.append("model", model);

  return _fetchWithFallback(PRIMARY_API, FALLBACK_API, formData);
}

async function _fetchWithFallback(primaryUrl, fallbackUrl, body) {
  try {
    const result = await _timedFetch(primaryUrl, body);
    return result;
  } catch (primaryErr) {
    console.warn(`Primary API failed (${primaryErr.message}), trying fallback...`);
    try {
      return await _timedFetch(fallbackUrl, body);
    } catch (fallbackErr) {
      throw new Error(`Both APIs failed. Primary: ${primaryErr.message}. Fallback: ${fallbackErr.message}`);
    }
  }
}

async function _timedFetch(url, body) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const resp = await fetch(url, {
      method: "POST",
      body,
      signal: controller.signal,
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text}`);
    }
    return await resp.json();
  } finally {
    clearTimeout(timer);
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/js/api.js
git commit -m "feat: add API client with HF Space + Render fallback"
```

---

## Task 4: Write `frontend/js/screening-mode.js`

**Files:**
- Create: `frontend/js/screening-mode.js`

- [ ] **Step 1: Write `screening-mode.js`**

```javascript
// frontend/js/screening-mode.js
// Screening mode — ASHA-facing UI logic.

import { predict } from "./api.js";

const REFERRAL_MESSAGES = {
  normal:   "No anemia detected. Continue regular monitoring.",
  mild:     "Mild anemia. Advise iron-rich diet and iron supplementation. Follow up in 4 weeks.",
  moderate: "Moderate anemia. Refer to PHC for iron supplementation and further evaluation.",
  severe:   "⚠️ Severe anemia. URGENT referral to PHC/hospital required.",
};

const BADGE_LABELS = {
  normal:   "✅ Normal",
  mild:     "⚠ Mild Anemia",
  moderate: "⚠️ Moderate Anemia",
  severe:   "🚨 Severe Anemia",
};

export function initScreeningMode() {
  const conjInput   = document.getElementById("conj-input");
  const nailInput   = document.getElementById("nail-input");
  const conjPreview = document.getElementById("conj-preview");
  const nailPreview = document.getElementById("nail-preview");
  const analyzeBtn  = document.getElementById("analyze-btn");
  const resultCard  = document.getElementById("result-card");
  const loading     = document.getElementById("loading");
  const errorBanner = document.getElementById("error-banner");

  // Wire capture buttons
  document.querySelectorAll(".btn-capture[data-target]").forEach(btn => {
    btn.addEventListener("click", () => {
      document.getElementById(btn.dataset.target).click();
    });
  });

  conjInput.addEventListener("change", () => {
    _showPreview(conjInput, conjPreview, "conj-capture");
    _updateAnalyzeBtn();
  });

  nailInput.addEventListener("change", () => {
    _showPreview(nailInput, nailPreview, "nail-capture");
    _updateAnalyzeBtn();
  });

  analyzeBtn.addEventListener("click", async () => {
    const conjFile = conjInput.files[0] || null;
    const nailFile = nailInput.files[0] || null;

    _setLoading(true);
    _hideResult();
    _hideError();

    try {
      const result = await predict(conjFile, nailFile, "ensemble");
      _showResult(result);
    } catch (err) {
      _showError(err.message);
    } finally {
      _setLoading(false);
    }
  });

  function _showPreview(input, preview, cardId) {
    if (!input.files[0]) return;
    const url = URL.createObjectURL(input.files[0]);
    preview.src = url;
    preview.hidden = false;
    document.getElementById(cardId).classList.add("has-image");
  }

  function _updateAnalyzeBtn() {
    const hasConj = conjInput.files.length > 0;
    const hasNail = nailInput.files.length > 0;
    analyzeBtn.disabled = !(hasConj || hasNail);
  }

  function _setLoading(on) {
    loading.hidden = !on;
    analyzeBtn.disabled = on;
  }

  function _showResult(data) {
    document.getElementById("result-hb-value").textContent =
      `${data.hb_estimate} g/dL`;
    document.getElementById("result-ci").textContent =
      `95% CI: ${data.hb_ci_95[0]}–${data.hb_ci_95[1]}`;

    const badge = document.getElementById("result-badge");
    const cls = data.classification;
    badge.textContent = BADGE_LABELS[cls] || cls;
    badge.className = `result-badge ${cls}`;

    document.getElementById("result-referral").textContent =
      REFERRAL_MESSAGES[cls] || "";

    const topProb = Math.max(...Object.values(data.class_probabilities));
    document.getElementById("result-confidence-value").textContent =
      `${(topProb * 100).toFixed(0)}%`;

    resultCard.hidden = false;
  }

  function _hideResult() { resultCard.hidden = true; }

  function _showError(msg) {
    errorBanner.textContent = `Analysis failed: ${msg}`;
    errorBanner.hidden = false;
  }

  function _hideError() { errorBanner.hidden = true; }
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/js/screening-mode.js
git commit -m "feat: add ASHA screening mode JS"
```

---

## Task 5: Write `frontend/js/advanced-mode.js`

**Files:**
- Create: `frontend/js/advanced-mode.js`

- [ ] **Step 1: Write `advanced-mode.js`**

```javascript
// frontend/js/advanced-mode.js
// Advanced (research/clinician) mode.

import { predict } from "./api.js";

export function initAdvancedMode() {
  const advAnalyzeBtn = document.getElementById("adv-analyze-btn");
  const advResults    = document.getElementById("adv-results");
  const advLoading    = document.getElementById("adv-loading");
  const advError      = document.getElementById("adv-error");
  const modelSelect   = document.getElementById("model-select");

  let lastResult = null;

  advAnalyzeBtn.addEventListener("click", async () => {
    const conjFile = document.getElementById("adv-conj-input").files[0] || null;
    const nailFile = document.getElementById("adv-nail-input").files[0] || null;
    const model    = modelSelect.value;

    if (!conjFile && !nailFile) {
      advError.textContent = "Please upload at least one image.";
      advError.hidden = false;
      return;
    }

    advLoading.hidden = false;
    advResults.hidden = true;
    advError.hidden = true;

    try {
      const result = await predict(conjFile, nailFile, model);
      lastResult = result;
      _renderResults(result);
      advResults.hidden = false;
    } catch (err) {
      advError.textContent = `Analysis failed: ${err.message}`;
      advError.hidden = false;
    } finally {
      advLoading.hidden = true;
    }
  });

  document.getElementById("export-json-btn").addEventListener("click", () => {
    if (!lastResult) return;
    const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `anemiascan_result_${Date.now()}.json`;
    a.click();
  });

  function _renderResults(data) {
    const table = document.getElementById("adv-results-table");
    table.innerHTML = _tableRows([
      ["Hb Estimate",     `${data.hb_estimate} g/dL`],
      ["95% CI",          `${data.hb_ci_95[0]}–${data.hb_ci_95[1]} g/dL`],
      ["Classification",  data.classification.replace("_", " ")],
      ...Object.entries(data.class_probabilities).map(([k, v]) =>
        [`P(${k})`, `${(v * 100).toFixed(1)}%`]
      ),
      ["Model version",   data.model_version],
    ]);

    const perModel = document.getElementById("adv-permodel-table");
    const rows = [];
    for (const [site, m] of Object.entries(data.per_model || {})) {
      rows.push([site, `${m.hb_estimate} g/dL — ${m.classification}`]);
    }
    perModel.innerHTML = rows.length ? _tableRows(rows) : "<tr><td>Single site</td></tr>";
  }

  function _tableRows(pairs) {
    return pairs.map(([k, v]) =>
      `<tr><td>${k}</td><td><strong>${v}</strong></td></tr>`
    ).join("");
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/js/advanced-mode.js
git commit -m "feat: add advanced research mode JS"
```

---

## Task 6: Write `frontend/js/app.js`

**Files:**
- Create: `frontend/js/app.js`

- [ ] **Step 1: Write `app.js`**

```javascript
// frontend/js/app.js
// Core orchestration: mode switching, PWA registration.

import { initScreeningMode } from "./screening-mode.js";
import { initAdvancedMode }  from "./advanced-mode.js";

// Register Service Worker for PWA offline shell
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/service-worker.js").catch(err => {
      console.warn("SW registration failed:", err);
    });
  });
}

// Mode toggle
const modeToggle      = document.getElementById("mode-toggle");
const screeningPanel  = document.getElementById("screening-mode");
const advancedPanel   = document.getElementById("advanced-mode");

let currentMode = "screening";

modeToggle.addEventListener("click", () => {
  if (currentMode === "screening") {
    screeningPanel.classList.remove("active");
    advancedPanel.classList.add("active");
    modeToggle.textContent = "Screening Mode";
    currentMode = "advanced";
  } else {
    advancedPanel.classList.remove("active");
    screeningPanel.classList.add("active");
    modeToggle.textContent = "Advanced Mode";
    currentMode = "screening";
  }
});

// Init both modes
initScreeningMode();
initAdvancedMode();
```

- [ ] **Step 2: Commit**

```bash
git add frontend/js/app.js
git commit -m "feat: add core app.js with mode switching and PWA registration"
```

---

## Task 7: Write PWA Files

**Files:**
- Create: `frontend/manifest.json`
- Create: `frontend/service-worker.js`

- [ ] **Step 1: Write `manifest.json`**

```json
{
  "name": "AnemiaScan",
  "short_name": "AnemiaScan",
  "description": "AI-powered non-invasive anemia screening",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#fafafa",
  "theme_color": "#b91c1c",
  "icons": [
    { "src": "assets/icons/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "assets/icons/icon-512.png", "sizes": "512x512", "type": "image/png" }
  ]
}
```

- [ ] **Step 2: Write `service-worker.js`**

```javascript
// frontend/service-worker.js
// Caches the UI shell for offline access.
// Images and API calls are always fetched fresh (network-first).

const CACHE_NAME = "anemiascan-shell-v1";
const SHELL_ASSETS = [
  "/",
  "/index.html",
  "/css/styles.css",
  "/js/app.js",
  "/js/screening-mode.js",
  "/js/advanced-mode.js",
  "/js/api.js",
  "/manifest.json",
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(SHELL_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", event => {
  // API calls: always network
  if (event.request.url.includes("/api/predict")) return;

  // Shell assets: cache-first
  event.respondWith(
    caches.match(event.request).then(cached => cached || fetch(event.request))
  );
});
```

- [ ] **Step 3: Create placeholder icons**

```bash
# Create minimal 1x1 PNG icons as placeholders
python - <<'EOF'
from PIL import Image
import pathlib
pathlib.Path("frontend/assets/icons").mkdir(parents=True, exist_ok=True)
for size in [192, 512]:
    img = Image.new("RGB", (size, size), color=(185, 28, 28))
    img.save(f"frontend/assets/icons/icon-{size}.png")
print("Placeholder icons created")
EOF
```

- [ ] **Step 4: Commit**

```bash
git add frontend/manifest.json frontend/service-worker.js frontend/assets/
git commit -m "feat: add PWA manifest and service worker"
```

---

## Task 7b: Add Language Toggle (Kannada / English)

**Files:**
- Create: `frontend/js/i18n.js`
- Modify: `frontend/index.html` (add lang toggle button)
- Modify: `frontend/js/app.js` (init i18n)

- [ ] **Step 1: Write `frontend/js/i18n.js`**

```javascript
// frontend/js/i18n.js
// Minimal Kannada/English toggle for ASHA-facing screening mode.

const STRINGS = {
  en: {
    title:        "AnemiaScan",
    conjunctiva:  "Conjunctiva",
    conjHint:     "Lower eyelid, everted",
    nailbed:      "Nail Bed",
    nailHint:     "Index finger, extended",
    captureBtn:   "📷 Capture / Upload",
    analyzeBtn:   "Analyze",
    loading:      "Analyzing image...",
    estHb:        "Estimated Hb",
    normal:       "✅ Normal",
    mild:         "⚠ Mild Anemia",
    moderate:     "⚠️ Moderate Anemia",
    severe:       "🚨 Severe Anemia",
    confidence:   "Confidence",
    disclaimer:   "⚠️ Research tool only — not a certified diagnostic device. Clinical confirmation required.",
    referral: {
      normal:   "No anemia detected. Continue regular monitoring.",
      mild:     "Mild anemia. Advise iron-rich diet and iron supplementation. Follow up in 4 weeks.",
      moderate: "Moderate anemia. Refer to PHC for iron supplementation and further evaluation.",
      severe:   "⚠️ Severe anemia. URGENT referral to PHC/hospital required.",
    },
  },
  kn: {
    title:        "ಅನೀಮಿಯಾಸ್ಕ್ಯಾನ್",
    conjunctiva:  "ಕಂಜಂಕ್ಟಿವಾ",
    conjHint:     "ಕೆಳ ರೆಪ್ಪೆ, ತಿರುಗಿಸಿ",
    nailbed:      "ಉಗುರಿನ ತಳ",
    nailHint:     "ತೋರು ಬೆರಳು, ಚಾಚಿ",
    captureBtn:   "📷 ಚಿತ್ರ ತೆಗೆಯಿರಿ / ಅಪ್ಲೋಡ್",
    analyzeBtn:   "ವಿಶ್ಲೇಷಿಸಿ",
    loading:      "ಚಿತ್ರ ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",
    estHb:        "ಅಂದಾಜು ಹಿಮೋಗ್ಲೋಬಿನ್",
    normal:       "✅ ಸಾಮಾನ್ಯ",
    mild:         "⚠ ಸ್ವಲ್ಪ ರಕ್ತಹೀನತೆ",
    moderate:     "⚠️ ಮಧ್ಯಮ ರಕ್ತಹೀನತೆ",
    severe:       "🚨 ತೀವ್ರ ರಕ್ತಹೀನತೆ",
    confidence:   "ವಿಶ್ವಾಸ",
    disclaimer:   "⚠️ ಇದು ಸಂಶೋಧನಾ ಸಾಧನ ಮಾತ್ರ — ವೈದ್ಯಕೀಯ ದೃಢೀಕರಣ ಅಗತ್ಯ.",
    referral: {
      normal:   "ರಕ್ತಹೀನತೆ ಕಂಡುಬಂದಿಲ್ಲ. ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಮುಂದುವರಿಸಿ.",
      mild:     "ಸ್ವಲ್ಪ ರಕ್ತಹೀನತೆ. ಕಬ್ಬಿಣಾಂಶ ಭರಿತ ಆಹಾರ ಮತ್ತು ಮಾತ್ರೆ ಶಿಫಾರಸು. 4 ವಾರದಲ್ಲಿ ಮರು ಪರೀಕ್ಷೆ.",
      moderate: "ಮಧ್ಯಮ ರಕ್ತಹೀನತೆ. PHC ಗೆ ಉಲ್ಲೇಖಿಸಿ.",
      severe:   "⚠️ ತೀವ್ರ ರಕ್ತಹೀನತೆ. ತಕ್ಷಣ PHC/ಆಸ್ಪತ್ರೆಗೆ ಕಳಿಸಿ.",
    },
  },
};

let currentLang = localStorage.getItem("anemiascan_lang") || "en";

export function t(key) {
  return STRINGS[currentLang]?.[key] ?? STRINGS.en[key] ?? key;
}

export function tReferral(cls) {
  return STRINGS[currentLang]?.referral?.[cls] ?? STRINGS.en.referral[cls] ?? "";
}

export function getCurrentLang() { return currentLang; }

export function setLang(lang) {
  if (!STRINGS[lang]) return;
  currentLang = lang;
  localStorage.setItem("anemiascan_lang", lang);
  applyTranslations();
}

export function applyTranslations() {
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.dataset.i18n;
    el.textContent = t(key);
  });
}

export function initI18n() {
  applyTranslations();
  const toggle = document.getElementById("lang-toggle");
  if (toggle) {
    toggle.textContent = currentLang === "en" ? "ಕನ್ನಡ" : "English";
    toggle.addEventListener("click", () => {
      const next = currentLang === "en" ? "kn" : "en";
      setLang(next);
      toggle.textContent = next === "en" ? "ಕನ್ನಡ" : "English";
    });
  }
}
```

- [ ] **Step 2: Add `data-i18n` attributes and lang toggle button to `index.html`**

In the header section of `frontend/index.html`, add the language toggle button:

```html
<!-- Replace the existing header-inner content with: -->
<div class="header-inner">
  <h1 class="app-title" data-i18n="title">🩸 AnemiaScan</h1>
  <div class="header-actions">
    <button id="lang-toggle" class="btn btn-ghost btn-sm">ಕನ್ನಡ</button>
    <button id="mode-toggle" class="btn btn-ghost" data-i18n="advancedMode">Advanced Mode</button>
  </div>
</div>
```

Add `data-i18n` attributes to key UI elements (capture labels, buttons, disclaimer, result labels).

- [ ] **Step 3: Update `frontend/js/app.js` to init i18n**

```javascript
// Add to top of app.js
import { initI18n } from "./i18n.js";

// Add to the module init block:
initI18n();
```

- [ ] **Step 4: Add lang toggle CSS to `styles.css`**

```css
.header-actions { display: flex; gap: 0.5rem; align-items: center; }
.btn-sm { padding: 0.35rem 0.75rem; font-size: 0.8rem; }
```

- [ ] **Step 5: Commit**

```bash
git add frontend/js/i18n.js
git commit -m "feat: add Kannada/English language toggle for ASHA screening mode"
```

---

## Task 8: Deploy to Netlify and Verify

- [ ] **Step 1: Verify `netlify.toml` is correct (from Plan 1)**

```bash
cat netlify.toml
```

Expected: `publish = "frontend"`.

- [ ] **Step 2: Push all frontend code to main**

```bash
git add frontend/
git status   # verify no secrets or large files included
git commit -m "feat: complete AnemiaScan frontend (screening + advanced modes, PWA)"
git push
```

- [ ] **Step 3: Connect repo to Netlify (one-time setup)**

```bash
# If Netlify CLI is installed:
netlify init
# Select: Connect to existing site or create new
# Confirm build dir: frontend
```

Or: netlify.com → New site from Git → select `hssling/anemia-detection-ai` → build: `frontend`.

- [ ] **Step 4: Set Netlify environment variables**

In Netlify dashboard → Site settings → Environment variables:

| Key | Value |
|-----|-------|
| `HF_SPACE_URL` | `https://hssling-anemia-screening.hf.space/api/predict` |
| `BACKUP_API_URL` | `https://anemiascan-inference-backup.onrender.com/api/predict` |

- [ ] **Step 5: Trigger deploy and verify**

```bash
netlify deploy --prod
```

Or wait for auto-deploy from GitHub push.

Open `https://anemiascan.netlify.app` in a mobile browser:
- [ ] Site loads
- [ ] Camera capture buttons work
- [ ] Disclaimer banner visible
- [ ] Mode toggle switches views
- [ ] Analysis with a test image returns a result

- [ ] **Step 6: Tag v0.5.0**

```bash
git tag v0.5.0 -m "Frontend complete: screening mode, advanced mode, PWA, Netlify deployed"
git push origin v0.5.0
```

---

## Completion Criteria

Plan 5 is complete when:
- [ ] Site loads at Netlify URL
- [ ] Screening mode: image upload + analysis works end-to-end
- [ ] Advanced mode: per-model breakdown renders
- [ ] JSON export works
- [ ] PWA: site can be added to home screen on Android
- [ ] Medical disclaimer visible on every result
- [ ] `v0.5.0` tag pushed

**Next:** Plan 6 — Full Automation (GitHub Actions CI/CD connecting all platforms)

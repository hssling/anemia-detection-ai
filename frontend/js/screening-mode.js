// frontend/js/screening-mode.js
import { predict } from "./api.js";
import { t, tReferral } from "./i18n.js";

const BADGE_LABELS = { normal: "Normal", mild: "Mild Anaemia", moderate: "Moderate Anaemia", severe: "Severe Anaemia" };

export function initScreeningMode() {
  const conjBtn   = document.getElementById("conj-btn");
  const conjInput = document.getElementById("conj-input");
  const conjPreview = document.getElementById("conj-preview");
  const nailBtn   = document.getElementById("nail-btn");
  const nailInput = document.getElementById("nail-input");
  const nailPreview = document.getElementById("nail-preview");
  const analyseBtn = document.getElementById("analyse-btn");
  const resultCard = document.getElementById("result-card");
  const spinner   = document.getElementById("spinner");

  let conjFile = null;
  let nailFile = null;

  function _wireCapture(btn, input, preview, onFile) {
    btn.addEventListener("click", () => input.click());
    input.addEventListener("change", () => {
      const file = input.files[0];
      if (!file) return;
      onFile(file);
      preview.src = URL.createObjectURL(file);
      preview.classList.remove("hidden");
      btn.closest(".capture-card").classList.add("has-image");
      _updateAnalyseBtn();
    });
  }

  _wireCapture(conjBtn, conjInput, conjPreview, (f) => { conjFile = f; });
  _wireCapture(nailBtn, nailInput, nailPreview, (f) => { nailFile = f; });

  function _updateAnalyseBtn() {
    analyseBtn.disabled = !conjFile && !nailFile;
  }

  analyseBtn.addEventListener("click", async () => {
    spinner.classList.remove("hidden");
    resultCard.classList.add("hidden");
    analyseBtn.disabled = true;

    try {
      const result = await predict(conjFile, nailFile);
      _renderResult(result);
    } catch (err) {
      _renderError(err.message);
    } finally {
      spinner.classList.add("hidden");
      _updateAnalyseBtn();
    }
  });

  function _renderResult(result) {
    const cls = result.classification;
    const hbEl    = document.getElementById("hb-value");
    const ciEl    = document.getElementById("ci-value");
    const badge   = document.getElementById("result-badge");
    const referral = document.getElementById("referral-msg");
    const confidence = document.getElementById("confidence-msg");

    hbEl.textContent = `${result.hb_estimate} g/dL`;
    ciEl.textContent  = `(95% CI: ${result.hb_ci_95[0]}–${result.hb_ci_95[1]})`;

    badge.textContent = BADGE_LABELS[cls] || cls;
    badge.className = `result-badge ${cls}`;

    referral.textContent = tReferral(cls);

    const probs = result.class_probabilities;
    const pct = Math.round((probs[cls] || 0) * 100);
    confidence.textContent = `Model confidence: ${pct}%`;

    resultCard.classList.remove("hidden");
  }

  function _renderError(msg) {
    const hbEl = document.getElementById("hb-value");
    hbEl.textContent = "Error";
    document.getElementById("ci-value").textContent = "";
    document.getElementById("result-badge").textContent = msg;
    document.getElementById("result-badge").className = "result-badge moderate";
    document.getElementById("referral-msg").textContent = t("retryMsg");
    document.getElementById("confidence-msg").textContent = "";
    document.getElementById("result-card").classList.remove("hidden");
  }
}

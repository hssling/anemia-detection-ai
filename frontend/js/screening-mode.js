import { predict } from "./api.js";
import { t, tReferral } from "./i18n.js";

const BADGE_TEXT = {
  normal: "Normal",
  mild: "Mild anemia",
  moderate: "Moderate anemia",
  severe: "Severe anemia",
};

function showPreview(input, preview, card, clearButton) {
  const file = input.files?.[0];
  if (!file) return;
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  clearButton.hidden = false;
  card.classList.add("has-image");
}

function clearImage(input, preview, card, clearButton) {
  input.value = "";
  preview.removeAttribute("src");
  preview.hidden = true;
  clearButton.hidden = true;
  card.classList.remove("has-image");
}

function buildProbabilityBars(probabilities) {
  return Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => `
      <div class="probability-row">
        <span>${label}</span>
        <div class="probability-track"><div class="probability-fill" style="width:${Math.max(value * 100, 4)}%"></div></div>
        <strong>${(value * 100).toFixed(1)}%</strong>
      </div>
    `)
    .join("");
}

export function initScreeningMode() {
  const conjInput = document.getElementById("conj-input");
  const nailInput = document.getElementById("nail-input");
  const conjPreview = document.getElementById("conj-preview");
  const nailPreview = document.getElementById("nail-preview");
  const conjCard = document.getElementById("conj-card");
  const nailCard = document.getElementById("nail-card");
  const conjClear = document.getElementById("conj-clear");
  const nailClear = document.getElementById("nail-clear");
  const analyzeButton = document.getElementById("analyze-btn");
  const loading = document.getElementById("loading");
  const errorBanner = document.getElementById("error-banner");
  const resultCard = document.getElementById("result-card");

  function updateAnalyzeState() {
    analyzeButton.disabled = !(conjInput.files?.length || nailInput.files?.length);
  }

  function hideError() {
    errorBanner.hidden = true;
    errorBanner.textContent = "";
  }

  function showError(message) {
    errorBanner.hidden = false;
    errorBanner.textContent = `Analysis failed: ${message}`;
  }

  function setLoading(isLoading) {
    loading.hidden = !isLoading;
    analyzeButton.disabled = isLoading || !(conjInput.files?.length || nailInput.files?.length);
  }

  function renderResult(result) {
    const badge = document.getElementById("result-badge");
    const confidence = Math.max(...Object.values(result.class_probabilities || { normal: 0 }));
    document.getElementById("result-hb-value").textContent = Number(result.hb_estimate).toFixed(2);
    document.getElementById("result-ci").textContent = `95% CI ${result.hb_ci_95[0]} to ${result.hb_ci_95[1]}`;
    badge.className = `result-badge ${result.classification}`;
    badge.textContent = BADGE_TEXT[result.classification] || result.classification;
    document.getElementById("result-confidence-value").textContent = `${(confidence * 100).toFixed(0)}%`;
    document.getElementById("result-referral").textContent = tReferral(result.classification);
    document.getElementById("probability-bars").innerHTML = buildProbabilityBars(result.class_probabilities || {});
    document.getElementById("result-source").textContent = result._served_by || "";

    const warningEl = document.getElementById("result-warning");
    const warningLines = [];
    if (Array.isArray(result.warnings) && result.warnings.length) warningLines.push(...result.warnings);
    if (result.disclaimer) warningLines.push(result.disclaimer);
    if (warningLines.length) {
      warningEl.hidden = false;
      warningEl.textContent = warningLines.join(" ");
    } else {
      warningEl.hidden = true;
      warningEl.textContent = "";
    }

    resultCard.hidden = false;
  }

  document.querySelectorAll(".capture-trigger").forEach((button) => {
    button.addEventListener("click", () => {
      document.getElementById(button.dataset.target)?.click();
    });
  });

  conjInput.addEventListener("change", () => {
    showPreview(conjInput, conjPreview, conjCard, conjClear);
    updateAnalyzeState();
  });

  nailInput.addEventListener("change", () => {
    showPreview(nailInput, nailPreview, nailCard, nailClear);
    updateAnalyzeState();
  });

  conjClear.addEventListener("click", () => {
    clearImage(conjInput, conjPreview, conjCard, conjClear);
    updateAnalyzeState();
  });

  nailClear.addEventListener("click", () => {
    clearImage(nailInput, nailPreview, nailCard, nailClear);
    updateAnalyzeState();
  });

  analyzeButton.addEventListener("click", async () => {
    hideError();
    resultCard.hidden = true;
    setLoading(true);
    try {
      const result = await predict(conjInput.files?.[0] || null, nailInput.files?.[0] || null, "ensemble", false);
      renderResult(result);
    } catch (error) {
      showError(error.message);
    } finally {
      setLoading(false);
    }
  });

  document.addEventListener("anemiascan:langchange", () => {
    const referral = document.getElementById("result-referral");
    const badge = document.getElementById("result-badge");
    analyzeButton.textContent = t("analyze");
    if (!resultCard.hidden) {
      const cls = [...badge.classList].find((name) => BADGE_TEXT[name]);
      if (cls) referral.textContent = tReferral(cls);
    }
  });

  updateAnalyzeState();
  setLoading(false);
}

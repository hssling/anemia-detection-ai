// frontend/js/advanced-mode.js
import { predict } from "./api.js";

export function initAdvancedMode() {
  const conjBtn     = document.getElementById("adv-conj-btn");
  const conjInput   = document.getElementById("adv-conj-input");
  const conjPreview = document.getElementById("adv-conj-preview");
  const nailBtn     = document.getElementById("adv-nail-btn");
  const nailInput   = document.getElementById("adv-nail-input");
  const nailPreview = document.getElementById("adv-nail-preview");
  const analyseBtn  = document.getElementById("adv-analyse-btn");
  const modelSelect = document.getElementById("model-select");
  const spinner     = document.getElementById("adv-spinner");
  const advResult   = document.getElementById("adv-result");
  const exportBtn   = document.getElementById("export-btn");

  let conjFile = null;
  let nailFile = null;
  let lastResult = null;

  function _wireCapture(btn, input, preview, onFile) {
    btn.addEventListener("click", () => input.click());
    input.addEventListener("change", () => {
      const file = input.files[0];
      if (!file) return;
      onFile(file);
      preview.src = URL.createObjectURL(file);
      preview.classList.remove("hidden");
      btn.closest(".capture-card").classList.add("has-image");
      analyseBtn.disabled = !conjFile && !nailFile;
    });
  }

  _wireCapture(conjBtn, conjInput, conjPreview, (f) => { conjFile = f; });
  _wireCapture(nailBtn, nailInput, nailPreview, (f) => { nailFile = f; });

  analyseBtn.addEventListener("click", async () => {
    spinner.classList.remove("hidden");
    advResult.classList.add("hidden");
    analyseBtn.disabled = true;

    try {
      const result = await predict(conjFile, nailFile, modelSelect.value);
      lastResult = result;
      _renderResult(result);
    } catch (err) {
      alert(`Prediction failed: ${err.message}`);
    } finally {
      spinner.classList.add("hidden");
      analyseBtn.disabled = !conjFile && !nailFile;
    }
  });

  function _renderResult(result) {
    const tbody = document.getElementById("result-tbody");
    tbody.innerHTML = "";
    const rows = [
      ["Hb Estimate", `${result.hb_estimate} g/dL`],
      ["95% CI", `${result.hb_ci_95[0]} – ${result.hb_ci_95[1]} g/dL`],
      ["Classification", result.classification],
    ];
    for (const [label, val] of Object.entries(result.class_probabilities || {})) {
      rows.push([`P(${label})`, `${(val * 100).toFixed(1)}%`]);
    }
    for (const [label, val] of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${label}</td><td>${val}</td>`;
      tbody.appendChild(tr);
    }

    const perModelDiv = document.getElementById("per-model-div");
    perModelDiv.innerHTML = "";
    for (const [site, siteResult] of Object.entries(result.per_model || {})) {
      const section = document.createElement("div");
      section.className = "per-model-section";
      section.innerHTML = `
        <h4>${site}</h4>
        <p>Hb: ${siteResult.hb_estimate} g/dL (CI: ${siteResult.hb_ci_95[0]}–${siteResult.hb_ci_95[1]})</p>
        <p>Class: ${siteResult.classification}</p>
      `;
      perModelDiv.appendChild(section);
    }

    advResult.classList.remove("hidden");
  }

  exportBtn.addEventListener("click", () => {
    if (!lastResult) return;
    const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `anemiascan-result-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });
}

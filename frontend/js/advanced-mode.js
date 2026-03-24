import { predict } from "./api.js";

function rowsFromPairs(pairs) {
  return pairs.map(([key, value]) => `<tr><td>${key}</td><td><strong>${value}</strong></td></tr>`).join("");
}

export function initAdvancedMode() {
  const analyzeButton = document.getElementById("adv-analyze-btn");
  const loading = document.getElementById("adv-loading");
  const errorBanner = document.getElementById("adv-error");
  const resultsPanel = document.getElementById("adv-results");
  const summaryTable = document.getElementById("adv-results-table");
  const perModelTable = document.getElementById("adv-permodel-table");
  const jsonPreview = document.getElementById("adv-json");
  const exportButton = document.getElementById("export-json-btn");
  const modelSelect = document.getElementById("model-select");
  const includeGradcam = document.getElementById("include-gradcam");
  let lastResult = null;

  function setLoading(on) {
    loading.hidden = !on;
    analyzeButton.disabled = on;
  }

  analyzeButton.addEventListener("click", async () => {
    const conjFile = document.getElementById("adv-conj-input").files?.[0] || null;
    const nailFile = document.getElementById("adv-nail-input").files?.[0] || null;

    if (!conjFile && !nailFile) {
      errorBanner.hidden = false;
      errorBanner.textContent = "Please upload at least one image.";
      return;
    }

    errorBanner.hidden = true;
    resultsPanel.hidden = true;
    setLoading(true);

    try {
      const result = await predict(conjFile, nailFile, modelSelect.value, includeGradcam.checked);
      lastResult = result;
      summaryTable.innerHTML = rowsFromPairs([
        ["Served by", result._served_by || "unknown"],
        ["Hb estimate", `${result.hb_estimate} g/dL`],
        ["95% CI", `${result.hb_ci_95[0]} to ${result.hb_ci_95[1]} g/dL`],
        ["Classification", result.classification],
        ["Model version", result.model_version || "n/a"],
        ...Object.entries(result.class_probabilities || {}).map(([key, value]) => [`P(${key})`, `${(value * 100).toFixed(2)}%`]),
      ]);

      const perModelRows = [];
      for (const [name, payload] of Object.entries(result.per_model || {})) {
        perModelRows.push([name, `${payload.hb_estimate} g/dL · ${payload.classification}`]);
      }
      perModelTable.innerHTML = perModelRows.length ? rowsFromPairs(perModelRows) : rowsFromPairs([["status", "single-model response"]]);
      jsonPreview.textContent = JSON.stringify(result, null, 2);
      resultsPanel.hidden = false;
    } catch (error) {
      errorBanner.hidden = false;
      errorBanner.textContent = `Advanced analysis failed: ${error.message}`;
    } finally {
      setLoading(false);
    }
  });

  exportButton.addEventListener("click", () => {
    if (!lastResult) return;
    const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `anemiascan-result-${Date.now()}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  });
}

// frontend/js/app.js
import { initScreeningMode } from "./screening-mode.js";
import { initAdvancedMode } from "./advanced-mode.js";
import { initI18n } from "./i18n.js";

// Register service worker
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/service-worker.js").catch((err) => {
      console.warn("Service worker registration failed:", err);
    });
  });
}

// Mode toggle
const modeToggleBtn = document.getElementById("mode-toggle");
const screeningPanel = document.getElementById("screening-panel");
const advancedPanel = document.getElementById("advanced-panel");

if (!modeToggleBtn || !screeningPanel || !advancedPanel) {
  console.error("app.js: required DOM elements not found");
} else {

let advancedMode = false;

modeToggleBtn.addEventListener("click", () => {
  advancedMode = !advancedMode;
  screeningPanel.classList.toggle("active", !advancedMode);
  screeningPanel.classList.toggle("hidden", advancedMode);
  advancedPanel.classList.toggle("active", advancedMode);
  advancedPanel.classList.toggle("hidden", !advancedMode);
  modeToggleBtn.dataset.i18n = advancedMode ? "screeningMode" : "advancedMode";
  // Re-apply translation
  import("./i18n.js").then(({ t, applyTranslations }) => {
    modeToggleBtn.textContent = t(modeToggleBtn.dataset.i18n);
    applyTranslations();
  });
});

initScreeningMode();
initAdvancedMode();
initI18n();
}

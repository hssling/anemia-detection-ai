import { initI18n } from "./i18n.js";
import { initScreeningMode } from "./screening-mode.js";
import { initAdvancedMode } from "./advanced-mode.js";

function initModeToggle() {
  const toggle = document.getElementById("mode-toggle");
  const screening = document.getElementById("screening-mode");
  const advanced = document.getElementById("advanced-mode");
  let currentMode = "screening";

  toggle.addEventListener("click", () => {
    currentMode = currentMode === "screening" ? "advanced" : "screening";
    screening.classList.toggle("active", currentMode === "screening");
    advanced.classList.toggle("active", currentMode === "advanced");
    toggle.textContent = currentMode === "screening" ? "Advanced Mode" : "Screening Mode";
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("./service-worker.js").catch((error) => {
      console.warn("Service worker registration failed", error);
    });
  });
}

initI18n();
initModeToggle();
initScreeningMode();
initAdvancedMode();

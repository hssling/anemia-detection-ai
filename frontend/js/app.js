import { initI18n } from "./i18n.js";
import { initScreeningMode } from "./screening-mode.js";

const THEME_KEY = "anemiascan_theme";
const MODE_KEY = "anemiascan_mode";

let advancedModeModuleLoaded = false;

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  document.querySelector('meta[name="theme-color"]')?.setAttribute(
    "content",
    theme === "dark" ? "#141d29" : "#9c2f1f",
  );
  const darkButton = document.getElementById("theme-dark");
  const lightButton = document.getElementById("theme-light");
  if (darkButton && lightButton) {
    const darkActive = theme === "dark";
    darkButton.classList.toggle("active", darkActive);
    lightButton.classList.toggle("active", !darkActive);
    darkButton.setAttribute("aria-pressed", darkActive ? "true" : "false");
    lightButton.setAttribute("aria-pressed", darkActive ? "false" : "true");
  }
  localStorage.setItem(THEME_KEY, theme);
}

async function ensureAdvancedMode() {
  if (advancedModeModuleLoaded) return;
  const { initAdvancedMode } = await import("./advanced-mode.js");
  initAdvancedMode();
  advancedModeModuleLoaded = true;
}

function setMode(mode) {
  const screening = document.getElementById("screening-mode");
  const advanced = document.getElementById("advanced-mode");
  const toggle = document.getElementById("mode-toggle");
  const isAdvanced = mode === "advanced";
  screening.classList.toggle("active", !isAdvanced);
  advanced.classList.toggle("active", isAdvanced);
  toggle.textContent = isAdvanced ? "Screening Mode" : "Advanced Mode";
  localStorage.setItem(MODE_KEY, mode);
  if (isAdvanced) {
    ensureAdvancedMode();
  }
}

function initThemeToggle() {
  const saved = localStorage.getItem(THEME_KEY);
  applyTheme(saved || "dark");
  document.getElementById("theme-dark")?.addEventListener("click", () => applyTheme("dark"));
  document.getElementById("theme-light")?.addEventListener("click", () => applyTheme("light"));
}

function initModeToggle() {
  const toggle = document.getElementById("mode-toggle");
  const initialMode = localStorage.getItem(MODE_KEY) || "screening";
  setMode(initialMode);

  toggle.addEventListener("click", async () => {
    const nextMode = document.getElementById("advanced-mode").classList.contains("active")
      ? "screening"
      : "advanced";
    setMode(nextMode);
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    const register = () => {
      navigator.serviceWorker.register("./service-worker.js").catch((error) => {
        console.warn("Service worker registration failed", error);
      });
    };

    if ("requestIdleCallback" in window) {
      window.requestIdleCallback(register);
    } else {
      window.setTimeout(register, 800);
    }
  });
}

initI18n();
initThemeToggle();
initModeToggle();
initScreeningMode();

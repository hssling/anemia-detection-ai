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
  const themeToggle = document.getElementById("theme-toggle");
  const themeToggleLabel = document.getElementById("theme-toggle-label");
  if (themeToggle) {
    themeToggle.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
  }
  if (themeToggleLabel) {
    themeToggleLabel.textContent = theme === "dark" ? "Dark Mode" : "Light Mode";
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
  const preferredDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  applyTheme(saved || (preferredDark ? "dark" : "light"));

  document.getElementById("theme-toggle")?.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(next);
  });
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

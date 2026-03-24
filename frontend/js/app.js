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

window.addEventListener("load", async () => {
  if (!("serviceWorker" in navigator)) return;
  try {
    const registrations = await navigator.serviceWorker.getRegistrations();
    await Promise.all(registrations.map((registration) => registration.unregister()));
    if ("caches" in window) {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((key) => key.startsWith("anemiascan-shell-"))
          .map((key) => caches.delete(key)),
      );
    }
  } catch (error) {
    console.warn("Service worker cleanup failed", error);
  }
});

initI18n();
initThemeToggle();
initModeToggle();
initScreeningMode();

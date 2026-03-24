// frontend/js/i18n.js
const STRINGS = {
  en: {
    title: "🩸 AnemiaScan",
    advancedMode: "Advanced Mode",
    screeningMode: "Screening Mode",
    disclaimer: "⚠️ Research tool only. Not a certified diagnostic device. Clinical confirmation required.",
    conjLabel: "Conjunctiva",
    nailLabel: "Nail-bed",
    captureBtn: "📷 Capture",
    analyseBtn: "Analyse",
    hbLabel: "Hb:",
    modelLabel: "Model:",
    metricHeader: "Metric",
    valueHeader: "Value",
    perModelHeader: "Per-Model Results",
    exportBtn: "Export JSON",
    retryMsg: "Please try again or contact support.",
    referral: {
      normal: "✅ Haemoglobin within normal range. No immediate referral needed.",
      mild: "⚠️ Mild anaemia detected. Dietary advice and follow-up recommended.",
      moderate: "🔶 Moderate anaemia. Refer to PHC for investigation and treatment.",
      severe: "🚨 Severe anaemia. Urgent referral to hospital required.",
    },
  },
  kn: {
    title: "🩸 ರಕ್ತಹೀನತಾ ಪರೀಕ್ಷೆ",
    advancedMode: "ಸುಧಾರಿತ ಮೋಡ್",
    screeningMode: "ತಪಾಸಣೆ ಮೋಡ್",
    disclaimer: "⚠️ ಸಂಶೋಧನಾ ಸಾಧನ ಮಾತ್ರ. ವೈದ್ಯಕೀಯ ದೃಢೀಕರಣ ಅಗತ್ಯ.",
    conjLabel: "ಕಣ್ಣಿನ ತಳ",
    nailLabel: "ಉಗುರು ಹಾಸಿಗೆ",
    captureBtn: "📷 ತೆಗೆಯಿರಿ",
    analyseBtn: "ವಿಶ್ಲೇಷಿಸಿ",
    hbLabel: "ಹಿಮೋಗ್ಲೋಬಿನ್:",
    modelLabel: "ಮಾದರಿ:",
    metricHeader: "ಮಾಪಕ",
    valueHeader: "ಮೌಲ್ಯ",
    perModelHeader: "ಪ್ರತಿ-ಮಾದರಿ ಫಲಿತಾಂಶ",
    exportBtn: "JSON ರಫ್ತು",
    retryMsg: "ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ ಅಥವಾ ಬೆಂಬಲ ಸಂಪರ್ಕಿಸಿ.",
    referral: {
      normal: "✅ ಹಿಮೋಗ್ಲೋಬಿನ್ ಸಾಮಾನ್ಯ ಮಿತಿಯಲ್ಲಿದೆ. ತಕ್ಷಣದ ಉಲ್ಲೇಖ ಅಗತ್ಯವಿಲ್ಲ.",
      mild: "⚠️ ಸಣ್ಣ ರಕ್ತಹೀನತೆ ಕಂಡಿದೆ. ಆಹಾರ ಸಲಹೆ ಮತ್ತು ಅನುಸರಣೆ ಶಿಫಾರಸು.",
      moderate: "🔶 ಮಧ್ಯಮ ರಕ್ತಹೀನತೆ. ತನಿಖೆಗಾಗಿ PHC ಗೆ ಉಲ್ಲೇಖಿಸಿ.",
      severe: "🚨 ತೀವ್ರ ರಕ್ತಹೀನತೆ. ಆಸ್ಪತ್ರೆಗೆ ತುರ್ತು ಉಲ್ಲೇಖ ಅಗತ್ಯ.",
    },
  },
};

let currentLang = "en";

export function t(key) {
  return STRINGS[currentLang]?.[key] ?? STRINGS.en[key] ?? key;
}

export function tReferral(cls) {
  return STRINGS[currentLang]?.referral?.[cls] ?? STRINGS.en.referral[cls] ?? "";
}

export function setLang(lang) {
  currentLang = lang;
  applyTranslations();
}

export function applyTranslations() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;
    el.textContent = t(key);
  });
  // Update lang toggle button label (show opposite language)
  const langBtn = document.getElementById("lang-toggle");
  if (langBtn) langBtn.textContent = currentLang === "en" ? "ಕನ್ನಡ" : "English";
}

export function initI18n() {
  const langBtn = document.getElementById("lang-toggle");
  if (!langBtn) return;
  langBtn.addEventListener("click", () => {
    setLang(currentLang === "en" ? "kn" : "en");
  });
  applyTranslations();
}

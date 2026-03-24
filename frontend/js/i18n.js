const STRINGS = {
  en: {
    title: "AnemiaScan",
    disclaimer: "Research tool only. This is not a certified diagnostic device. Clinical confirmation is required.",
    conjunctiva: "Conjunctiva",
    conj_hint: "Lower eyelid, gently everted",
    nailbed: "Nail Bed",
    nail_hint: "Index finger, extended and steady",
    capture_button: "Capture / Upload",
    analyze: "Analyze",
    loading: "Analyzing image...",
    estimated_hb: "Estimated Hb",
    confidence: "Confidence",
    referral: {
      normal: "No anemia detected. Continue regular monitoring.",
      mild: "Mild anemia. Recommend iron-rich diet, supplementation, and follow-up.",
      moderate: "Moderate anemia. Refer to PHC for evaluation and treatment.",
      severe: "Severe anemia. Urgent referral to PHC or hospital is required.",
    },
  },
  kn: {
    title: "ಅನೀಮಿಯಾಸ್ಕ್ಯಾನ್",
    disclaimer: "ಇದು ಸಂಶೋಧನಾ ಸಾಧನ ಮಾತ್ರ. ಇದು ಪ್ರಮಾಣಿತ ವೈದ್ಯಕೀಯ ನಿರ್ಧಾರ ಸಾಧನ ಅಲ್ಲ. ವೈದ್ಯಕೀಯ ದೃಢೀಕರಣ ಅಗತ್ಯ.",
    conjunctiva: "ಕಂಜಂಕ್ಟಿವಾ",
    conj_hint: "ಕೆಳ ರೆಪ್ಪೆ, ಸೌಮ್ಯವಾಗಿ ತಿರುಗಿಸಿ",
    nailbed: "ಉಗುರಿನ ತಳ",
    nail_hint: "ತೋರಿಬೆರಳು, ನೇರವಾಗಿ ಮತ್ತು ಸ್ಥಿರವಾಗಿ",
    capture_button: "ಚಿತ್ರ ತೆಗೆಯಿರಿ / ಅಪ್ಲೋಡ್",
    analyze: "ವಿಶ್ಲೇಷಿಸಿ",
    loading: "ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",
    estimated_hb: "ಅಂದಾಜು ಹಿಮೋಗ್ಲೋಬಿನ್",
    confidence: "ವಿಶ್ವಾಸ",
    referral: {
      normal: "ರಕ್ತಹೀನತೆ ಕಂಡುಬಂದಿಲ್ಲ. ನಿಯಮಿತ ಮೇಲ್ವಿಚಾರಣೆ ಮುಂದುವರಿಸಿ.",
      mild: "ಸ್ವಲ್ಪ ರಕ್ತಹೀನತೆ. ಕಬ್ಬಿಣಯುಕ್ತ ಆಹಾರ, ಪೂರಕ ಚಿಕಿತ್ಸೆ ಮತ್ತು ಮರುಪರಿಶೀಲನೆ ಶಿಫಾರಸು.",
      moderate: "ಮಧ್ಯಮ ರಕ್ತಹೀನತೆ. ಹೆಚ್ಚಿನ ಪರಿಶೀಲನೆಗಾಗಿ PHC ಗೆ ಕಳುಹಿಸಿ.",
      severe: "ತೀವ್ರ ರಕ್ತಹೀನತೆ. ತುರ್ತು PHC ಅಥವಾ ಆಸ್ಪತ್ರೆ ರೆಫರಲ್ ಅಗತ್ಯ.",
    },
  },
};

let currentLanguage = localStorage.getItem("anemiascan_lang") || "en";

export function t(key) {
  return STRINGS[currentLanguage]?.[key] || STRINGS.en[key] || key;
}

export function tReferral(classification) {
  return STRINGS[currentLanguage]?.referral?.[classification] || STRINGS.en.referral[classification] || "";
}

export function initI18n() {
  const toggle = document.getElementById("lang-toggle");

  function apply() {
    document.documentElement.lang = currentLanguage;
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = t(node.dataset.i18n);
    });
    if (toggle) toggle.textContent = currentLanguage === "en" ? "ಕನ್ನಡ" : "English";
    document.dispatchEvent(new CustomEvent("anemiascan:langchange", { detail: currentLanguage }));
  }

  if (toggle) {
    toggle.addEventListener("click", () => {
      currentLanguage = currentLanguage === "en" ? "kn" : "en";
      localStorage.setItem("anemiascan_lang", currentLanguage);
      apply();
    });
  }

  apply();
}

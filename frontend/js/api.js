const PRIMARY_API = window._ENV?.PRIMARY_API_URL || "/api/predict";
const FALLBACK_API = window._ENV?.BACKUP_API_URL || "https://anemia-detection-ai-1.onrender.com/api/predict";
const HF_SPACE_API = window._ENV?.HF_SPACE_URL || "https://hssling-anemia-screening.hf.space/api/predict";
const TIMEOUT_MS = 45000;

function makeTimedRequest(url, body) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), TIMEOUT_MS);
  return fetch(url, { method: "POST", body, signal: controller.signal })
    .then(async (response) => {
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`HTTP ${response.status}: ${text}`);
      }
      return response.json();
    })
    .catch((error) => {
      if (error?.name === "AbortError") {
        throw new Error(`request timed out after ${TIMEOUT_MS / 1000}s`);
      }
      throw error;
    })
    .finally(() => window.clearTimeout(timer));
}

function cloneFormData(original) {
  const next = new FormData();
  for (const [key, value] of original.entries()) next.append(key, value);
  return next;
}

export async function predict(conjFile, nailFile, model = "ensemble", includeGradcam = false) {
  const formData = new FormData();
  if (conjFile) formData.append("conjunctiva_image", conjFile);
  if (nailFile) formData.append("nailbed_image", nailFile);
  formData.append("model", model);
  if (includeGradcam) formData.append("include_gradcam", "true");

  const attempts = [
    { label: "Netlify proxy", url: PRIMARY_API },
    { label: "HF Space", url: HF_SPACE_API },
    { label: "Render backup", url: FALLBACK_API },
  ];
  const errors = [];

  for (const attempt of attempts) {
    try {
      const result = await makeTimedRequest(attempt.url, cloneFormData(formData));
      return { ...result, _served_by: attempt.label };
    } catch (error) {
      errors.push(`${attempt.label}: ${error.message}`);
    }
  }
  throw new Error(errors.join(" | "));
}

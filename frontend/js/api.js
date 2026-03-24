// frontend/js/api.js
const PRIMARY_API = window._ENV?.HF_SPACE_URL || "https://hssling-anemia-screening.hf.space/api/predict";
const FALLBACK_API = window._ENV?.BACKUP_API_URL || "";

/**
 * @param {File|null} conjFile
 * @param {File|null} nailFile
 * @param {string} model - "ensemble"|"conjunctiva"|"nailbed"
 * @returns {Promise<object>}
 */
export async function predict(conjFile, nailFile, model = "ensemble") {
  const form = new FormData();
  if (conjFile) form.append("conjunctiva_image", conjFile);
  if (nailFile) form.append("nailbed_image", nailFile);
  form.append("_model", model);

  return _fetchWithFallback(PRIMARY_API, FALLBACK_API, form);
}

async function _fetchWithFallback(primaryUrl, fallbackUrl, form) {
  try {
    return await _timedFetch(primaryUrl, form, 45_000);
  } catch (primaryErr) {
    if (fallbackUrl) {
      console.warn("Primary API failed, trying fallback:", primaryErr.message);
      return _timedFetch(fallbackUrl, form, 45_000);
    }
    throw primaryErr;
  }
}

async function _timedFetch(url, form, timeoutMs) {
  const controller = new AbortController();
  const timerId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { method: "POST", body: form, signal: controller.signal });
    if (!res.ok) {
      const detail = await res.text().catch(() => res.statusText);
      throw new Error(`API error ${res.status}: ${detail}`);
    }
    return res.json();
  } finally {
    clearTimeout(timerId);
  }
}

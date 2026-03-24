const CACHE_NAME = "anemiascan-shell-v3";
const SHELL_FILES = [
  "/",
  "/index.html",
  "/css/styles.css",
  "/js/app.js",
  "/js/api.js",
  "/js/i18n.js",
  "/js/screening-mode.js",
  "/js/advanced-mode.js",
  "/manifest.json",
  "/assets/icons/icon.svg",
  "/assets/icons/icon-192.svg",
  "/assets/icons/icon-512.svg"
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_FILES)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key)))));
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  if (event.request.url.includes("/api/predict")) return;
  event.respondWith(caches.match(event.request).then((cached) => cached || fetch(event.request)));
});

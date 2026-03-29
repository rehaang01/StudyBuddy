// ─────────────────────────────────────────────────────────────────────────────
// StudyBuddy Service Worker — Offline-Capable PWA
// ─────────────────────────────────────────────────────────────────────────────
// Strategy:
//   Navigation requests           → Network-First, cached app shell fallback
//   Versioned static assets       → Network-First, cache fallback
//   API calls                     → Network-First, cache fallback
//   Blob images / CDN assets      → Cache-First
// ─────────────────────────────────────────────────────────────────────────────

const BUILD_ID = new URL(self.location.href).searchParams.get("build") || "dev";
const CACHE_VERSION = `sb-${BUILD_ID}`;
const SHELL_CACHE = `shell-${CACHE_VERSION}`;
const API_CACHE = `api-${CACHE_VERSION}`;
const IMAGE_CACHE = `images-${CACHE_VERSION}`;

const MAX_CACHED_IMAGES = 60;

const CACHEABLE_API_PATHS = [
  "/quiz/history",
  "/quiz/",
  "/diagrams/history",
  "/goals/",
  "/settings/",
  "/settings/dismissed-weak-topics",
  "/settings/account",
  "/settings/billing",
  "/settings/connectors",
  "/sessions/weekly",
  "/chat/conversations",
  "/chat/history/",
  "/coins",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE).then((cache) => {
      return cache.addAll([
        "/",
        "/manifest.json",
        "/favicon.ico",
      ]).catch(() => {
        // Best effort only.
      });
    })
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== SHELL_CACHE && key !== API_CACHE && key !== IMAGE_CACHE)
          .map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

function isCacheableAPI(url) {
  return CACHEABLE_API_PATHS.some((path) => url.pathname.includes(path));
}

function isBlobImage(url) {
  return (
    url.hostname.includes("blob.core.windows.net") ||
    url.pathname.includes("/upload/view-file")
  );
}

function isStaticAsset(url) {
  return (
    url.pathname.endsWith(".js") ||
    url.pathname.endsWith(".css") ||
    url.pathname.endsWith(".woff2") ||
    url.pathname.endsWith(".svg") ||
    url.pathname.endsWith(".ico") ||
    url.pathname.endsWith(".png") ||
    url.pathname.endsWith(".jpg") ||
    url.pathname.endsWith(".jpeg") ||
    url.pathname.endsWith(".webp")
  );
}

async function trimImageCache() {
  const cache = await caches.open(IMAGE_CACHE);
  const keys = await cache.keys();
  if (keys.length > MAX_CACHED_IMAGES) {
    const toDelete = keys.slice(0, keys.length - MAX_CACHED_IMAGES);
    await Promise.all(toDelete.map((key) => cache.delete(key)));
  }
}

async function cacheResponse(cacheName, request, response) {
  if (!response || !response.ok) return response;
  const cache = await caches.open(cacheName);
  await cache.put(request, response.clone());
  return response;
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.method !== "GET") return;

  if (request.mode === "navigate" && url.origin === self.location.origin) {
    event.respondWith(
      fetch(request)
        .then(async (response) => {
          if (response.ok) {
            const cache = await caches.open(SHELL_CACHE);
            await cache.put("/", response.clone());
          }
          return response;
        })
        .catch(async () => {
          return (await caches.match("/")) || new Response("Offline", { status: 503 });
        })
    );
    return;
  }

  if (isBlobImage(url)) {
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) return cached;
        return fetch(request, { mode: "cors", credentials: "omit" })
          .then(async (response) => {
            if (response && (response.ok || response.type === "opaque")) {
              const cache = await caches.open(IMAGE_CACHE);
              await cache.put(request, response.clone());
              await trimImageCache();
            }
            return response;
          })
          .catch(() => new Response("Offline — image not cached", { status: 503 }));
      })
    );
    return;
  }

  if (url.hostname === "cdnjs.cloudflare.com") {
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) return cached;
        return fetch(request)
          .then((response) => cacheResponse(SHELL_CACHE, request, response))
          .catch(() => new Response("Offline — CDN resource not cached", { status: 503 }));
      })
    );
    return;
  }

  if (isCacheableAPI(url)) {
    event.respondWith(
      fetch(request)
        .then((response) => cacheResponse(API_CACHE, request, response))
        .catch(async () => {
          const cached = await caches.match(request);
          if (cached) {
            const headers = new Headers(cached.headers);
            headers.set("X-SW-Cached", "true");
            return new Response(cached.body, {
              status: cached.status,
              statusText: cached.statusText,
              headers,
            });
          }
          return new Response(
            JSON.stringify({ error: "offline", message: "No cached data available" }),
            { status: 503, headers: { "Content-Type": "application/json" } }
          );
        })
    );
    return;
  }

  if (url.origin === self.location.origin && isStaticAsset(url)) {
    event.respondWith(
      fetch(request)
        .then((response) => cacheResponse(SHELL_CACHE, request, response))
        .catch(async () => {
          return (await caches.match(request)) || new Response("Offline", { status: 503 });
        })
    );
  }
});

self.addEventListener("message", (event) => {
  if (event.data === "skipWaiting") {
    self.skipWaiting();
  }
});

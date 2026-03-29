from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

app = FastAPI(title="StudyBuddy API", version="1.0", redirect_slashes=False)

# Read frontend URL from .env (optional)
frontend_url = os.getenv("FRONTEND_URL", "").strip()
azure_frontend_url = os.getenv("AZURE_FRONTEND_URL", "").strip()

# Start with a sensible dev-origin list for common localhost usage
# (explicit origins + a regex to cover any localhost / 127.0.0.1 port)
origins = []

# If the user supplied a FRONTEND_URL, add it exactly
if frontend_url:
    # ensure scheme is present for clarity; otherwise allow as-is
    parsed = urlparse(frontend_url)
    if parsed.scheme:
        origins.append(frontend_url)
    else:
        # if they provided e.g. "localhost:5173", add both http/https variants
        origins.append(f"http://{frontend_url}")
        origins.append(f"https://{frontend_url}")

if azure_frontend_url:
    parsed = urlparse(azure_frontend_url)
    if parsed.scheme:
        origins.append(azure_frontend_url)
    else:
        origins.append(f"http://{azure_frontend_url}")
        origins.append(f"https://{azure_frontend_url}")

# Always allow the common local dev hosts (explicit entries)
origins.extend(
    [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]
)

# Allow any localhost / 127.0.0.1 origin on any port (useful during development)
allow_origin_regex = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=allow_origin_regex,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Log configured CORS values for easier debugging
@app.on_event("startup")
async def _print_cors_info():
    print("CORS configured with allow_origins:")
    for o in origins:
        print("  -", o)
    print("CORS configured with allow_origin_regex:", allow_origin_regex)

    # Ensure settings container exists in Cosmos DB
    try:
        from app.services.settings_service import ensure_settings_container
        await ensure_settings_container()
    except Exception as e:
        print(f"[startup] Settings container init error (non-fatal): {e}")

    # Ensure coins container exists in Cosmos DB
    try:
        from app.services.coins_service import ensure_coins_container
        await ensure_coins_container()
    except Exception as e:
        print(f"[startup] Coins container init error (non-fatal): {e}")

    # Ensure sessions container exists in Cosmos DB
    try:
        from app.services.sessions_service import ensure_sessions_container
        await ensure_sessions_container()
    except Exception as e:
        print(f"[startup] Sessions container init error (non-fatal): {e}")

    # Ensure flashcards container exists in Cosmos DB
    try:
        from app.services.cosmos_service import ensure_flashcards_container
        await ensure_flashcards_container()
    except Exception as e:
        print(f"[startup] Flashcards container init error (non-fatal): {e}")

    # Start the notification scheduler (APScheduler — 9 PM goal reminders, weekly goals)
    try:
        from app.routers.notifications import start_scheduler
        start_scheduler()
    except Exception as e:
        print(f"[startup] Notification scheduler error (non-fatal): {e}")

    try:
        from app.services.email_service import get_email_service_status
        email_status = get_email_service_status()
        if email_status["configured"]:
            print(
                "[startup] Email service configured "
                f"({email_status['smtp_host']}:{email_status['smtp_port']})."
            )
        else:
            print("[startup] Email service not configured; notification emails will fail until SMTP env vars are set.")
    except Exception as e:
        print(f"[startup] Email status check error (non-fatal): {e}")

    # Ensure Azure AI Search index exists before any request hits it.
    try:
        from app.services.search_service import create_index_if_not_exists
        create_index_if_not_exists()
        print("[startup] Azure AI Search index ready.")
    except Exception as e:
        print(f"[startup] Search index init error (non-fatal): {e}")

# ── Routers ────────────────────────────────────────────────────────────────────
from app.routers.upload import router as upload_router
app.include_router(upload_router)

# Phase 2 + 3 routers will be added here as we build them:
from app.routers.chat import router as chat_router
from app.routers.quiz import router as quiz_router
from app.routers.diagrams import router as diagrams_router
from app.routers.study_plans import router as study_plans_router
from app.routers.flashcards import router as flashcards_router
from app.routers.goals import router as goals_router
from app.routers.settings import router as settings_router
from app.routers.sessions import router as sessions_router
from app.routers.graph import router as graph_router
from app.routers.notifications import router as notifications_router
from app.routers.coins import router as coins_router
app.include_router(chat_router)
app.include_router(quiz_router)
app.include_router(diagrams_router)
app.include_router(study_plans_router)
app.include_router(flashcards_router)
app.include_router(goals_router)
app.include_router(settings_router)
app.include_router(sessions_router)
app.include_router(graph_router)
app.include_router(notifications_router)
app.include_router(coins_router)


@app.on_event("shutdown")
async def _shutdown():
    try:
        from app.routers.notifications import stop_scheduler
        stop_scheduler()
    except Exception:
        pass


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0"}

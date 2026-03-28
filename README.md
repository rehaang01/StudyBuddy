# StudyBuddy

AI-native study assistant with chat, quizzes, study plans, diagramming, and multimodal document understanding. The project ships a FastAPI backend for RAG + task orchestration and a Vite/React/TypeScript frontend with a polished UI and simple design.

## Highlights
- Chat with RAG over uploaded study material (Gemini + Azure AI Search).
- Quiz generator, diagram/flowchart builder, AI image generation, and study-plan creator.
- Document ingestion pipeline (Azure Blob + Document Intelligence) with chunking and embeddings.
- Multilingual support: translation and text-to-speech (Azure Translator + Speech).
- Goals, reminders, and settings with Cosmos DB persistence.

## Architecture
- **Backend (FastAPI)**: RAG chat, upload pipeline, quiz, diagrams/images, study plans, goals, settings, TTS/translation, email reminders. Routers live under [Backend/app/routers](Backend/app/routers) with orchestration/services under [Backend/app/services](Backend/app/services).
- **Search + Storage**: Azure Blob for uploads and generated assets; Azure Document Intelligence for OCR; Azure AI Search for hybrid/vector retrieval; Azure Cosmos DB for conversations, quizzes, diagrams, goals, and settings.
- **Frontend (Vite + React + TypeScript)**: UI in [Frontend/src](Frontend/src) with shadcn/ui components, React Router pages, React Query data layer, Tailwind styling, Vitest for tests.

## Tech Stack
- Frontend: Vite, React 18, TypeScript, Tailwind, shadcn/ui (Radix), React Router, TanStack Query, Vitest + Testing Library, KaTeX, Mermaid.
- Backend: FastAPI, Uvicorn, Pydantic v2, Azure SDKs (Blob, Cosmos, AI Search, Translator, Speech, Document Intelligence), Google Gemini (genai SDK), python-dotenv.
- Infra/Services: Azure Blob Storage, Azure AI Search, Azure Cosmos DB (NoSQL), Azure Document Intelligence, Azure Translator, Azure Speech, SMTP (optional).

## Prerequisites
- Node.js 18+ (or Bun 1.1+) and npm/pnpm/bun for the frontend.
- Python 3.11+ for the backend.
- Azure resources: Storage account, AI Search, Cosmos DB (NoSQL), Document Intelligence, Translator, Speech. Google Gemini API key.

## Quick Start
1) **Backend**
```bash
cd Backend
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2) **Frontend** (npm shown; pnpm/bun also work)
```bash
cd Frontend
npm install
npm run dev -- --host --port 5173
```

3) Open the app at http://localhost:5173 and ensure the backend is reachable at http://localhost:8000.

## Environment
Create `.env` files in `Backend/` (and optionally `Frontend/`). Minimum backend variables:

```
GEMINI_API_KEY=
AZURE_STORAGE_CONNECTION_STRING=
AZURE_STORAGE_CONTAINER_NAME=studybuddy-files  # optional override
AZURE_COSMOS_CONNECTION_STRING=
AZURE_COSMOS_DB_NAME=studybuddy                # optional override
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_KEY=
AZURE_SEARCH_INDEX_NAME=studybuddy-index       # optional override
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=
AZURE_DOCUMENT_INTELLIGENCE_KEY=
AZURE_TRANSLATOR_KEY=
AZURE_TRANSLATOR_REGION=
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
FRONTEND_URL=http://localhost:5173             # for CORS allowlist
```

Optional (email reminders):

```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
EMAIL_SENDER=
EMAIL_PASSWORD=
```

Notes:
- Azure Speech and Translator keys/regions are required for TTS and translation endpoints.
- If email creds are omitted, reminders are logged to the console (demo mode).

## Key Flows
- **Upload → RAG**: `/upload/file` stores to Blob, runs Document Intelligence OCR, chunks text, embeds with Gemini, and indexes into Azure AI Search scoped by conversation.
- **Chat**: `/chat/message` performs intent classification, RAG retrieval, and streams Gemini responses; supports translations, TTS, study-plan/quiz/diagram dispatch.
- **Quizzes**: `/quiz/generate`, `/quiz/preclassify`, `/quiz/submit`, `/quiz/history` provide document-grounded or general quizzes with weak-area labels.
- **Diagrams & Images**: `/diagrams/generate`, `/diagrams/history`, `/diagrams/generate-image` build Mermaid diagrams and AI images, persisted to Blob/Cosmos.
- **Study Plans**: `/study_plans/parse_intent`, `/study_plans/generate`, `/study_plans/mark_goal_saved` generate structured plans and link to goals.
- **Goals**: CRUD under `/goals`, plus one-click reminder email.
- **Settings**: `/settings` endpoints manage profile, notifications, AI prefs, appearance, connectors, and billing plan stubs.

Router sources: <br>
[Backend/app/routers/chat.py](Backend/app/routers/chat.py), [Backend/app/routers/quiz.py](Backend/app/routers/quiz.py), [Backend/app/routers/diagrams.py](Backend/app/routers/diagrams.py), [Backend/app/routers/study_plans.py](Backend/app/routers/study_plans.py), [Backend/app/routers/goals.py](Backend/app/routers/goals.py), [Backend/app/routers/upload.py](Backend/app/routers/upload.py), [Backend/app/routers/settings.py](Backend/app/routers/settings.py).

## Scripts
- Frontend: `npm run dev`, `npm run build`, `npm run preview`, `npm run lint`, `npm run test`, `npm run test:watch`.
- Backend: run with `uvicorn app.main:app --reload`. Add `--port` to avoid clashes with Vite.

## Testing
- Frontend: `npm run test` (Vitest + Testing Library; setup in [Frontend/test/setup.ts](Frontend/test/setup.ts)).
- Backend: no automated tests yet; consider adding FastAPI route/unit tests for services.

## Production Notes
- Create the Azure AI Search index once via `create_index_if_not_exists()` in [Backend/app/services/search_service.py](Backend/app/services/search_service.py) (safe to run on startup).
- Ensure Blob containers and Cosmos containers exist; `ensure_settings_container()` runs at startup to create the `settings` container.
- Use separate `.env` per environment; set `FRONTEND_URL` to the deployed web app for CORS.

## Project Layout
- Backend source: [Backend/app](Backend/app)
- Frontend source: [Frontend/src](Frontend/src)
- Shared assets: [Frontend/public](Frontend/public), [Frontend/src/assets](Frontend/src/assets)

## Project Tree

```
├── 📁 Backend
│   ├── 📁 app
│   │   ├── 📁 models
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 routers
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 chat.py
│   │   │   ├── 🐍 diagrams.py
│   │   │   ├── 🐍 goals.py
│   │   │   ├── 🐍 quiz.py
│   │   │   ├── 🐍 settings.py
│   │   │   ├── 🐍 study_plans.py
│   │   │   └── 🐍 upload.py
│   │   ├── 📁 services
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 blob_service.py
│   │   │   ├── 🐍 cosmos_service.py
│   │   │   ├── 🐍 doc_intelligence_service.py
│   │   │   ├── 🐍 email_service.py
│   │   │   ├── 🐍 gemini_service.py
│   │   │   ├── 🐍 goals_service.py
│   │   │   ├── 🐍 search_service.py
│   │   │   ├── 🐍 settings_service.py
│   │   │   ├── 🐍 study_plan_service.py
│   │   │   ├── 🐍 translator_service.py
│   │   │   └── 🐍 tts_service.py
│   │   ├── 📁 utils
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 chunking.py
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 main.py
│   ├── ⚙️ .env.example
│   └── 📄 requirements.txt
├── 📁 Frontend
│   ├── 📁 public
│   │   ├── 📄 favicon.ico
│   │   ├── 🖼️ placeholder.svg
│   │   └── 📄 robots.txt
│   ├── 📁 src
│   │   ├── 📁 assets
│   │   │   └── ⚙️ loading.json
│   │   ├── 📁 components
│   │   │   ├── 📁 layout
│   │   │   │   ├── 📄 AppHeader.tsx
│   │   │   │   ├── 📄 AppLayout.tsx
│   │   │   │   └── 📄 AppSidebar.tsx
│   │   │   ├── 📁 ui
│   │   │   │   ├── 📄 accordion.tsx
│   │   │   │   ├── 📄 alert-dialog.tsx
│   │   │   │   ├── 📄 alert.tsx
│   │   │   │   ├── 📄 aspect-ratio.tsx
│   │   │   │   ├── 📄 avatar.tsx
│   │   │   │   ├── 📄 badge.tsx
│   │   │   │   ├── 📄 breadcrumb.tsx
│   │   │   │   ├── 📄 button.tsx
│   │   │   │   ├── 📄 calendar.tsx
│   │   │   │   ├── 📄 card.tsx
│   │   │   │   ├── 📄 carousel.tsx
│   │   │   │   ├── 📄 chart.tsx
│   │   │   │   ├── 📄 checkbox.tsx
│   │   │   │   ├── 📄 collapsible.tsx
│   │   │   │   ├── 📄 command.tsx
│   │   │   │   ├── 📄 context-menu.tsx
│   │   │   │   ├── 📄 dialog.tsx
│   │   │   │   ├── 📄 drawer.tsx
│   │   │   │   ├── 📄 dropdown-menu.tsx
│   │   │   │   ├── 📄 form.tsx
│   │   │   │   ├── 📄 hover-card.tsx
│   │   │   │   ├── 📄 input-otp.tsx
│   │   │   │   ├── 📄 input.tsx
│   │   │   │   ├── 📄 label.tsx
│   │   │   │   ├── 📄 menubar.tsx
│   │   │   │   ├── 📄 navigation-menu.tsx
│   │   │   │   ├── 📄 pagination.tsx
│   │   │   │   ├── 📄 popover.tsx
│   │   │   │   ├── 📄 progress.tsx
│   │   │   │   ├── 📄 radio-group.tsx
│   │   │   │   ├── 📄 resizable.tsx
│   │   │   │   ├── 📄 scroll-area.tsx
│   │   │   │   ├── 📄 select.tsx
│   │   │   │   ├── 📄 separator.tsx
│   │   │   │   ├── 📄 sheet.tsx
│   │   │   │   ├── 📄 sidebar.tsx
│   │   │   │   ├── 📄 skeleton.tsx
│   │   │   │   ├── 📄 slider.tsx
│   │   │   │   ├── 📄 sonner.tsx
│   │   │   │   ├── 📄 switch.tsx
│   │   │   │   ├── 📄 table.tsx
│   │   │   │   ├── 📄 tabs.tsx
│   │   │   │   ├── 📄 textarea.tsx
│   │   │   │   ├── 📄 toast.tsx
│   │   │   │   ├── 📄 toaster.tsx
│   │   │   │   ├── 📄 toggle-group.tsx
│   │   │   │   ├── 📄 toggle.tsx
│   │   │   │   ├── 📄 tooltip.tsx
│   │   │   │   └── 📄 use-toast.ts
│   │   │   ├── 📄 LoadingDots.tsx
│   │   │   └── 📄 NavLink.tsx
│   │   ├── 📁 contexts
│   │   │   ├── 📄 AppearanceContext.tsx
│   │   │   └── 📄 LanguageContext.tsx
│   │   ├── 📁 hooks
│   │   │   ├── 📄 use-mobile.tsx
│   │   │   └── 📄 use-toast.ts
│   │   ├── 📁 lib
│   │   │   └── 📄 utils.ts
│   │   ├── 📁 pages
│   │   │   ├── 📄 ChatPage.tsx
│   │   │   ├── 📄 DashboardPage.tsx
│   │   │   ├── 📄 GoalsPage.tsx
│   │   │   ├── 📄 ImagesPage.tsx
│   │   │   ├── 📄 Index.tsx
│   │   │   ├── 📄 LandingPage.tsx
│   │   │   ├── 📄 NotFound.tsx
│   │   │   ├── 📄 QuizzesPage.tsx
│   │   │   └── 📄 SettingsPage.tsx
│   │   ├── 📁 test
│   │   │   ├── 📄 example.test.ts
│   │   │   └── 📄 setup.ts
│   │   ├── 🎨 App.css
│   │   ├── 📄 App.tsx
│   │   ├── 🎨 index.css
│   │   ├── 📄 main.tsx
│   │   └── 📄 vite-env.d.ts
│   ├── ⚙️ .gitignore
│   ├── 📄 bun.lockb
│   ├── ⚙️ components.json
│   ├── 📄 eslint.config.js
│   ├── 🌐 index.html
│   ├── ⚙️ package-lock.json
│   ├── ⚙️ package.json
│   ├── 📄 postcss.config.js
│   ├── 📄 tailwind.config.ts
│   ├── ⚙️ tsconfig.app.json
│   ├── ⚙️ tsconfig.json
│   ├── ⚙️ tsconfig.node.json
│   ├── 📄 vite.config.ts
│   └── 📄 vitest.config.ts
├── ⚙️ .gitignore
└── 📝 README.md
```

## Contributing
- Use feature branches and conventional commits.
- Keep backend endpoints documented in the routers above.
- Add tests for new logic (Vitest for frontend; propose pytest for backend).

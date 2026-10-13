# Chat With Your Docs — Frontend

A standalone, zero-dependency frontend for **Chat With Your Docs**.  
Plain HTML + CSS + JavaScript (ES2020, one IIFE). No build step, no npm.

---

## Quick Start

### 1. Start the Backend

In the **backend** project directory:

```powershell
python server.py
```

The API runs at `http://127.0.0.1:8000`.

### 2. Start the Frontend Dev Server

In **this** directory (`chat-with-your-docs-frontend`):

```powershell
python -m http.server 5173
```

Open **http://localhost:5173** in your browser.

---

## Development with the Mock Backend

No Ollama? No problem. The included `mock_server.py` fakes all 7 endpoints with realistic delays and NDJSON streaming:

```powershell
# Terminal 1 — mock backend
python mock_server.py

# Terminal 2 — static file server
python -m http.server 5173
```

Open http://localhost:5173.  
Click **"Load sample notes"** in the Files menu, then ask a question.

---

## Configuration

### Changing the API URL

Edit [`config.js`](config.js):

```js
window.CHAT_CONFIG = {
  API_BASE: "http://127.0.0.1:8000"  // ← change this
};
```

### Backend CORS

The backend must allow requests from the frontend origin.  
In the backend's `.env` or environment, set:

```
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Or edit the `_DEFAULT_CORS_ORIGINS` list in `server.py`.

### Why `file://` Doesn't Work

Browsers enforce **same-origin policy** for `fetch()` from `file://` URLs. The frontend must be served over HTTP — even a simple `python -m http.server` works.

---

## Files

| File | Purpose |
|------|---------|
| `index.html` | Semantic HTML shell — three-panel layout |
| `style.css` | All styles — dark + light themes via CSS custom properties |
| `app.js` | Application logic — state, API, streaming, rendering |
| `config.js` | API base URL (the only thing you might need to edit) |
| `mock_server.py` | Dev-only fake backend (stdlib only, port 8000) |
| `README.md` | This file |

---

## Features

- **Three-panel layout**: Files · Chat · Sources & Trace (collapses to tabs on mobile)
- **File upload**: Drag-and-drop or click; supports PDF, DOCX, MD, TXT up to 50 MB
- **Streaming chat**: NDJSON stream with live token rendering, blinking caret
- **Mode switch**: Fast / Balanced / Thorough (persisted in localStorage)
- **Citation chips**: `[file.pdf#2.0]` → `[1]` clickable chip; scrolls to source card
- **Sources panel**: Appears before the answer finishes; collapsible excerpt with highlighted search terms
- **Trace panel**: Query rewrite, retrieval flow (6-step), Self-RAG decisions, timing bars — all update live
- **Dark + light themes** via `prefers-color-scheme`
- **Mobile responsive** (tabs at ≤ 900 px, no horizontal overflow at 360 px)
- **Accessibility**: ARIA live regions, keyboard-operable drop zone, focus indicators

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| **Red banner: "Can't reach the backend"** | Start the backend with `python server.py` (or `python mock_server.py`). Check `API_BASE` in `config.js`. |
| **CORS error in the browser console** | Add `http://localhost:5173` to `CORS_ORIGINS` in the backend config. |
| **Ollama not running / model not found** | Start Ollama (`ollama serve`) and pull the models listed in the top bar. |
| **First question is very slow** | The reranker model loads on first use — subsequent queries are faster. |
| **Upload fails silently** | Open DevTools → Network tab to see the request/response. |
| **`file://` gives fetch errors** | Serve with `python -m http.server 5173` instead of opening the file directly. |

---

## Manual Test Checklist

> Run these against `mock_server.py` if the real backend isn't available.

- [ ] Page loads with welcome card, no error banner
- [ ] **Files menu** → "Load sample notes" populates the file list with 3 items
- [ ] **Upload**: drag a `.pdf` → appears with "Indexing…" spinner → success toast → "Indexed · N chunks"
- [ ] **Upload**: drag a `.exe` → error toast "unsupported type"
- [ ] **Remove file** → hover row → ✕ button appears → click → file removed
- [ ] **Remove all** → Files menu → confirm strip appears → Cancel works → Remove All clears list
- [ ] **Mode switch** Fast / Balanced / Thorough persists after page reload
- [ ] **Chat**: ask a question → user bubble appears right-aligned
- [ ] Sources panel fills **before** the answer finishes streaming
- [ ] Answer streams token by token with blinking caret
- [ ] Citations like `[1]` appear as clickable chips in the finished answer
- [ ] Clicking a citation chip highlights the matching source card on the right
- [ ] **Stop button** halts streaming mid-response
- [ ] **Trace panel** — all sections: Query Rewrite, Retrieval (6-step flow), Self-RAG Decisions, Timing bars
- [ ] **Timing**: bar widths are proportional to ms values; total shown
- [ ] Clicking an earlier assistant message updates Sources & Trace panel
- [ ] **Mobile (≤ 900 px)**: tabs Files / Chat / Trace switch panels correctly
- [ ] **Dark mode** and **light mode** both look correct
- [ ] **Kill the mock**: stop `mock_server.py` → red error banner appears with Retry button → Retry re-hides banner when mock is back

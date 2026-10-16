/* ═══════════════════════════════════════════════════════════════════════════
   Chat With Your Docs — app.js
   Plain ES2020 IIFE. No modules, no build step, no external libraries.
   Sections: STATE · API · RENDERING · EVENTS
   ═══════════════════════════════════════════════════════════════════════════ */
(function () {
  'use strict';

  /* ══════════════════════════════════════════════════════════════════════════
     §1  STATE
     ══════════════════════════════════════════════════════════════════════════ */
  const API_BASE = (window.CHAT_CONFIG || {}).API_BASE || 'http://127.0.0.1:8000';

  const state = {
    // Backend state
    files: [],          // [{name, size, chunks, pages, type}]
    ready: false,
    error: null,
    pipeline: '',
    modes: ['fast', 'balanced', 'thorough'],
    models: {},
    hasSamples: false,

    // Auth & Portal State
    token: null,
    user: null,
    portal: 'student',   // 'student' or 'admin'

    // UI state
    mode: 'balanced',           // current retrieval mode
    streaming: false,           // is a chat request in-flight?
    abortController: null,      // AbortController for active stream

    // Conversation history (last 8 assistant turns)
    history: [],                // [{role:'user'|'assistant', content}]

    // Each entry in messages:
    // { id, role, content, sources:[], terms:[], trace:{}, timingMs, usedRetrieval, el }
    messages: [],
    selectedMsgId: null,        // which assistant message is "selected" in Sources panel

    // Citation map per message: citationKey -> sequential number
    // citationKey = "filename#score_id" from [file.pdf#2.0]
  };

  // Read saved settings
  try {
    const savedMode = localStorage.getItem('chat_mode');
    if (['fast', 'balanced', 'thorough'].includes(savedMode)) state.mode = savedMode;

    const savedToken = localStorage.getItem('doctalk_token');
    const savedUser = localStorage.getItem('doctalk_user');
    const savedPortal = localStorage.getItem('doctalk_portal');
    if (savedToken && savedUser) {
      state.token = savedToken;
      state.user = JSON.parse(savedUser);
      state.portal = savedPortal || (state.user && state.user.role) || 'student';
    } else {
      // Default demo initial session
      state.user = {
        id: 'demo-student',
        name: 'Alex Student',
        email: 'student@doctalk.com',
        role: 'student'
      };
      state.portal = 'student';
    }
  } catch (_) { /* ignore */ }

  /* ══════════════════════════════════════════════════════════════════════════
     §2  API
     ══════════════════════════════════════════════════════════════════════════ */

  /** Core fetch helper. Throws {message, network:true} on network failure. */
  async function request(path, opts = {}) {
    const url = API_BASE + path;
    opts.headers = opts.headers || {};
    if (state.token) {
      if (opts.headers instanceof Headers) {
        if (!opts.headers.has('Authorization')) {
          opts.headers.set('Authorization', `Bearer ${state.token}`);
        }
      } else if (!opts.headers['Authorization']) {
        opts.headers['Authorization'] = `Bearer ${state.token}`;
      }
    }
    try {
      const res = await fetch(url, opts);
      if (res.status === 401 && !path.startsWith('/api/auth/')) {
        showAuthModal('Session expired. Please sign in again.');
      }
      return res;
    } catch (err) {
      const e = new Error(`Can't reach the backend at ${API_BASE}`);
      e.network = true;
      throw e;
    }
  }

  async function apiLogin(email, password) {
    const res = await request('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Invalid credentials' }));
      throw new Error(err.detail || 'Login failed');
    }
    return res.json();
  }

  async function apiRegister(name, email, password, role) {
    const res = await request('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password, role })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Registration failed' }));
      throw new Error(err.detail || 'Registration failed');
    }
    return res.json();
  }

  async function apiGetState() {
    const res = await request('/api/state');
    if (!res.ok) throw new Error(`State fetch failed: ${res.status}`);
    return res.json();
  }

  async function apiUpload(files) {
    const fd = new FormData();
    for (const f of files) fd.append('files', f);
    const res = await request('/api/upload', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
  }

  async function apiDeleteFile(name) {
    const res = await request(`/api/files/${encodeURIComponent(name)}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
    return res.json();
  }

  async function apiDeleteAll() {
    const res = await request('/api/files', { method: 'DELETE' });
    if (!res.ok) throw new Error(`Delete all failed: ${res.status}`);
    return res.json();
  }

  async function apiLoadSamples() {
    const res = await request('/api/samples', { method: 'POST' });
    if (!res.ok) throw new Error(`Load samples failed: ${res.status}`);
    return res.json();
  }

  /* ══════════════════════════════════════════════════════════════════════════
     §3  RENDERING UTILITIES
     ══════════════════════════════════════════════════════════════════════════ */

  /** Escape HTML. Used everywhere server text is inserted. */
  function esc(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /** Format bytes to human-readable */
  function fmtBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  }

  /** File type → emoji icon */
  function fileIcon(name) {
    const ext = (name || '').split('.').pop().toLowerCase();
    return { pdf: '📄', docx: '📝', md: '📋', txt: '📃' }[ext] || '📄';
  }

  /** Show a toast notification */
  function toast(message, type = 'info', durationMs = 3500) {
    const container = document.getElementById('toast-container');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.setAttribute('role', 'status');
    const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
    el.innerHTML = `<span class="toast-icon" aria-hidden="true">${icon}</span><span>${esc(message)}</span>`;
    container.appendChild(el);
    setTimeout(() => {
      el.classList.add('hiding');
      el.addEventListener('animationend', () => el.remove());
    }, durationMs);
  }

  /** Show / hide error banner */
  function showBanner(msg) {
    const banner = document.getElementById('error-banner');
    document.getElementById('error-banner-text').textContent = msg;
    banner.classList.add('visible');
  }
  function hideBanner() {
    document.getElementById('error-banner').classList.remove('visible');
  }

  /* ─── Render answer markdown (safe) ───────────────────────────────────── */

  /**
   * Render escaped answer text:
   *  - **bold** → <strong>
   *  - `code`   → <code>
   *  - leading "- " → <ul><li>
   *  - leading "N. " → <ol><li>
   *  - blank line    → paragraph break
   */
  function renderMarkdown(text) {
    // Split into paragraphs by blank lines
    const paragraphs = text.split(/\n\n+/);
    return paragraphs.map(para => {
      const lines = para.split('\n');
      // Check if all non-empty lines are bullet list items
      const isBullet = lines.every(l => l === '' || /^[-*]\s/.test(l));
      const isOrdered = lines.every(l => l === '' || /^\d+\.\s/.test(l));

      let html;
      if (isBullet && lines.some(l => /^[-*]\s/.test(l))) {
        const items = lines.filter(l => /^[-*]\s/.test(l))
          .map(l => `<li>${inlineMarkdown(l.replace(/^[-*]\s/, ''))}</li>`).join('');
        html = `<ul>${items}</ul>`;
      } else if (isOrdered && lines.some(l => /^\d+\.\s/.test(l))) {
        const items = lines.filter(l => /^\d+\.\s/.test(l))
          .map(l => `<li>${inlineMarkdown(l.replace(/^\d+\.\s/, ''))}</li>`).join('');
        html = `<ol>${items}</ol>`;
      } else {
        html = `<p>${inlineMarkdown(lines.join('\n'))}</p>`;
      }
      return html;
    }).join('');
  }

  function inlineMarkdown(text) {
    // Escape first, then apply inline transforms
    return esc(text)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/`([^`]+)`/g, '<code>$1</code>');
  }

  /**
   * Post-process rendered HTML to replace citation markers like [file.pdf#2.0]
   * with clickable chip spans. Returns {html, citationMap}.
   */
  function processCitations(html, sources, existingMap) {
    const map = existingMap || {};  // citationKey -> sequential number
    let counter = Object.keys(map).length;

    // Regex: [anything#number] — the backend pattern
    const re = /\[([^\]]+#[\d.]+)\]/g;
    const result = html.replace(re, (_, key) => {
      if (!map[key]) {
        counter++;
        map[key] = counter;
      }
      const num = map[key];
      // Find matching source
      const srcIdx = sources.findIndex(s =>
        key.startsWith(s.source) || key === s.id ||
        s.source === key.split('#')[0]);
      const srcId = srcIdx >= 0 ? sources[srcIdx].id : null;
      const dataId = srcId ? ` data-src-id="${esc(srcId)}"` : '';
      return `<button class="citation-chip" type="button" aria-label="Source ${num}"${dataId}>[${num}]</button>`;
    });
    return { html: result, citationMap: map };
  }

  /* ─── Highlight terms in text (for source excerpts) ──────────────────── */
  function highlightTerms(text, terms) {
    if (!terms || terms.length === 0) return esc(text);
    // Build safe regex from terms
    const escaped = terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const re = new RegExp(`\\b(${escaped.join('|')})\\b`, 'gi');
    return esc(text).replace(re, '<mark>$1</mark>');
  }

  /* ══════════════════════════════════════════════════════════════════════════
     §4  TOP BAR
     ══════════════════════════════════════════════════════════════════════════ */
  function renderModelChips() {
    const container = document.getElementById('model-chips');
    container.innerHTML = '';
    const { models } = state;
    if (!models) return;

    const chips = [
      { label: 'LLM', value: models.llm },
      { label: 'Embedding', value: models.embedding },
      { label: 'Reranker', value: models.reranker },
    ].filter(c => c.value);

    chips.forEach(c => {
      const div = document.createElement('div');
      div.className = 'model-chip';
      div.setAttribute('role', 'listitem');
      div.innerHTML = `<span class="chip-dot" aria-hidden="true"></span>${esc(c.label)} <strong>${esc(c.value)}</strong>`;
      container.appendChild(div);
    });
  }

  /* ══════════════════════════════════════════════════════════════════════════
     §5  FILES PANEL
     ══════════════════════════════════════════════════════════════════════════ */
  function renderFiles() {
    const list = document.getElementById('file-list');
    const noDocsEl = document.getElementById('no-docs-yet');
    const samplesBtn = document.getElementById('load-samples-btn');

    list.innerHTML = '';
    const files = state.files;

    // Toggle sample button
    samplesBtn.style.display = state.hasSamples ? '' : 'none';

    if (!files || files.length === 0) {
      noDocsEl.style.display = '';
    } else {
      noDocsEl.style.display = 'none';
      files.forEach(f => {
        const row = createFileRow(f);
        list.appendChild(row);
      });
    }

    updateChatInputState();
  }

  function createFileRow(file, indexing = false) {
    const div = document.createElement('div');
    div.className = 'file-row' + (indexing ? ' indexing' : '');
    div.setAttribute('role', 'listitem');
    div.setAttribute('data-filename', file.name);

    const chunkInfo = file.chunks ? `${file.chunks} chunks` : '';
    const pageInfo = file.pages ? `· ${file.pages} pages` : '';
    const metaText = indexing
      ? '' // filled below
      : (file.chunks ? `<span class="indexed">Indexed · ${esc(chunkInfo)}${pageInfo ? ' ' + esc(pageInfo) : ''}</span>`
                     : `${esc(fmtBytes(file.size || 0))}`);

    div.innerHTML = `
      <div class="file-icon" aria-hidden="true">${fileIcon(file.name)}</div>
      <div class="file-info">
        <div class="file-name" title="${esc(file.name)}">${esc(file.name)}</div>
        <div class="file-meta">
          ${indexing
            ? `<span class="indexing-badge"><span class="indexing-spinner" aria-hidden="true"></span>Indexing…</span>`
            : metaText}
        </div>
      </div>
      ${!indexing ? `
      <button class="file-remove-btn" type="button"
              aria-label="Remove ${esc(file.name)}"
              data-filename="${esc(file.name)}">✕</button>
      ` : ''}
    `;

    if (!indexing) {
      div.querySelector('.file-remove-btn').addEventListener('click', e => {
        e.stopPropagation();
        handleDeleteFile(file.name);
      });
    }

    return div;
  }

  async function handleDeleteFile(name) {
    try {
      const newState = await apiDeleteFile(name);
      applyState(newState);
      toast(`Removed ${name}`, 'success');
    } catch (err) {
      toast(err.message, 'error');
    }
  }

  /* ══════════════════════════════════════════════════════════════════════════
     §6  CHAT INPUT STATE
     ══════════════════════════════════════════════════════════════════════════ */
  function updateChatInputState() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const welcomeCard = document.getElementById('welcome-card');
    const noFilesNotice = document.getElementById('no-files-notice');
    const backendNotReady = document.getElementById('backend-not-ready');

    // Remove existing notices
    if (noFilesNotice) noFilesNotice.remove();
    if (backendNotReady) backendNotReady.remove();

    const hasFiles = state.files && state.files.length > 0;

    if (!hasFiles) {
      // No files uploaded
      input.disabled = true;
      input.placeholder = 'Upload a document to start';
      sendBtn.disabled = true;

      // Show arrow hint in welcome card area if welcome is visible
      if (welcomeCard && !welcomeCard.classList.contains('hidden')) {
        const notice = document.createElement('div');
        notice.id = 'no-files-notice';
        notice.innerHTML = `
          <div class="arrow-hint" aria-hidden="true">👈</div>
          <p>Upload a document in the <strong>Files</strong> panel to start chatting.</p>
        `;
        notice.style.cssText = 'padding:20px 24px;text-align:center;color:var(--text-secondary);font-size:13px;display:flex;flex-direction:column;align-items:center;gap:8px;';
        const msgList = document.getElementById('message-list');
        msgList.appendChild(notice);
      }
    } else if (!state.ready && state.error) {
      // Files exist but pipeline not ready
      input.disabled = true;
      input.placeholder = 'Backend not ready…';
      sendBtn.disabled = true;

      const notice = document.createElement('div');
      notice.id = 'backend-not-ready';
      notice.innerHTML = `<span style="color:var(--red)">⚠️ ${esc(state.error)}</span>`;
      notice.style.cssText = 'padding:16px 24px;font-size:13px;';
      const msgList = document.getElementById('message-list');
      msgList.appendChild(notice);
    } else if (!state.streaming) {
      input.disabled = false;
      input.placeholder = 'Ask a question about your documents…';
      sendBtn.disabled = false;
    }
  }

  /* ══════════════════════════════════════════════════════════════════════════
     §7  APPLY STATE (after any API call that returns state)
     ══════════════════════════════════════════════════════════════════════════ */
  function applyState(s) {
    state.files    = s.files    || [];
    state.ready    = s.ready    || false;
    state.error    = s.error    || null;
    state.pipeline = s.pipeline || '';
    state.modes    = s.modes    || ['fast', 'balanced', 'thorough'];
    state.models   = s.models   || {};
    state.hasSamples = s.has_samples || false;

    hideBanner();
    renderModelChips();
    renderFiles();
  }

  /* ══════════════════════════════════════════════════════════════════════════
     §8  STREAMING CHAT
     ══════════════════════════════════════════════════════════════════════════ */

  async function sendChat(question) {
    if (state.streaming) return;
    if (!question.trim()) return;

    // Hide welcome card
    const wc = document.getElementById('welcome-card');
    if (wc) wc.classList.add('hidden');

    // Build history from last 8 turns
    const history = state.history.slice(-8);

    // Add user message to UI
    const userMsgId = 'msg-' + Date.now();
    addUserBubble(userMsgId, question);
    state.history.push({ role: 'user', content: question });

    // Create assistant bubble placeholder
    const asMsgId = 'msg-' + (Date.now() + 1);
    const { wrapper: asWrapper, content: asContent, footer: asFooter } =
      addAssistantBubble(asMsgId);

    // Set streaming state
    state.streaming = true;
    state.abortController = new AbortController();

    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-btn');
    const input   = document.getElementById('chat-input');
    sendBtn.style.display = 'none';
    stopBtn.style.display = 'flex';
    input.disabled = true;

    // Streaming accumulators
    let draftText = '';
    let sources = [];
    let terms = [];
    let traceData = {};
    let citationMap = {};
    let caret = document.createElement('span');
    caret.className = 'streaming-caret';
    caret.setAttribute('aria-hidden', 'true');
    asContent.appendChild(caret);

    const startTime = Date.now();

    try {
      const res = await request('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, history, mode: state.mode }),
        signal: state.abortController.signal,
      });

      if (!res.ok) {
        const errText = await res.text();
        asContent.innerHTML = `<span class="error-inline">Error: ${esc(errText)}</span>`;
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Process all complete newline-delimited JSON lines
        let nl;
        while ((nl = buffer.indexOf('\n')) !== -1) {
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          if (!line) continue;

          let event;
          try { event = JSON.parse(line); } catch { continue; }

          switch (event.type) {
            case 'token':
              draftText += event.text;
              renderDraft(asContent, draftText, caret);
              scrollToBottom();
              break;

            case 'reset':
              draftText = '';
              asContent.innerHTML = '';
              caret = document.createElement('span');
              caret.className = 'streaming-caret';
              caret.setAttribute('aria-hidden', 'true');
              asContent.appendChild(caret);
              break;

            case 'sources':
              sources = event.sources || [];
              terms   = event.terms   || [];
              // Show sources in right panel immediately
              renderSourcesPanel(sources, terms, null);
              // Switch to sources tab on mobile
              activateMobileTab('panel-sources');
              break;

            case 'trace':
              if (event.trace) {
                mergeTrace(traceData, event.trace);
                renderTracePanel(traceData);
              }
              break;

            case 'error':
              asContent.innerHTML = `<span class="error-inline">⚠️ ${esc(event.message)}</span>`;
              if (caret.parentNode) caret.remove();
              break;

            case 'done': {
              if (caret.parentNode) caret.remove();
              // Final answer
              const finalAnswer = event.answer || draftText;
              sources = event.sources || sources;
              terms   = event.terms   || terms;
              traceData = event.trace || traceData;
              draftText = finalAnswer;

              // Render final answer with citations
              const { html, citationMap: cm } = processCitations(
                renderMarkdown(finalAnswer), sources, {}
              );
              citationMap = cm;
              asContent.innerHTML = html;

              // Bind citation chips
              bindCitationChips(asContent, sources);

              // Footer
              const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
              const srcCount = sources.filter(s => s.used).length;
              asFooter.textContent = `${srcCount} source${srcCount !== 1 ? 's' : ''} · ${elapsed} s`;

              // Store in state
              const msgEntry = {
                id: asMsgId,
                role: 'assistant',
                content: finalAnswer,
                sources,
                terms,
                trace: traceData,
                timingMs: event.trace && event.trace.timings_ms ? event.trace.timings_ms : null,
                usedRetrieval: event.used_retrieval,
                el: asWrapper,
                citationMap,
              };
              state.messages.push({ id: userMsgId, role: 'user', content: question });
              state.messages.push(msgEntry);

              // Auto-select this message
              selectMessage(asMsgId);

              // Push to history
              state.history.push({ role: 'assistant', content: finalAnswer });

              // Update source + trace panels (already rendered, just ensure latest)
              renderSourcesPanel(sources, terms, traceData);
              break;
            }
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        // User stopped — finalize partial
        if (caret.parentNode) caret.remove();
        if (draftText) {
          const { html } = processCitations(renderMarkdown(draftText), sources, {});
          asContent.innerHTML = html + ' <em style="color:var(--text-muted);font-size:12px;">[stopped]</em>';
          bindCitationChips(asContent, sources);
        }
      } else if (err.network) {
        showBanner(`${err.message}. Start it (python server.py) or edit API_BASE in config.js.`);
        asContent.innerHTML = `<span class="error-inline">⚠️ ${esc(err.message)}</span>`;
        if (caret.parentNode) caret.remove();
      } else {
        asContent.innerHTML = `<span class="error-inline">⚠️ ${esc(err.message)}</span>`;
        if (caret.parentNode) caret.remove();
      }
    } finally {
      state.streaming = false;
      state.abortController = null;
      sendBtn.style.display = 'flex';
      stopBtn.style.display = 'none';
      input.disabled = false;
      input.focus();
    }
  }

  /** Render streamed text into the content el, keeping caret at the end */
  function renderDraft(el, text, caret) {
    // Temporarily remove caret, render text, re-append caret
    if (caret.parentNode) caret.remove();
    el.innerHTML = renderMarkdown(text);
    el.appendChild(caret);
  }

  function scrollToBottom() {
    const list = document.getElementById('message-list');
    list.scrollTop = list.scrollHeight;
  }

  /** Merge a partial trace event into the accumulated trace object */
  function mergeTrace(target, source) {
    for (const [k, v] of Object.entries(source)) {
      if (v !== null && typeof v === 'object' && !Array.isArray(v)
          && typeof target[k] === 'object' && target[k] !== null) {
        mergeTrace(target[k], v);
      } else {
        target[k] = v;
      }
    }
  }

  /* ══════════════════════════════════════════════════════════════════════════
     §9  MESSAGE BUBBLE BUILDERS
     ══════════════════════════════════════════════════════════════════════════ */
  function addUserBubble(id, text) {
    const list = document.getElementById('message-list');
    const wrapper = document.createElement('div');
    wrapper.className = 'msg-wrapper user';
    wrapper.id = id;
    wrapper.innerHTML = `
      <div class="msg-bubble">
        <div class="answer-content">${esc(text)}</div>
      </div>
    `;
    list.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
  }

  function addAssistantBubble(id) {
    const list = document.getElementById('message-list');
    const wrapper = document.createElement('div');
    wrapper.className = 'msg-wrapper assistant';
    wrapper.id = id;

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    const content = document.createElement('div');
    content.className = 'answer-content';
    bubble.appendChild(content);

    const footer = document.createElement('div');
    footer.className = 'bubble-footer';

    wrapper.appendChild(bubble);
    wrapper.appendChild(footer);
    list.appendChild(wrapper);

    // Click to select
    bubble.addEventListener('click', () => selectMessage(id));

    scrollToBottom();
    return { wrapper, content, footer };
  }

  /** Select an assistant message — highlight it and show its sources/trace */
  function selectMessage(id) {
    // Deselect previous
    if (state.selectedMsgId) {
      const prev = document.getElementById(state.selectedMsgId);
      if (prev) prev.classList.remove('selected');
    }
    state.selectedMsgId = id;
    const el = document.getElementById(id);
    if (el) el.classList.add('selected');

    const msg = state.messages.find(m => m.id === id && m.role === 'assistant');
    if (msg) {
      renderSourcesPanel(msg.sources || [], msg.terms || [], msg.trace || {});
      activateMobileTab('panel-sources');
    }
  }

  /* ══════════════════════════════════════════════════════════════════════════
     §10  SOURCES PANEL
     ══════════════════════════════════════════════════════════════════════════ */
  function renderSourcesPanel(sources, terms, trace) {
    const body = document.getElementById('sources-panel-body');
    const empty = document.getElementById('sources-empty');
    if (empty) empty.style.display = 'none';

    // Clear previous sources/trace but keep empty placeholder
    body.innerHTML = '';

    const hasRetrieval = sources && sources.length > 0;

    if (!hasRetrieval) {
      const notice = document.createElement('div');
      notice.className = 'no-retrieval';
      notice.innerHTML = `<span aria-hidden="true">💬</span> No sources: answered directly.`;
      body.appendChild(notice);
    } else {
      // ── Sources heading ──
      const heading = document.createElement('div');
      heading.className = 'sources-heading';
      heading.textContent = `Sources (${sources.length})`;
      body.appendChild(heading);

      sources.forEach((src, i) => {
        body.appendChild(createSourceCard(src, i + 1, terms));
      });
    }

    // ── Trace sections ──
    if (trace && Object.keys(trace).length > 0) {
      body.appendChild(buildTraceSection(trace));
    }
  }

  function renderTracePanel(trace) {
    // Find existing trace section or create it
    const body = document.getElementById('sources-panel-body');
    let traceSection = body.querySelector('.trace-main-section');
    if (!traceSection) {
      traceSection = buildTraceSection(trace);
      traceSection.classList.add('trace-main-section');
      body.appendChild(traceSection);
    } else {
      // Update in-place
      const updated = buildTraceSection(trace);
      updated.classList.add('trace-main-section');
      body.replaceChild(updated, traceSection);
    }
  }

  function createSourceCard(src, num, terms) {
    const card = document.createElement('div');
    card.className = 'source-card';
    card.id = `source-${src.id || num}`;
    card.setAttribute('data-src-id', src.id || '');

    const page  = src.page  ? ` · p. ${src.page}` : '';
    const score = src.score != null ? src.score.toFixed(3) : null;
    const denseR  = src.dense_rank  != null ? `dense #${src.dense_rank}`  : null;
    const sparseR = src.sparse_rank != null ? `sparse #${src.sparse_rank}` : null;

    const badgeUsed = src.used
      ? `<span class="badge badge-used">used</span>`
      : `<span class="badge badge-unused">not used</span>`;
    const badgeScore = score
      ? `<span class="badge badge-score">score ${score}</span>`
      : '';
    const badgeDense  = denseR  ? `<span class="badge badge-rank">${esc(denseR)}</span>`  : '';
    const badgeSparse = sparseR ? `<span class="badge badge-rank">${esc(sparseR)}</span>` : '';

    const excerptHighlighted = highlightTerms(src.text || '', terms);

    card.innerHTML = `
      <div class="source-card-header" role="button" tabindex="0" aria-expanded="true"
           aria-label="Source ${num}: ${esc(src.source)}">
        <div class="source-file-icon" aria-hidden="true">${fileIcon(src.source || '')}</div>
        <div class="source-file-info">
          <div class="source-file-name" title="${esc(src.source || '')}">[${num}] ${esc(src.source || '')}${esc(page)}</div>
          <div class="source-section-name">${esc(src.section || '')}</div>
        </div>
      </div>
      <div class="source-badges">
        ${badgeUsed}${badgeScore}${badgeDense}${badgeSparse}
      </div>
      <div class="source-excerpt collapsed">${excerptHighlighted}</div>
    `;

    // Collapse/expand excerpt on header click
    const header = card.querySelector('.source-card-header');
    const excerpt = card.querySelector('.source-excerpt');
    let expanded = false;
    const toggleExcerpt = () => {
      expanded = !expanded;
      excerpt.classList.toggle('collapsed', !expanded);
      header.setAttribute('aria-expanded', String(expanded));
    };
    header.addEventListener('click', toggleExcerpt);
    header.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleExcerpt(); }
    });

    return card;
  }

  /** Scroll to and highlight a source card */
  function highlightSourceCard(srcId) {
    // Remove previous highlight
    document.querySelectorAll('.source-card.highlighted').forEach(c =>
      c.classList.remove('highlighted'));

    const card = document.getElementById(`source-${srcId}`) ||
      document.querySelector(`[data-src-id="${CSS.escape(srcId)}"]`);
    if (!card) return;
    card.classList.add('highlighted');
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /** Bind citation chip click handlers inside a bubble content element */
  function bindCitationChips(el, sources) {
    el.querySelectorAll('.citation-chip').forEach(chip => {
      chip.addEventListener('click', e => {
        e.stopPropagation();
        const srcId = chip.dataset.srcId;
        if (!srcId) return;

        // In Student Portal: open the clean distraction-free citation modal
        if (state.portal === 'student') {
          const src = (sources || []).find(s => s.id === srcId);
          if (src) {
            showCitationModal(src, chip.textContent);
            return;
          }
        }

        // In Admin Studio: open and scroll the deep Sources & Trace inspector
        activateMobileTab('panel-sources');
        highlightSourceCard(srcId);
      });
    });
  }

  /* ══════════════════════════════════════════════════════════════════════════
     §11  TRACE PANEL
     ══════════════════════════════════════════════════════════════════════════ */
  function buildTraceSection(trace) {
    const section = document.createElement('div');
    section.className = 'trace-section';

    // ── Query Rewrite ──
    section.appendChild(buildCollapsibleTrace('Query Rewrite', buildQueryRewriteContent(trace)));

    // ── Retrieval ──
    section.appendChild(buildCollapsibleTrace('Retrieval', buildRetrievalContent(trace)));

    // ── Self-RAG Decisions ──
    section.appendChild(buildCollapsibleTrace('Self-RAG Decisions', buildSelfRAGContent(trace)));

    // ── Timing ──
    if (trace.timings_ms) {
      section.appendChild(buildCollapsibleTrace('Timing', buildTimingContent(trace.timings_ms)));
    }

    return section;
  }

  function buildCollapsibleTrace(title, contentEl) {
    const wrapper = document.createElement('div');
    const storageKey = `trace-open-${title}`;
    let isOpen = true;
    try { const s = localStorage.getItem(storageKey); if (s === 'false') isOpen = false; } catch (_) {}

    const toggle = document.createElement('button');
    toggle.className = 'trace-toggle' + (isOpen ? '' : ' collapsed');
    toggle.type = 'button';
    toggle.setAttribute('aria-expanded', String(isOpen));
    toggle.innerHTML = `<span>${esc(title)}</span><span class="caret" aria-hidden="true">▾</span>`;

    const content = document.createElement('div');
    content.className = 'trace-content' + (isOpen ? '' : ' collapsed');
    content.appendChild(contentEl);

    toggle.addEventListener('click', () => {
      isOpen = !isOpen;
      toggle.classList.toggle('collapsed', !isOpen);
      content.classList.toggle('collapsed', !isOpen);
      toggle.setAttribute('aria-expanded', String(isOpen));
      try { localStorage.setItem(storageKey, String(isOpen)); } catch (_) {}
    });

    wrapper.appendChild(toggle);
    wrapper.appendChild(content);
    return wrapper;
  }

  function traceRow(label, valueHtml) {
    const row = document.createElement('div');
    row.className = 'trace-row';
    row.innerHTML = `<span class="trace-label">${esc(label)}</span><span class="trace-value">${valueHtml}</span>`;
    return row;
  }

  function pill(text, cls = '') {
    return `<span class="trace-pill ${cls}">${esc(String(text))}</span>`;
  }

  function buildQueryRewriteContent(trace) {
    const div = document.createElement('div');
    if (!trace.original && !trace.query) {
      div.innerHTML = '<span class="waiting-text">Waiting…</span>';
      return div;
    }
    const original  = trace.original || '';
    const rewritten = trace.query    || '';
    const same = original.trim().toLowerCase() === rewritten.trim().toLowerCase();

    div.appendChild(traceRow('Original', esc(original)));
    div.appendChild(traceRow('Rewritten',
      same ? pill('unchanged') : esc(rewritten)));

    if (trace.queries && trace.queries.length > 1) {
      const qList = trace.queries.map(q => `<li>${esc(q)}</li>`).join('');
      div.appendChild(traceRow('Expanded', `<ul style="padding-left:16px;margin:0;font-size:11px;">${qList}</ul>`));
    }
    if (trace.hyde) {
      div.appendChild(traceRow('HyDE', `<em style="font-size:11px;color:var(--text-secondary);">${esc(trace.hyde.slice(0, 200))}…</em>`));
    }
    return div;
  }

  function buildRetrievalContent(trace) {
    const div = document.createElement('div');

    if (!trace.retrieval_mode) {
      div.innerHTML = '<span class="waiting-text">Waiting…</span>';
      return div;
    }

    const mode     = trace.retrieval_mode || '';
    const reranker = trace.reranker || null;
    const s        = trace.search   || {};

    div.appendChild(traceRow('Mode', pill(mode, 'accent')));
    div.appendChild(traceRow('Reranker',
      reranker ? pill(reranker) : pill('no reranker')));

    // Retrieval flow
    const flow = document.createElement('div');
    flow.className = 'retrieval-flow';

    const showDense  = mode !== 'sparse';
    const showSparse = mode !== 'dense';

    if (showDense) {
      flow.appendChild(flowStep('Dense path',
        s.dense_hits != null ? s.dense_hits : '?',
        `top-${s.dense_k || 30} per query · cosine/HNSW`, s.dense_hits != null));
    }
    if (showSparse) {
      flow.appendChild(flowStep('Sparse path',
        s.sparse_hits != null ? s.sparse_hits : '?',
        `top-${s.sparse_k || 30} per query · BM25${s.stemming ? ' + stemming' : ''}`, s.sparse_hits != null));
    }
    if (showDense && showSparse) {
      flow.appendChild(flowStep('Found by both',
        s.both != null ? s.both : '?', '', s.both != null));
      flow.appendChild(flowStep('RRF fusion',
        s.fused != null ? s.fused : '?',
        s.duplicates_removed != null ? `${s.duplicates_removed} duplicates removed` : '',
        s.fused != null));
    }
    flow.appendChild(flowStep('Candidates',
      s.to_reranker != null ? s.to_reranker : (trace.candidates != null ? trace.candidates : '?'),
      `(cap ${s.max_candidates || 50})`,
      s.to_reranker != null || trace.candidates != null));

    if (reranker) {
      const scored = s.to_reranker || 0;
      flow.appendChild(flowStep('Reranker',
        `scored ${scored} → top 5`, '', scored > 0));
    } else {
      flow.appendChild(flowStep('Reranker', 'off', '', true));
    }

    div.appendChild(flow);

    // Final chunks
    if (trace.retrieved && trace.retrieved.length > 0) {
      const ul = document.createElement('ul');
      ul.style.cssText = 'padding-left:16px;margin:6px 0 0;font-size:11px;color:var(--text-secondary);';
      trace.retrieved.forEach(id => {
        const rscore = trace.rerank_scores && trace.rerank_scores[id] != null
          ? trace.rerank_scores[id].toFixed(3) : null;
        const paths  = trace.paths && trace.paths[id] ? trace.paths[id] : {};
        const li = document.createElement('li');
        li.innerHTML = `${esc(id)}` +
          (rscore ? ` <span style="color:var(--amber)">↑${rscore}</span>` : '') +
          (paths.dense  != null ? ` dense #${paths.dense}`   : '') +
          (paths.sparse != null ? ` sparse #${paths.sparse}` : '');
        ul.appendChild(li);
      });
      const finalLabel = document.createElement('div');
      finalLabel.style.cssText = 'font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;margin-top:8px;';
      finalLabel.textContent = 'Final chunks:';
      div.appendChild(finalLabel);
      div.appendChild(ul);
    }

    return div;
  }

  function flowStep(label, count, detail, active) {
    const step = document.createElement('div');
    step.className = 'flow-step' + (active ? ' active' : '');
    step.innerHTML = `
      <span class="flow-dot" aria-hidden="true"></span>
      <span class="flow-label">${esc(label)}</span>
      <span class="flow-count">${esc(String(count))}</span>
      ${detail ? `<span class="flow-detail">${esc(detail)}</span>` : ''}
    `;
    return step;
  }

  function buildSelfRAGContent(trace) {
    const div = document.createElement('div');

    if (!trace.decision) {
      div.innerHTML = '<span class="waiting-text">Waiting…</span>';
      return div;
    }

    const d = trace.decision;
    div.appendChild(traceRow('Decision',
      d.retrieve
        ? pill('retrieve', 'green')
        : pill('answer directly', 'amber')));
    if (d.reason) {
      div.appendChild(traceRow('Reason', `<em style="font-size:11px;">${esc(d.reason)}</em>`));
    }

    // Grading
    if (trace.grading !== undefined) {
      div.appendChild(traceRow('Grading', pill(trace.grading || 'LLM', 'accent')));
    }
    if (trace.graded_out && trace.graded_out.length > 0) {
      const list = trace.graded_out.map(id => `<li>${esc(id)}</li>`).join('');
      div.appendChild(traceRow('Dropped',
        `<ul style="padding-left:14px;margin:0;font-size:11px;color:var(--red);">${list}</ul>`));
    }

    // Groundedness
    if (trace.groundedness) {
      const g = trace.groundedness;
      div.appendChild(traceRow('Grounded', g.grounded ? pill('✓ supported', 'green') : pill('unsupported', 'red')));
      if (g.unsupported && g.unsupported.length > 0) {
        const list = g.unsupported.map(s => `<li>${esc(s)}</li>`).join('');
        div.appendChild(traceRow('Unsupported',
          `<ul style="padding-left:14px;margin:0;font-size:11px;color:var(--amber);">${list}</ul>`));
      }
      if (trace.regenerated) {
        div.appendChild(traceRow('Regenerated', pill('yes', 'amber')));
      }
    } else if (!trace.grading) {
      const el = document.createElement('div');
      el.className = 'skipped-text';
      el.textContent = 'Skipped in this mode.';
      div.appendChild(el);
    }

    return div;
  }

  function buildTimingContent(timings) {
    const div = document.createElement('div');
    if (!timings) {
      div.innerHTML = '<span class="waiting-text">Waiting…</span>';
      return div;
    }

    const STAGES = [
      ['decide',   'Decide'],
      ['rewrite',  'Rewrite'],
      ['expand',   'Expand'],
      ['search',   'Search'],
      ['rerank',   'Rerank'],
      ['grade',    'Grade'],
      ['generate', 'Answer'],
      ['verify',   'Verify'],
    ];

    const total = timings.total || 1;
    const rows = document.createElement('div');
    rows.className = 'timing-bars';

    STAGES.forEach(([key, label]) => {
      const ms = timings[key];
      if (ms == null) return; // stage didn't run

      const pct = Math.min((ms / total) * 100, 100).toFixed(1);
      const row = document.createElement('div');
      row.className = 'timing-row';
      row.innerHTML = `
        <span class="timing-name">${esc(label)}</span>
        <div class="timing-bar-track" aria-hidden="true">
          <div class="timing-bar-fill" style="width:${pct}%"></div>
        </div>
        <span class="timing-ms">${ms} ms</span>
      `;
      rows.appendChild(row);
    });

    const totalRow = document.createElement('div');
    totalRow.className = 'timing-total';
    totalRow.textContent = `Total: ${total} ms`;
    rows.appendChild(totalRow);

    div.appendChild(rows);
    return div;
  }

  /* ══════════════════════════════════════════════════════════════════════════
     §12  MOBILE TAB SWITCHING
     ══════════════════════════════════════════════════════════════════════════ */
  function activateMobileTab(panelId) {
    // Only do something if we're actually in mobile view
    if (window.innerWidth >= 900) return;
    const tabBtns = document.querySelectorAll('.tab-btn');
    const panels  = document.querySelectorAll('.panel');

    panels.forEach(p => p.classList.remove('tab-active'));
    tabBtns.forEach(b => {
      b.classList.remove('active');
      b.setAttribute('aria-selected', 'false');
    });

    const panel = document.getElementById(panelId);
    if (panel) panel.classList.add('tab-active');

    const tabId = {
      'panel-files':   'tab-files',
      'panel-chat':    'tab-chat',
      'panel-sources': 'tab-sources',
    }[panelId];
    const btn = tabId ? document.getElementById(tabId) : null;
    if (btn) {
      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
    }
  }

  /* ══════════════════════════════════════════════════════════════════════════
     §13  EVENTS
     ══════════════════════════════════════════════════════════════════════════ */
  function initEvents() {
    // ── Retry button ──
    document.getElementById('retry-btn').addEventListener('click', loadState);

    // ── Files menu dropdown ──
    const menuBtn  = document.getElementById('files-menu-btn');
    const dropdown = document.getElementById('files-dropdown');
    menuBtn.addEventListener('click', e => {
      e.stopPropagation();
      const open = dropdown.classList.toggle('open');
      menuBtn.setAttribute('aria-expanded', String(open));
    });
    document.addEventListener('click', () => {
      dropdown.classList.remove('open');
      menuBtn.setAttribute('aria-expanded', 'false');
    });

    // ── Load samples ──
    document.getElementById('load-samples-btn').addEventListener('click', async () => {
      dropdown.classList.remove('open');
      try {
        const s = await apiLoadSamples();
        applyState(s);
        toast('Sample notes loaded!', 'success');
      } catch (err) {
        toast(err.message, 'error');
      }
    });

    // ── Remove all (with inline confirm) ──
    document.getElementById('remove-all-btn').addEventListener('click', () => {
      dropdown.classList.remove('open');
      document.getElementById('confirm-strip').classList.add('visible');
    });
    document.getElementById('confirm-cancel-btn').addEventListener('click', () => {
      document.getElementById('confirm-strip').classList.remove('visible');
    });
    document.getElementById('confirm-ok-btn').addEventListener('click', async () => {
      document.getElementById('confirm-strip').classList.remove('visible');
      try {
        const s = await apiDeleteAll();
        applyState(s);
        toast('All files removed.', 'success');
      } catch (err) {
        toast(err.message, 'error');
      }
    });

    // ── Docs section toggle ──
    const docsToggle = document.getElementById('docs-toggle');
    const docsBody   = document.getElementById('docs-body');
    docsToggle.addEventListener('click', () => {
      const open = docsToggle.getAttribute('aria-expanded') === 'true';
      docsToggle.setAttribute('aria-expanded', String(!open));
      docsToggle.classList.toggle('collapsed', open);
      docsBody.classList.toggle('collapsed', open);
    });

    // ── Drop zone ──
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
    });

    dropZone.addEventListener('dragover', e => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      handleFiles([...e.dataTransfer.files]);
    });

    fileInput.addEventListener('change', () => {
      handleFiles([...fileInput.files]);
      fileInput.value = '';
    });

    // ── Upload handler ──
    async function handleFiles(files) {
      const ALLOWED_TYPES = ['application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/markdown', 'text/plain', 'text/x-markdown'];
      const ALLOWED_EXT = ['.pdf', '.docx', '.md', '.txt'];
      const MAX_SIZE = 50 * 1024 * 1024;

      const valid = [];
      for (const f of files) {
        const ext = '.' + f.name.split('.').pop().toLowerCase();
        if (!ALLOWED_EXT.includes(ext)) {
          toast(`${f.name}: unsupported type (${ext})`, 'error');
          continue;
        }
        if (f.size > MAX_SIZE) {
          toast(`${f.name}: exceeds 50 MB limit`, 'error');
          continue;
        }
        valid.push(f);
      }
      if (valid.length === 0) return;

      // Show indexing rows
      const fileList = document.getElementById('file-list');
      document.getElementById('no-docs-yet').style.display = 'none';
      const placeholders = valid.map(f => {
        const row = createFileRow({ name: f.name, size: f.size }, true);
        fileList.appendChild(row);
        return row;
      });

      try {
        const result = await apiUpload(valid);

        // Remove placeholders
        placeholders.forEach(p => p.remove());

        // Show per-file results
        (result.results || []).forEach(r => {
          if (r.ok) {
            toast(`${r.name}: indexed (${r.chunks} chunks)`, 'success');
          } else {
            toast(`${r.name}: ${r.error}`, 'error');
          }
        });

        applyState(result);
      } catch (err) {
        placeholders.forEach(p => p.remove());
        if (err.network) {
          showBanner(`${err.message}. Start it (python server.py) or edit API_BASE in config.js.`);
        }
        toast(err.message, 'error');
        renderFiles();
      }
    }

    // ── Mode switch ──
    document.querySelectorAll('.mode-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        state.mode = btn.dataset.mode;
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        try { localStorage.setItem('chat_mode', state.mode); } catch (_) {}
      });
    });

    // ── Set initial active mode button ──
    document.querySelectorAll('.mode-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.mode === state.mode);
    });

    // ── Chat input ──
    const chatInput = document.getElementById('chat-input');
    const sendBtn   = document.getElementById('send-btn');
    const stopBtn   = document.getElementById('stop-btn');

    // Auto-grow textarea
    chatInput.addEventListener('input', () => {
      chatInput.style.height = 'auto';
      chatInput.style.height = Math.min(chatInput.scrollHeight, 180) + 'px';
    });

    chatInput.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        doSend();
      }
    });

    sendBtn.addEventListener('click', doSend);
    stopBtn.addEventListener('click', () => {
      if (state.abortController) state.abortController.abort();
    });

    function doSend() {
      const q = chatInput.value.trim();
      if (!q || state.streaming) return;
      chatInput.value = '';
      chatInput.style.height = 'auto';
      sendChat(q);
    }

    // ── Example questions ──
    document.querySelectorAll('.example-q').forEach(btn => {
      btn.addEventListener('click', () => {
        if (state.files.length === 0) return;
        const text = btn.textContent.trim().replace(/^[^\w]+/, '').trim();
        chatInput.value = text;
        chatInput.dispatchEvent(new Event('input'));
        chatInput.focus();
      });
    });

    // ── Mobile tabs ──
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        activateMobileTab(btn.dataset.panel);
      });
    });

    // ── Theme toggle (Light/White by default) ──
    const themeBtn = document.getElementById('theme-toggle-btn');
    if (themeBtn) {
      const savedTheme = localStorage.getItem('chat_theme') || 'light';
      if (savedTheme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        themeBtn.textContent = '☀️';
      } else {
        document.documentElement.removeAttribute('data-theme');
        themeBtn.textContent = '🌙';
      }
      themeBtn.addEventListener('click', () => {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        if (isDark) {
          document.documentElement.removeAttribute('data-theme');
          themeBtn.textContent = '🌙';
          localStorage.setItem('chat_theme', 'light');
        } else {
          document.documentElement.setAttribute('data-theme', 'dark');
          themeBtn.textContent = '☀️';
          localStorage.setItem('chat_theme', 'dark');
        }
      });
    }

    // ── User Dropdown & Portal Switching ──
    const userBtn = document.getElementById('user-btn');
    const userDropdown = document.getElementById('user-dropdown');
    if (userBtn && userDropdown) {
      userBtn.addEventListener('click', e => {
        e.stopPropagation();
        const open = userDropdown.classList.contains('open');
        userDropdown.classList.toggle('open', !open);
        userBtn.setAttribute('aria-expanded', String(!open));
      });

      document.addEventListener('click', e => {
        if (!userDropdown.contains(e.target) && !userBtn.contains(e.target)) {
          userDropdown.classList.remove('open');
          userBtn.setAttribute('aria-expanded', 'false');
        }
      });
    }

    const switchPortalBtn = document.getElementById('switch-portal-btn');
    if (switchPortalBtn) {
      switchPortalBtn.addEventListener('click', () => {
        if (userDropdown) userDropdown.classList.remove('open');
        const nextPortal = state.portal === 'student' ? 'admin' : 'student';
        setPortal(nextPortal);
        toast(`Switched to ${nextPortal === 'admin' ? 'Admin Studio' : 'Student Portal'}`, 'info');
      });
    }

    const openLoginBtn = document.getElementById('open-login-btn');
    if (openLoginBtn) {
      openLoginBtn.addEventListener('click', () => {
        if (userDropdown) userDropdown.classList.remove('open');
        showAuthModal();
      });
    }

    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
      logoutBtn.addEventListener('click', () => {
        if (userDropdown) userDropdown.classList.remove('open');
        state.token = null;
        state.user = {
          name: 'Guest User',
          email: 'guest@doctalk.com',
          role: 'student'
        };
        try {
          localStorage.removeItem('doctalk_token');
          localStorage.removeItem('doctalk_user');
        } catch (_) {}
        setPortal('student');
        toast('Signed out', 'info');
        showAuthModal();
      });
    }

    // ── Quick Demo Login Buttons ──
    const demoStudentBtn = document.getElementById('demo-student-btn');
    if (demoStudentBtn) {
      demoStudentBtn.addEventListener('click', async () => {
        try {
          const res = await apiLogin('student@doctalk.com', 'student123');
          state.token = res.token;
          state.user = res.user;
          try {
            localStorage.setItem('doctalk_token', res.token);
            localStorage.setItem('doctalk_user', JSON.stringify(res.user));
          } catch (_) {}
          setPortal('student');
          hideAuthModal();
          toast('Logged in as Alex Student (Student Portal)', 'success');
        } catch (err) {
          // If offline/mock server fallback
          state.user = { name: 'Alex Student', email: 'student@doctalk.com', role: 'student' };
          setPortal('student');
          hideAuthModal();
          toast('Entered Student Portal (Demo Mode)', 'success');
        }
      });
    }

    const demoAdminBtn = document.getElementById('demo-admin-btn');
    if (demoAdminBtn) {
      demoAdminBtn.addEventListener('click', async () => {
        try {
          const res = await apiLogin('admin@doctalk.com', 'admin123');
          state.token = res.token;
          state.user = res.user;
          try {
            localStorage.setItem('doctalk_token', res.token);
            localStorage.setItem('doctalk_user', JSON.stringify(res.user));
          } catch (_) {}
          setPortal('admin');
          hideAuthModal();
          toast('Logged in as Prof. Sarah (Admin Studio)', 'success');
        } catch (err) {
          state.user = { name: 'Prof. Sarah (Admin)', email: 'admin@doctalk.com', role: 'admin' };
          setPortal('admin');
          hideAuthModal();
          toast('Entered Admin Studio (Demo Mode)', 'success');
        }
      });
    }

    // ── Auth Modal Tabs ──
    const tabLogin = document.getElementById('tab-login');
    const tabRegister = document.getElementById('tab-register');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');

    if (tabLogin && tabRegister && loginForm && registerForm) {
      tabLogin.addEventListener('click', () => {
        tabLogin.classList.add('active');
        tabRegister.classList.remove('active');
        loginForm.classList.remove('hidden');
        registerForm.classList.add('hidden');
      });
      tabRegister.addEventListener('click', () => {
        tabRegister.classList.add('active');
        tabLogin.classList.remove('active');
        registerForm.classList.remove('hidden');
        loginForm.classList.add('hidden');
      });
    }

    // ── Login Form Submit ──
    if (loginForm) {
      loginForm.addEventListener('submit', async e => {
        e.preventDefault();
        const email = document.getElementById('login-email').value.trim();
        const password = document.getElementById('login-password').value;
        const errBox = document.getElementById('login-error');
        errBox.classList.add('hidden');
        try {
          const res = await apiLogin(email, password);
          state.token = res.token;
          state.user = res.user;
          try {
            localStorage.setItem('doctalk_token', res.token);
            localStorage.setItem('doctalk_user', JSON.stringify(res.user));
          } catch (_) {}
          setPortal(res.user.role || 'student');
          hideAuthModal();
          toast(`Welcome, ${res.user.name}!`, 'success');
        } catch (err) {
          errBox.textContent = err.message;
          errBox.classList.remove('hidden');
        }
      });
    }

    // ── Register Form Submit ──
    if (registerForm) {
      registerForm.addEventListener('submit', async e => {
        e.preventDefault();
        const name = document.getElementById('reg-name').value.trim();
        const email = document.getElementById('reg-email').value.trim();
        const password = document.getElementById('reg-password').value;
        const role = document.getElementById('reg-role').value;
        const errBox = document.getElementById('register-error');
        errBox.classList.add('hidden');
        try {
          const res = await apiRegister(name, email, password, role);
          state.token = res.token;
          state.user = res.user;
          try {
            localStorage.setItem('doctalk_token', res.token);
            localStorage.setItem('doctalk_user', JSON.stringify(res.user));
          } catch (_) {}
          setPortal(res.user.role || 'student');
          hideAuthModal();
          toast(`Account created! Welcome, ${res.user.name}`, 'success');
        } catch (err) {
          errBox.textContent = err.message;
          errBox.classList.remove('hidden');
        }
      });
    }

    // ── Modal Close Handlers ──
    const authCloseBtn = document.getElementById('auth-close-btn');
    if (authCloseBtn) authCloseBtn.addEventListener('click', hideAuthModal);

    const citationCloseBtn = document.getElementById('citation-close-btn');
    const citationDoneBtn = document.getElementById('citation-done-btn');
    if (citationCloseBtn) citationCloseBtn.addEventListener('click', hideCitationModal);
    if (citationDoneBtn) citationDoneBtn.addEventListener('click', hideCitationModal);

    // Initial Portal setup
    setPortal(state.portal || 'student');
  }

  /* ══════════════════════════════════════════════════════════════════════════
     §13B PORTAL & MODAL HELPERS
     ══════════════════════════════════════════════════════════════════════════ */
  function updateUserUI() {
    const user = state.user || { name: 'Guest', email: 'guest@doctalk.com', role: 'student' };
    const userName = document.getElementById('user-name');
    const userAvatar = document.getElementById('user-avatar');
    const dropName = document.getElementById('dropdown-user-name');
    const dropEmail = document.getElementById('dropdown-user-email');
    const dropRole = document.getElementById('dropdown-user-role-pill');

    if (userName) userName.textContent = user.name;
    if (dropName) dropName.textContent = user.name;
    if (dropEmail) dropEmail.textContent = user.email;

    const isAdmin = state.portal === 'admin';
    if (userAvatar) userAvatar.textContent = isAdmin ? '🛡️' : '🎓';

    if (dropRole) {
      dropRole.textContent = isAdmin ? 'Admin / Instructor' : 'Student';
      dropRole.className = `role-pill ${isAdmin ? 'admin' : 'student'}`;
    }
  }

  function setPortal(portalRole) {
    state.portal = portalRole;
    try { localStorage.setItem('doctalk_portal', portalRole); } catch (_) {}

    const appShell = document.getElementById('app-shell');
    if (appShell) {
      appShell.classList.remove('portal-student', 'portal-admin');
      appShell.classList.add(`portal-${portalRole}`);
    }

    const badge = document.getElementById('portal-badge');
    const switchBtn = document.getElementById('switch-portal-btn');
    const filesTitle = document.querySelector('#panel-files .panel-title');

    if (portalRole === 'student') {
      if (badge) {
        badge.textContent = '🎓 Student Portal';
        badge.className = 'portal-badge student';
      }
      if (switchBtn) switchBtn.textContent = '🛡️ Switch to Admin Studio';
      if (filesTitle) filesTitle.textContent = 'Course Notes';
    } else {
      if (badge) {
        badge.textContent = '🛡️ Admin Studio';
        badge.className = 'portal-badge admin';
      }
      if (switchBtn) switchBtn.textContent = '🎓 Switch to Student Portal';
      if (filesTitle) filesTitle.textContent = 'Document Manager';
    }

    updateUserUI();
  }

  function showAuthModal(msg) {
    const overlay = document.getElementById('auth-overlay');
    if (!overlay) return;
    const errBox = document.getElementById('login-error');
    if (errBox) {
      if (msg) {
        errBox.textContent = msg;
        errBox.classList.remove('hidden');
      } else {
        errBox.classList.add('hidden');
      }
    }
    overlay.classList.remove('hidden');
  }

  function hideAuthModal() {
    const overlay = document.getElementById('auth-overlay');
    if (overlay) overlay.classList.add('hidden');
  }

  function showCitationModal(src, label) {
    const overlay = document.getElementById('citation-overlay');
    if (!overlay) return;

    const badgeNum = document.getElementById('citation-badge-num');
    const docName = document.getElementById('citation-source-doc');
    const pageTag = document.getElementById('citation-source-page');
    const textEl = document.getElementById('citation-text');

    if (badgeNum) badgeNum.textContent = label || '[1]';
    if (docName) docName.textContent = `📄 ${src.source || 'Document'}`;
    if (pageTag) pageTag.textContent = src.page ? `Page ${src.page}` : (src.section || 'Excerpt');
    if (textEl) textEl.textContent = src.text || 'No text excerpt provided.';

    overlay.classList.remove('hidden');
  }

  function hideCitationModal() {
    const overlay = document.getElementById('citation-overlay');
    if (overlay) overlay.classList.add('hidden');
  }

  /* ══════════════════════════════════════════════════════════════════════════
     §14  INIT
     ══════════════════════════════════════════════════════════════════════════ */
  async function loadState() {
    try {
      const s = await apiGetState();
      applyState(s);
    } catch (err) {
      if (err.network) {
        showBanner(`${err.message}. Start it (python server.py) or edit API_BASE in config.js.`);
      } else {
        showBanner(err.message);
      }
      updateChatInputState();
    }
  }

  function init() {
    initEvents();
    loadState();
  }

  // Kick off when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

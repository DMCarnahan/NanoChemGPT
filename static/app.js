document.addEventListener('DOMContentLoaded', () => {
  // Spinner overlay for long-running requests
  const spinnerOverlay = document.createElement('div');
  spinnerOverlay.id = 'globalSpinnerOverlay';
  spinnerOverlay.style = `
    display: none;
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background: rgba(255,255,255,0.5);
    z-index: 9999;
    justify-content: center;
    align-items: center;
  `;
  const spinner = document.createElement('div');
  spinner.style = `
    width: 48px; height: 48px;
    border: 6px solid #ccc;
    border-top: 6px solid #1976d2;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  `;
  spinnerOverlay.appendChild(spinner);
  document.body.appendChild(spinnerOverlay);
  // Add spinner animation CSS
  const style = document.createElement('style');
  style.textContent = `@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`;
  document.head.appendChild(style);
  const $ = (id) => document.getElementById(id);

  // Elements
  const askBtn = $('askBtn');
  const parseBtn = $('parseBtn');
  const uploadBtn = $('uploadBtn');
  const fileInput = $('fileInput');
  const qInput = $('question');
  const askMsg = $('askMsg');
  const answerPre = $('answerPre');
  const rationalePre = $('rationalePre');
  const uplList = $('uplList');
  const uplMsg = $('uplMsg');
  const historyBtn = $('historyBtn');
  const historyList = $('historyList');
  const modeRobot = $('modeRobot');
  const modeReason = $('modeReason');
  const saveTxtBtn = $('saveTxtBtn');
  const builtinDrop = $('builtinDrop');
  const builtinFile = $('builtinFile');
  const builtinMsg = $('builtinMsg');

  let mode = 'robot';

  // Mode toggle (accessibility: ARIA attributes, keyboard)
  modeRobot?.addEventListener('click', () => {
    mode = 'robot';
    modeRobot.setAttribute('aria-checked', 'true');
    modeReason?.setAttribute('aria-checked', 'false');
    modeRobot.classList.add('active');
    modeReason?.classList.remove('active');
  });
  modeReason?.addEventListener('click', () => {
  // Keyboard accessibility for mode toggles
  [modeRobot, modeReason].forEach(el => {
    el?.setAttribute('tabindex', '0');
    el?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') el.click();
    });
  });
    mode = 'reasoning';
    modeReason.setAttribute('aria-checked', 'true');
    modeRobot?.setAttribute('aria-checked', 'false');
    modeReason.classList.add('active');
    modeRobot?.classList.remove('active');
  });

  /**
   * Reads the CSRF token from meta tag or cookie.
   * @returns {string|undefined}
   */
  function readCsrfToken() {
    return document.querySelector('meta[name="csrf-token"]')?.content ||
      (document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/)?.[1]);
  }

  // Small helper to sandbox optional UI so it can't crash the flow
  function safe(fn){ try { fn(); } catch(e){ console.warn('Optional UI failed:', e); } }

  // Ask button
  /**
   * Handles the Ask button click: sends question to backend and updates UI.
   */
  // Ask button
  askBtn?.addEventListener('click', async () => {
    const question = qInput?.value.trim();
    if (!question) return;

  askBtn.disabled = true;
  askMsg.classList.remove('hidden');
  askMsg.textContent = 'Asking…';
  spinner.style.display = 'block';

  try {
      const headers = { 'Content-Type': 'application/json' };
      const csrf = readCsrfToken();
      if (csrf) headers['X-CSRFToken'] = csrf;

      const res = await fetch('/ask', {
        method: 'POST',
        headers,
        body: JSON.stringify({ question, mode })
      });

      const raw = await res.text();
      let data;
      try { data = JSON.parse(raw); } catch { data = { answer: raw }; }

      answerPre.textContent = data.answer ?? '(no answer)';
      rationalePre.textContent = data.rationale ?? '';
      renderRefsFromData(data);
      if (!res.ok) {
        askMsg.textContent = `Error ${res.status}: ${data.error || raw}`;
      } else if (data.answer && data.answer.toLowerCase().includes('error')) {
        askMsg.textContent = `Backend error: ${data.answer}`;
      } else {
        askMsg.textContent = 'Done.';
      }
    } catch (err) {
      console.error(err);
      askMsg.textContent = `Error: ${err.message || err}`;
    } finally {
      askBtn.disabled = false;
      spinner.style.display = 'none';
    }
  });

  // Parse button (Convert to JSON + Download)
  /**
   * Handles the Parse button click: converts answer to JSON and triggers download.
   */
  // Parse button (Convert to JSON + Download)
  parseBtn?.addEventListener('click', async () => {
    const text = answerPre?.textContent || '';
    if (!text) return;

    parseBtn.disabled = true;
    const oldLabel = parseBtn.textContent;
    parseBtn.textContent = 'Converting…';

  try {
      const headers = { 'Content-Type': 'application/json' };
      const csrf = readCsrfToken();
      if (csrf) headers['X-CSRFToken'] = csrf;

      const res = await fetch('/parse', {
        method: 'POST',
        headers,
        body: JSON.stringify({ text })
      });

      // Try to read JSON; if not JSON, throw
      const data = await res.json();
      if (!data || !data.ok || !data.data) {
        parseBtn.textContent = `Error: ${data?.error || 'Parse failed'}`;
        setTimeout(()=>{ parseBtn.textContent = oldLabel || 'Parse'; }, 2000);
        return;
      }

      const pretty = JSON.stringify(data.data, null, 2);

      // ---- DOWNLOAD ONLY (no preview) ----
      const stamp = new Date();
      const pad = (n)=> String(n).padStart(2,'0');
      const fname = `answer-${stamp.getFullYear()}${pad(stamp.getMonth()+1)}${pad(stamp.getDate())}-${pad(stamp.getHours())}${pad(stamp.getMinutes())}${pad(stamp.getSeconds())}.json`;

      const blob = new Blob([pretty], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fname;
      document.body.appendChild(a); // Required for Firefox
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      // ------------------------------------
    } catch (err) {
      console.error(err);
      parseBtn.textContent = `Error: ${err.message || err}`;
      setTimeout(()=>{ parseBtn.textContent = oldLabel || 'Parse'; }, 2000);
      return;
    } finally {
      parseBtn.disabled = false;
      parseBtn.textContent = oldLabel || 'Parse';
    }
  });

  /**
   * Renders references from data object into the UI.
   * @param {object} data
   */
  
  /**
   * Renders references from data object into the UI.
   * Accepts multiple shapes, prefers ACS block, falls back to structured list.
   * @param {object} data
   */
  function renderRefsFromData(data) {
    const refsSection = document.getElementById('refsSection');
    const refsList    = document.getElementById('refsList');      // candidates list (debug)
    if (!refsSection) return;

    // Accept multiple shapes
    const block  = (data?.reference_block || data?.references_block || '').trim();
    const arrRaw = (data?.references || data?.refs || data?.citations || data?.used_refs) || null;
    const used   = Array.isArray(data?.used_ref_indexes)
      ? data.used_ref_indexes.map(Number).filter(Number.isFinite)
      : [];

    // Debug: what did backend return?
    try {
      console.debug('[refs] blockLen=', block.length,
                    'used=', used,
                    'candidates=', Array.isArray(arrRaw) ? arrRaw.length : 0);
    } catch {}

    // Reset UI
    if (refsList) refsList.innerHTML = '';
    refsSection.querySelector('.refs-pre')?.remove();
    refsSection.classList.add('hidden');

    // 1) Prefer the used-only ACS block from backend
    if (block) {
      const pre = document.createElement('pre');
      pre.className = 'refs-pre';
      pre.textContent = block;
      refsSection.appendChild(pre);
      refsSection.classList.remove('hidden');
      try { window.initRefsToggle?.({ btnSelector: '#refsToggleBtn', panelSelector: '#refsSection' }); } catch {}
      return; // short-circuit so we do NOT render candidates
    }

    // 2) Fallback: structured candidates (optionally filter to "used" if indexes present)
    if (Array.isArray(arrRaw) && arrRaw.length && refsList) {
      const items = used.length
        ? arrRaw.map((r, i) => ({ r, i: i + 1 }))
              .filter(x => used.includes(x.i))
              .map(x => x.r)
        : arrRaw.slice(0, 6); // cap to top-6 to keep UI tidy when no block

      if (items.length) {
        items.forEach((r, idx) => {
          const li = document.createElement('li');

          if (typeof r === 'string') {
            li.textContent = r;
            refsList.appendChild(li);
            return;
          }

          const title   = r.title || r.citation || r.name || r.label || `Reference ${idx+1}`;
          const authors = Array.isArray(r.authors) ? r.authors.join(', ') : (r.authors || '');
          const journal = (r.biblio?.journal) || r.journal || r.source || '';
          const year    = r.year || r.biblio?.year || '';
          const doi     = r.doi ? String(r.doi).replace(/^https?:\/\/doi\.org\//, '') : '';
          const url     = r.url || (doi ? `https://doi.org/${doi}` : '');

          const strong = document.createElement('strong');
          strong.textContent = title;
          li.appendChild(strong);

          const metaBits = [authors, journal, year].filter(Boolean);
          if (metaBits.length) {
            const small = document.createElement('small');
            small.textContent = ' — ' + metaBits.join(', ');
            li.appendChild(small);
          }

          if (url) {
            li.appendChild(document.createTextNode(' '));
            const a = document.createElement('a');
            a.href = url; a.target = '_blank'; a.rel = 'noopener';
            a.textContent = '(link)';
            li.appendChild(a);
          }

          refsList.appendChild(li);
        });

        refsSection.classList.remove('hidden');
        try { window.initRefsToggle?.({ btnSelector: '#refsToggleBtn', panelSelector: '#refsSection' }); } catch {}
        return;
      }
    }

    // 3) Nothing to show
    refsSection.classList.add('hidden');
  }



  // Upload button
  /**
   * Handles the Upload button click: uploads a file to the backend.
   */
  // Upload button
  uploadBtn?.addEventListener('click', async () => {
    const file = fileInput?.files?.[0];
    if (!file) return;
    uploadBtn.disabled = true;
    uplMsg.textContent = 'Uploading…';
    uplMsg.classList.remove('hidden');

  try {
      const headers = {};
      const csrf = readCsrfToken();
      if (csrf) headers['X-CSRFToken'] = csrf;

      const fd = new FormData();
      fd.append('file', file);

      const res = await fetch('/upload', {
        method: 'POST',
        headers,
        body: fd
      });

      const json = await res.json();
      if (json.ok) {
        uplMsg.textContent = 'Uploaded OK';
        if (json.filename) {
          const li = document.createElement('li');
          li.textContent = json.filename;
          uplList?.appendChild(li);
        }
      } else {
        uplMsg.textContent = `Upload error: ${json.error || 'unknown'}`;
      }

    } catch (err) {
      console.error(err);
      uplMsg.textContent = `Upload failed: ${err.message || err}`;
    } finally {
      uploadBtn.disabled = false;
    }
  });

  // History button
  /**
   * Handles the History button click: loads and displays previous Q&A.
   */
  // History button
  historyBtn?.addEventListener('click', async () => {
  try {
      const res = await fetch('/api/history');
      if (!res.ok) {
        historyList.innerHTML = `<li>Error loading history: ${res.status}</li>`;
        return;
      }
      const data = await res.json();
      const items = data.items || [];
      historyList.innerHTML = items.map(r =>
        `<li><a href="#" data-id="${r._id}">${r.question || '(no question)'}</a></li>`
      ).join('');

      // Add click handler for each link
      historyList.querySelectorAll('a[data-id]').forEach(a => {
        a.addEventListener('click', async (e) => {
          e.preventDefault();
          const id = a.getAttribute('data-id');
          if (!id) return;
          try {
            const res = await fetch(`/api/history/${id}`);
            if (!res.ok) {
              answerPre.textContent = `Error loading answer: ${res.status}`;
              rationalePre.textContent = '';
              return;
            }
            const data = await res.json();
            answerPre.textContent = data.answer ?? '(no answer)';
            rationalePre.textContent = data.rationale ?? '';
            renderRefsFromData(data);
          } catch (err) {
            answerPre.textContent = `Error: ${err.message || err}`;
            rationalePre.textContent = '';
            console.error(err);
          }
        });
      });
    } catch (err) {
      historyList.innerHTML = `<li>Error: ${err.message || err}</li>`;
      console.error(err);
    }
  });

  // Save as TXT button (Export answer)
  /**
   * Handles the Save as TXT button click: downloads the answer as a text file.
   */
  // Save as TXT button (Export answer)
  saveTxtBtn?.addEventListener('click', () => {
    const text = answerPre?.textContent || '';
    if (!text) return;

    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'answer.txt';
    document.body.appendChild(a); // Required for Firefox
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  // Built-in file upload
  // Built-in file upload (drag & drop, keyboard, click)
  builtinDrop?.addEventListener('click', () => builtinFile?.click());
  builtinDrop?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') builtinFile?.click();
  });
  builtinDrop?.addEventListener('dragover', (e) => {
    e.preventDefault();
    builtinDrop.classList.add('dragover');
  });
  builtinDrop?.addEventListener('dragleave', () => {
    builtinDrop.classList.remove('dragover');
  });
  builtinDrop?.addEventListener('drop', (e) => {
    e.preventDefault();
    builtinDrop.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      builtinFile.files = e.dataTransfer.files;
      uploadBuiltinFiles(e.dataTransfer.files);
    }
  });
  builtinFile?.addEventListener('change', () => {
    if (builtinFile.files.length) uploadBuiltinFiles(builtinFile.files);
  });

  /**
   * Uploads built-in files to the backend.
   * @param {FileList|Array} files
   */
  /**
   * Uploads built-in files to the backend.
   * @param {FileList|Array} files
   */
  async function uploadBuiltinFiles(files) {
    if (!files.length) return;
    builtinMsg.textContent = 'Uploading…';
    builtinMsg.classList.remove('hidden');
    try {
      const fd = new FormData();
      for (const file of files) fd.append('file', file);

      // Add CSRF token to headers
      const headers = {};
      const csrf = readCsrfToken();
      if (csrf) headers['X-CSRFToken'] = csrf;

      const res = await fetch('/upload_builtin', {
        method: 'POST',
        headers,
        body: fd
      });
      const json = await res.json();
      builtinMsg.textContent = json.ok ? 'Uploaded OK' : 'Error: ' + (json.error || 'unknown');
    } catch (err) {
  builtinMsg.textContent = `Upload failed: ${err.message || err}`;
  console.error(err);
    }
  }

  // Auto-load history on page load
  if (historyBtn) historyBtn.click();
});

function renderRefsFromData(data) {
  try {
    if (!data || typeof data !== 'object') {
      console.warn('renderRefsFromData: expected object, got', data);
      return;
    }
    const refsSection = document.getElementById('refsSection');
    const refsList    = document.getElementById('refsList');
    if (!refsSection) return;

    const block  = (data.reference_block || data.references_block || '').trim?.() || '';
    const arrRaw = (data.references || data.refs || data.citations || data.used_refs) || null;
    const used   = Array.isArray(data.used_ref_indexes) ? data.used_ref_indexes.map(Number).filter(Number.isFinite) : [];

    try {
      console.debug('[refs] blockLen=', block.length, 'used=', used, 'candidates=', Array.isArray(arrRaw) ? arrRaw.length : 0);
    } catch {}

    // Reset UI
    if (refsList) refsList.innerHTML = '';
    const oldPre = refsSection.querySelector('.refs-pre');
    if (oldPre) oldPre.remove();
    refsSection.classList.add('hidden');

    // 1) Prefer used-only ACS block
    if (block) {
      const pre = document.createElement('pre');
      pre.className = 'refs-pre';
      pre.textContent = block;
      refsSection.appendChild(pre);
      refsSection.classList.remove('hidden');
      try { window.initRefsToggle?.({ btnSelector: '#refsToggleBtn', panelSelector: '#refsSection' }); } catch {}
      return;
    }

    // 2) Fallback: structured candidates (optionally filtered by used)
    if (Array.isArray(arrRaw) && arrRaw.length && refsList) {
      const items = used.length
        ? arrRaw.map((r, i) => ({ r, i: i + 1 })).filter(x => used.includes(x.i)).map(x => x.r)
        : arrRaw.slice(0, 6); // cap to keep tidy when no block

      if (items.length) {
        items.forEach((r, idx) => {
          const li = document.createElement('li');

          if (typeof r === 'string') {
            li.textContent = r;
            refsList.appendChild(li);
            return;
          }

          const title   = r.title || r.citation || r.name || r.label || `Reference ${idx+1}`;
          const authors = Array.isArray(r.authors) ? r.authors.join(', ') : (r.authors || '');
          const journal = (r.biblio && r.biblio.journal) || r.journal || r.source || '';
          const year    = r.year || (r.biblio && r.biblio.year) || '';
          const doi     = r.doi ? String(r.doi).replace(/^https?:\/\/doi\.org\//, '') : '';
          const url     = r.url || (doi ? `https://doi.org/${doi}` : '');

          const strong = document.createElement('strong');
          strong.textContent = title;
          li.appendChild(strong);

          const metaBits = [authors, journal, year].filter(Boolean);
          if (metaBits.length) {
            const small = document.createElement('small');
            small.textContent = ' — ' + metaBits.join(', ');
            li.appendChild(small);
          }

          if (url) {
            li.appendChild(document.createTextNode(' '));
            const a = document.createElement('a');
            a.href = url; a.target = '_blank'; a.rel = 'noopener';
            a.textContent = '(link)';
            li.appendChild(a);
          }

          refsList.appendChild(li);
        });

        refsSection.classList.remove('hidden');
        try { window.initRefsToggle?.({ btnSelector: '#refsToggleBtn', panelSelector: '#refsSection' }); } catch {}
        return;
      }
    }

    // 3) Nothing to show
    refsSection.classList.add('hidden');
  } catch (e) {
    try { console.error('renderRefsFromData error:', e); } catch {}
  }
}

function renderRefsFromData(data) {
  try {
    if (!data || typeof data !== 'object') {
      console.warn('renderRefsFromData: expected object, got', data);
      return;
    }
    const refsSection = document.getElementById('refsSection');
    const refsList    = document.getElementById('refsList');
    const candPanel   = document.getElementById('candPanel'); // optional <details>
    if (!refsSection) return;

    const block  = (data.reference_block || data.references_block || '').trim?.() || '';
    const arrRaw = (data.references || data.refs || data.citations || data.used_refs) || null;
    const used   = Array.isArray(data.used_ref_indexes) ? data.used_ref_indexes.map(Number).filter(Number.isFinite) : [];

    try {
      console.debug('[refs] blockLen=', block.length, 'used=', used, 'candidates=', Array.isArray(arrRaw) ? arrRaw.length : 0);
    } catch {}

    // Reset UI
    if (refsList) refsList.innerHTML = '';
    const oldPre = refsSection.querySelector('.refs-pre');
    if (oldPre) oldPre.remove();
    refsSection.classList.add('hidden');
    if (candPanel) candPanel.classList.add('hidden');

    // If we have a block, render it first
    if (block) {
      const pre = document.createElement('pre');
      pre.className = 'refs-pre';
      pre.textContent = block;
      refsSection.appendChild(pre);
      refsSection.classList.remove('hidden');
    }

    // Populate candidates (even if block exists), keep panel collapsed by default
    if (Array.isArray(arrRaw) && arrRaw.length && refsList) {
      const items = used.length
        ? arrRaw.map((r, i) => ({ r, i: i + 1 })).filter(x => used.includes(x.i)).map(x => x.r)
        : arrRaw.slice(0, 8);

      if (items.length) {
        items.forEach((r, idx) => {
          const li = document.createElement('li');

          if (typeof r === 'string') {
            li.textContent = r;
            refsList.appendChild(li);
            return;
          }

          const title   = r.title || r.citation || r.name || r.label || `Reference ${idx+1}`;
          const authors = Array.isArray(r.authors) ? r.authors.join(', ') : (r.authors || '');
          const journal = (r.biblio && r.biblio.journal) || r.journal || r.source || '';
          const year    = r.year || (r.biblio && r.biblio.year) || '';
          const doi     = r.doi ? String(r.doi).replace(/^https?:\/\/doi\.org\//, '') : '';
          const url     = r.url || (doi ? `https://doi.org/${doi}` : '');

          const strong = document.createElement('strong');
          strong.textContent = title;
          li.appendChild(strong);

          const metaBits = [authors, journal, year].filter(Boolean);
          if (metaBits.length) {
            const small = document.createElement('small');
            small.textContent = ' — ' + metaBits.join(', ');
            li.appendChild(small);
          }

          if (url) {
            li.appendChild(document.createTextNode(' '));
            const a = document.createElement('a');
            a.href = url; a.target = '_blank'; a.rel = 'noopener';
            a.textContent = '(link)';
            li.appendChild(a);
          }

          refsList.appendChild(li);
        });

        if (candPanel) candPanel.classList.remove('hidden');
        refsSection.classList.remove('hidden');
      }
    }

    // If neither block nor candidates, keep hidden
    try { window.initRefsToggle?.({ btnSelector: '#refsToggleBtn', panelSelector: '#refsSection' }); } catch {}
  } catch (e) {
    try { console.error('renderRefsFromData error:', e); } catch {}
  }
}
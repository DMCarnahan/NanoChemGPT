document.addEventListener('DOMContentLoaded', () => {
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
  const refsSection = $('refsSection');
  const refsList = $('refsList');
  const jsonBlock = $('jsonPre');
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

  // Mode toggle
  modeRobot?.addEventListener('click', () => {
    mode = 'robot';
    modeRobot.setAttribute('aria-checked', 'true');
    modeReason?.setAttribute('aria-checked', 'false');
    modeRobot.classList.add('active');
    modeReason?.classList.remove('active');
  });
  modeReason?.addEventListener('click', () => {
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
  askBtn?.addEventListener('click', async () => {
    const question = qInput?.value.trim();
    if (!question) return;

    askBtn.disabled = true;
    askMsg.classList.remove('hidden');
    askMsg.textContent = 'Asking…';

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
askMsg.textContent = res.ok ? 'Done.' : `Error ${res.status}`;
    } catch (err) {
      console.error(err);
      askMsg.textContent = 'Error. Check console.';
    } finally {
      askBtn.disabled = false;
    }
  });

  // Parse button (Convert to JSON + Download)
  /**
   * Handles the Parse button click: converts answer to JSON and triggers download.
   */
  parseBtn?.addEventListener('click', async () => {
  const text = answerPre?.textContent || '';
  if (!text) return;

  parseBtn.disabled = true;
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

    const data = await res.json();
    if (!data || !data.ok || !data.data) {
      throw new Error(data?.error || 'Parse failed');
    }

    const pretty = JSON.stringify(data.data, null, 2);

    if (jsonBlock) jsonBlock.textContent = pretty;
        document.getElementById('jsonBlock')?.classList.remove('hidden');
      } catch (err) {
        console.error(err);
        parseBtn.textContent = 'Error';
      } finally {
        parseBtn.disabled = false;
        parseBtn.textContent = 'Parse';
      }
    });

  /**
   * Renders references from data object into the UI.
   * @param {object} data
   */
  
  function renderRefsFromData(data) {
    // Accept multiple shapes: references/ref arrays, citations, used_refs; and blocks under reference_block/references_block
    if (!refsSection) return;

    const block = (data && (data.reference_block || data.references_block)) || '';
    const arrRaw = (data && (data.references || data.refs || data.citations || data.used_refs)) || null;

    // Used indexes (1-based), e.g., [1,2,5]
    const used = Array.isArray(data?.used_ref_indexes) ? data.used_ref_indexes.map(Number).filter(n => Number.isFinite(n)) : [];

    // Clear previous content
    refsList && (refsList.innerHTML = '');
    refsSection?.querySelector('.refs-pre')?.remove();

    // Prefer structured refs if present
    if (Array.isArray(arrRaw) && arrRaw.length && refsList) {
      const items = used.length
        ? arrRaw.map((r, i) => ({ r, i: i + 1 })).filter(x => used.includes(x.i)).map(x => x.r)
        : arrRaw;

      if (items.length) {
        items.forEach((r, idx) => {
          const li = document.createElement('li');

          // String reference
          if (typeof r === 'string') {
            li.textContent = r;
            refsList.appendChild(li);
            return;
          }

          const title   = r.title || r.citation || r.name || r.label || `Reference ${idx+1}`;
          const authors = Array.isArray(r.authors) ? r.authors.join(', ') : (r.authors || '');
          const journal = (r.biblio && r.biblio.journal) || r.journal || r.source || '';
          const year    = r.year || (r.biblio && r.biblio.year) || '';
          const doiUrl  = r.doi ? String(r.doi).replace(/^https?:\/\/doi\.org\//, '') : '';
          const url     = r.url || (doiUrl ? `https://doi.org/${doiUrl}` : '');

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
        // Initialize optional toggle if available
        safe(() => window.initRefsToggle?.({ btnSelector: '#refsToggleBtn', panelSelector: '#refsSection' }));
        return;
      }
    }

    // Fallback: preformatted block
    if (typeof block === 'string' && block.trim()) {
      refsList.innerHTML = '';
      const pre = document.createElement('pre');
      pre.className = 'refs-pre';
      pre.textContent = block.trim();
      refsSection.appendChild(pre);
      refsSection.classList.remove('hidden');
      safe(() => window.initRefsToggle?.({ btnSelector: '#refsToggleBtn', panelSelector: '#refsSection' }));
      return;
    }

    // Nothing to show
    refsSection.classList.add('hidden');
  }

  // Upload button
  /**
   * Handles the Upload button click: uploads a file to the backend.
   */
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
        uplMsg.textContent = 'Upload error: ' + (json.error || 'unknown');
      }

    } catch (err) {
      console.error(err);
      uplMsg.textContent = 'Upload failed';
    } finally {
      uploadBtn.disabled = false;
    }
  });

  // History button
  /**
   * Handles the History button click: loads and displays previous Q&A.
   */
  historyBtn?.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/history');
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
            const data = await res.json();
            answerPre.textContent = data.answer ?? '(no answer)';
            rationalePre.textContent = data.rationale ?? '';
            renderRefsFromData(data);
          } catch (err) {
            console.error(err);
          }
        });
      });
    } catch (err) {
      console.error(err);
    }
  });

  // Save as TXT button (Export answer)
  /**
   * Handles the Save as TXT button click: downloads the answer as a text file.
   */
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
  builtinDrop?.addEventListener('click', () => builtinFile?.click());
  builtinDrop?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') builtinFile?.click();
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
      builtinMsg.textContent = 'Upload failed';
      console.error(err);
    }
  }

  // Auto-load history on page load
if (historyBtn) historyBtn.click()});
document.addEventListener('DOMContentLoaded', () => {
  const $ = (id) => document.getElementById(id);

  // --- Elements (all optional-chained) ---
  const askBtn        = $('askBtn');
  const parseBtn      = $('parseBtn');
  const uploadBtn     = $('uploadBtn');
  const fileInput     = $('fileInput');
  const qInput        = $('question');
  const askMsg        = $('askMsg');
  const answerPre     = $('answerPre');
  const rationalePre  = $('rationalePre');
  const refsSection   = $('refsSection');
  const refsList      = $('refsList');
  const jsonPre       = $('jsonPre');
  const historyBtn    = $('historyBtn');
  const historyList   = $('historyList');
  const builtinBtn    = $('builtinBtn');
  const builtinFile   = $('builtinFile');
  const builtinMsg    = $('builtinMsg');

  const modeReason    = $('modeReason');
  const modeRobot     = $('modeRobot');
  const convertBtn    = $('convertBtn');

  // -------- Utilities --------
  function escapeHtml(s) {
    if (s == null) return '';
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;')
      .replace(/\//g, '&#x2F;');
  }

  function readCsrfToken() {
    const m = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/);
    if (!m) return null;
    try {
      return decodeURIComponent(m[1]);
    } catch {
      return m[1];
    }
  }

  function currentMode() {
    // If you have explicit radio/toggle controls, detect them here.
    // Fallback: "reason" unless a #modeRobot exists and is active.
    const reasonActive = !!modeReason && modeReason.getAttribute('aria-checked') === 'true' || modeReason?.classList?.contains('active');
    const robotActive  = !!modeRobot && modeRobot.getAttribute('aria-checked') === 'true'  || modeRobot?.classList?.contains('active');
    return robotActive ? 'robot' : 'reason';
  }

  function renderRefsFromData(data) {
    // Supports either `data.references_block` (preformatted string)
    // OR an array `data.references` of {title, url, ...}
    if (!refsSection) return;

    const block = data?.references_block;
    const arr   = data?.references;

    // Clear old
    if (refsList) refsList.innerHTML = '';

    if (block && typeof block === 'string') {
      // Render block safely inside a <pre>, but as list if it starts with "1. "
      const lines = block.split(/\r?\n/).filter(Boolean);
      if (refsList && lines.length) {
        lines.forEach(line => {
          const li = document.createElement('li');
          li.textContent = line.replace(/^\s*\d+\.\s*/, '');
          refsList.appendChild(li);
        });
        refsSection?.classList?.remove('hidden');
      } else {
        refsSection?.classList?.add('hidden');
      }
      return;
    }

    if (Array.isArray(arr) && arr.length && refsList) {
      arr.forEach(r => {
        const li = document.createElement('li');
        // Prefer title + link if available
        if (r?.url) {
          const a = document.createElement('a');
          a.href = r.url;
          a.textContent = r.title ? r.title : (r.url);
          a.target = '_blank';
          a.rel = 'noopener noreferrer';
          li.appendChild(a);
          if (r?.meta) {
            const small = document.createElement('small');
            small.textContent = ' ' + r.meta;
            li.appendChild(small);
          }
        } else {
          li.textContent = r?.title ? r.title : JSON.stringify(r);
        }
        refsList.appendChild(li);
      });
      refsSection?.classList?.remove('hidden');
      return;
    }

    // Nothing to show
    refsSection?.classList?.add('hidden');
  }

  function download(filename, content, type='application/octet-stream') {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  async function fetchJSON(url, options = {}) {
    const headers = options.headers ? { ...options.headers } : {};
    const csrf = readCsrfToken();
    if (csrf) headers['X-CSRF-Token'] = csrf;
    return fetch(url, { ...options, headers });
  }

  // -------- Event handlers --------

  // Ask
  askBtn?.addEventListener('click', async () => {
    const question = qInput?.value?.trim();
    if (!question) return;

    askBtn.disabled = true;
    askMsg?.classList?.remove('hidden');
    askMsg && (askMsg.textContent = 'Asking…');

    try {
      const res = await fetchJSON('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, mode: currentMode() })
      });
      const data = await res.json();

      // Update UI
      answerPre && (answerPre.textContent = data.answer || '');
      rationalePre && (rationalePre.textContent = data.rationale || '');
      renderRefsFromData(data);

      // If server returns history id, store it on the answer pre for later
      if (data?._id && answerPre) answerPre.setAttribute('data-id', data._id);

    } catch (err) {
      console.error(err);
      askMsg && (askMsg.textContent = 'Error while asking.');
    } finally {
      askBtn.disabled = false;
      setTimeout(() => askMsg && (askMsg.textContent = ''), 1200);
    }
  });

  // Convert → JSON (parse current answer)
  parseBtn?.addEventListener('click', async () => {
    const answerText = answerPre?.textContent?.trim();
    if (!answerText) return;

    parseBtn.disabled = true;
    try {
      const res = await fetchJSON('/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: answerText })
      });
      const data = await res.json();
      const pretty = JSON.stringify(data, null, 2);
      jsonPre && (jsonPre.textContent = pretty);
      // Offer a download of the JSON too
      download('answer.json', pretty, 'application/json');
    } catch (err) {
      console.error(err);
    } finally {
      parseBtn.disabled = false;
    }
  });

  // Upload a PDF or file to /upload
  uploadBtn?.addEventListener('click', async () => {
    const file = fileInput?.files?.[0];
    if (!file) return;

    uploadBtn.disabled = true;
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetchJSON('/upload', {
        method: 'POST',
        body: fd
      });
      const data = await res.json();
      alert(data?.ok ? 'Upload OK' : ('Upload failed: ' + (data?.error || 'unknown')));
    } catch (err) {
      console.error(err);
      alert('Upload failed.');
    } finally {
      uploadBtn.disabled = false;
    }
  });

  // Load history
  historyBtn?.addEventListener('click', async () => {
    try {
      const res = await fetchJSON('/api/history', { method: 'GET' });
      const items = await res.json();
      if (!Array.isArray(items)) return;

      // Clear list safely
      if (historyList) historyList.innerHTML = '';

      items.forEach(rec => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = '#';
        if (rec?._id) a.setAttribute('data-id', rec._id);
        a.textContent = rec?.question || '(no question)';
        li.appendChild(a);
        historyList?.appendChild(li);
      });
    } catch (err) {
      console.error(err);
    }
  });

  // Click on a history item → load the full record
  historyList?.addEventListener('click', async (e) => {
    const target = e.target;
    if (!(target instanceof Element)) return;
    if (target.tagName !== 'A') return;
    e.preventDefault();
    const id = target.getAttribute('data-id');
    if (!id) return;

    try {
      const res = await fetchJSON(`/api/history/${encodeURIComponent(id)}`, { method: 'GET' });
      const data = await res.json();

      answerPre && (answerPre.textContent = data.answer || '');
      rationalePre && (rationalePre.textContent = data.rationale || '');
      renderRefsFromData(data);
    } catch (err) {
      console.error(err);
    }
  });

  // Optional: upload a built-in dataset or config
  builtinBtn?.addEventListener('click', async () => {
    const file = builtinFile?.files?.[0];
    if (!file) return;
    builtinBtn.disabled = true;
    if (builtinMsg) builtinMsg.textContent = 'Uploading…';
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetchJSON('/upload_builtin', {
        method: 'POST',
        body: fd
      });
      const json = await res.json();
      if (builtinMsg) {
        builtinMsg.textContent = json.ok ? 'Uploaded OK' : 'Error: ' + (json.error || 'unknown');
      }
    } catch (err) {
      if (builtinMsg) builtinMsg.textContent = 'Upload failed';
      console.error(err);
    } finally {
      builtinBtn.disabled = false;
    }
  });

  // Default mode toggle 
  modeReason?.addEventListener('click', () => {
    modeReason?.setAttribute('aria-checked', 'true');
    modeRobot?.setAttribute('aria-checked', 'false');
    modeReason?.classList?.add('active');
    modeRobot?.classList?.remove('active');
  });

  modeRobot?.addEventListener('click', () => {
    modeRobot?.setAttribute('aria-checked', 'true');
    modeReason?.setAttribute('aria-checked', 'false');
    modeRobot?.classList?.add('active');
    modeReason?.classList?.remove('active');
  });

  // Auto-load history on page load
  historyBtn?.click();
});

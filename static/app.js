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

  

// ---- Harvest progress UI ----
let lastJobId = null;
function ensureHarvestUI(){
  let area = document.getElementById('harvestArea');
  if (!area){
    area = document.createElement('div');
    area.id = 'harvestArea';
    area.className = ''; area.style.marginTop = '12px'; area.style.border = '1px solid #1f2937'; area.style.padding = '10px'; area.style.borderRadius = '10px'; area.style.background = 'rgba(15,21,32,0.6)';
    area.innerHTML = `
      <div id="harvestHeader" style="font-size:12px;color:#cfd8e3;margin-bottom:6px">Indexing evidence…</div>
      <div style="width:100%;background:#374151;border-radius:4px;height:10px;margin-bottom:8px;"><div id="harvestBar" style="background:#3b82f6;height:10px;border-radius:4px;width:0%"></div></div>
      <pre id="harvestLog" style="font-size:12px;background:rgba(0,0,0,0.4);padding:8px;border-radius:6px;max-height:160px;overflow:auto"></pre>
      <div class="mt-2 flex gap-2">
        <button id="harvestRetryBtn" style="display:none;padding:6px 10px;border-radius:8px;background:#2563eb;color:white;font-size:12px;border:0;cursor:pointer">Re-run with updated index</button>
      </div>
    `;
    const mount = document.querySelector('#sec-answer') || document.body;
    mount.parentNode.insertBefore(area, mount.nextSibling);
  }
  return area;
}

async function pollHarvest(jid, originalQuestion){
  lastJobId = jid;
  const area = ensureHarvestUI();
  area.style.display = 'block';
  const bar = area.querySelector('#harvestBar');
  const logEl = area.querySelector('#harvestLog');
  const header = area.querySelector('#harvestHeader');
  const retry = area.querySelector('#harvestRetryBtn');
  retry.onclick = () => {
    if (originalQuestion){
      (document.getElementById('question') || {}).value = originalQuestion;
      document.getElementById('askBtn')?.click();
    }
  };
  let done = false;
  while(!done){
    try{
      const r = await fetch(`/status/${jid}`);
      if (!r.ok){ throw new Error('status ' + r.status); }
      const j = await r.json();
      const pct = Math.max(0, Math.min(100, Number(j.progress || 0)));
      bar.style.width = pct + '%';
      header.textContent = `Indexing evidence… ${pct}%`;
      if (Array.isArray(j.logs)){
        logEl.textContent = j.logs.slice(-200).join('\n');
        logEl.scrollTop = logEl.scrollHeight;
      }
      if (j.status === 'done' || j.status === 'error'){
        done = true;
        header.textContent = j.status === 'done' ? 'Indexing complete.' : ('Indexing error: ' + (j.error || ''));
        retry.style.display = 'inline-block';
      }
    }catch(e){
      console.warn('pollHarvest error', e);
      done = true;
      header.textContent = 'Indexing status unavailable.';
    }
    if (!done) await new Promise(r => setTimeout(r, 2000));
  }
}

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
      if (data.harvest_job_id) { pollHarvest(data.harvest_job_id, q); }
      // ----- References (used-only), with fallback to preformatted block -----
  const usedIdxSet = new Set(
    Array.isArray(data.used_ref_indexes) ? data.used_ref_indexes.map(Number) : []
  );
  const haveStructured = Array.isArray(data.refs) && data.refs.length > 0;
  const haveBlock = typeof data.references_block === 'string' && data.references_block.trim().length > 0;

  // clear any previous <pre> block if we switch back to list mode
  refsSection.querySelector('.refs-pre')?.remove();

  if (haveStructured && usedIdxSet.size > 0) {
    refsSection.classList.remove('hidden');
    refsList.innerHTML = data.refs
      .map((r, i) => ({ r, i: i + 1 }))
      .filter(x => usedIdxSet.has(x.i))
      .map(({ r, i }) => {
        const authors = Array.isArray(r.authors) ? r.authors.join(', ') : (r.authors || '');
        const title = r.title || `Reference ${i}`;
        const journal = (r.biblio && r.biblio.journal) || r.journal || '';
        const year = r.year || (r.biblio && r.biblio.year) || '';
        const volume = (r.biblio && r.biblio.volume) || r.volume || '';
        const pages = (r.biblio && r.biblio.pages) || r.pages || '';
        const doiUrl = r.doi ? `https://doi.org/${r.doi}` : '';
        const url = r.url || doiUrl || '#';
        const acs = `${authors}. <i>${title}</i>. <b>${journal}</b> ${year}, ${volume}, ${pages}. ` +
                    `<a href="${url}" target="_blank" rel="noopener">${r.doi || url}</a>`;
        return `<li>${acs}</li>`;
      })
      .join('');
  } else if (haveBlock) {
    refsSection.classList.remove('hidden');
    refsList.innerHTML = '';
    const pre = document.createElement('pre');
    pre.className = 'refs-pre';
    pre.textContent = data.references_block.trim();
    refsSection.appendChild(pre);
  } else {
    refsSection.classList.add('hidden');
    refsList.innerHTML = '';
  }

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

  /**
   * Renders references from data object into the UI.
   * @param {object} data
   */
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

    // Trigger download
    const blob = new Blob([pretty], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = (data.filename || 'converted') + '.json';
    document.body.appendChild(a); // needed for Firefox
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

  } catch (err) {
    console.error(err);
    if (jsonBlock) jsonBlock.textContent = 'Request failed';
    document.getElementById('jsonBlock')?.classList.remove('hidden');
  } finally {
    parseBtn.disabled = false;
    parseBtn.textContent = 'Convert → JSON';
  }
});

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
  if (historyBtn) historyBtn.click();
});
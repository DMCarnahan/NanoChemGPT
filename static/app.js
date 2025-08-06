// ------- DOM helpers -------
const qs  = (s) => document.querySelector(s);
const qsa = (s) => Array.from(document.querySelectorAll(s));

// ------- Health indicator -------
function setStatus(ok) {
  const el = qs('#healthStatus');
  if (!el) return;
  el.classList.remove('ok', 'err');
  el.classList.add(ok ? 'ok' : 'err');
  const t = el.querySelector('.status-txt');
  if (t) t.textContent = ok ? 'healthy' : 'unhealthy';
}

async function checkHealth() {
  try {
    const r = await fetch('/health', { cache: 'no-store' });
    setStatus(r.ok);
  } catch {
    setStatus(false);
  }
}

// ------- UI helpers -------
function showAlert(id, kind, text) {
  const el = qs(id);
  if (!el) return;
  el.classList.remove('hidden');
  el.className = `alert ${kind}`;
  el.textContent = text;
}

function hide(el) {
  el?.classList.add('hidden');
}

function setProgress(pct) {
  const bar = qs('#progressInner');
  if (bar) bar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

// ------- Text/HTML helpers -------
function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function renderRationaleWithCitations(text, refs) {
  // Escape everything first
  let html = escapeHtml(text || '');

  // Normalize full-width brackets to ASCII and footnotes [^n] -> [n]
  html = html.replaceAll('【', '[').replaceAll('】', ']');
  html = html.replace(/\[\^(\d{1,4})\]/g, '[$1]');

  // Link numeric citations [1], [2], ...
  html = html.replace(/\[(\d{1,4})\]/g, (m, n) => {
    const idx = parseInt(n, 10) - 1;
    const r = Array.isArray(refs) ? refs[idx] : null;
    const url = r && r.url ? r.url : (r && r.doi ? `https://doi.org/${r.doi}` : null);
    return url ? `<a href="${url}" target="_blank" rel="noopener">[${n}]</a>` : m;
  });

  // Style tags: [CTX], [GEN], and new [DB], [PARSED]
  html = html
    .replace(/\[CTX\]/g, '<span class="badge badge-ctx">[CTX]</span>')
    .replace(/\[GEN\]/g, '<span class="badge badge-gen">[GEN]</span>')
    .replace(/\[DB\]/g, '<span class="badge badge-db">[DB]</span>')
    .replace(/\[PARSED\]/g, '<span class="badge badge-parsed">[PARSED]</span>');

  return html;
}

function renderReferencesList(refs) {
  if (!Array.isArray(refs)) return '';
  return refs.map((r, i) => {
    const title = escapeHtml(r.title || '(no title)');
    const url   = r.url || (r.doi ? `https://doi.org/${r.doi}` : '#');
    const meta  = [r.venue, r.year, r.source].filter(Boolean).join(' • ');
    const mh    = meta ? ` <span class="muted">(${escapeHtml(meta)})</span>` : '';
    return `<li>[${i + 1}] <a href="${url}" target="_blank" rel="noopener">${title}</a>${mh}</li>`;
  }).join('');
}

function showRefs(refs) {
  const list = qs('#refsList');
  const sec  = qs('#refsSection');
  if (!list || !sec) return;

  if (Array.isArray(refs) && refs.length) {
    list.innerHTML = renderReferencesList(refs);
    sec.classList.remove('hidden');
  } else {
    list.innerHTML = '';
    sec.classList.add('hidden');
  }
}

// ------- Sources-used viewer -------

function updateSourcesUsed(data) {
  console.log('updateSourcesUsed got:', data); // <-- see ctx_* in console

  const s1 = document.getElementById('srcCtxVs');
  const s2 = document.getElementById('srcCtxParsed');
  const s3 = document.getElementById('srcCtxDb');
  const usedEl = document.getElementById('srcUsedSummary');
  const panel = document.getElementById('sourcesPanel');

  const v1 = (data.ctx_vs || data.ctxVS || data.ctx_uploads || '').trim();
  const v2 = (data.ctx_parsed || '').trim();
  const v3 = (data.ctx_db || '').trim();

  if (s1) s1.textContent = (v1 ? v1 : '(empty)').slice(0, 4000);
  if (s2) s2.textContent = (v2 ? v2 : '(empty)').slice(0, 4000);
  if (s3) s3.textContent = (v3 ? v3 : '(empty)').slice(0, 4000);

  // Show usage summary if provided by server
  if (usedEl) {
    try {
      const used = data.used || {};
      const refs = used.refs || data.refs_used || [];
      const tags = used.tags || {};
      const bits = [];
      if (Array.isArray(refs) && refs.length) bits.push(`refs [${refs.join(', ')}]`);
      const tparts = Object.entries(tags).filter(([k,v]) => v > 0).map(([k,v]) => `${k}×${v}`);
      if (tparts.length) bits.push(tparts.join(' • '));
      usedEl.textContent = bits.length ? bits.join(' | ') : '(none detected)';
    } catch (e) {
      usedEl.textContent = '(none detected)';
    }
  }

  if (panel && (v1 || v2 || v3)) panel.open = true;
}

// ------- Upload flow -------
async function uploadFile(file) {
  const fd = new FormData();
  fd.append('file', file);

  // Fake progress until server returns a job id
  let pct = 10;
  const fake = setInterval(() => {
    if (pct < 85) { pct += 5; setProgress(pct); }
  }, 150);

  try {
    const resp = await fetch('/upload', { method: 'POST', body: fd });
    clearInterval(fake);

    if (!resp.ok) {
      const txt = await resp.text();
      showAlert('#result', 'error', `Upload failed (${resp.status}): ${txt || 'Unknown error'}`);
      return;
    }

    const data = await resp.json();
    showAlert('#result', 'success', `Uploaded: ${data.filename || file.name}`);

    if (data.job_id) {
      for (;;) {
        const r = await fetch(`/status/${data.job_id}`, { cache: 'no-store' });
        if (!r.ok) { showAlert('#result', 'error', `Status error: ${r.status}`); break; }
        const st = await r.json();
        if (typeof st.progress === 'number') setProgress(st.progress);
        if (st.status === 'done') { showAlert('#result', 'success', `Processed ${st.filename || ''}`); break; }
        if (st.status === 'error') { showAlert('#result', 'error', `Processing error: ${st.error || 'unknown'}`); break; }
        await new Promise((res) => setTimeout(res, 1000));
      }
    } else {
      setProgress(100);
    }
  } catch (err) {
    clearInterval(fake);
    showAlert('#result', 'error', `Network error: ${err}`);
  }
}

// ------- Ask flow -------
let lastAnswer   = '';
let lastQuestion = '';

async function askQuestion() {
  hide(qs('#askMsg'));
  hide(qs('#jsonBlock'));

  const q = (qs('#questionInput').value || '').trim();
  if (!q) {
    showAlert('#askMsg', 'error', 'Type a question first.');
    return;
  }

  const btn = qs('#askBtn');
  btn.disabled = true;
  btn.querySelector('.spinner')?.classList.remove('hidden');
  qs('#parseBtn').disabled   = true;
  qs('#saveTxtBtn').disabled = true;

  let data;
  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question: q })
    });

    if (!r.ok) {
      const txt = await r.text();
      showAlert('#askMsg', 'error', `Ask failed (${r.status}): ${txt || 'Unknown error'}`);
      return;
    }

    data = await r.json();

    lastAnswer   = data.answer || '';
    lastQuestion = q;

    // Fill UI
    qs('#answerPre').textContent = lastAnswer || '(empty)';
    qs('#rationalePre').innerHTML = renderRationaleWithCitations(
      data.rationale || '',
      data.references || []
    );
    showRefs(data.references || []);

    // Fill Sources-used viewer if server returned ctx fields
    updateSourcesUsed(data);

    // Buttons
    qs('#parseBtn').disabled   = !lastAnswer;
    qs('#saveTxtBtn').disabled = !lastAnswer;

    showAlert('#askMsg', 'success', 'Answer ready.');

    // Fallback: fetch context from history if missing
    if ((!data.ctx_vs && !data.ctx_parsed && !data.ctx_db) && data.qa_id) {
      try {
        const r2 = await fetch(`/api/history/${data.qa_id}`, { cache: 'no-store' });
        if (r2.ok) {
          const doc = await r2.json();
          updateSourcesUsed({
            ctx_vs: doc.ctx_vs || '',
            ctx_parsed: doc.ctx_parsed || '',
            ctx_db: doc.ctx_db || ''
          });
        }
      } catch {}
    }

  } catch (err) {
    showAlert('#askMsg', 'error', `Network error: ${err}`);
  } finally {
    btn.disabled = false;
    btn.querySelector('.spinner')?.classList.add('hidden');
  }
}

// ------- Download helpers -------
function downloadFile(filename, text, mime = 'application/json') {
  const blob = new Blob([text], { type: mime });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function safeName(s) {
  return (s || 'parsed')
    .toLowerCase()
    .replace(/[^a-z0-9\-_.]+/g, '-')
    .replace(/-+/g, '-')
    .slice(0, 80);
}

// ------- JSON preview toggle -------
function showJsonPreview(pretty) {
  const block = qs('#jsonBlock');
  if (!block) return;
  const want = qs('#showJsonPreview')?.checked;
  if (want) {
    qs('#jsonPre').textContent = pretty;
    block.classList.remove('hidden');
  } else {
    block.classList.add('hidden');
  }
}

// ------- Convert answer to JSON -------
async function parseAnswer() {
  const robot = !!qs('#robotMode')?.checked;

  let text = (typeof lastAnswer === 'string' && lastAnswer.trim())
    ? lastAnswer
    : (qs('#answerPre')?.textContent || '').trim();

  hide(qs('#jsonBlock'));

  if (!text) {
    showAlert('#askMsg', 'error', 'No answer to parse yet. Click “Ask” first.');
    return;
  }

  try {
    const r = await fetch('/parse', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text, robot, question: lastQuestion || '' })
    });


    const raw = await r.text();

    if (!r.ok) {
      showAlert('#askMsg', 'error', `Parse failed (${r.status}): ${raw || 'Unknown error'}`);
      return;
    }

    // Tolerant: parse or show raw
    let obj;
    try {
      obj = JSON.parse(raw);
    } catch {
      qs('#jsonPre').textContent = raw;
      qs('#jsonBlock').classList.remove('hidden');
      showAlert('#askMsg', 'error', 'Server returned non‑JSON. Showing raw text.');
      return;
    }

    const pretty = JSON.stringify(obj, null, 2);
    showJsonPreview(pretty);
    showAlert('#askMsg', 'success', 'Parsed to JSON (downloaded).');

    // Auto‑download
    const base = safeName(lastQuestion || 'answer');
    downloadFile(`${base || 'answer'}.json`, pretty, 'application/json');
  } catch (err) {
    showAlert('#askMsg', 'error', `Network/JS error: ${err}`);
  }
}

// ------- Save answer to TXT -------
async function saveTxt() {
  if (!lastAnswer) {
    showAlert('#askMsg', 'error', 'No answer to save yet.');
    return;
  }
  try {
    const r = await fetch('/save_txt', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ answer: lastAnswer, question: lastQuestion })
    });

    if (!r.ok) {
      const txt = await r.text();
      showAlert('#askMsg', 'error', `Save failed (${r.status}): ${txt || 'Unknown error'}`);
      return;
    }

    const blob = await r.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url;
    a.download = 'answer.txt';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    showAlert('#askMsg', 'success', 'Saved as TXT.');
  } catch (err) {
    showAlert('#askMsg', 'error', `Network error: ${err}`);
  }
}

// ------- Literature search -------
async function runSearch() {
  const q = (qs('#searchInput')?.value || '').trim();
  if (!q) {
    showAlert('#searchMsg', 'error', 'Enter a search query.');
    return;
  }

  showAlert('#searchMsg', 'success', 'Searching…');

  try {
    const r = await fetch('/search', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ q, n: 6 })
    });

    const raw = await r.text();

    if (!r.ok) {
      showAlert('#searchMsg', 'error', `Search failed (${r.status}): ${raw}`);
      return;
    }

    const data = JSON.parse(raw);
    const ol   = qs('#searchResults');
    ol.innerHTML = (data.results || []).map((it, i) => {
      const title = escapeHtml(it.title || '(no title)');
      const url   = it.url || (it.doi ? `https://doi.org/${it.doi}` : '#');
      const meta  = [it.venue, it.year, it.source].filter(Boolean).join(' • ');
      return `<li>[${i + 1}] <a href="${url}" target="_blank" rel="noopener">${title}</a>` +
             (meta ? ` <span class="muted">(${escapeHtml(meta)})</span>` : '') +
             `</li>`;
    }).join('');

    showAlert('#searchMsg', 'success', `Found ${(data.results || []).length} results.`);
  } catch (e) {
    showAlert('#searchMsg', 'error', `Network/JS error: ${e}`);
  }
}

// ------- History (Q&A) -------
let histSkip  = 0;
const histLimit = 10;

function renderHistItems(items, append = false) {
  const ol = qs('#histList');
  if (!ol) return;

  const html = (items || []).map((d) => {
    const dt = d.created_at ? new Date(d.created_at).toISOString().slice(0, 19).replace('T', ' ') : '';
    const qx = (d.question || '').slice(0, 120).replace(/\s+/g, ' ');
    return `<li><a href="#" data-hist-id="${d._id}">[${dt}] ${escapeHtml(qx)}</a></li>`;
  }).join('');

  if (append) ol.insertAdjacentHTML('beforeend', html);
  else ol.innerHTML = html;
}

async function loadHistory(reset = false) {
  if (reset) {
    histSkip = 0;
    const list = qs('#histList');
    if (list) list.innerHTML = '';
  }

  const query = (qs('#histQuery')?.value || '').trim();
  const url   = `/api/history?skip=${histSkip}&limit=${histLimit}` +
                (query ? `&q=${encodeURIComponent(query)}` : '');

  try {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) { showAlert('#histMsg', 'error', `History error (${r.status})`); return; }
    const data = await r.json();

    renderHistItems(data.items, (histSkip > 0));
    if (data.items && data.items.length) histSkip += data.items.length;

    showAlert('#histMsg', 'success', `Loaded ${data.items?.length || 0} items.`);
  } catch (e) {
    showAlert('#histMsg', 'error', `History load failed: ${e}`);
  }
}

async function loadHistoryItem(id) {
  try {
    const r = await fetch(`/api/history/${id}`, { cache: 'no-store' });
    if (!r.ok) { showAlert('#histMsg', 'error', `Open failed (${r.status})`); return; }
    const d = await r.json();

    // Fill UI
    lastQuestion = d.question || '';
    lastAnswer   = d.answer   || '';

    qs('#questionInput').value  = d.question || '';
    qs('#answerPre').textContent = lastAnswer || '(empty)';
    qs('#rationalePre').innerHTML = renderRationaleWithCitations(d.rationale || '', d.references || []);
    showRefs(d.references || []);

    // Also populate the Sources‑Used viewer when loading from history (if fields exist)
    updateSourcesUsed({
      ctx_vs:     d.ctx_vs,
      ctx_parsed: d.ctx_parsed,
      ctx_db:     d.ctx_db
    });

    qs('#parseBtn').disabled   = !lastAnswer;
    qs('#saveTxtBtn').disabled = !lastAnswer;

    showAlert('#askMsg', 'success', 'Loaded from history.');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch (e) {
    showAlert('#histMsg', 'error', `Open failed: ${e}`);
  }
}

// ------- Upload Browser -------
async function loadUploads() {
  try {
    const r = await fetch('/api/uploads', { cache: 'no-store' });
    if (!r.ok) { showAlert('#uplMsg', 'error', `Uploads error (${r.status})`); return; }

    const data = await r.json();
    const ol   = qs('#uplList');

    ol.innerHTML = (data.items || []).map((u) => {
      const ts  = u.indexed_at || u.ts || '';
      const dt  = ts ? new Date(ts).toISOString().slice(0, 19).replace('T', ' ') : '';
      const info = [u.status, u.kind, (u.n_pages ? `${u.n_pages} pp` : '')].filter(Boolean).join(' • ');
      return `<li>${escapeHtml(u.filename || '(unknown)')} ` +
             `<span class="muted">— ${escapeHtml(info)} ${dt ? ` • ${dt}` : ''}</span></li>`;
    }).join('');

    showAlert('#uplMsg', 'success', `Loaded ${data.items?.length || 0} uploads.`);
  } catch (e) {
    showAlert('#uplMsg', 'error', `Uploads load failed: ${e}`);
  }
}

// ------- Wire UI -------
function wireUI() {
  checkHealth();
  setInterval(checkHealth, 30000);

  // Upload
  const input  = qs('#fileInput');
  const button = qs('#uploadBtn');

  input?.addEventListener('change', () => {
    const f = input.files?.[0];
    const hint = qs('#fileHint');
    if (hint && f) hint.textContent = `${f.name} • ${(f.size / 1024 / 1024).toFixed(2)} MB`;
  });

  button?.addEventListener('click', async (e) => {
    e.preventDefault();
    const f = input?.files?.[0];
    if (!f) { showAlert('#result', 'error', 'Choose a file first.'); return; }
    button.disabled = true;
    button.querySelector('.spinner')?.classList.remove('hidden');
    await uploadFile(f);
    button.disabled = false;
    button.querySelector('.spinner')?.classList.add('hidden');
    loadUploads(); // refresh uploads list after a new upload
  });

  // Ask / parse / save / search
  qs('#askBtn')?.addEventListener('click',  (e) => { e.preventDefault(); askQuestion(); });
  qs('#parseBtn')?.addEventListener('click',(e) => { e.preventDefault(); parseAnswer(); });
  qs('#saveTxtBtn')?.addEventListener('click',(e)=> { e.preventDefault(); saveTxt(); });
  qs('#searchBtn')?.addEventListener('click',(e)=> { e.preventDefault(); runSearch(); });

  // History
  qs('#histRefreshBtn')?.addEventListener('click', (e) => { e.preventDefault(); loadHistory(true); });
  qs('#histMoreBtn')?.addEventListener('click',    (e) => { e.preventDefault(); loadHistory(false); });
  qs('#histList')?.addEventListener('click', (e) => {
    const a = e.target.closest('a[data-hist-id]');
    if (!a) return;
    e.preventDefault();
    loadHistoryItem(a.getAttribute('data-hist-id'));
  });

  // Uploads
  qs('#uplRefreshBtn')?.addEventListener('click', (e) => { e.preventDefault(); loadUploads(); });

  // Initial data
  loadHistory(true);
  loadUploads();
}

document.addEventListener('DOMContentLoaded', wireUI);

// ----- Builtin uploader → /admin/upload_builtin -----
(() => {
  const dz   = document.getElementById('builtinDrop');
  const inp  = document.getElementById('builtinFile');
  const msg  = document.getElementById('builtinMsg');

  if (!dz || !inp) return;

  // 🔐 Put your ADMIN_UPLOAD_SECRET here while testing, or prompt each time.
  // Better: render it into the page server-side for a private /admin route only.
  let ADMIN_TOKEN = ""; // e.g. "paste-temporary-token-here"

  function setMsg(txt, ok=false) {
    msg.classList.remove('hidden');
    msg.textContent = txt;
    msg.classList.toggle('error', !ok);
  }

  async function uploadOne(file) {
    const fd = new FormData();
    fd.append('file', file);

    const res = await fetch('/admin/upload_builtin', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer ' + (ADMIN_TOKEN || prompt('Admin token:') || '') },
      body: fd
    });

    let body;
    try { body = await res.json(); } catch { body = {error: await res.text()} }

    if (!res.ok || body.error) {
      setMsg(`❌ ${file.name}: ${body.error || res.statusText}`, false);
    } else {
      const where = body.decompressed || body.saved;
      setMsg(`✅ ${file.name} uploaded → ${where}`, true);
    }
  }

  function handleFiles(files) {
    [...files].forEach(f => uploadOne(f));
  }

  dz.addEventListener('click', () => inp.click());
  dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); inp.click(); } });
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('hover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('hover'));
  dz.addEventListener('drop', (e) => { e.preventDefault(); dz.classList.remove('hover'); handleFiles(e.dataTransfer.files); });
  inp.addEventListener('change', () => handleFiles(inp.files));
})();

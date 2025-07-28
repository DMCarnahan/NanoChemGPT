const qs  = (s) => document.querySelector(s);
const qsa = (s) => Array.from(document.querySelectorAll(s));

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
  } catch { setStatus(false); }
}

function humanSize(bytes) {
  if (!Number.isFinite(bytes)) return '';
  const units = ['B','KB','MB','GB','TB'];
  let i = 0, n = bytes;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

function showAlert(id, kind, text) {
  const el = qs(id);
  if (!el) return;
  el.classList.remove('hidden');
  el.className = `alert ${kind}`;
  el.textContent = text;
}

function hide(el) { el?.classList.add('hidden'); }
function setProgress(pct) { const bar = qs('#progressInner'); if (bar) bar.style.width = `${Math.max(0, Math.min(100, pct))}%`; }
function resetProgress() { setProgress(0); }
function downloadFile(filename, text, mime = 'application/json') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  a.remove(); URL.revokeObjectURL(url);
}
function safeName(s) {
  return (s || 'parsed')
    .toLowerCase().replace(/[^a-z0-9\-_.]+/g, '-').replace(/-+/g, '-')
    .slice(0, 80);
}

// ---- Upload flow ----
async function uploadFile(file) {
  resetProgress();

  const fd = new FormData(); fd.append('file', file);

  // Fake progress until server returns a job id
  let pct = 10;
  const fake = setInterval(() => { if (pct < 85) { pct += 5; setProgress(pct); } }, 150);

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
    if (data.job_id) await pollStatus(data.job_id);
    else setProgress(100);
  } catch (err) {
    clearInterval(fake);
    showAlert('#result', 'error', `Network error: ${err}`);
  }
}

async function pollStatus(jobId) {
  for (;;) {
    const r = await fetch(`/status/${jobId}`, { cache: 'no-store' });
    if (!r.ok) { showAlert('#result', 'error', `Status error: ${r.status}`); break; }
    const st = await r.json();
    if (typeof st.progress === 'number') setProgress(st.progress);
    if (st.status === 'done') { showAlert('#result', 'success', `Processed ${st.filename || ''}`); break; }
    if (st.status === 'error') { showAlert('#result', 'error', `Processing error: ${st.error || 'unknown'}`); break; }
    await new Promise(res => setTimeout(res, 1000));
  }
}

// ---- Ask flow ----
let lastAnswer = '';
let lastQuestion = '';

async function askQuestion() {
  hide(qs('#askMsg')); hide(qs('#jsonBlock'));
  const q = (qs('#questionInput').value || '').trim();
  if (!q) { showAlert('#askMsg', 'error', 'Type a question first.'); return; }

  const btn = qs('#askBtn');
  btn.disabled = true; btn.querySelector('.spinner')?.classList.remove('hidden');
  qs('#parseBtn').disabled = true; qs('#saveTxtBtn').disabled = true;

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
    const data = await r.json();
    lastAnswer = data.answer || '';
    lastQuestion = q;

    qs('#answerPre').textContent = lastAnswer || '(empty)';
    qs('#rationalePre').textContent = data.rationale || '(no rationale returned)';
    qs('#parseBtn').disabled = !lastAnswer;
    qs('#saveTxtBtn').disabled = !lastAnswer;

    showAlert('#askMsg', 'success', 'Answer ready.');
  } catch (err) {
    showAlert('#askMsg', 'error', `Network error: ${err}`);
  } finally {
    btn.disabled = false; btn.querySelector('.spinner')?.classList.add('hidden');
  }
}

// ---- Convert answer to JSON ----
async function parseAnswer() {
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
      body: JSON.stringify({ text })
    });

    const raw = await r.text();
    if (!r.ok) {
      showAlert('#askMsg', 'error', `Parse failed (${r.status}): ${raw || 'Unknown error'}`);
      return;
    }

    // Tolerant: parse or show raw
    let obj;
    try { obj = JSON.parse(raw); }
    catch {
      qs('#jsonPre').textContent = raw;
      qs('#jsonBlock').classList.remove('hidden');
      showAlert('#askMsg', 'error', 'Server returned non‑JSON. Showing raw text.');
      return;
    }

    const pretty = JSON.stringify(obj, null, 2);
    qs('#jsonPre').textContent = pretty;
    qs('#jsonBlock').classList.remove('hidden');
    showAlert('#askMsg', 'success', 'Parsed to JSON.');

    // Auto‑download
    const base = safeName(lastQuestion || 'answer');
    downloadFile(`${base || 'answer'}.json`, pretty, 'application/json');
  } catch (err) {
    showAlert('#askMsg', 'error', `Network/JS error: ${err}`);
  }
}


// ---- Save answer to TXT ----
async function saveTxt() {
  if (!lastAnswer) { showAlert('#askMsg', 'error', 'No answer to save yet.'); return; }
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
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'answer.txt';
    document.body.appendChild(a); a.click();
    a.remove(); URL.revokeObjectURL(url);
    showAlert('#askMsg', 'success', 'Saved as TXT.');
  } catch (err) {
    showAlert('#askMsg', 'error', `Network error: ${err}`);
  }
}

// ---- Clear uploaded memory ----
async function clearUploads() {
  try {
    const r = await fetch('/clear_uploads', { method: 'POST' });
    if (!r.ok) {
      const txt = await r.text();
      showAlert('#clearMsg', 'error', `Clear failed (${r.status}): ${txt || 'Unknown error'}`);
      return;
    }
    showAlert('#clearMsg', 'success', 'Uploads cleared.');
  } catch (err) {
    showAlert('#clearMsg', 'error', `Network error: ${err}`);
  }
}

// ---- Wire UI ----
function wireUI() {
  checkHealth();
  setInterval(checkHealth, 30000);

  const input  = qs('#fileInput');
  const button = qs('#uploadBtn');
  input?.addEventListener('change', () => {
    const f = input.files?.[0];
    const hint = qs('#fileHint');
    if (hint && f) hint.textContent = `${f.name} • ${humanSize(f.size)}`;
  });
  button?.addEventListener('click', async (e) => {
    e.preventDefault();
    const f = input?.files?.[0];
    if (!f) { showAlert('#result', 'error', 'Choose a file first.'); return; }
    button.disabled = true; button.querySelector('.spinner')?.classList.remove('hidden');
    await uploadFile(f);
    button.disabled = false; button.querySelector('.spinner')?.classList.add('hidden');
  });

  qs('#askBtn')?.addEventListener('click', (e) => { e.preventDefault(); askQuestion(); });
  qs('#parseBtn')?.addEventListener('click', (e) => { e.preventDefault(); parseAnswer(); });
  qs('#saveTxtBtn')?.addEventListener('click', (e) => { e.preventDefault(); saveTxt(); });
  qs('#clearBtn')?.addEventListener('click', (e) => { e.preventDefault(); clearUploads(); });
}

document.addEventListener('DOMContentLoaded', wireUI);

function escapeHtml(s) {
  return (s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function runSearch() {
  const q = (qs('#searchInput')?.value || '').trim();
  if (!q) { showAlert('#searchMsg', 'error', 'Enter a search query.'); return; }
  showAlert('#searchMsg', 'success', 'Searching…');
  try {
    const r = await fetch('/search', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ q, n: 6 })
    });
    const raw = await r.text();
    if (!r.ok) { showAlert('#searchMsg', 'error', `Search failed (${r.status}): ${raw}`); return; }
    const data = JSON.parse(raw);
    renderSearchResults(data.results || []);
    showAlert('#searchMsg', 'success', `Found ${ (data.results||[]).length } results.`);
  } catch (e) {
    showAlert('#searchMsg', 'error', `Network/JS error: ${e}`);
  }
}

function renderSearchResults(items) {
  const ol = qs('#searchResults');
  if (!ol) return;
  ol.innerHTML = (items || []).map((it, i) => {
    const title = escapeHtml(it.title || '(no title)');
    const url = it.url || (it.doi ? `https://doi.org/${it.doi}` : '#');
    const meta = [it.venue, it.year, it.source].filter(Boolean).join(' • ');
    return `<li>[${i+1}] <a href="${url}" target="_blank" rel="noopener">${title}</a>` +
           (meta ? ` <span class="muted">(${escapeHtml(meta)})</span>` : '') +
           `</li>`;
  }).join('');
}

qs('#searchBtn')?.addEventListener('click', (e) => { e.preventDefault(); runSearch(); });

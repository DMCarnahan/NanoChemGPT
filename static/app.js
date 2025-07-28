function qs(sel) { return document.querySelector(sel); }
function qsa(sel) { return Array.from(document.querySelectorAll(sel)); }

function setStatus(ok) {
  const el = qs('#healthStatus');
  if (!el) return;
  el.classList.remove('ok', 'err');
  el.classList.add(ok ? 'ok' : 'err');
  const text = el.querySelector('.status-txt');
  if (text) text.textContent = ok ? 'healthy' : 'unhealthy';
}

async function checkHealth() {
  try {
    const r = await fetch('/health', { cache: 'no-store' });
    setStatus(r.ok);
  } catch {
    setStatus(false);
  }
}

function humanSize(bytes) {
  if (!Number.isFinite(bytes)) return '';
  const units = ['B','KB','MB','GB','TB'];
  let i = 0, n = bytes;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

function resetProgress() {
  const bar = qs('#progressInner');
  if (bar) bar.style.width = '0%';
}

function setProgress(pct) {
  const bar = qs('#progressInner');
  if (bar) bar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

function showMessage(kind, text) {
  const out = qs('#result');
  if (!out) return;
  out.className = `alert ${kind}`;
  out.textContent = text;
}

async function uploadFile(file) {
  resetProgress();
  setProgress(10);

  const fd = new FormData();
  fd.append('file', file);

  // Note: Native fetch doesn't give real upload progress without XHR.
  // We fake a little progress then jump to 100% on completion.
  const fake = setInterval(() => {
    const bar = qs('#progressInner');
    if (!bar) return;
    const curr = parseFloat(bar.style.width || '0');
    if (curr < 85) setProgress(curr + 5);
  }, 150);

  try {
    const resp = await fetch('/upload', { method: 'POST', body: fd });
    clearInterval(fake);
    setProgress(100);

    const contentType = resp.headers.get('content-type') || '';
    if (!resp.ok) {
      const txt = await resp.text();
      showMessage('error', `Upload failed (${resp.status}): ${txt || 'Unknown error'}`);
      return;
    }

    if (contentType.includes('application/json')) {
      const data = await resp.json();
      const fname = data.filename || file.name;
      let details = '';
      if (typeof data.chars === 'number') details = ` • ${data.chars} chars`;
      if (data.kind) details = ` (${data.kind})` + details;
      showMessage('success', `Uploaded: ${fname}${details}`);
    } else {
      const txt = await resp.text();
      showMessage('success', `Uploaded: ${file.name}\n${txt.slice(0, 500)}`);
    }
  } catch (err) {
    clearInterval(fake);
    showMessage('error', `Network error: ${err}`);
  }
}

function wireUI() {
  // Health badge wires itself
  checkHealth();
  // ping health periodically
  setInterval(checkHealth, 30000);

  const input = qs('#fileInput');
  const button = qs('#uploadBtn');

  if (input) {
    input.addEventListener('change', () => {
      const f = input.files?.[0];
      const hint = qs('#fileHint');
      if (hint && f) {
        hint.textContent = `${f.name} • ${humanSize(f.size)}`;
      }
    });
  }

  if (button && input) {
    button.addEventListener('click', async (e) => {
      e.preventDefault();
      const f = input.files?.[0];
      if (!f) {
        showMessage('error', 'Choose a file first (PDF/JSON/TXT).');
        return;
      }
      button.disabled = true;
      qsa('.btn .spinner').forEach(s => s.classList.remove('hidden'));
      await uploadFile(f);
      qsa('.btn .spinner').forEach(s => s.classList.add('hidden'));
      button.disabled = false;
    });
  }
}

document.addEventListener('DOMContentLoaded', wireUI);

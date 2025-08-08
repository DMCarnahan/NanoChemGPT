// ---- config ----
const ENDPOINTS = {
  health: '/health',
  ask: '/ask',
  upload: '/upload',
  uploadBuiltin: '/upload_builtin',
  uploadsList: '/uploads',
  clearUploads: '/clear_uploads',
  historyList: '/history'
};

// ---- DOM helpers ----
const $ = (id) => document.getElementById(id);
const on = (el, ev, fn) => el && el.addEventListener(ev, fn);
const show = (el, yes = true) => el && el.classList[yes ? 'remove' : 'add']('hidden');
const text = (el, v = '') => { if (el) el.textContent = v; };
const html = (el, v = '') => { if (el) el.innerHTML = v; };

function setBusy(btn, busy = true) {
  if (!btn) return;
  const spin = btn.querySelector?.('.spinner');
  if (spin) show(spin, busy);
  btn.disabled = busy;
}

function toast(el, msg, type = 'success') {
  if (!el) return;
  el.className = `alert ${type}`;
  text(el, msg);
  show(el, true);
}

function ensureInteractive() {
  ['modeRobot', 'modeReason', 'parseBtn', 'saveTxtBtn'].forEach(id => {
    const el = $(id);
    if (!el) return;
    el.disabled = false;
    el.style.pointerEvents = 'auto';
    el.style.opacity = '1';
    el.classList?.remove('disabled');
  });
}

// ---- Health check ----
async function checkHealth() {
  const hs = $('healthStatus');
  try {
    const res = await fetch(ENDPOINTS.health, { cache: 'no-store', credentials: 'include' });
    const ok = res.ok;
    hs?.classList.toggle('ok', ok);
    text(hs?.querySelector('.status-txt'), ok ? 'healthy' : 'degraded');
  } catch {
    hs?.classList.remove('ok');
    text(hs?.querySelector('.status-txt'), 'offline');
  }
}

// ---- Ask handler ----
async function handleAsk() {
  const btn = $('askBtn');
  const msg = $('askMsg');
  const qEl = $('questionInput');
  const q = qEl?.value?.trim() || '';
  const mode = $('modeReason')?.getAttribute('aria-checked') === 'true' ? 'reason' : 'robot';

  if (!q) { toast(msg, 'Please enter a question.', 'error'); return; }
  show(msg, false);
  setBusy(btn, true);

  try {
    const res = await fetch(ENDPOINTS.ask, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, mode })
    });

    if (!res.ok) {
      let body = ''; try { body = await res.text(); } catch {}
      throw new Error(`HTTP ${res.status}${body ? ` – ${body.slice(0, 200)}` : ''}`);
    }

    const raw = await res.text();
    let data; try { data = JSON.parse(raw); } catch { data = { answer: raw }; }

    const answer = (data.answer ?? data.result ?? data.response ?? data.message ?? '').toString();
    const rationale = (data.rationale ?? data.explanation ?? '').toString();
    const refs = Array.isArray(data.refs) ? data.refs :
                 (Array.isArray(data.references) ? data.references : []);
    const sources = data.sources || {};
    const usage = data.usage || {};

    text($('answerPre'), answer);
    text($('rationalePre'), rationale);
    html($('refsList'), refs.map(r => `<li>${r}</li>`).join(''));
    show($('refsSection'), refs.length > 0);

    text($('srcUsedSummary'), usage.summary || (Object.keys(usage).length ? JSON.stringify(usage, null, 2) : ''));
    text($('srcCtxVs'), (sources.ctx_vs && JSON.stringify(sources.ctx_vs, null, 2)) || '');
    text($('srcCtxParsed'), (sources.ctx_parsed && JSON.stringify(sources.ctx_parsed, null, 2)) || '');
    text($('srcCtxDb'), (sources.ctx_db && JSON.stringify(sources.ctx_db, null, 2)) || '');

    const hasAnswer = !!$('answerPre')?.textContent?.trim();
    if ($('parseBtn')) $('parseBtn').disabled = !hasAnswer;
    if ($('saveTxtBtn')) $('saveTxtBtn').disabled = !hasAnswer;

    ensureInteractive();
    toast(msg, 'Done.', 'success');
  } catch (err) {
    toast(msg, `Ask failed: ${err.message}`, 'error');
  } finally {
    setBusy(btn, false);
  }
}

// ---- Mode toggle ----
function toggleMode(which) {
  const robot = $('modeRobot');
  const reason = $('modeReason');
  const r = which === 'robot';
  robot?.setAttribute('aria-checked', r ? 'true' : 'false');
  reason?.setAttribute('aria-checked', r ? 'false' : 'true');
  ensureInteractive();
}

// ---- Export answer ----
function exportAnswer() {
  const blob = new Blob([$('answerPre')?.textContent || ''], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'answer.txt';
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  on($('askBtn'), 'click', handleAsk);
  on($('modeRobot'), 'click', () => toggleMode('robot'));
  on($('modeReason'), 'click', () => toggleMode('reason'));
  on($('saveTxtBtn'), 'click', exportAnswer);
  // Add parseBtn wiring when parser is ready
});

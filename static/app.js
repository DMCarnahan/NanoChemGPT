(function () {
  // Run after DOM is ready
  document.addEventListener('DOMContentLoaded', () => {
    const $ = (id) => document.getElementById(id);

    // Core elements (IDs must match index.html)
    const askBtn = $('askBtn');
    const qInput = $('question');           // textarea
    const answerPre = $('answerPre');       // <pre> for answer (markdown text for now)
    const rationalePre = $('rationalePre'); // <pre> optional rationale
    const askMsg = $('askMsg');             // status alert
    const reasoningBtn = $('reasoningBtn'); // mode toggle
    const robotBtn = $('robotBtn');         // mode toggle

    if (!askBtn || !qInput) {
      console.warn('[NanoChemGPT] Missing #askBtn or #question in DOM.');
      return;
    }

    let mode = 'robot';
    function setMode(next) {
      mode = next;
      if (robotBtn) robotBtn.setAttribute('aria-checked', String(next === 'robot'));
      if (reasoningBtn) reasoningBtn.setAttribute('aria-checked', String(next === 'reasoning'));
    }
    robotBtn?.addEventListener('click', () => setMode('robot'));
    reasoningBtn?.addEventListener('click', () => setMode('reasoning'));
    setMode('robot'); // initialize

    function readCsrfToken() {
      const meta = document.querySelector('meta[name="csrf-token"]')?.content;
      if (meta) return meta;
      const m = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/);
      return m ? decodeURIComponent(m[1]) : null;
    }

    async function callAsk(questionText) {
      const headers = { 'Content-Type': 'application/json' };
      const csrf = readCsrfToken();
      if (csrf) headers['X-CSRFToken'] = csrf;

      const body = JSON.stringify({ question: questionText /*, mode*/ });

      const res = await fetch('/ask', { method: 'POST', headers, body });
      const raw = await res.text();
      let data;
      try { data = JSON.parse(raw); } catch { data = { answer: raw }; }
      return { ok: res.ok, status: res.status, data };
    }

    function setStatus(msg, show = true) {
      if (!askMsg) return;
      if (show) {
        askMsg.classList.remove('hidden');
        askMsg.textContent = msg;
      } else {
        askMsg.textContent = msg || '';
        askMsg.classList.add('hidden');
      }
    }

    function renderAnswer(payload) {
      const { answer, rationale } = payload || {};
      if (answerPre) {
        
        answerPre.textContent = (answer ?? '').trim() || '(no answer)';
      }
      if (rationalePre) {
        rationalePre.textContent = (rationale ?? '').trim();
      }
    }

    askBtn.addEventListener('click', async (e) => {
      e.preventDefault();
      const question = (qInput.value || '').trim();
      if (!question) {
        setStatus('Please enter a question.');
        qInput.focus();
        return;
      }

      askBtn.disabled = true;
      setStatus('Asking…');

      try {
        const resp = await callAsk(question);
        renderAnswer(resp.data);
        setStatus(resp.ok ? 'Done.' : `Error ${resp.status}`);
        if (resp.status === 401) {
        }
      } catch (err) {
        console.error('[NanoChemGPT] /ask failed', err);
        setStatus('Request failed. See console.');
      } finally {
        askBtn.disabled = false;
      }
    });

    qInput.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        askBtn.click();
      }
    });

    const yr = document.getElementById('year');
    if (yr) yr.textContent = String(new Date().getFullYear());
  });
})();
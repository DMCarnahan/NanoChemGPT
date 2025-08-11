(function () {
  document.addEventListener('DOMContentLoaded', () => {
    const $  = (id) => document.getElementById(id);
    const by = (sel) => document.querySelector(sel);

    // Core elements
    const askBtn = $('askBtn');
    const qInput = $('question');
    const answerPre = $('answerPre');
    const rationalePre = $('rationalePre');
    const askMsg = $('askMsg');
    const parseBtn = $('parseBtn');
    const saveTxtBtn = $('saveTxtBtn');
    const jsonBlock = $('jsonBlock');
    const jsonPre = $('jsonPre');
    const refsSection = $('refsSection');
    const refsList = $('refsList');

    // Edited-file → JSON widgets
    const editedFile = $('editedFile');
    const editedConvertBtn = $('editedConvertBtn');
    const editedDownloadJson = $('editedDownloadJson');

    // Mode toggles
    const reasoningBtn = document.getElementById('reasoningBtn');
    const robotBtn     = document.getElementById('robotBtn');
    let mode = 'robot';

    function setMode(next) {
      mode = next;
      const rb = robotBtn, rs = reasoningBtn;
      rb?.setAttribute('aria-checked', String(next === 'robot'));
      rs?.setAttribute('aria-checked', String(next === 'reasoning'));
      rb?.classList.toggle('is-selected', next === 'robot');
      rs?.classList.toggle('is-selected', next === 'reasoning');
    }
    robotBtn?.addEventListener('click', () => setMode('robot'));
    reasoningBtn?.addEventListener('click', () => setMode('reasoning'));
    setMode('robot');

    const clearBtn = document.getElementById('clearBtn');
    const uplMsg   = document.getElementById('uplMsg');

    clearBtn?.addEventListener('click', async () => {
      try {
        clearBtn.disabled = true;
        if (uplMsg) {
          uplMsg.classList.remove('hidden');
          uplMsg.textContent = 'Clearing…';
        }

        const headers = { 'X-CSRFToken': readCsrfToken() }; 
        const res = await fetch('/clear_uploads', { method: 'POST', headers });

        if (!res.ok) throw new Error('HTTP ' + res.status);
        if (uplMsg) uplMsg.textContent = 'Uploads cleared (vector memory)';

        if (typeof loadUploads === 'function') loadUploads();
      } catch (err) {
        if (uplMsg) uplMsg.textContent = 'Clear failed: ' + err.message;
        console.error('[Clear uploads]', err);
      } finally {
        clearBtn.disabled = false;
      }
    });

    // Helpers
    function readCsrfToken() {
      return document.querySelector('meta[name="csrf-token"]')?.content
          || (document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/)?.[1] || '');
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
      const answerPre    = document.getElementById('answerPre');
      const rationalePre = document.getElementById('rationalePre');
      const refsSection  = document.getElementById('refsSection');
      const refsList     = document.getElementById('refsList');

      const { answer, rationale } = payload || {};
      if (answerPre)    answerPre.textContent    = (answer ?? '').trim() || '(no answer)';
      if (rationalePre) rationalePre.textContent = (rationale ?? '').trim();

      // Accept multiple possible keys for references
      let refs = payload?.refs
              || payload?.references
              || payload?.sources
              || payload?.citations
              || [];

      // Normalize refs into [{title, url}] if possible
      if (typeof refs === 'string') {
        refs = refs.split(/\n+/).map(s => ({title: s.trim()})).filter(x => x.title);
      } else if (refs && typeof refs === 'object' && !Array.isArray(refs)) {
        refs = Object.values(refs);
      }
      if (!Array.isArray(refs)) refs = [];

      if (refs.length && refsSection && refsList) {
        refsSection.classList.remove('hidden');
        refsList.innerHTML = refs.map(r => {
          const title = (r.title || r.name || r.url || r.doi || 'reference').toString();
          const url   = r.url || (r.doi ? `https://doi.org/${r.doi}` : '');
          return url ? `<li><a href="${url}" target="_blank" rel="noopener">${title}</a></li>`
                    : `<li>${title}</li>`;
        }).join('');
      } else if (refsSection && refsList) {
        refsSection.classList.add('hidden');
        refsList.innerHTML = '';
      }
    }

    async function postJSON(url, payload) {
      const headers = { 'Content-Type': 'application/json' };
      const csrf = readCsrfToken();
      if (csrf) headers['X-CSRFToken'] = csrf;
      const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload) });
      const text = await res.text();
      let data; try { data = JSON.parse(text); } catch { data = { raw: text }; }
      return {status: res.status, ok: res.ok, data};
    }
    async function tryEndpoints(endpoints, payload) {
      for (const url of endpoints) {
        try {
          const r = await postJSON(url, payload);
          if (r.status !== 404 && r.status !== 405) return {...r, url};
        } catch (e) {
          // continue
        }
      }
      throw new Error('All endpoints returned 404/405 or failed');
    }

    // ASK flow
    askBtn?.addEventListener('click', async (e) => {
      e.preventDefault();
      const question = (qInput?.value || '').trim();
      if (!question) { setStatus('Please enter a question.'); qInput?.focus(); return; }
      askBtn.disabled = true; setStatus('Asking…');
      try {
        const r = await tryEndpoints(['/ask'], { question /*, mode*/ });
        renderAnswer(r.data);
        setStatus(r.ok ? 'Done.' : `Error ${r.status}`);
      } catch (err) {
        console.error('[ASK]', err);
        setStatus('Request failed. See console.');
      } finally {
        askBtn.disabled = false;
      }
    });

    // Convert answer → JSON
    parseBtn?.addEventListener('click', async () => {
      const answerText = (document.getElementById('answerPre')?.textContent || '').trim();
      if (!answerText || answerText === '(no answer)') {
        setStatus('Nothing to convert. Ask a question first.', true);
        return;
      }
      parseBtn.disabled = true; setStatus('Converting to JSON…');

      async function postJSON(url, body) {
        const headers = {'Content-Type':'application/json'};
        const csrf = readCsrfToken(); if (csrf) headers['X-CSRFToken'] = csrf;
        const res = await fetch(url, {method:'POST', headers, body: JSON.stringify(body)});
        const text = await res.text();
        let data; try { data = JSON.parse(text); } catch { data = {raw: text}; }
        return {res, data};
      }
      async function postTextPlain(url, textBody) {
        const headers = {'Content-Type':'text/plain'};
        const csrf = readCsrfToken(); if (csrf) headers['X-CSRFToken'] = csrf;
        const res = await fetch(url, {method:'POST', headers, body: textBody});
        const text = await res.text();
        let data; try { data = JSON.parse(text); } catch { data = {raw: text}; }
        return {res, data};
      }

      const endpoints = ['/convert', '/parse', '/api/convert'];
      const shapes = [
        (t)=>({ text: t, mode }),
        (t)=>({ answer: t, mode }),
        (t)=>({ markdown: t, mode }),
        (t)=>({ content: t, mode }),
        (t)=>({ raw: t, mode }),
      ];

      try {
        let final = null;

        // Try JSON shapes first
        outer: for (const ep of endpoints) {
          for (const shape of shapes) {
            const {res, data} = await postJSON(ep, shape(answerText));
            if (res.status !== 404 && res.status !== 405 && res.status !== 422) {
              final = {res, data}; break outer;
            }
            // If 200 OK, also accept and break
            if (res.ok) { final = {res, data}; break outer; }
          }
        }

        // Fallback: text/plain to /convert
        if (!final) {
          const {res, data} = await postTextPlain('/convert', answerText);
          if (res.ok || (res.status !== 404 && res.status !== 405)) final = {res, data};
        }

        if (!final) throw new Error('No compatible conversion endpoint/shape found');

        const {res, data} = final;
        const pretty = JSON.stringify(data, null, 2);
        document.getElementById('jsonBlock')?.classList.remove('hidden');
        const jp = document.getElementById('jsonPre'); if (jp) jp.textContent = pretty;
        setStatus(res.ok ? 'Converted.' : `Convert error ${res.status}`);
      } catch (err) {
        console.error('[CONVERT answer]', err);
        setStatus('Convert failed. See console.');
      } finally {
        parseBtn.disabled = false;
      }
    });

    // Export answer → .txt
    saveTxtBtn?.addEventListener('click', () => {
      const txt = (answerPre?.textContent || '').trim();
      if (!txt || txt === '(no answer)') { setStatus('Nothing to export.', true); return; }
      const blob = new Blob([txt], {type: 'text/plain'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'answer.txt';
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
    });

    // Edited file → JSON
    editedConvertBtn?.addEventListener('click', async () => {
      const f = editedFile?.files && editedFile.files[0];
      if (!f) { setStatus('Choose a TXT/MD file to convert.', true); return; }
      editedConvertBtn.disabled = true; setStatus('Converting file…');
      try {
        const text = await f.text();
        const r = await tryEndpoints(['/convert', '/parse', '/api/convert'], { text, mode });
        const pretty = JSON.stringify(r.data, null, 2);
        // Show a download link with the JSON
        const blob = new Blob([pretty], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        if (editedDownloadJson) {
          editedDownloadJson.classList.remove('hidden');
          editedDownloadJson.href = url;
          editedDownloadJson.download = (f.name.replace(/\.(txt|md)$/i, '') || 'converted') + '.json';
        }
        // Also display in the main JSON block for visibility
        if (jsonPre && jsonBlock) {
          jsonBlock.classList.remove('hidden');
          jsonPre.textContent = pretty;
        }
        setStatus(r.ok ? 'File converted.' : `Convert error ${r.status}`);
      } catch (err) {
        console.error('[CONVERT file]', err);
        setStatus('File convert failed. See console.');
      } finally {
        editedConvertBtn.disabled = false;
      }
    });

    // UX niceties
    qInput?.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') askBtn?.click();
    });

    // Footer year
    const yr = $('year'); if (yr) yr.textContent = String(new Date().getFullYear());
  });
})();
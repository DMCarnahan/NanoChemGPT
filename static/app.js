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
  const jsonBlock = $('jsonPre'); // Should match HTML id for JSON output
  const uplList = $('uplList');
  const uplMsg = $('uplMsg');
  const historyBtn = $('historyBtn');
  const historyList = $('historyList');
  const modeRobot = $('modeRobot');
  const modeReason = $('modeReason');

  let mode = 'robot';

  // Mode toggle
  modeRobot?.addEventListener('click', () => {
    mode = 'robot';
    modeRobot.setAttribute('aria-checked', 'true');
    modeReason?.setAttribute('aria-checked', 'false');
  });
  modeReason?.addEventListener('click', () => {
    mode = 'reasoning';
    modeReason.setAttribute('aria-checked', 'true');
    modeRobot?.setAttribute('aria-checked', 'false');
  });

  function readCsrfToken() {
    return document.querySelector('meta[name="csrf-token"]')?.content ||
      (document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/)?.[1]);
  }

  // Ask button
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
      if (Array.isArray(data.references) && data.references.length) {
        refsSection.classList.remove('hidden');
        refsList.innerHTML = data.references.map(r => `<li>${r.title || 'ref'}</li>`).join('');
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

  // Parse button
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
      if (data.ok && data.data) {
        const pretty = JSON.stringify(data.data, null, 2);
        jsonBlock.textContent = pretty;

        const blob = new Blob([pretty], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'converted.json';
        a.click();
        URL.revokeObjectURL(url);
      } else {
        jsonBlock.textContent = 'Error: ' + (data.error || 'unknown');
      }

    } catch (err) {
      console.error(err);
      jsonBlock.textContent = 'Request failed';
    } finally {
      parseBtn.disabled = false;
      parseBtn.textContent = 'Convert → JSON';
    }
  });

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
  historyBtn?.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/history');
      const data = await res.json();
      const items = data.items || [];
      historyList.innerHTML = items.map(r => `<li>${r.question || ''}</li>`).join('');
    } catch (err) {
      console.error(err);
    }
  });

});
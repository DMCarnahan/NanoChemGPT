/* --- Global references renderer (ES5-safe) --- */
window.renderRefsFromData = function(data){
  try { console.debug('[refs]', data); } catch(e){}
  var refsSection = document.getElementById('refsSection');
  var refsList    = document.getElementById('refsList');
  if (!refsSection || !refsList) return;

  // clear old
  refsList.innerHTML = '';
  var oldPre = refsSection.querySelector ? refsSection.querySelector('.refs-pre') : null;
  if (oldPre && oldPre.parentNode) oldPre.parentNode.removeChild(oldPre);

  // Accept a handful of alias keys
  var block = (data && (data.references_block || data.referencesBlock || data.references_str || data.referencesText || '')) || '';
  var arr   = (data && (data.references || data.referencesAll || null)) || [];

  if (typeof block === 'string' && block.trim().length > 0) {
    var pre = document.createElement('pre');
    pre.className = 'refs-pre';
    pre.textContent = block.trim();
    refsSection.appendChild(pre);
    if (refsSection.classList) refsSection.classList.remove('hidden');
    return;
  }

  if (Object.prototype.toString.call(arr) === '[object Array]' && arr.length > 0) {
    var html = arr.map(function(r, i){
      r = r || {};
      var authors = (Array.isArray(r.authors) ? r.authors.join(', ') : (r.authors || ''));
      var title = r.title || ('Reference ' + (i+1));
      var journal = (r.biblio && r.biblio.journal) || r.journal || '';
      var year = r.year || (r.biblio && r.biblio.year) || '';
      var volume = (r.biblio && r.biblio.volume) || r.volume || '';
      var pages = (r.biblio && r.biblio.pages) || r.pages || '';
      var doiUrl = r.doi ? ('https://doi.org/' + r.doi) : '';
      var url = r.url || doiUrl || '#';
      var acs = authors + '. <i>' + title + '</i>. <b>' + journal + '</b> ' + year + ', ' + volume + ', ' + pages + '. ' +
                '<a href="' + url + '" target="_blank" rel="noopener">' + (r.doi || url) + '</a>';
      return '<li>' + acs + '</li>';
    }).join('');
    refsList.innerHTML = html;
    if (refsSection.classList) refsSection.classList.remove('hidden');
    return;
  }

  // fallback to data.refs
  if (data && Object.prototype.toString.call(data.refs) === '[object Array]' && data.refs.length > 0) {
    data.references = data.refs;
    return window.renderRefsFromData(data);
  }

  // If nothing: show a tiny hint so you can see it's alive
  var hint = document.createElement('div');
  hint.className = 'small';
  hint.textContent = 'No references in response.';
  refsSection.appendChild(hint);
  if (refsSection.classList) refsSection.classList.remove('hidden');
};
/* --- End global references renderer --- */


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

    // Nothing to show
    if (refsSection && refsSection.classList) refsSection.classList.add('hidden');
  // Upload button
  /**
   * Handles the Upload button click: uploads a file to the backend.
   */
  if (uploadBtn) {
    uploadBtn.addEventListener('click', async () => {
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
  }

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
            window.renderRefsFromData(data);
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
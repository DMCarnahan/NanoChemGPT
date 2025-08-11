(function () {
  document.addEventListener('DOMContentLoaded', () => {
    const $  = (id) => document.getElementById(id);
    const by = (sel) => document.querySelector(sel);

    // --- tiny CSS to show selected mode ---
    (function injectModeCSS(){
      const css = `.btn.tertiary.is-selected{border-color:var(--brand);color:var(--brand);background:color-mix(in srgb,var(--brand),transparent 90%)} .segmented, .segmented .btn{pointer-events:auto}`;
      const el = document.createElement('style'); el.textContent = css; document.head.appendChild(el);
    })();

    function readCsrfToken() {
      return by('meta[name="csrf-token"]')?.content
          || (document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/)?.[1] || '');
    }
    function setStatus(msg, show = true) {
      const askMsg = $('askMsg'); if (!askMsg) return;
      if (show) { askMsg.classList.remove('hidden'); askMsg.textContent = msg; }
      else { askMsg.textContent = msg || ''; askMsg.classList.add('hidden'); }
    }

    // --- Extract and flatten Synthesis Protocol into Key: Value lines for /parse ---
    function extractProtocolForParse(fullText) {
      if (!fullText) return "";
      let txt = String(fullText);

      // 1) Trim to Synthesis Protocol block if present
      const h = txt.indexOf("## Synthesis Protocol");
      if (h >= 0) {
        txt = txt.slice(h + "## Synthesis Protocol".length);
      }

      // Stop at a rationale/code fence if present
      const stopIdxs = [];
      const fenceIdx = txt.indexOf("```");
      if (fenceIdx >= 0) stopIdxs.push(fenceIdx);
      const ratH = txt.search(/\n#{1,3}\s*(rationale|reasoning)\b/i);
      if (ratH >= 0) stopIdxs.push(ratH);
      const stopAt = stopIdxs.length ? Math.min(...stopIdxs) : -1;
      if (stopAt > 0) txt = txt.slice(0, stopAt);

      const lines = txt.split(/\r?\n/);

      // 2) Parse sections like: "1. **Hardware & Glassware**:" etc.
      const sections = {};
      const titleMap = {
        "hardware & glassware": "Hardware & Glassware",
        "materials": "Materials",
        "procedure": "Procedure",
      };

      let current = null;
      const headingRE = /^\s*\d+\.\s*\*\*(.+?)\*\*\s*:?\s*$/i;
      for (let i=0; i<lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const m = line.match(headingRE);
        if (m) {
          const raw = m[1].toLowerCase();
          const key = titleMap[raw] || m[1];
          current = key;
          if (!sections[current]) sections[current] = [];
          continue;
        }
        if (!current) continue;
        // collect content lines under current section
        sections[current].push(line);
      }

      // If no sections detected, fall back: strip leading markdown symbols and return plain text
      if (Object.keys(sections).length === 0) {
        return lines
          .map(l => l.replace(/^\s*[-*•]\s*/, '').replace(/\[(CTX|GEN|PARSED|DB|\d+)\]/g, '').trim())
          .filter(Boolean)
          .join("\n");
      }

      // 3) Clean bullets/numbering/tags and flatten each section into a single Key: value line
      function cleanJoin(arr) {
        return arr.map(s => s
            .replace(/^\s*[-*•]\s*/, '')        // bullets
            .replace(/^\s*\d+\.\s*/, '')        // numbers "1. "
            .replace(/\s*\[(CTX|GEN|PARSED|DB|\d+)\]\s*/g, '') // tags
            .replace(/\*\*(.*?)\*\*/g, '$1')    // bold
            .trim()
          )
          .filter(Boolean)
          .join('; ');
      }

      const out = [];
      for (const key of Object.keys(sections)) {
        const val = cleanJoin(sections[key]);
        if (val) out.push(`${key}: ${val}`);
      }
      return out.join("\n");
    }

    function renderAnswer(payload) {
      const answerPre    = $('answerPre');
      const rationalePre = $('rationalePre');
      const refsSection  = $('refsSection');
      const refsList     = $('refsList');
      const { answer, rationale } = payload || {};
      if (answerPre)    answerPre.textContent    = (answer ?? '').trim() || '(no answer)';
      if (rationalePre) rationalePre.textContent = (rationale ?? '').trim();

      // Accept refs under refs|references|sources|citations
      let refs = payload?.refs || payload?.references || payload?.sources || payload?.citations || [];
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
          return url ? `<li><a href="${url}" target="_blank" rel="noopener">${title}</a></li>` : `<li>${title}</li>`;
        }).join('');
      } else if (refsSection && refsList) {
        refsSection.classList.add('hidden');
        refsList.innerHTML = '';
      }
    }

    async function postJSON(url, obj, headersExtra={}) {
      const headers = {'Content-Type':'application/json', ...headersExtra};
      const csrf = readCsrfToken(); if (csrf) headers['X-CSRFToken'] = csrf;
      const res = await fetch(url, {method:'POST', headers, body: JSON.stringify(obj)});
      const text = await res.text();
      let data; try { data = JSON.parse(text); } catch { data = {raw: text}; }
      return {res, data, url, sent: obj};
    }

    // Elements
    const askBtn = $('askBtn');
    const qInput = $('question');
    const parseBtn = $('parseBtn');
    const saveTxtBtn = $('saveTxtBtn');
    const jsonBlock = $('jsonBlock');
    const jsonPre = $('jsonPre');
    const clearBtn = $('clearBtn');
    const uplMsg = $('uplMsg');
    const editedFile = $('editedFile');
    const editedConvertBtn = $('editedConvertBtn');
    const editedDownloadJson = $('editedDownloadJson');
    const robotBtn = $('robotBtn');
    const reasoningBtn = $('reasoningBtn');

    // Mode
    let mode = 'robot';
    function setMode(next) {
      mode = next;
      robotBtn?.setAttribute('aria-checked', String(next === 'robot'));
      reasoningBtn?.setAttribute('aria-checked', String(next === 'reasoning'));
      robotBtn?.classList.toggle('is-selected', next === 'robot');
      reasoningBtn?.classList.toggle('is-selected', next === 'reasoning');

      // Disable parse when in reasoning mode (no structured protocol present)
      const parseBtn = document.getElementById('parseBtn');
      if (parseBtn) {
        parseBtn.disabled = (next === 'reasoning');
        parseBtn.title = (next === 'reasoning')
          ? 'Parsing is only available for structured protocols (Robot mode).'
          : '';
      }
    }
    robotBtn?.removeAttribute('disabled');
    reasoningBtn?.removeAttribute('disabled');
    robotBtn?.addEventListener('click', () => setMode('robot'));
    reasoningBtn?.addEventListener('click', () => setMode('reasoning'));
    setMode('robot'); // default on load

    // Ask
    askBtn?.addEventListener('click', async (e) => {
      e.preventDefault();
      const question = (qInput?.value || '').trim();
      if (!question) { setStatus('Please enter a question.'); qInput?.focus(); return; }
      askBtn.disabled = true; setStatus('Asking…');
      try {
        const {res, data} = await postJSON('/ask', { question, mode, want_inline_citations: true });
        renderAnswer(data);
        setStatus(res.ok ? 'Done.' : `Error ${res.status}`);
      } catch (err) {
        console.error('[ASK]', err); setStatus('Request failed. See console.');
      } finally { askBtn.disabled = false; }
    });

    // Convert displayed answer → JSON via /parse (with extraction)
    parseBtn?.addEventListener('click', async () => {
      const full = ($('answerPre')?.textContent || '').trim();
      if (!full || full === '(no answer)') { setStatus('Nothing to convert. Ask first.', true); return; }
      const text = extractProtocolForParse(full);
      if (!text) { setStatus('Could not extract protocol section.', true); return; }

      parseBtn.disabled = true; setStatus('Converting to JSON…');
      try {
        const {res, data} = await postJSON('/parse', { text });
        if (jsonBlock && jsonPre) {
          jsonBlock.classList.remove('hidden');
          jsonPre.textContent = JSON.stringify(data, null, 2);
        }
        if (!res.ok || data?.ok === False || data?.ok === false) {
          const code = res.status;
          const msg = (data && (data.error || data.message)) || 'Parser error';
          setStatus(`Convert error ${code}: ${msg}`);
        } else {
          setStatus('Converted.');
        }
      } catch (err) {
        console.error('[PARSE]', err); setStatus('Convert failed. See console.');
      } finally { parseBtn.disabled = false; }
    });

    // Export answer → .txt
    saveTxtBtn?.addEventListener('click', () => {
      const txt = ($('answerPre')?.textContent || '').trim();
      if (!txt || txt === '(no answer)') { setStatus('Nothing to export.', true); return; }
      const blob = new Blob([txt], {type:'text/plain'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = 'answer.txt';
      document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    });

    // Edited file → JSON (uses /parse with extraction too if needed)
    editedConvertBtn?.addEventListener('click', async () => {
      const f = editedFile?.files && editedFile.files[0];
      if (!f) { setStatus('Choose a TXT/MD file to convert.', true); return; }
      editedConvertBtn.disabled = true; setStatus('Converting file…');
      try {
        let text = await f.text();
        // If the edited file contains the same Synthesis Protocol markdown, extract it
        if (/##\s*Synthesis Protocol/.test(text)) {
          text = extractProtocolForParse(text);
        }
        const {res, data} = await postJSON('/parse', { text });
        const pretty = JSON.stringify(data, null, 2);
        if (jsonBlock && jsonPre) { jsonBlock.classList.remove('hidden'); jsonPre.textContent = pretty; }
        if (editedDownloadJson) {
          const blob = new Blob([pretty], {type:'application/json'});
          const url = URL.createObjectURL(blob);
          editedDownloadJson.classList.remove('hidden');
          editedDownloadJson.href = url;
          editedDownloadJson.download = (f.name.replace(/\.(txt|md)$/i,'') || 'converted') + '.json';
        }
        if (!res.ok || data?.ok === False || data?.ok === false) {
          const code = res.status;
          const msg = (data && (data.error || data.message)) || 'Parser error';
          setStatus(`Convert error ${code}: ${msg}`);
        } else {
          setStatus('File converted.');
        }
      } catch (err) {
        console.error('[PARSE file]', err); setStatus('File convert failed. See console.');
      } finally { editedConvertBtn.disabled = false; }
    });

    // Clear uploads (CSRF)
    const clearBtn = $('clearBtn'); const uplMsg = $('uplMsg');
    clearBtn?.addEventListener('click', async () => {
      try {
        clearBtn.disabled = true;
        if (uplMsg) { uplMsg.classList.remove('hidden'); uplMsg.textContent = 'Clearing…'; }
        const headers = { 'X-CSRFToken': readCsrfToken() };
        const res = await fetch('/clear_uploads', { method:'POST', headers });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        if (uplMsg) uplMsg.textContent = 'Uploads cleared (vector memory)';
      } catch (err) {
        console.error('[Clear uploads]', err);
        if (uplMsg) uplMsg.textContent = 'Clear failed: ' + err.message;
      } finally { clearBtn.disabled = false; }
    });

    // Keyboard shortcut
    const qInput = $('question');
    qInput?.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') $('askBtn')?.click();
    });

    // Footer year
    const yr = $('year'); if (yr) yr.textContent = String(new Date().getFullYear());
  });
})();
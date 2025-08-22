(function() {
  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function parseJSONPre(jsonPre) {
    if (!jsonPre) return null;
    const raw = jsonPre.textContent || "";
    if (!raw.trim()) return null;
    try {
      return JSON.parse(raw);
    } catch (e) {
      const first = raw.indexOf("{");
      const last = raw.lastIndexOf("}");
      if (first >= 0 && last > first) {
        try { return JSON.parse(raw.slice(first, last + 1)); } catch {}
      }
      return null;
    }
  }

  function buildRefsList(refs, usedSet, showUnused) {
    const ul = document.createElement("ul");
    ul.id = "refsEnhancedList";
    ul.style.listStyle = "none";
    ul.style.paddingLeft = "0";

    (refs || []).forEach((r, idx) => {
      const n = idx + 1;
      const used = usedSet.has(n);
      if (!used && !showUnused) return;

      const li = document.createElement("li");
      li.dataset.index = String(n);
      li.style.marginBottom = "6px";
      if (!used) li.style.opacity = "0.6";

      const aURL = (r.url && typeof r.url === "string" && r.url.trim()) ? r.url
                 : (r.doi && typeof r.doi === "string" && r.doi.trim() ? ("https://doi.org/" + r.doi) : "");

      const title = esc(r.title || "(no title)");
      const year = esc(r.year || "");
      const prefix = used ? "[" + n + "]" : "[" + n + " \u2022 unused]";

      const span = document.createElement("span");
      span.innerHTML = `<strong>${esc(prefix)}</strong> ${title}${year ? " ("+year+")" : ""}`;

      li.appendChild(span);
      if (aURL) {
        const link = document.createElement("a");
        link.href = aURL;
        link.target = "_blank";
        link.rel = "noopener";
        link.textContent = " — link";
        link.style.marginLeft = "6px";
        li.appendChild(link);
      }
      ul.appendChild(li);
    });

    if (!ul.children.length) {
      const empty = document.createElement("div");
      empty.textContent = "No references available.";
      empty.className = "text-muted";
      ul.appendChild(empty);
    }
    return ul;
  }

  function renderEnhancedRefs(data) {
    const refsSection = document.getElementById("refsSection");
    const refsList = document.getElementById("refsList");
    if (!refsSection || !refsList) return;

    // Extract data
    const refs = Array.isArray(data && data.refs) ? data.refs : [];
    const usedArr = Array.isArray(data && data.used_ref_indexes) ? data.used_ref_indexes : [];
    const references_block = (data && typeof data.references_block === "string") ? data.references_block : "";
    const usedSet = new Set(usedArr);

    // Remove prior enhanced container if present
    const prior = document.getElementById("refsEnhancedContainer");
    if (prior) prior.remove();

    const container = document.createElement("div");
    container.id = "refsEnhancedContainer";
    container.style.marginTop = "8px";

    if (references_block && references_block.trim()) {
      const details = document.createElement("details");
      details.open = true;

      const summary = document.createElement("summary");
      summary.textContent = "References (ACS block)";
      summary.style.cursor = "pointer";
      summary.style.marginBottom = "6px";
      details.appendChild(summary);

      const pre = document.createElement("pre");
      pre.className = "refsBlock";
      pre.style.whiteSpace = "pre-wrap";
      pre.style.margin = 0;
      pre.textContent = references_block.trim();
      details.appendChild(pre);

      container.appendChild(details);
    }

    if (refs.length) {
      const toggleRow = document.createElement("div");
      toggleRow.id = "refsToggleRow";
      toggleRow.style.display = "flex";
      toggleRow.style.alignItems = "center";
      toggleRow.style.gap = "8px";
      toggleRow.style.margin = "8px 0";

      const usedCount = usedSet.size;
      const totalCount = refs.length;

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.id = "toggleUnusedRefs";
      cb.checked = false; // default: hide unused

      const label = document.createElement("label");
      label.setAttribute("for", "toggleUnusedRefs");
      label.textContent = `Show unused refs (${Math.max(0, totalCount - usedCount)})`;

      toggleRow.appendChild(cb);
      toggleRow.appendChild(label);
      container.appendChild(toggleRow);

      const listMount = document.createElement("div");
      listMount.id = "refsEnhancedMount";
      container.appendChild(listMount);

      function draw() {
        listMount.innerHTML = "";
        listMount.appendChild(buildRefsList(refs, usedSet, cb.checked));
      }

      cb.addEventListener("change", draw);
      draw();
    }

    refsList.insertAdjacentElement("afterend", container);
  }

  function tryRender() {
    const jsonPre = document.getElementById("jsonPre");
    const data = parseJSONPre(jsonPre);
    if (data) renderEnhancedRefs(data);
  }

  document.addEventListener("DOMContentLoaded", () => {
    // Initial attempt in case JSON is already present
    tryRender();

    // Re-render whenever the JSON payload changes
    const jsonPre = document.getElementById("jsonPre");
    if (jsonPre) {
      const mo = new MutationObserver(() => { tryRender(); });
      mo.observe(jsonPre, { childList: true, subtree: true, characterData: true });
    }

    // Also observe the refsList container as a fallback signal that /ask completed
    const refsList = document.getElementById("refsList");
    if (refsList) {
      const mo2 = new MutationObserver(() => { tryRender(); });
      mo2.observe(refsList, { childList: true, subtree: true });
    }
  });
})();

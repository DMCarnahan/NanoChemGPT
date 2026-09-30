(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.NanoChemMarkdown = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function renderInline(value) {
    // Split out inline-code spans first so Markdown inside code stays literal.
    return String(value == null ? '' : value)
      .split(/(`[^`\n]+`)/g)
      .map((part) => {
        if (part.length >= 2 && part.startsWith('`') && part.endsWith('`')) {
          return `<code>${escapeHtml(part.slice(1, -1))}</code>`;
        }

        let safe = escapeHtml(part);
        safe = safe.replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>');
        safe = safe.replace(/__([^_\n]+?)__/g, '<strong>$1</strong>');
        safe = safe.replace(
          /(^|[\s(])\*([^*\n]+?)\*(?=$|[\s).,;:!?])/g,
          '$1<em>$2</em>'
        );
        safe = safe.replace(
          /(^|[\s(])_([^_\n]+?)_(?=$|[\s).,;:!?])/g,
          '$1<em>$2</em>'
        );
        return safe;
      })
      .join('');
  }

  function isBlockStart(line) {
    return (
      /^\s*$/.test(line) ||
      /^\s*```/.test(line) ||
      /^\s{0,3}#{1,6}\s+/.test(line) ||
      /^\s{0,3}(?:[-+*]\s+|\d+[.)]\s+)/.test(line) ||
      /^\s{0,3}>\s?/.test(line) ||
      /^\s{0,3}(?:---+|___+|\*\*\*+)\s*$/.test(line)
    );
  }

  function renderMarkdown(value) {
    const source = String(value == null ? '' : value).replace(/\r\n?/g, '\n');
    const lines = source.split('\n');
    const output = [];
    let index = 0;

    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        index += 1;
        continue;
      }

      const fence = line.match(/^\s*```([A-Za-z0-9_-]*)\s*$/);
      if (fence) {
        const code = [];
        index += 1;
        while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) {
          code.push(lines[index]);
          index += 1;
        }
        if (index < lines.length) index += 1;
        const language = fence[1]
          ? ` class="language-${escapeHtml(fence[1])}"`
          : '';
        output.push(`<pre><code${language}>${escapeHtml(code.join('\n'))}</code></pre>`);
        continue;
      }

      const heading = line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
      if (heading) {
        const level = heading[1].length;
        output.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
        index += 1;
        continue;
      }

      if (/^\s{0,3}(?:---+|___+|\*\*\*+)\s*$/.test(line)) {
        output.push('<hr>');
        index += 1;
        continue;
      }

      const quote = line.match(/^\s{0,3}>\s?(.*)$/);
      if (quote) {
        const quoted = [];
        while (index < lines.length) {
          const match = lines[index].match(/^\s{0,3}>\s?(.*)$/);
          if (!match) break;
          quoted.push(renderInline(match[1]));
          index += 1;
        }
        output.push(`<blockquote>${quoted.join('<br>')}</blockquote>`);
        continue;
      }

      const listItem = line.match(/^\s{0,3}([-+*]|(\d+)[.)])\s+(.+)$/);
      if (listItem) {
        const ordered = Boolean(listItem[2]);
        const tag = ordered ? 'ol' : 'ul';
        const start = ordered && Number(listItem[2]) !== 1
          ? ` start="${Number(listItem[2])}"`
          : '';
        const items = [];

        while (index < lines.length) {
          const match = lines[index].match(/^\s{0,3}([-+*]|(\d+)[.)])\s+(.+)$/);
          if (!match || Boolean(match[2]) !== ordered) break;
          items.push(`<li>${renderInline(match[3])}</li>`);
          index += 1;
        }

        output.push(`<${tag}${start}>${items.join('')}</${tag}>`);
        continue;
      }

      const paragraph = [];
      while (index < lines.length && !isBlockStart(lines[index])) {
        paragraph.push(renderInline(lines[index]));
        index += 1;
      }
      // A non-empty line always advances, but keep this guard for malformed input.
      if (!paragraph.length) {
        paragraph.push(renderInline(line));
        index += 1;
      }
      output.push(`<p>${paragraph.join('<br>')}</p>`);
    }

    return output.join('\n');
  }

  return { escapeHtml, renderInline, renderMarkdown };
});

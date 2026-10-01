import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_markdown_renderer_formats_bold_and_escapes_html():
    if not shutil.which("node"):
        pytest.skip("Node.js is not available")

    sample = (
        "## Synthesis Protocol:\n"
        "1. **Hardware & Glassware**:\n"
        "- flask\n\n"
        "<script>alert('xss')</script>"
    )
    script = (
        "const {renderMarkdown}=require('./static/markdown.js');"
        "process.stdout.write(renderMarkdown(process.argv[1]));"
    )
    result = subprocess.run(
        ["node", "-e", script, sample],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "<h2>Synthesis Protocol:</h2>" in result.stdout
    assert "<strong>Hardware &amp; Glassware</strong>" in result.stdout
    assert "**Hardware" not in result.stdout
    assert "&lt;script&gt;" in result.stdout
    assert "<script>" not in result.stdout


def test_page_loads_renderer_before_app_and_uses_wrapped_reference_block():
    template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")

    assert template.index("filename='markdown.js'") < template.index(
        "filename='app.js'"
    )
    assert 'id="answerPre" class="pre markdown-output"' in template
    assert 'id="refsBlock"' in template
    assert "#refsBlock,.refs-pre,.refsBlock" in template
    assert "white-space:pre-wrap" in template


def test_attachments_survive_followups_and_require_explicit_removal():
    if not shutil.which("node"):
        pytest.skip("Node.js is not available")
    result = subprocess.run(
        ["node", "tests/frontend_attachment_flow.cjs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

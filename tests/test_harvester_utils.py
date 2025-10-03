import os
import tempfile
from harvester import utils as hu


def test_ensure_dir_and_write_json_and_safe_slug(tmp_path):
    d = tmp_path / "subdir"
    hu.ensure_dir(d)
    p = tmp_path / "out.json"
    hu.write_json(p, {"a": 1})
    assert p.exists()
    slug = hu.safe_slug("A test/title: with ? chars")
    assert " " not in slug

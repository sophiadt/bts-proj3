"""tools/check_changes.py: the deletion guard run on pull requests.

Each test builds a small throwaway git repository, so it needs only git.

    .venv\\Scripts\\python -m pytest tests/test_check_changes.py -v
"""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_changes.py"
spec = importlib.util.spec_from_file_location("check_changes", TOOL)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(repo, *args):
    return subprocess.run(("git", "-c", "user.name=test", "-c", "user.email=test@example.com",
                           "-c", "core.autocrlf=false") + args,
                          cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo, message):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def repo(tmp_path):
    """A repo shaped like this one: key files, a pipeline/ folder, 15 notes."""
    _git(tmp_path, "init", "-q")
    for name in ("CLAUDE.md", "README.md", ".gitignore", "pyproject.toml"):
        (tmp_path / name).write_text(name + "\n")
    (tmp_path / "pipeline").mkdir()
    for i in range(3):
        (tmp_path / "pipeline" / "step{0}.py".format(i)).write_text("x = {0}\n".format(i))
    (tmp_path / "notes").mkdir()
    for i in range(15):
        (tmp_path / "notes" / "n{0:02d}.md".format(i)).write_text("note {0}\n".format(i) * 20)
    base = _commit(tmp_path, "base")
    return tmp_path, base


def _check(repo_path, base, monkeypatch, allow=False):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    if allow:
        monkeypatch.setenv("ALLOW_DELETIONS", "1")
    else:
        monkeypatch.delenv("ALLOW_DELETIONS", raising=False)
    return cc.main([base, "HEAD"], cwd=str(repo_path))


def test_mass_deletion_fails(repo, monkeypatch, capsys):
    path, base = repo
    shutil.rmtree(path / "notes")                      # 15 files, like the 2 Nov commit made from a partial copy
    _commit(path, "oops")
    assert _check(path, base, monkeypatch) == 1
    assert "15 files deleted" in capsys.readouterr().out


def test_deleting_a_key_file_fails(repo, monkeypatch, capsys):
    path, base = repo
    (path / "CLAUDE.md").unlink()
    _commit(path, "remove CLAUDE.md")
    assert _check(path, base, monkeypatch) == 1
    assert "key file deleted: CLAUDE.md" in capsys.readouterr().out


def test_emptying_a_key_folder_fails(repo, monkeypatch, capsys):
    path, base = repo
    shutil.rmtree(path / "pipeline")                   # only 3 files, but the whole folder
    _commit(path, "no pipeline")
    assert _check(path, base, monkeypatch) == 1
    assert "folder emptied: pipeline/" in capsys.readouterr().out


def test_moving_a_folder_is_not_a_deletion(repo, monkeypatch, capsys):
    path, base = repo
    _git(path, "mv", "notes", "docs")
    _commit(path, "move notes to docs")
    assert _check(path, base, monkeypatch) == 0
    out = capsys.readouterr().out
    assert "15 renamed" in out and "-> OK" in out


def test_ordinary_changes_pass(repo, monkeypatch):
    path, base = repo
    (path / "pipeline" / "step0.py").write_text("x = 42\n")
    (path / "pipeline" / "new.py").write_text("y = 1\n")
    (path / "notes" / "n00.md").unlink()              # one deletion is fine
    _commit(path, "edit")
    assert _check(path, base, monkeypatch) == 0


def test_maya_scene_under_template_export_fails(repo, monkeypatch, capsys):
    path, base = repo
    (path / "template_export" / "scenes").mkdir(parents=True)
    (path / "template_export" / "scenes" / "s01_ui.mb").write_bytes(b"\0" * 64)
    _commit(path, "add a fitted scene")
    assert _check(path, base, monkeypatch) == 1
    assert "Maya scene added under template_export/" in capsys.readouterr().out


def test_allow_deletions_accepts_an_intended_cleanup(repo, monkeypatch, capsys):
    path, base = repo
    shutil.rmtree(path / "notes")
    _commit(path, "clean up old notes on purpose")
    assert _check(path, base, monkeypatch, allow=True) == 0
    assert "accepted: ALLOW_DELETIONS" in capsys.readouterr().out


def test_size_rules():
    changes = [("A", "big.bin", None), ("M", "medium.npz", None), ("A", "small.txt", None)]
    sizes = {"big.bin": 120_000_000, "medium.npz": 60_000_000, "small.txt": 10}
    errors, warnings, deleted = cc.evaluate(changes, list(sizes), sizes)
    assert any("big.bin" in e for e in errors) and not deleted
    assert any("medium.npz" in w for w in warnings) and not any("small.txt" in m for m in errors + warnings)


def test_parse_name_status_handles_renames_and_spaces():
    raw = b"M\0README.md\0R087\0Capture Data/a.json\0captures/s01/a.json\0D\0TrueDepth App/x.swift\0"
    assert cc.parse_name_status(raw) == [("M", "README.md", None),
                                         ("R", "Capture Data/a.json", "captures/s01/a.json"),
                                         ("D", "TrueDepth App/x.swift", None)]

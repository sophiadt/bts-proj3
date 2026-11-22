"""Capture naming, the capture.json manifest and tools/ingest_capture.py.

numpy only for the synthetic captures (pipeline/synthetic.py); the ingest tool
itself must stay standard-library only. Runs under .venv or mayapy:

    .venv\\Scripts\\python -m pytest tests/test_captures.py -v
"""

import ast
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

import config
from pipeline import capture as cap
from pipeline import synthetic
from tools import ingest_capture as ic

T0 = 1793295000
STAMP = ic.utc_stamp(T0)             # 20261029T173000Z


@pytest.fixture(scope="module")
def guided_src(tmp_path_factory):
    """One synthetic guided capture in the app's flat layout, with timestamps."""
    folder = tmp_path_factory.mktemp("synthetic") / "face_1793295000"
    synthetic.make_guided(folder)
    for k, view in enumerate(("front", "left", "right")):
        meta_path = folder / (view + "_meta.json")
        meta = json.loads(meta_path.read_text())
        meta["timestamp"] = T0 + 6 * k
        meta_path.write_text(json.dumps(meta))
    return folder


def copy_of(src, dest):
    shutil.copytree(src, dest)
    return dest


def set_meta(folder, **fields):
    for meta_path in Path(folder).glob("*meta.json"):
        meta = json.loads(meta_path.read_text())
        meta.update(fields)
        meta_path.write_text(json.dumps(meta))


def run(root, *argv):
    return ic.main(list(argv) + ["--root", str(root)])


# --- naming and lookup ----------------------------------------------------------

@pytest.mark.parametrize("cid,ok", [
    ("s01_20261110T212937Z_burst", True),
    ("s123_20261030T170049Z_guided", True),
    ("s01_20261110T212937Z_record", True),
    ("S01_20261110T212937Z_burst", False),          # subject codes are lower case
    ("s1_20261110T212937Z_burst", False),           # at least two digits
    ("william_20261110T212937Z_burst", False),      # names never go in the repo
    ("s01_20261110T212937_burst", False),           # UTC 'Z' required
    ("s01_20261110T212937Z_video", False),          # unknown mode
])
def test_capture_id_pattern(cid, ok):
    assert bool(re.match(config.CAPTURE_ID_RE, cid)) is ok


def test_resolve_capture_by_path_and_id(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CAPTURES_DIR", tmp_path)
    folder = tmp_path / "s07" / "s07_20261110T212937Z_burst"
    folder.mkdir(parents=True)
    assert cap.resolve_capture(str(folder)) == folder
    assert cap.resolve_capture("s07_20261110T212937Z_burst") == folder
    with pytest.raises(FileNotFoundError):
        cap.resolve_capture("s07_20261111T000000Z_burst")
    with pytest.raises(FileNotFoundError):
        cap.resolve_capture("not-a-capture")


def test_manifest_present_and_missing(tmp_path):
    assert cap.read_manifest(tmp_path) is None
    assert cap.capture_id(tmp_path) == tmp_path.name
    (tmp_path / "capture.json").write_text(json.dumps({"capture_id": "s01_20261110T212937Z_burst"}))
    assert cap.capture_id(tmp_path) == "s01_20261110T212937Z_burst"


def test_load_capture_ignores_extras(guided_src, tmp_path):
    folder = copy_of(guided_src, tmp_path / "c")
    (folder / "extras").mkdir()
    shutil.copy(folder / "front_landmarks_3d_mm.json", folder / "extras" / "front_landmarks_3d_mm.json")
    assert sorted(cap.load_capture(folder)) == ["front", "left", "right"]


# --- ingest: layouts ------------------------------------------------------------

def test_ingest_flat_folder(guided_src, tmp_path):
    root = tmp_path / "captures"
    assert run(root, str(guided_src), "--subject", "s01") == 0
    out = root / "s01" / ("s01_" + STAMP + "_guided")
    manifest = json.loads((out / "capture.json").read_text())
    assert manifest["capture_id"] == out.name and manifest["mode"] == "guided"
    assert manifest["views"] == ["front", "left", "right"] and manifest["labels"] == "unchecked"
    assert manifest["created_utc"] == "2026-10-29T17:30:00Z" and STAMP == "20261029T173000Z"
    assert sorted(cap.load_capture(out)) == ["front", "left", "right"]
    assert guided_src.exists()                       # explicit sources are never moved


@pytest.mark.parametrize("prefixed", [True, False])
def test_ingest_view_subfolders(guided_src, tmp_path, prefixed):
    src = tmp_path / "Capture Data"
    for view in ("front", "left", "right"):
        (src / view.title()).mkdir(parents=True)
        for f in guided_src.glob(view + "_*"):
            name = f.name if prefixed else f.name[len(view) + 1:]
            shutil.copy(f, src / view.title() / name)
    root = tmp_path / "captures"
    assert run(root, str(src), "--subject", "s01", "--labels", "swapped") == 0
    out = root / "s01" / ("s01_" + STAMP + "_guided")
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [p.name for p in guided_src.iterdir()] + ["capture.json"])
    assert json.loads((out / "capture.json").read_text())["labels"] == "swapped"
    a, b = cap.load_capture(guided_src), cap.load_capture(out)
    for k in a:
        np.testing.assert_array_equal(a[k].landmarks, b[k].landmarks)


def test_ingest_detects_burst(guided_src, tmp_path):
    src = copy_of(guided_src, tmp_path / "burst")
    set_meta(src, depth_frames_merged=16, depth_merge="per-pixel median")
    root = tmp_path / "captures"
    assert run(root, str(src), "--subject", "s02") == 0
    out = root / "s02" / ("s02_" + STAMP + "_burst")
    assert json.loads((out / "capture.json").read_text())["depth_frames_merged"] == 16


def test_ingest_refuses_to_overwrite(guided_src, tmp_path):
    root = tmp_path / "captures"
    assert run(root, str(guided_src), "--subject", "s01") == 0
    out = root / "s01" / ("s01_" + STAMP + "_guided")
    before = (out / "capture.json").read_text()
    assert run(root, str(guided_src), "--subject", "s01") == 1
    assert (out / "capture.json").read_text() == before


def test_ingest_puts_other_files_in_extras_with_ascii_names(guided_src, tmp_path):
    src = copy_of(guided_src, tmp_path / "with_extras")
    (src / "Screenshot 2026-11-10 at 2.37.00ΓÇ»PM.png").write_bytes(b"png")
    (src / "registration.json").write_text("{}")
    root = tmp_path / "captures"
    assert run(root, str(src), "--subject", "s01") == 0
    out = root / "s01" / ("s01_" + STAMP + "_guided")
    names = sorted(p.name for p in (out / "extras").iterdir())
    assert names == ["Screenshot_2026-11-10_at_2.37.00PM.png", "registration.json"]
    assert all(n.isascii() for n in names)


def test_ingest_rejects_wrong_depth_size(guided_src, tmp_path):
    src = copy_of(guided_src, tmp_path / "bad")
    depth = next(src.glob("front_depth_*.bin"))
    depth.write_bytes(depth.read_bytes()[:-4])
    assert run(tmp_path / "captures", str(src), "--subject", "s01") == 1
    assert not (tmp_path / "captures" / "s01").exists() or not any((tmp_path / "captures" / "s01").iterdir())


def test_ingest_needs_subject_without_manifest(guided_src, tmp_path):
    assert run(tmp_path / "captures", str(guided_src)) == 1


# --- ingest: the app's zip and the inbox ------------------------------------------

def app_zip(guided_src, folder, cid):
    """A zip as the new app shares it: <cid>.zip holding <cid>/ with capture.json."""
    session = copy_of(guided_src, folder / "build" / cid)
    set_meta(session, depth_frames_merged=16)
    (session / "capture.json").write_text(json.dumps({
        "format": 1, "capture_id": cid, "subject": cid.split("_")[0], "mode": "burst",
        "app": {"version": "1.0", "build": "7"}, "device": "iPhone15,2"}))
    zpath = folder / (cid + ".zip")
    with zipfile.ZipFile(zpath, "w") as z:
        for f in sorted(session.rglob("*")):
            z.write(f, Path(cid) / f.relative_to(session))
    shutil.rmtree(folder / "build")
    return zpath


def test_inbox_zip_from_the_app_needs_no_flags(guided_src, tmp_path):
    root = tmp_path / "captures"
    inbox = root / "_inbox"
    inbox.mkdir(parents=True)
    cid = "s03_" + STAMP + "_burst"
    app_zip(guided_src, inbox, cid)
    assert run(root) == 0
    out = root / "s03" / cid
    manifest = json.loads((out / "capture.json").read_text())
    assert manifest["app"] == {"version": "1.0", "build": "7"} and manifest["device"] == "iPhone15,2"
    assert manifest["created_from"] == "app capture.json"
    assert (inbox / "done" / (cid + ".zip")).exists() and not (inbox / (cid + ".zip")).exists()


def test_from_folder_takes_only_capture_names(guided_src, tmp_path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    cid = "s04_" + STAMP + "_burst"
    app_zip(guided_src, downloads, cid)
    (downloads / "holiday_photos.zip").write_bytes(b"not a capture")
    root = tmp_path / "captures"
    assert run(root, "--from", str(downloads)) == 0
    assert (root / "s04" / cid / "capture.json").exists()
    assert (downloads / "holiday_photos.zip").exists()
    assert (root / "_inbox" / "done" / (cid + ".zip")).exists()


def test_bad_inbox_item_stays_and_others_still_run(guided_src, tmp_path):
    root = tmp_path / "captures"
    inbox = root / "_inbox"
    inbox.mkdir(parents=True)
    good = "s05_" + STAMP + "_burst"
    app_zip(guided_src, inbox, good)
    (inbox / "broken.zip").write_bytes(b"this is not a zip")
    assert run(root) == 1
    assert (root / "s05" / good).exists()
    assert (inbox / "broken.zip").exists()


def test_loose_files_in_inbox(guided_src, tmp_path):
    root = tmp_path / "captures"
    inbox = root / "_inbox"
    inbox.mkdir(parents=True)
    for f in guided_src.iterdir():
        shutil.copy(f, inbox / f.name)
    assert run(root, "--subject", "s06") == 0
    assert (root / "s06" / ("s06_" + STAMP + "_guided")).exists()
    assert not any(p.is_file() for p in inbox.iterdir())


def test_dry_run_changes_nothing(guided_src, tmp_path):
    root = tmp_path / "captures"
    inbox = root / "_inbox"
    inbox.mkdir(parents=True)
    app_zip(guided_src, inbox, "s08_" + STAMP + "_burst")
    assert run(root, "--dry-run") == 0
    assert sorted(p.name for p in root.iterdir()) == ["_inbox"]
    assert len(list(inbox.iterdir())) == 1


@pytest.mark.skipif(not hasattr(sys, "stdlib_module_names"), reason="needs Python 3.10+ (mayapy has 3.9)")
def test_ingest_tool_is_standard_library_only():
    """The person AirDropping captures may only have macOS's python3."""
    allowed = set(sys.stdlib_module_names) | {"config"}
    for path in (Path(ic.__file__), Path(config.__file__)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            else:
                continue
            for name in names:
                assert name.split(".")[0] in allowed, "{0} imports {1}".format(path.name, name)


def test_mirror_check_records_a_reviewed_verdict(tmp_path):
    from pipeline import mirror_check as mc
    (tmp_path / "capture.json").write_text(json.dumps({"capture_id": "s01_20261110T212937Z_burst",
                                                       "labels": "unchecked", "mirror_check": None}))
    assert mc.record_verdict(tmp_path, {"verdict": "LABELS_SWAPPED"}, tmp_path / "run") == "swapped"
    m = json.loads((tmp_path / "capture.json").read_text())
    assert m["labels"] == "swapped" and m["mirror_check"]["verdict"] == "LABELS_SWAPPED"
    assert mc.record_verdict(tmp_path, {"verdict": "MIRROR_TWIN"}, tmp_path / "run") == "unchecked"

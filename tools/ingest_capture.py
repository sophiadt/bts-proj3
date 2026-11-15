"""File a phone capture into captures/<subject>/<capture_id>/ (standard library only).

Runs with any Python 3.8+: macOS's built-in python3, the repo's .venv, or
mayapy. It never needs numpy, so the person AirDropping captures needs
nothing installed beyond git.

    python3 tools/ingest_capture.py                      # everything in captures/_inbox/
    python3 tools/ingest_capture.py --from ~/Downloads   # every s##_..._<mode>.zip / folder there
    python3 tools/ingest_capture.py PATH [PATH ...]      # specific zips or folders
    python3 tools/ingest_capture.py --subject s02 ...    # old-app captures with no capture.json

What it accepts:
  - <capture_id>.zip, as the app shares it (one file per session);
  - a <capture_id>/ folder;
  - older captures: a flat folder of front_/left_/right_ files, Front/Left/Right
    subfolders (prefixed or not), or loose files dropped straight into _inbox/.
    These need --subject because nothing in them says who the subject is.

What it does, per capture:
  1. finds the views and checks each one (478 landmarks, intrinsics in
     meta.json, depth file size = W x H x 4 bytes);
  2. names it <subject>_<YYYYMMDD>T<HHMMSS>Z_<mode>: the app's capture.json if
     present, else the earliest meta.json timestamp (UTC) and the detected mode;
  3. copies the app's files flat into captures/<subject>/<capture_id>/, puts
     anything else (screenshots, other tools' outputs) in extras/ with plain
     ASCII names, and writes capture.json;
  4. refuses to overwrite an existing capture, and moves a processed _inbox/ or
     --from item into _inbox/done/ (nothing is ever deleted).
A failing capture is reported and left where it is; the others still run.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402  (stdlib only)

N_LANDMARKS = 478
VIEW_NAMES = ("front", "left", "right")
APP_FILE_RE = re.compile(
    r"^(?:(?P<view>front|left|right)_)?"
    r"(?P<kind>landmarks_3d_mm\.json|meta\.json|depth_(?P<w>\d+)x(?P<h>\d+)_float32_m\.bin|"
    r"rgb\.jpg|rgb_upright\.jpg|landmarks\.ply|face_crop\.ply|blendshapes\.json)$")
RECORD_FILES = ("landmarks_sequence_mm.json", "recording_meta.json")
RECORD_DIRS = ("raw",)
IGNORED = (".DS_Store", "Thumbs.db", "desktop.ini")
REQUIRED_META = ("fx", "fy", "cx", "cy", "intrinsics_ref_width")
ID_RE = re.compile(config.CAPTURE_ID_RE)
# macOS puts a narrow no-break space (U+202F) before AM/PM in screenshot names;
# read as cp437 after a transfer it becomes "ΓÇ»". Both forms, and the plain
# no-break space and its mangled form, are dropped from extras' names.
MANGLED_SPACES = ("ΓÇ»", " ", "┬á", " ")


class CaptureError(Exception):
    """This item cannot be filed; the message says why."""


# --- naming -------------------------------------------------------------------

def utc_stamp(unix_seconds):
    return datetime.datetime.fromtimestamp(float(unix_seconds), datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def stamp_to_iso(stamp):
    return datetime.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%dT%H:%M:%SZ")


def make_capture_id(subject, stamp, mode):
    cid = "{0}_{1}_{2}".format(subject, stamp, mode)
    if not ID_RE.match(cid):
        raise CaptureError("bad capture id {0!r}: subject must look like s01, mode one of {1}".format(
            cid, ", ".join(config.CAPTURE_MODES)))
    return cid


def ascii_name(name):
    """A plain-ASCII file name: accents dropped, anything else becomes '_'.

    AirDrop and zip tools mangle macOS's narrow no-break space (screenshot
    names) into characters like 'ΓÇ»'; those must not reach the repo."""
    for junk in MANGLED_SPACES:
        name = name.replace(junk, "")
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "file"
    ext = re.sub(r"[^A-Za-z0-9]+", "", ext).lower()
    return stem + ("." + ext if ext else "")


# --- finding the files ----------------------------------------------------------

class Found(object):
    """What one source holds: app files by destination name, record extras, others."""

    def __init__(self):
        self.app = {}          # destination name -> source path
        self.record_dirs = {}  # "raw" -> source dir
        self.extras = []       # source paths of non-app files
        self.manifest = None   # the app's capture.json, if any


def _visible(p):
    return not p.name.startswith(".") and p.name not in IGNORED and p.name != "__MACOSX"


def _add_app(found, dest, src):
    if dest in found.app:
        raise CaptureError("two copies of {0}: {1} and {2}".format(dest, found.app[dest], src))
    found.app[dest] = src


def scan_folder(folder):
    """Classify a capture folder's contents (flat, or one level of view folders)."""
    found = Found()
    for p in sorted(folder.iterdir(), key=lambda q: q.name):
        if not _visible(p):
            continue
        if p.is_dir():
            if p.name in RECORD_DIRS:
                found.record_dirs[p.name] = p
                continue
            sub_view = p.name.lower() if p.name.lower() in VIEW_NAMES else None
            for q in sorted(p.rglob("*"), key=lambda r: str(r)):
                rel = q.relative_to(folder).parts
                if not q.is_file() or any(part.startswith(".") or part in IGNORED or part == "__MACOSX"
                                          for part in rel):
                    continue
                m = APP_FILE_RE.match(q.name) if q.parent == p else None
                if m and m.group("view"):
                    _add_app(found, q.name, q)
                elif m and sub_view:
                    _add_app(found, sub_view + "_" + q.name, q)       # unprefixed file inside Front/ etc.
                elif q.name == config.CAPTURE_MANIFEST and q.parent == p and found.manifest is None:
                    found.manifest = _read_json(q)
                else:
                    found.extras.append(q)
            continue
        if p.name == config.CAPTURE_MANIFEST:
            found.manifest = _read_json(p)
        elif APP_FILE_RE.match(p.name) or p.name in RECORD_FILES:
            _add_app(found, p.name, p)
        else:
            found.extras.append(p)
    return found


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as e:
        raise CaptureError("{0}: not readable JSON ({1})".format(path.name, e))


def views_of(found):
    """{view name: {kind: path}}; an unprefixed set is the single view 'single'."""
    views = {}
    for dest, src in found.app.items():
        m = APP_FILE_RE.match(dest)
        if not m:
            continue
        views.setdefault(m.group("view") or "single", {})[m.group("kind")] = src
    return views


# --- checks ---------------------------------------------------------------------

def check_view(name, files):
    """Raises CaptureError unless the view can be fitted. Returns its meta.json."""
    if "landmarks_3d_mm.json" not in files:
        raise CaptureError("{0}: no landmarks_3d_mm.json".format(name))
    if "meta.json" not in files:
        raise CaptureError("{0}: no meta.json".format(name))
    lm = _read_json(files["landmarks_3d_mm.json"])
    if not isinstance(lm, list) or len(lm) != N_LANDMARKS:
        raise CaptureError("{0}: {1} landmarks, expected {2}".format(
            name, len(lm) if isinstance(lm, list) else "no list of", N_LANDMARKS))
    for i, p in enumerate(lm):
        if p is not None and not (isinstance(p, list) and len(p) == 3
                                  and all(isinstance(c, (int, float)) for c in p)):
            raise CaptureError("{0}: landmark {1} is not [x, y, z] or null".format(name, i))
    meta = _read_json(files["meta.json"])
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise CaptureError("{0}: meta.json lacks {1}".format(name, ", ".join(missing)))
    for kind, path in files.items():
        m = APP_FILE_RE.match(kind)
        if m and m.group("w"):
            want = int(m.group("w")) * int(m.group("h")) * 4
            if path.stat().st_size != want:
                raise CaptureError("{0}: {1} is {2} bytes, expected {3}".format(
                    name, path.name, path.stat().st_size, want))
    return meta


def detect_mode(found, views, metas):
    if any(name in found.app for name in RECORD_FILES):
        return "record"
    if any(int(m.get("depth_frames_merged") or 1) > 1 for m in metas.values()):
        return "burst"
    if list(views) == ["single"]:
        return "photo"
    return "guided"


# --- one capture ----------------------------------------------------------------

def plan_capture(folder, subject=None, mode="auto", labels=None, notes=None, source_name=None):
    """Work out the capture id, file list and manifest for one source folder."""
    found = scan_folder(folder)
    views = views_of(found)
    is_record = any(name in found.app for name in RECORD_FILES)
    if not views and not is_record:
        raise CaptureError("no capture files (no *landmarks_3d_mm.json or landmarks_sequence_mm.json)")
    if "single" in views and len(views) > 1:
        raise CaptureError("mixes unprefixed files with front_/left_/right_ views")
    metas = {name: check_view(name, files) for name, files in views.items()}
    for name in RECORD_FILES:
        if name in found.app:
            _read_json(found.app[name])

    app = found.manifest or {}
    name_match = ID_RE.match(source_name or "")
    if subject is None:
        subject = app.get("subject") or (name_match.group("subject") if name_match else None)
    if subject is None:
        raise CaptureError("no capture.json and no s##_... name: pass --subject (e.g. --subject s02)")
    if not re.match(config.SUBJECT_RE, subject):
        raise CaptureError("subject {0!r} must be an anonymous code like s01".format(subject))

    stamps = [m["timestamp"] for m in metas.values() if isinstance(m.get("timestamp"), (int, float))]
    if app.get("capture_id") and ID_RE.match(app["capture_id"]):
        stamp, stamp_source = ID_RE.match(app["capture_id"]).group("stamp"), "app capture.json"
    elif stamps:
        stamp, stamp_source = utc_stamp(min(stamps)), "earliest meta.json timestamp"
    elif name_match:
        stamp, stamp_source = name_match.group("stamp"), "source name"
    else:
        oldest = min(p.stat().st_mtime for p in found.app.values())
        stamp, stamp_source = utc_stamp(oldest), "file modification time"

    if mode == "auto":
        mode = app.get("mode") or detect_mode(found, views, metas)
    if mode not in config.CAPTURE_MODES:
        raise CaptureError("mode {0!r} is not one of {1}".format(mode, ", ".join(config.CAPTURE_MODES)))
    cid = make_capture_id(subject, stamp, mode)
    if app.get("capture_id") and app["capture_id"] != cid:
        raise CaptureError("app capture.json says {0} but the files give {1}".format(app["capture_id"], cid))

    extras, used = {}, set(found.app)
    for src in found.extras:
        rel = src.relative_to(folder)
        base = ascii_name(src.name if len(rel.parts) == 1 else "_".join(rel.parts))
        dest, k = base, 2
        while dest in used or dest in extras:
            stem, dot, ext = base.rpartition(".")
            dest = "{0}_{1}{2}{3}".format(stem or base, k, dot, ext)
            k += 1
        extras[dest] = src

    merged = sorted({int(m.get("depth_frames_merged") or 1) for m in metas.values()}) or [1]
    manifest = dict(app)
    manifest.update({
        "format": 1,
        "capture_id": cid,
        "subject": subject,
        "mode": mode,
        "created_utc": stamp_to_iso(stamp),
        "created_from": stamp_source,
        "views": sorted(views) if views else [],
        "depth_frames_merged": merged[-1],
        "labels": labels or app.get("labels") or "unchecked",
        "mirror_check": app.get("mirror_check"),
        "notes": notes if notes is not None else app.get("notes", ""),
        "app": app.get("app", "unknown (no capture.json from the app)"),
        "files": sorted(found.app) + sorted(d + "/" for d in found.record_dirs),
        "extras": sorted("extras/" + d for d in extras),
        "ingested": {"utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "source": source_name or folder.name, "tool": "tools/ingest_capture.py"},
    })
    return {"capture_id": cid, "subject": subject, "found": found, "extras": extras, "manifest": manifest}


def write_capture(plan, root):
    """Copy into root/<subject>/<capture_id>/ via a staging folder; never overwrite."""
    dest = Path(root) / plan["subject"] / plan["capture_id"]
    if dest.exists():
        raise CaptureError("{0} already exists; not overwriting".format(dest))
    stage = dest.parent / (".staging_" + plan["capture_id"])
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        for name, src in plan["found"].app.items():
            shutil.copy2(src, stage / name)
        for name, src in plan["found"].record_dirs.items():
            shutil.copytree(src, stage / name)
        if plan["extras"]:
            (stage / "extras").mkdir()
            for name, src in plan["extras"].items():
                shutil.copy2(src, stage / "extras" / name)
        with open(stage / config.CAPTURE_MANIFEST, "w", encoding="utf-8") as fh:
            json.dump(plan["manifest"], fh, indent=2)
            fh.write("\n")
        # Re-check what landed before it becomes visible.
        again = plan_capture(stage, plan["subject"], plan["manifest"]["mode"],
                             source_name=plan["capture_id"])
        if again["capture_id"] != plan["capture_id"]:
            raise CaptureError("copied capture reads back as {0}".format(again["capture_id"]))
        os.replace(stage, dest)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return dest


# --- sources --------------------------------------------------------------------

def _unpack(src, tmp):
    """A folder to scan for one source: the folder itself, or an unzipped zip."""
    if src.is_dir():
        return src
    if src.suffix.lower() != ".zip":
        raise CaptureError("not a folder or .zip")
    out = Path(tmp) / src.stem
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            target = (out / info.filename).resolve()
            if not str(target).startswith(str(out.resolve())):
                raise CaptureError("zip entry {0!r} points outside the archive".format(info.filename))
        z.extractall(out)
    inner = [p for p in out.iterdir() if _visible(p)]
    if len(inner) == 1 and inner[0].is_dir():
        return inner[0]          # zip of a folder: <capture_id>/...
    return out


def inbox_items(inbox):
    """(name, path or list of loose files) for everything waiting in _inbox/."""
    items, loose = [], []
    if not inbox.exists():
        return items
    for p in sorted(inbox.iterdir(), key=lambda q: q.name):
        if not _visible(p) or p.name == "done":
            continue
        if p.is_dir() or p.suffix.lower() == ".zip":
            items.append((p.name, p))
        else:
            loose.append(p)
    if loose:
        items.append(("loose files in _inbox", loose))
    return items


def from_items(folder):
    """Zips and folders in `folder` whose names are capture ids."""
    out = []
    for p in sorted(Path(folder).expanduser().iterdir(), key=lambda q: q.name):
        stem = p.stem if p.suffix.lower() == ".zip" else p.name
        if _visible(p) and (p.is_dir() or p.suffix.lower() == ".zip") and ID_RE.match(stem):
            out.append((p.name, p))
    return out


def file_away(src, done_dir):
    """Move a processed source into _inbox/done/ (never delete)."""
    done_dir.mkdir(parents=True, exist_ok=True)
    srcs = src if isinstance(src, list) else [src]
    for s in srcs:
        target, k = done_dir / s.name, 2
        while target.exists():
            target = done_dir / "{0}_{1}{2}".format(s.stem, k, s.suffix)
            k += 1
        shutil.move(str(s), str(target))


def ingest(name, src, args, root, done_dir):
    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(src, list):                       # loose files: gather them into one folder
            folder = Path(tmp) / "loose"
            folder.mkdir()
            for f in src:
                shutil.copy2(f, folder / f.name)
            source_name = name
        else:
            folder = _unpack(src, tmp)
            source_name = src.stem if src.suffix.lower() == ".zip" else src.name
        plan = plan_capture(folder, args.subject, args.mode, args.labels, args.notes, source_name)
        print("{0}\n  -> {1}/{2}/  ({3} files, {4} extras, views: {5})".format(
            name, plan["subject"], plan["capture_id"], len(plan["manifest"]["files"]),
            len(plan["extras"]), ", ".join(plan["manifest"]["views"]) or "recording"))
        if args.dry_run:
            return plan["capture_id"]
        write_capture(plan, root)
    if done_dir is not None:
        file_away(src, done_dir)
    return plan["capture_id"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", help="zips or folders (default: everything in captures/_inbox/)")
    ap.add_argument("--from", dest="from_dir", help="pick up s##_..._<mode> zips/folders from this folder")
    ap.add_argument("--subject", help="anonymous subject code (needed for captures without capture.json)")
    ap.add_argument("--mode", default="auto", help="auto (default) or one of " + ", ".join(config.CAPTURE_MODES))
    ap.add_argument("--labels", choices=("unchecked", "ok", "swapped"), help="left/right labels, if already known")
    ap.add_argument("--notes", help="free text stored in capture.json")
    ap.add_argument("--root", default=str(config.CAPTURES_DIR), help=argparse.SUPPRESS)
    ap.add_argument("--dry-run", action="store_true", help="show what would happen, change nothing")
    args = ap.parse_args(argv)

    root = Path(args.root)
    inbox = root / config.CAPTURE_INBOX.name
    done = inbox / "done"
    if args.sources:
        items, done = [(Path(s).name, Path(s)) for s in args.sources], None   # explicit paths stay put
    elif args.from_dir:
        items = from_items(args.from_dir)
    else:
        items = inbox_items(inbox)
    if not items:
        print("nothing to ingest" + ("" if args.sources or args.from_dir else " in {0}".format(inbox)))
        return 0

    ok, failed = [], []
    for name, src in items:
        try:
            ok.append(ingest(name, src, args, root, done))
        except (CaptureError, OSError, zipfile.BadZipFile) as e:
            failed.append(name)
            print("{0}\n  !! not filed: {1}".format(name, e))
    if ok and not args.dry_run:
        subjects = sorted({cid.split("_")[0] for cid in ok})
        print("\nNext:\n  git add " + " ".join("captures/" + s for s in subjects))
        print('  git commit -m "Add capture{0} {1}"'.format("s" if len(ok) > 1 else "", ", ".join(ok)))
        print("  git push")
        print("  then on the pipeline side: python pipeline/mirror_check.py <capture_id> --record")
    if failed:
        print("\n{0} item(s) not filed: {1}".format(len(failed), ", ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

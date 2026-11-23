"""Guard: does a change delete too much of the repo? (standard library only)

    python tools/check_changes.py BASE HEAD
    python tools/check_changes.py origin/main HEAD      # before opening a pull request

Compares HEAD with BASE (git diff BASE...HEAD: what HEAD adds since the two
split, with renames detected, so moving a folder is not a deletion) and fails
when:

  - more than MAX_DELETED files are deleted;
  - a key file is deleted (CLAUDE.md, README.md, .gitignore, the lab scene, ...);
  - every tracked file under a key folder is gone (pipeline/, tests/, captures/, ...);
  - a Maya scene is added under template_export/ (each is ~430 MB);
  - a file over FAIL_MB is added or changed (GitHub refuses files over 100 MB).

It warns for a file over WARN_MB. ALLOW_DELETIONS=1 accepts the deletions, for
a clean-up made on purpose (on GitHub: the pull request's `allow-deletions`
label); the size rules still apply. Exit 1 on a failure, else 0.

Why: on 2 Nov 2026 a commit made from a copy of the repo that lacked most of
its files deleted 94 files from main (d48444e; reverted in 676901a). A deleting
commit can always be undone with `git revert`; this stops it before it lands.
"""

import os
import subprocess
import sys

MAX_DELETED = 10
WARN_MB = 50.0
FAIL_MB = 95.0
KEY_FILES = ("CLAUDE.md", "README.md", ".gitignore", ".gitattributes", "pyproject.toml",
             "config/landmark_to_vertex.json", "captures/README.md")
KEY_DIRS = ("pipeline/", "rig/", "core/", "checks/", "tests/", "tools/", "config/", "captures/",
            "template_export/", "TrueDepth App/")
SHOW_MAX = 40


def git(*args, cwd=None):
    return subprocess.run(("git",) + args, cwd=cwd, check=True, capture_output=True).stdout


def parse_name_status(raw):
    """`git diff --name-status -z` output -> [(status letter, path, new path or None)]."""
    parts = raw.decode("utf-8", "replace").split("\0")
    out, i = [], 0
    while i < len(parts) and parts[i]:
        code = parts[i][0]
        if code in "RC":
            out.append((code, parts[i + 1], parts[i + 2]))
            i += 3
        else:
            out.append((code, parts[i + 1], None))
            i += 2
    return out


def tracked_names(ref, cwd=None):
    """Paths tracked at ref. Names only: needs no file contents, so it is cheap in a
    partial clone (the GitHub check clones history without file contents)."""
    return [p for p in git("ls-tree", "-r", "--name-only", "-z", ref, cwd=cwd).decode("utf-8", "replace").split("\0")
            if p]


def tracked_sizes(ref, cwd=None):
    """{path: bytes} of every file tracked at ref (ref's files must be present: the checkout)."""
    sizes = {}
    for line in git("ls-tree", "-r", "-l", "-z", ref, cwd=cwd).decode("utf-8", "replace").split("\0"):
        if not line:
            continue
        meta, path = line.split("\t", 1)
        kind, size = meta.split()[1], meta.split()[3]
        if kind == "blob":
            sizes[path] = int(size) if size.isdigit() else 0
    return sizes


def evaluate(changes, base_files, head_sizes, allow_deletions=False):
    """-> (errors, warnings, deleted paths). changes: parse_name_status() output;
    base_files: paths tracked at BASE; head_sizes: {path: bytes} at HEAD."""
    errors, warnings = [], []
    deleted = sorted(p for code, p, _new in changes if code == "D")
    deletion_errors = []
    if len(deleted) > MAX_DELETED:
        deletion_errors.append("{0} files deleted (more than {1})".format(len(deleted), MAX_DELETED))
    gone = set(deleted)
    for key in KEY_FILES:
        if key in gone:
            deletion_errors.append("key file deleted: " + key)
    for p in deleted:
        if "/" not in p and p.lower().endswith(".mb"):
            deletion_errors.append("lab scene deleted: " + p)
    for d in KEY_DIRS:
        had = [p for p in base_files if p.startswith(d)]
        if had and not any(p.startswith(d) for p in head_sizes):
            deletion_errors.append("folder emptied: {0} ({1} files)".format(d, len(had)))
    if deletion_errors and allow_deletions:
        warnings += [e + " (accepted: ALLOW_DELETIONS)" for e in deletion_errors]
    else:
        errors += deletion_errors
    for code, p, new in changes:
        path = new or p
        if code not in "AMRC" or path not in head_sizes:
            continue
        if path.startswith("template_export/") and path.lower().endswith(".mb"):
            errors.append("Maya scene added under template_export/ (rebuilt by rig/write_rest.py; too big for git): "
                          + path)
        mb = head_sizes[path] / 1e6
        if mb > FAIL_MB:
            errors.append("{0:.0f} MB file: {1} (GitHub refuses files over 100 MB)".format(mb, path))
        elif mb > WARN_MB:
            warnings.append("{0:.0f} MB file: {1} (GitHub warns over 50 MB)".format(mb, path))
    return errors, warnings, deleted


def report(base, head, changes, errors, warnings, deleted):
    counts = {}
    for code, _p, _n in changes:
        counts[code] = counts.get(code, 0) + 1
    names = (("A", "added"), ("M", "modified"), ("R", "renamed"), ("C", "copied"), ("D", "deleted"),
             ("T", "type changed"))
    summary = ", ".join("{0} {1}".format(counts[c], n) for c, n in names if counts.get(c)) or "no file changes"
    lines = ["check_changes: {0}...{1}: {2}".format(base, head, summary)]
    if deleted:
        lines.append("deleted files:")
        lines += ["  - " + p for p in deleted[:SHOW_MAX]]
        if len(deleted) > SHOW_MAX:
            lines.append("  ... and {0} more".format(len(deleted) - SHOW_MAX))
    lines += ["WARNING: " + w for w in warnings]
    lines += ["FAIL: " + e for e in errors]
    lines.append("-> " + ("FAIL" if errors else "OK"))
    if errors:
        lines.append("If these deletions are intended, set ALLOW_DELETIONS=1 (on a pull request: add the "
                     "`allow-deletions` label). To undo a bad commit: git revert <sha>, never push --force.")
    return "\n".join(lines)


def main(argv=None, cwd=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    base, head = argv
    changes = parse_name_status(git("diff", "--name-status", "-M", "-z", "{0}...{1}".format(base, head), cwd=cwd))
    base_files = tracked_names(base, cwd=cwd)
    head_sizes = tracked_sizes(head, cwd=cwd)
    allow = os.environ.get("ALLOW_DELETIONS", "") not in ("", "0", "false", "False")
    errors, warnings, deleted = evaluate(changes, base_files, head_sizes, allow_deletions=allow)
    text = report(base, head, changes, errors, warnings, deleted)
    print(text)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        for w in warnings:
            print("::warning title=check_changes::" + w)
        for e in errors:
            print("::error title=check_changes::" + e)
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_file:
            with open(summary_file, "a", encoding="utf-8") as fh:
                fh.write("### Deletion guard\n\n```\n" + text + "\n```\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

# Captures

Every phone capture goes in one place, under one name, so the fitting pipeline can find it by id and a
capture is never mixed up with another.

```
captures/
  _inbox/                          drop AirDropped zips here (git-ignored; processed ones move to _inbox/done/)
  s01/                             one folder per subject (anonymous code)
    s01_20261030T170049Z_guided/   one folder per capture: <subject>_<UTC time>_<mode>
      capture.json                 who / what / when, app build, left/right verdict
      front_*  left_*  right_*     the app's files, flat
      extras/                      anything else: screenshots, other tools' outputs
    s01_20261110T212937Z_burst/
```

## Naming rules

- **Subject:** `s01`, `s02`, ... Always a code, never a name or initials. The list of which code is which
  person is kept privately by the repo owner, never in this repo. Only capture people who have agreed to
  it.
- **Capture id:** `<subject>_<YYYYMMDD>T<HHMMSS>Z_<mode>`. The time is UTC, at the start of the session.
  For example: `s02_20261118T190512Z_burst`.
- **Mode:**

  | Mode | What it is |
  | --- | --- |
  | `burst` | Guided capture (front, left, right) with 16 depth frames merged per pose. The default now; use it for fitting |
  | `guided` | The same with single-frame depth (app builds before burst) |
  | `photo` | One frame, no prefix |
  | `record` | A landmark recording (`landmarks_sequence_mm.json`, optional `raw/`); not fitted yet |

The app names its session folder and zip with this id itself. The ingest tool works it out for older
captures. Nobody types an id by hand.

## From the phone to the repo

This is written for the Mac the iPhone AirDrops to. Windows works the same way: replace `python3` with
`python` and `~/Downloads` with your Downloads folder.

### Once

```bash
# Clone without downloading the 432 MB lab scene (Git LFS); captures don't need it.
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/sophiadt/bts-proj3
cd bts-proj3
```

On Windows: `set GIT_LFS_SKIP_SMUDGE=1` (cmd) or `$env:GIT_LFS_SKIP_SMUDGE=1` (PowerShell), then
`git clone ...`. Keep the clone outside other repositories. Nothing needs installing: the ingest tool
uses only the Python that ships with macOS.

### Each session

1. **In the app,** type the subject code in the **SUBJECT** field (it turns orange if it isn't like
   `s01`). Ask the repo owner for the next free code.
2. **Tap guided capture** and follow the prompts (front, then turn to *your* left, then *your* right).
   Keep a neutral face, mouth closed.
3. **Tap Share and AirDrop to the Mac.** One file arrives in `~/Downloads`, named like
   `s02_20261118T190512Z_burst.zip`.
4. **File it:**
   ```bash
   git pull
   python3 tools/ingest_capture.py --from ~/Downloads
   ```
   Or drag the zip into `captures/_inbox/` and run `python3 tools/ingest_capture.py` with no arguments.
   The tool:
   - checks every view: 478 landmarks, intrinsics, depth file size;
   - writes `captures/s02/<capture_id>/`;
   - moves the zip to `captures/_inbox/done/`;
   - prints the capture id.

   Anything it can't file stays where it is, with the reason printed. `--dry-run` shows what would
   happen without changing anything.
5. **Commit and push** (the tool prints these lines):
   ```bash
   git add captures/s02
   git commit -m "Add capture s02_20261118T190512Z_burst"
   git push
   ```

### Then, on the pipeline side

```bash
git pull
python pipeline/mirror_check.py s02_20261118T190512Z_burst --record   # read the verdict; --record saves it
mayapy pipeline/fit_subject.py s02_20261118T190512Z_burst              # Ziva workstation
```

`--record` writes the reviewed verdict into the capture's `capture.json` (`labels: ok` or `swapped`).
`fit_subject.py` relabels a `swapped` capture on its own.

### Older captures (no zip, no `capture.json`)

Older app builds, or AirDropping files one by one, leave loose files or a folder with a different name.
Put them in a folder inside `captures/_inbox/` (any name; `Front/`, `Left/` and `Right/` subfolders are
fine), then give the subject:

```bash
python3 tools/ingest_capture.py --subject s02
```

The time comes from the earliest `meta.json` timestamp. The mode is `burst` if the depth was merged,
otherwise `guided`.

## `capture.json`

| Field | Meaning |
| --- | --- |
| `capture_id`, `subject`, `mode`, `created_utc` | The name, split into its parts |
| `views` | Poses present: `front`, `left`, `right` (or `single`) |
| `depth_frames_merged` | 16 for burst, 1 for single-frame depth |
| `app`, `device` | App version/build and iPhone model (written by the app; `unknown` for old captures) |
| `labels` | `unchecked` until `mirror_check.py --record`; then `ok` or `swapped` |
| `mirror_check` | The recorded verdict, its run folder and date |
| `notes` | Free text (`--notes` when ingesting) |
| `files`, `extras`, `ingested` | What was filed, what went to `extras/`, and when and from where |

## Storage

Captures are ordinary git files (about 7 MB per pose), not Git LFS. The free LFS quota is 1 GB, and the lab
scene already uses 432 MB of it. If the repo grows too large, move `captures/**/*.jpg` and `*.bin` to LFS
together.

"""Build the human-proportioned base head once: the lab head fitted to
MediaPipe's average human face.

The lab head is stylised (about 1.45x a real face, eyes 1.6-1.7x, a long, wide
lower face), so fitting it straight to a person needs a large reshape -- the
jaw sides squeeze by up to 30 mm, which is where most folded skin comes from.
Fitting it once to an average human face, and starting every person from that
result, leaves each person's fit a much smaller change.

The average face is MediaPipe's canonical mesh, rendered as a clean synthetic
guided capture (pipeline/synthetic.py: front and +/-30 deg turns, no noise).
Its lips are parted (5.8 mm), so the fit runs in base mode: lip landmarks,
lip-region scan data and the mouth opening are left out and the lab head's
closed mouth is kept. The eyeballs come out at 24 mm (fit_subject places
them). Same meshes and vertex order as the lab head, so the fixture, the
importance map and the rig all still apply.

    mayapy pipeline/build_base.py      # -> template_export/subjects/base_human/
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import fit_subject as fs                 # noqa: E402
from pipeline import synthetic                         # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        cap_dir = synthetic.make_guided(Path(tmp) / "average_face")
        fs.fit(cap_dir, fs.BASE_SUBJECT, base=None, base_mode=True)


if __name__ == "__main__":
    main()

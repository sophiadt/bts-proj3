"""Stand-in for the `paths` module the lab's keyboard UI imports.

The UI (a3-stochastic-V1/b99-.../h99-UI-mayascipt-V1_comsciV1.py) does
`from paths import ...`, but no paths.py exists on this machine. This module
points at the lab repo's folders for reading, and sends everything the UI
writes to this repo's template_export/ui_results, so nothing of the lab's is
written to. rig/keyboard_ui.py registers it as sys.modules["paths"].

    LAB_ROOT          $BTS_LAB_UI_ROOT, else C:\\Users\\zpa18\\a3-stochastic-V1
"""

import os
from pathlib import Path

LAB_ROOT = Path(os.environ.get("BTS_LAB_UI_ROOT", r"C:\Users\zpa18\a3-stochastic-V1"))
RESULTS = Path(__file__).resolve().parents[1] / "template_export" / "ui_results"

a3_stochastic_relative_path = LAB_ROOT
a1_blendshape_path = LAB_ROOT / "a1-blendshape-guided-animation-pipeline"
b98_sampling_path = LAB_ROOT / "b98-sampling-joint"
b99_frame1to4_path = LAB_ROOT / "b99-frame1to4-get-b1to3"
# Only added to sys.path by the UI ("b31 dataV22 backup hub"); not found on
# this machine, and nothing the UI imports for playback lives there.
backup_path = LAB_ROOT
results_path = RESULTS

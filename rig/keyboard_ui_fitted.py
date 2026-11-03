# Run the lab's keyboard UI on a fitted head. In Maya's Script Editor (Python tab):
#
#     exec(open(r"C:\Users\syd3\bts-proj3\rig\keyboard_ui_fitted.py").read())
#
# It opens template_export/scenes/<SUBJECT>_ui.mb and the UI window; type a
# sentence such as "not even you" and press Go, or use keyboard mode.
# Change SUBJECT to play another fitted person.

SUBJECT = "capture_01"

import importlib
import sys

REPO = r"C:\Users\syd3\bts-proj3"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from rig import keyboard_ui
importlib.reload(keyboard_ui)
keyboard_ui.launch(SUBJECT)

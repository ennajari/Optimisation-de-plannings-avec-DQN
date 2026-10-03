import os
import sys
import runpy

# Add ui/ to Python search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(CURRENT_DIR, "ui")
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

# Run the full interactive application
runpy.run_path(os.path.join(UI_DIR, "app.py"), run_name="__main__")

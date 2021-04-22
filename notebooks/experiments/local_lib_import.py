from pathlib import Path
import sys


root_lib_path_str = str(Path.cwd().parent.parent)
if str(root_lib_path_str) not in sys.path:
    sys.path.insert(0, root_lib_path_str)

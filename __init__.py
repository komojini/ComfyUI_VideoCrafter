import os
import sys
from pathlib import Path

current_path = os.path.abspath(os.path.dirname(__file__))

EXTENSION_PATH = Path(__file__).parent
sys.path.insert(0, str(EXTENSION_PATH.resolve()))


sys.path.insert(0, os.path.join(current_path, "scripts/evaluation"))
sys.path.insert(1, os.path.join(sys.path[0], 'lvdm'))


from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

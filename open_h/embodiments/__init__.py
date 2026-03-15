"""Auto-discover and register all Open-H embodiment configs.

Importing this module finds every *_config.py file under each embodiment
subdirectory and executes it, which triggers register_modality_config()
calls that populate the global MODALITY_CONFIGS registry.
"""

import importlib.util
from pathlib import Path


_embodiments_dir = Path(__file__).parent

for _config_file in sorted(_embodiments_dir.glob("*/*_config.py")):
    _spec = importlib.util.spec_from_file_location(_config_file.stem, _config_file)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)

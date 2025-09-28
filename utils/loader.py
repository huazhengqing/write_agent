from functools import lru_cache
import importlib
from typing import Tuple, Any


@lru_cache(maxsize=30)
def load_prompts(module_path: str, *component_names: str) -> Tuple[Any, ...]:
    module = importlib.import_module(module_path)
    return tuple(getattr(module, name) for name in component_names)

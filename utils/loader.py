import importlib
from typing import Tuple, Any


def load_prompts(category: str, module_name: str, *component_names: str) -> Tuple[Any, ...]:
    module_path = f"prompts.{category}.{module_name}"
    module = importlib.import_module(module_path)
    return tuple(getattr(module, name) for name in component_names)


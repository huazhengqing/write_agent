import importlib
from typing import Tuple, Any


def load_prompts(category: str, module_name: str, *component_names: str) -> Tuple[Any, ...]:
    try:
        module_path = f"prompts.{category}.{module_name}"
        module = importlib.import_module(module_path)
        return tuple(getattr(module, name) for name in component_names)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"无法从 {module_path} 加载组件 {component_names}: {e}")

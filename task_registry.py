"""
Simple task registry.

Usage in other files (like tasks.py):

    from orchestrator.core.task_registry import register_task

    @register_task()
    def load_data(context):
        ...

Every registered task can be called later by name from the pipeline engine.
"""

from typing import Callable, Dict, Any, Optional

# Internal registry dictionary
_TASK_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_task(name: Optional[str] = None):
    """
    Decorator to register a function as a task.

    Example:
        @register_task()
        def my_task(context):
            ...

        @register_task("custom_name")
        def another_task(context):
            ...
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        task_name = name or func.__name__

        if task_name in _TASK_REGISTRY:
            raise ValueError(f"Task '{task_name}' is already registered")

        _TASK_REGISTRY[task_name] = func
        return func

    return decorator


def get_task(name: str) -> Callable[..., Any]:
    """
    Return the task function registered under the given name.
    Raises KeyError if not found.
    """
    try:
        return _TASK_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Task '{name}' is not registered") from exc


def list_tasks() -> list[str]:
    """
    Return a sorted list of all registered task names.
    """
    return sorted(_TASK_REGISTRY.keys())

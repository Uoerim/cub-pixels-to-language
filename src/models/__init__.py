from importlib import import_module

from src.models.base_encoder import BaseEncoder
from src.models.scratch_cnn import ScratchCNN


MODEL_REGISTRY = {"scratch": ScratchCNN}


def register_model(name, constructor):
    MODEL_REGISTRY[name] = constructor


def build_model(config, for_inference=False):
    model_type = config.get("model_type", "scratch")
    if model_type not in MODEL_REGISTRY:
        module_name = config.get("model_module", f"src.models.{model_type}_cnn")
        class_name = config.get("model_class", f"{model_type.title()}CNN")
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name == module_name:
                raise ValueError(f"Add {module_name}.{class_name} or register '{model_type}' before using it.") from error
            raise
        constructor = getattr(module, class_name, None)
        if constructor is None:
            raise ValueError(f"{module_name} must expose {class_name}.")
    else:
        constructor = MODEL_REGISTRY[model_type]
    kwargs = dict(config.get("model_kwargs") or {})
    kwargs.setdefault("num_classes", config.get("num_classes", 200))
    kwargs.setdefault("dropout", config.get("dropout", 0.5))
    if for_inference and (model_type == "transfer" or "pretrained" in kwargs):
        kwargs["pretrained"] = False
    return constructor(**kwargs)


__all__ = ["BaseEncoder", "ScratchCNN", "build_model", "register_model"]

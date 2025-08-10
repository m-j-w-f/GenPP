from omegaconf import OmegaConf


def class_resolver(full_path):
    module_path, class_name = full_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def enum_resolver(full_path_and_value):
    enum_path, enum_value = full_path_and_value.rsplit(".", 1)
    enum_class = class_resolver(enum_path)
    return getattr(enum_class, enum_value)


def register_resolvers() -> None:
    """Register custom resolvers for OmegaConf."""
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("enum", enum_resolver)
    OmegaConf.register_new_resolver("class", class_resolver)
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))

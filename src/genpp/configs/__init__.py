from omegaconf import OmegaConf

from genpp.data import MetadataVars


def class_resolver(full_path):
    module_path, class_name = full_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def enum_resolver(full_path_and_value):
    enum_path, enum_value = full_path_and_value.rsplit(".", 1)
    enum_class = class_resolver(enum_path)
    return getattr(enum_class, enum_value)


def eval_print_error(expr):
    try:
        return eval(expr)
    except Exception as e:
        print(f"Error evaluating expression '{expr}': {e}")
        raise


def count_meta_features(preprocessing_list, include_pixel_idx):
    """Count meta features from preprocessing list.

    Args:
        preprocessing_list: List of preprocessing configurations
        include_pixel_idx: If False, excludes PIXEL_IDX from count
    """
    if preprocessing_list is None:
        return 0

    total = 0
    for processor in preprocessing_list:
        if "AddMetadataPreprocessor" in processor.get("_target_", ""):
            meta_features = processor.get("meta_features", [])
            if include_pixel_idx:
                total += len(meta_features)
            else:
                # Filter out PIXEL_IDX using the actual enum
                filtered = [f for f in meta_features if f != MetadataVars.PIXEL_IDX]
                total += len(filtered)
    return total


def register_resolvers() -> None:
    """Register custom resolvers for OmegaConf."""
    # Note: use_cache=False prevents caching issues with dynamic values
    OmegaConf.register_new_resolver("eval", eval_print_error, use_cache=False)
    OmegaConf.register_new_resolver("enum", enum_resolver)
    OmegaConf.register_new_resolver("class", class_resolver)
    OmegaConf.register_new_resolver("tuple", lambda *args: tuple(args))
    OmegaConf.register_new_resolver("count_meta_features", count_meta_features)


def add_y_kwargs(cfg, y_kwargs):
    # If there are no y_kwargs, then add them
    # NOTE: This is here only for compatibility with older configs
    y_kwargs_oc = OmegaConf.create({"y_kwargs": y_kwargs})

    if "y_kwargs" not in cfg.data.module.dataset_config.train:
        print("Adding y_kwargs to dataset_config.train")
        OmegaConf.set_struct(cfg, False)
        cfg.data.module.dataset_config.train = OmegaConf.merge(
            cfg.data.module.dataset_config.train, y_kwargs_oc
        )
        OmegaConf.set_struct(cfg, True)


def del_key(cfg, key):
    """Delete a key from the OmegaConf config given its dot-separated path."""
    OmegaConf.set_struct(cfg, False)
    del cfg[key]
    print(f"Deleted key '{key}' from config.")
    OmegaConf.set_struct(cfg, True)

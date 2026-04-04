import importlib.resources
from pathlib import Path


def _resolve_config(filename: str, local_name: str | None = None) -> Path:
    if local_name is not None:
        local_config = Path.cwd() / local_name
        if local_config.exists():
            return local_config

    user_config = Path.home() / ".config" / "sheptun" / filename
    if user_config.exists():
        return user_config

    with importlib.resources.as_file(
        importlib.resources.files("sheptun.config").joinpath(filename)
    ) as default_config:
        return Path(default_config)


def get_config_path(config: Path | None = None) -> Path:
    if config is not None:
        return config
    return _resolve_config("commands.yaml", local_name="sheptun.yaml")


def get_replacements_path(config: Path | None = None) -> Path:
    if config is not None:
        return config
    return _resolve_config("replacements.yaml", local_name="replacements.yaml")

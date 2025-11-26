import importlib.resources
from pathlib import Path


def get_config_path(config: Path | None = None) -> Path:
    if config is not None:
        return config

    local_config = Path.cwd() / "sheptun.yaml"
    if local_config.exists():
        return local_config

    user_config = Path.home() / ".config" / "sheptun" / "commands.yaml"
    if user_config.exists():
        return user_config

    with importlib.resources.as_file(
        importlib.resources.files("sheptun.config").joinpath("commands.yaml")
    ) as default_config:
        return Path(default_config)

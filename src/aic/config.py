"""Configuration loading.

A single ``Config`` object is threaded through the whole system so that paths and
model choices live in ``configs/default.yaml`` rather than being scattered as
literals. Paths are resolved against the repo root, so the code works the same
whether it is launched from the repo, from ``app/``, or from a notebook.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _resolve(value: str) -> Path:
    """Resolve a configured path against the repo root unless it is absolute."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


class _Section:
    """Dict wrapper giving attribute access and path resolution."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError as exc:  # pragma: no cover - developer error
            raise AttributeError(f"no config key {name!r} in {sorted(self._data)}") from exc
        return _Section(value) if isinstance(value, dict) else value

    def __getitem__(self, name: str) -> Any:
        return self.__getattr__(name)

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def path(self, name: str) -> Path:
        return _resolve(self.__getattr__(name))

    def as_dict(self) -> dict[str, Any]:
        return self._data


@dataclass
class Config:
    raw: dict[str, Any] = field(repr=False)

    @property
    def paths(self) -> _Section:
        return _Section(self.raw["paths"])

    @property
    def embedding(self) -> _Section:
        return _Section(self.raw["embedding"])

    @property
    def fusion(self) -> _Section:
        return _Section(self.raw["fusion"])

    @property
    def submission(self) -> _Section:
        return _Section(self.raw["submission"])

    @property
    def trake(self) -> _Section:
        return _Section(self.raw["trake"])

    @property
    def query(self) -> _Section:
        return _Section(self.raw["query"])

    @property
    def verify(self) -> _Section:
        return _Section(self.raw["verify"])

    # -- convenience path accessors -------------------------------------------------

    def raw_path(self, key: str) -> Path:
        return _resolve(self.raw["paths"]["raw"][key])

    def derived_path(self, key: str) -> Path:
        return _resolve(self.raw["paths"]["derived"][key])

    @property
    def catalog_path(self) -> Path:
        return self.derived_path("catalog")

    @property
    def submissions_dir(self) -> Path:
        return _resolve(self.raw["paths"]["submissions"])

    @property
    def active_space(self) -> _Section:
        name = self.raw["embedding"]["active"]
        spaces = self.raw["embedding"]["spaces"]
        if name not in spaces:
            raise KeyError(
                f"embedding.active={name!r} is not defined under embedding.spaces "
                f"({sorted(spaces)})"
            )
        return _Section({**spaces[name], "name": name})

    @property
    def index_path(self) -> Path:
        """FAISS index file for the currently active embedding space."""
        return self.derived_path("index") / f"{self.raw['embedding']['active']}.faiss"


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration, optionally overriding the file via ``AIC_CONFIG``."""
    resolved = Path(path) if path else Path(os.environ.get("AIC_CONFIG", DEFAULT_CONFIG_PATH))
    with open(resolved, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return Config(raw=data)

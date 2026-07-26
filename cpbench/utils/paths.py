"""
paths.py
--------
The single resolution point for dataset locations.

Every dataset path in this repository derives from one value, resolved here.
Nothing else should contain an absolute dataset literal: a path repeated in
eleven config files and five sbatch scripts is a path that is wrong in at
least one of them, and the failure surfaces on a cluster, hours after
submission, as a FileNotFoundError nobody can attribute.

Resolution order (first that is set wins)::

    1. an explicit ``data_root=`` config override on the command line
    2. the ``CPBENCH_DATA_ROOT`` environment variable
    3. the ``data_root`` key in the package's ``configs/config.yaml``
    4. ``DEFAULT_DATA_ROOT`` below

The environment variable sits above the config key because the config key is
checked into git and shared across machines, while the environment is the
machine talking about itself -- an sbatch script exporting one value should
not also have to pass an override to every entry point it calls. An explicit
CLI override still wins over both, because a value typed at the point of use
is never something the user wants silently ignored.

``DEFAULT_DATA_ROOT`` is a local convenience, not a claim about where anyone
else's data lives. On a machine that is not the UT EEMCS cluster it is simply
a directory that does not exist, and :func:`require_dataset_root` says which
environment variable to set rather than quoting a path from someone else's
filesystem.

Layout under the root -- ``RELATIVE`` maps an adapter name to its
subdirectory, so a dataset config declares which dataset it wants and never
where the tree is::

    <data_root>/opencood/opv2v        OPV2V   (OpenCOOD)
    <data_root>/opencood/v2xset       V2XSet  (OpenCOOD)
    <data_root>/air-thu/dair-v2x-c    DAIR-V2X-C
    <data_root>/huggingface/griffin   Griffin

Usage::

    from cpbench.utils.paths import data_root, require_dataset_root

    base = data_root(cfg.get("data_root"))
    root = require_dataset_root(cfg["dataset"]["root"])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

#: Environment variable consulted by :func:`data_root`.
ENV_VAR = "CPBENCH_DATA_ROOT"

#: Fallback when neither the environment nor the config supplies a value.
#: The UT EEMCS shared CV dataset tree -- convenient here, meaningless
#: elsewhere, and never the only thing standing between a job and its data.
DEFAULT_DATA_ROOT = "/datasets/eemcs/ps/cv"

#: Adapter name -> path relative to the resolved root. Both spellings of the
#: DAIR adapter appear across the packages; neither is worth a rename.
RELATIVE: Dict[str, str] = {
    "opv2v": "opencood/opv2v",
    "v2xset": "opencood/v2xset",
    "dair-v2x": "air-thu/dair-v2x-c",
    "dair_v2x": "air-thu/dair-v2x-c",
    "griffin": "huggingface/griffin",
}


def data_root(config_value: Optional[Any] = None,
              env: Optional[Mapping[str, str]] = None) -> Path:
    """Resolve the dataset base directory.

    Parameters
    ----------
    config_value
        The ``data_root`` key from a loaded config, or ``None``. Consulted
        only when the environment variable is unset.
    env
        Environment mapping to read; defaults to ``os.environ``. Injectable
        so the precedence rules are testable without mutating the process.

    Examples
    --------
    >>> data_root(env={"CPBENCH_DATA_ROOT": "/mnt/data"})
    PosixPath('/mnt/data')
    >>> data_root("/from/config", env={"CPBENCH_DATA_ROOT": "/mnt/data"})
    PosixPath('/mnt/data')
    >>> data_root("/from/config", env={})
    PosixPath('/from/config')
    >>> data_root(env={}) == Path(DEFAULT_DATA_ROOT)
    True
    """
    environ = os.environ if env is None else env
    from_env = environ.get(ENV_VAR)
    if from_env:
        return Path(from_env).expanduser()
    if config_value:
        return Path(str(config_value)).expanduser()
    return Path(DEFAULT_DATA_ROOT)


def dataset_root(adapter: str, config_value: Optional[Any] = None,
                 env: Optional[Mapping[str, str]] = None) -> Path:
    """``data_root() / RELATIVE[adapter]`` -- the default root for one dataset.

    >>> dataset_root("opv2v", env={"CPBENCH_DATA_ROOT": "/mnt/data"})
    PosixPath('/mnt/data/opencood/opv2v')
    >>> dataset_root("nuscenes", env={})            # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    KeyError
    """
    try:
        relative = RELATIVE[adapter]
    except KeyError:
        raise KeyError(
            f"no known layout for adapter {adapter!r}; known adapters are "
            f"{sorted(RELATIVE)}. Set dataset.root explicitly, or add the "
            f"subdirectory to cpbench.utils.paths.RELATIVE.") from None
    return data_root(config_value, env) / relative


def describe_source(config_value: Optional[Any] = None,
                    env: Optional[Mapping[str, str]] = None) -> str:
    """Say which rule supplied the root.

    Used in error messages so a wrong path explains *why* it is that path,
    instead of leaving the reader to guess whether the environment or a
    config file won.

    >>> describe_source(env={"CPBENCH_DATA_ROOT": "/mnt/data"})
    '$CPBENCH_DATA_ROOT=/mnt/data'
    >>> describe_source("/from/config", env={})
    'the data_root config key (/from/config)'
    """
    environ = os.environ if env is None else env
    from_env = environ.get(ENV_VAR)
    if from_env:
        return f"${ENV_VAR}={from_env}"
    if config_value:
        return f"the data_root config key ({config_value})"
    return f"the built-in default ({DEFAULT_DATA_ROOT})"


def missing_root_message(path: Any, *, config_value: Optional[Any] = None,
                         what: str = "dataset") -> str:
    """The text :func:`require_dataset_root` raises with.

    Names the environment variable rather than a literal path, because the
    reader is on a machine whose layout this file cannot know.
    """
    layouts = ", ".join(sorted(set(RELATIVE.values())))
    return (
        f"{what} root {path} does not exist on this machine.\n"
        f"It was derived from {describe_source(config_value)}.\n"
        f"Point the resolver at your data:\n"
        f"    export {ENV_VAR}=/path/to/your/datasets\n"
        f"which is expected to contain: {layouts}\n"
        f"Or override this one path directly: dataset.root=/path/to/dataset")


def require_dataset_root(root: Any, *, config_value: Optional[Any] = None,
                         what: str = "dataset") -> Path:
    """Return ``root`` as a Path, or raise with the resolver in the message."""
    path = Path(str(root)).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            missing_root_message(path, config_value=config_value, what=what))
    return path

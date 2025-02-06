"""
Configuration for RelevanceRelated project.

Note, store private configs in the same folder as 'config.toml', namely: "./[PRIVATE_PREFIX]_configs.toml"

Author: Simon M. Hofmann | <[firstname].[lastname][at]cbs.mpg.de> | 2022
"""

# %% Imports
from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import toml

# %% Config class & functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class CONFIG:
    """Configuration object."""

    def __init__(self, config_dict: dict | None = None):
        """Initialise CONFIG class object."""
        if config_dict is not None:
            self.update(config_dict)

    def __repr__(self):
        """Implement __repr__ of CONFIG."""
        str_out = "CONFIG("
        list_attr = [k for k in self.__dict__ if not k.startswith("_")]
        ctn = 0  # counter for visible attributes only
        for key, val in self.__dict__.items():
            if key.startswith("_"):
                # ignore hidden attributes
                continue
            ctn += 1
            str_out += f"{key}="
            if isinstance(val, CONFIG):
                str_out += val.__str__()
            else:
                str_out += f"'{val}'" if isinstance(val, str) else f"{val}"

            str_out += ", " if ctn < len(list_attr) else ""
        return str_out + ")"

    def update(self, new_configs: dict[str, Any]):
        """Update the config object with new entries."""
        for k, val in new_configs.items():
            if isinstance(val, (list, tuple)):
                setattr(self, k, [CONFIG(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, k, CONFIG(val) if isinstance(val, dict) else val)

    def show(self, indent: int = 0):
        """Display configurations in nested way."""
        for key, val in self.__dict__.items():
            if isinstance(val, CONFIG):
                print("\t" * indent + f"{key}:")
                val.show(indent=indent + 1)
            else:
                print("\t" * indent + f"{key}: " + (f"'{val}'" if isinstance(val, str) else f"{val}"))

    def asdict(self):
        """Convert the config object to a dict."""
        dict_out = {}
        for key, val in self.__dict__.items():
            if isinstance(val, CONFIG):
                dict_out.update({key: val.asdict()})
            else:
                dict_out.update({key: val})
        return dict_out

    def update_paths(self, parent_path: str | None = None):
        """Update relative paths to PROJECT_ROOT dir."""
        # Use project root dir as the parent path if not specified
        parent_path = self.PROJECT_ROOT if hasattr(self, "PROJECT_ROOT") else parent_path

        if parent_path is not None:
            parent_path = str(Path(parent_path).absolute())

            for key, path in self.__dict__.items():
                if isinstance(path, str) and not Path(path).is_absolute():
                    self.__dict__.update({key: str(Path(parent_path).joinpath(path))})

                elif isinstance(path, CONFIG):
                    path.update_paths(parent_path=parent_path)

        else:
            print("Paths can't be converted to absolute paths, since no PROJECT_ROOT is found!")


def set_wd(new_dir: str) -> None:
    """
    Set the given directory as new working directory of the project.

    :param new_dir: name of new working directory (must be in project folder)
    """
    if PROJECT_NAME not in str(Path.cwd()):
        msg = f'Current working dir "{Path.cwd()}" is outside of project "{PROJECT_NAME}".'
        raise FileNotFoundError(msg)

    # Remove '/' if new_dir == 'folder/' OR '/folder'
    new_dir = "".join(new_dir.split("/"))

    print("\033[94m" + f"Current working dir:\t{Path.cwd()}" + "\033[0m")  # print blue

    found = new_dir == str(Path.cwd()).split("/")[-1]

    # First look down the tree
    if not found:
        # Note: This works only for unique folder names
        if new_dir == Path(PROJECT_ROOT).name:
            os.chdir(PROJECT_ROOT)
            found = True
        else:
            for path in Path(PROJECT_ROOT).glob(f"**/{new_dir}"):  # 2. '_' == files
                os.chdir(path)
                found = True
                break
        if found:
            print("\033[93m" + f"New working dir:\t{Path.cwd()}\n" + "\033[0m")  # yellow print
        else:
            print(f"\033[91mGiven folder not found. Working dir remains:\t{Path.cwd()}\n\033[0m")  # red print


# %% Setup configuration object << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Instantiate config object
config = CONFIG()

# Load config file(s)
for config_file in Path(__file__).parent.glob("../configs/*config.toml"):
    config.update(new_configs=toml.load(str(config_file)))

# Extract some useful globals
PROJECT_NAME = config.PROJECT_NAME
PROJECT_ROOT = __file__[: __file__.find(PROJECT_NAME) + len(PROJECT_NAME)]

# Set root path to the config file and update paths
config.paths.PROJECT_ROOT = PROJECT_ROOT
config.paths.update_paths()

# Specifically for RelevanceRelated
if platform.system().lower() == "linux" and platform.node() != "experience":
    # same check as utils.check_system() == "MPI"
    config.paths.DATA = ".../Data"

# Extract paths
paths = config.paths
params = config.params

# Welcome
_w = 95
print("\n" + ("*" * _w + "\n") * 2 + "\n" + f"{PROJECT_NAME:^{_w}}" + "\n" * 2 + ("*" * _w + "\n") * 2)

# Set project working directory
set_wd(PROJECT_NAME)

# <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> ooo <<<<<<<<<<< ooo >>>>>>>>>>>>>> END

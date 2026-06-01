# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
from pathlib import Path
import sys

import tyro

import open_h.embodiments  # noqa: F401 — registers Open-H embodiment configs

from gr00t.configs.base_config import Config, get_default_config
from gr00t.experiment.experiment import run


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--load-config-path")
    parsed_args, remaining_args = parser.parse_known_args(argv)

    config_default = get_default_config()
    if parsed_args.load_config_path:
        config_path = Path(parsed_args.load_config_path)
        assert config_path.exists(), f"Config path does not exist: {config_path}"
        if remaining_args:
            parser.error(
                "`--load-config-path` cannot be combined with additional CLI overrides. "
                "Put overrides in the YAML config instead."
            )
        config = config_default.load(config_path)
        config.load_config_path = None
        logging.info(f"Loaded config from {config_path}")
    else:
        # Use tyro for clean CLI
        config = tyro.cli(Config, default=config_default, args=remaining_args, description=__doc__)
    run(config)

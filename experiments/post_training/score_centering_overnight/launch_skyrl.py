# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use the native SkyRL launcher with the existing CoreWeave filesystem setup."""

import argparse
from pathlib import Path

from cloud.iris.launch import execute_launch
from cloud.iris.launch_config import load_launch_config
from rigging.filesystem.s3_compat import configure_coreweave_s3
from skyrl_train.utils.utils import validate_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    validate_cfg(load_launch_config(args.config).skyrl)
    print(execute_launch(args.config))


if __name__ == "__main__":
    main()

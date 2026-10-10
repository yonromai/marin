# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use the native SkyRL launcher with the existing CoreWeave filesystem setup."""

import argparse
from pathlib import Path

from cloud.iris.launch import execute_launch
from rigging.filesystem.s3_compat import configure_coreweave_s3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    print(execute_launch(args.config))


if __name__ == "__main__":
    main()

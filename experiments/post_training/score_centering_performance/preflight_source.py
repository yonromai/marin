# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select a fixed, gradeable subset without changing the retained source bytes."""

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import fsspec
import pyarrow.parquet as pq
from cloud.iris.launch_config import load_launch_config
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rolloutengine.contracts import ModelTurn
from skyrl_gym.answer_tasks import grade_aime, grade_gsm8k, grade_reasoning_gym
from taskcompendium.grading_result import Outcome
from verifyit.grade import InvalidTask

GRADERS = {"aime": grade_aime, "gsm8k": grade_gsm8k, "reasoning_gym": grade_reasoning_gym}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--count", type=int, default=192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    if args.source_file is not None:
        raw = args.source_file.read_bytes()
    else:
        with fsspec.open(args.source, "rb") as source:
            raw = source.read()
    rows = pq.read_table(io.BytesIO(raw)).to_pylist()
    cfg = load_launch_config(args.config).skyrl
    selected, excluded = [], []
    for index, row in enumerate(rows):
        session = row["env_class"]
        expected = row["reward_spec"]["ground_truth"]
        try:
            for text in ("Answer: 0\n#### 0", "No final answer"):
                turn = ModelTurn(
                    message={"role": "assistant", "content": text},
                    prompt_token_ids=(1,),
                    response_token_ids=(2,),
                    logprobs=(0.0,),
                    stop_reason="stop",
                    text=text,
                )
                # Use the real session grader and configured options. Reward
                # values do not select rows; only invalid task data is excluded.
                result = GRADERS[session](turn, dict(cfg.environment.task_sessions.get(session, {})), row)
                if result.grade.status != Outcome.GRADED:
                    raise InvalidTask(f"Session grader returned {result.grade.status}")
        except InvalidTask as error:
            excluded.append({"source_index": index, "expected": expected, "error": str(error)})
            continue
        selected.append(index)
        if len(selected) == args.count:
            break
    assert len(selected) == args.count and selected[0] == 0
    selected_rows = [rows[index] for index in selected]
    content = json.dumps(selected_rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result = {
        "source_uri": args.source,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_rows": len(rows),
        "selected_source_indices": selected,
        "campaign_source_rows": [int(row["campaign_source_row"]) for row in selected_rows],
        "selected_rows_sha256": hashlib.sha256(content).hexdigest(),
        "selected_reference_prompt_lengths": [len(row["reference_prompt_ids"]) for row in selected_rows],
        "selected_task_sessions": dict(Counter(row["env_class"] for row in selected_rows)),
        "excluded": excluded,
        "rule": (
            "First192 retained rows whose actual configured session grader accepts task data; no source edits, "
            "generated-output selection or learning comparison. First32 groups define timing and qualification; "
            "the same fixed192-group subset supplies six natural periodic windows."
        ),
    }
    assert max(result["selected_reference_prompt_lengths"]) < 4096
    assert len(set(result["campaign_source_rows"])) == args.count
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selected": len(selected), "excluded": excluded, "source_sha256": result["source_sha256"]}))


if __name__ == "__main__":
    main()

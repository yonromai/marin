"""Record the installed worker checkout after the standard editable bootstrap."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
paths = [
    "vendor/skyrl/uv.lock",
    "vendor/skyrl/skyrl-train/skyrl_train/objective/score_centering.py",
    "vendor/skyrl/skyrl-train/skyrl_train/trajectory_runners/selected_topk.py",
    "vendor/skyrl/skyrl-train/skyrl_train/trajectory_runners/model_clients.py",
    "vendor/skyrl/skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py",
    "vendor/skyrl/skyrl-train/skyrl_train/rollouts/task_worker.py",
    "vendor/skyrl/skyrl-train/skyrl_train/trainer.py",
    "vendor/skyrl/skyrl-train/skyrl_train/entrypoints/score_centering_performance.py",
    "lib/verifyit/src/verifyit/modes/grade_math.py",
]
grader = Path(importlib.util.find_spec("verifyit.modes.grade_math").origin)
record = {
    "python": sys.executable,
    "versions": {name: importlib.metadata.version(name) for name in
                 ("torch", "vllm", "megatron-core", "marin-verifyit", "marin-rigging", "marin-taskcompendium")},
    "hashes": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths},
    "grader_path": str(grader),
    "grader_sha256": hashlib.sha256(grader.read_bytes()).hexdigest(),
    "scalar_guard_present": "or not value.is_commutative" in grader.read_text(),
    "direct_urls": {name: json.loads(importlib.metadata.distribution(name).read_text("direct_url.json"))
                    for name in ("marin-verifyit", "marin-rigging", "marin-taskcompendium")},
}
print("TIM_RUNTIME_IDENTITY " + json.dumps(record, sort_keys=True), flush=True)
if output := os.environ.get("IRIS_OUTPUT_DIR"):
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "tim-runtime-identity.json").write_text(json.dumps(record, indent=2) + "\n")

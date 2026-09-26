#!/usr/bin/env python3
"""Prepare a private Kaggle script-kernel directory; never pushes it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[2] / "CrispASR/tools/kaggle/kaggle_harness.py"


def prepare(config: dict, output: Path, harness: Path = HARNESS) -> dict:
    account = config.get("account", "")
    slug = config.get("slug", "")
    run_id = config.get("run_id", "")
    dataset = config.get("input_dataset", "")
    archive = config.get("archive_name", "")
    command = config.get("command")
    if not all(re.fullmatch(r"[a-z0-9][a-z0-9-]*", x) for x in (account, slug, run_id)):
        raise ValueError("explicit account, slug and unique run_id are required")
    if not dataset.startswith(account + "/") or not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", dataset):
        raise ValueError("input dataset must be owned by the selected account")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.zip", archive):
        raise ValueError("archive_name must be a simple .zip filename")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        raise ValueError("command must be an explicit JSON string array")
    if re.search(r"KGAT_[A-Za-z0-9]+|hf_[A-Za-z0-9]{20,}|gh[opsu]_[A-Za-z0-9]{20,}", json.dumps(config)):
        raise ValueError("configuration appears to contain a credential")
    if config.get("hf_downloads"):
        raise ValueError("HF downloads are disabled until pinned hashes, sizes and repo types are specified")
    if output.exists() or not harness.is_file():
        raise ValueError("fresh output and local CrispASR harness are required")
    source = harness.read_text()
    if re.search(r"KGAT_[A-Za-z0-9]+|hf_[A-Za-z0-9]{20,}", source):
        raise ValueError("harness appears to contain a credential")
    downloads = config.get("hf_downloads", [])
    if not isinstance(downloads, list) or any(not isinstance(x, dict) or set(x) != {"repo_id", "filename"} or
                                             not all(isinstance(v, str) for v in x.values()) for x in downloads):
        raise ValueError("hf_downloads must contain repo_id/filename string objects")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", x["repo_id"]) or
           not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", x["filename"]) for x in downloads):
        raise ValueError("invalid HF artifact path")
    run_config = {"account": account, "run_id": run_id, "input_dataset": dataset,
                  "archive_name": archive, "command": command,
                  "hf_downloads": downloads, "timeout_seconds": config.get("timeout_seconds", 1800),
                  "harness_sha256": hashlib.sha256(source.encode()).hexdigest()}
    if not isinstance(run_config["timeout_seconds"], int) or not 1 <= run_config["timeout_seconds"] <= 7200:
        raise ValueError("timeout_seconds must be 1..7200")
    template = (HERE / "kaggle_kernel.py.txt").read_text()
    code = template.replace("__CONFIG_JSON__", repr(json.dumps(run_config, sort_keys=True))).replace("__HARNESS_SOURCE__", repr(source))
    sources = [dataset]
    metadata = {"id": f"{account}/{slug}", "title": slug.replace("-", " ").title(),
                "code_file": "remote_quality.py", "language": "python", "kernel_type": "script",
                "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
                "competition_sources": [], "dataset_sources": sources, "kernel_sources": [], "model_sources": []}
    output.mkdir(parents=True)
    (output / "remote_quality.py").write_text(code)
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    # This copy is useful for a notebook/manual launch; the script embeds the same fallback.
    (output / "kaggle_harness.py").write_text(source)
    return {"status": "staged_only", "account": account, "run_id": run_id, "directory": str(output),
            "dataset_sources": sources, "launch": "disabled"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(json.loads(args.config.read_text()), args.output), sort_keys=True))


if __name__ == "__main__":
    main()

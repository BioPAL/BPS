#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 European Space Agency (ESA) - ACRI-ST
# SPDX-License-Identifier: Apache-2.0
"""Compute CI tier (0/1/2) for biomass-bps ci.yml tier-decision job.

Tier model (downstream jobs in ci.yml):
  0 — Baseline only (no Extended / Heavy).
  1 — Baseline + Extended (integration / SME path).
  2 — Baseline + Extended + Heavy (deep validation).

Policy is loaded from `.github/tier-policy.yml` at base_sha (merge target on PRs,
or `develop` on workflow_dispatch without PR) so the PR head cannot weaken rules.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
from pathlib import PurePosixPath
from typing import Any

import yaml

# Known dependency manifest / lock files (used to detect Dependabot bumps).
MANIFEST_LOCK_NAMES = (
    "pyproject.toml",
    "poetry.lock",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "Pipfile.lock",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "go.mod",
    "go.sum",
    "Cargo.toml",
    "Cargo.lock",
)


def gh_json(path: str, env: dict[str, str]) -> Any:
    """Single-page GitHub REST call via `gh api`."""
    proc = subprocess.run(
        ["gh", "api", path],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    return json.loads(proc.stdout)


def gh_json_paginate(path: str, env: dict[str, str]) -> Any:
    """Paginated GitHub REST call (e.g. PR file lists > 1 page)."""
    proc = subprocess.run(
        ["gh", "api", path, "--paginate"],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    return json.loads(proc.stdout)


def path_matches(patterns: list[str], path: str) -> bool:
    # Match one changed file against gitwildmatch-style globs from tier-policy.yml.
    p = PurePosixPath(path)
    for pat in patterns:
        if p.match(pat):
            return True
        if pat.startswith("**/") and p.match(pat[3:]):
            return True
    return False


def any_path_matches(patterns: list[str], paths: list[str]) -> bool:
    """True if any changed file hits a policy glob."""
    return bool(patterns) and any(path_matches(patterns, p) for p in paths)


def is_manifest_or_lock(path: str) -> bool:
    return any(path.endswith(name) or f"/{name}" in path for name in MANIFEST_LOCK_NAMES)


def is_version_file(path: str) -> bool:
    return path == "VERSION" or path.endswith("/VERSION")


def load_policy(repo: str, base_sha: str, env: dict[str, str]) -> dict[str, Any]:
    # Read policy from merge target — PR cannot tamper with its own rules.
    blob = gh_json(f"repos/{repo}/contents/.github/tier-policy.yml?ref={base_sha}", env)
    raw = base64.b64decode(blob["content"]).decode("utf-8")
    return yaml.safe_load(raw) or {}


def list_changed_files(
    repo: str,
    pr_number: int | None,
    base_sha: str,
    head_sha: str,
    env: dict[str, str],
) -> tuple[list[str], dict[str, Any] | None]:
    # PR event: list files + metadata. Dispatch: git compare base…head, no PR context.
    if pr_number is not None:
        files = gh_json_paginate(f"repos/{repo}/pulls/{pr_number}/files", env)
        changed = [f.get("filename", "") for f in files if f.get("filename")]
        pr_data = gh_json(f"repos/{repo}/pulls/{pr_number}", env)
        return changed, pr_data

    compare = gh_json(f"repos/{repo}/compare/{base_sha}...{head_sha}", env)
    changed = [f.get("filename", "") for f in compare.get("files", []) if f.get("filename")]
    return changed, None


def dependabot_major_bump(pr_data: dict[str, Any], changed: list[str]) -> bool:
    """True when Dependabot opens a semver-major dependency update."""
    if pr_data.get("user", {}).get("login") != "dependabot[bot]":
        return False
    body = pr_data.get("body", "") or ""
    title = pr_data.get("title", "") or ""
    # Prefer Dependabot metadata; fall back to title "from X to Y" major jump.
    metadata_major = bool(
        re.search(r"(?im)update-type:\s*version-update:semver-major", body)
        or re.search(r"(?i)\bsemver-major\b", body)
    )
    version_jump = re.search(
        r"from\s+v?(\d+)\.[^\s]*\s+to\s+v?(\d+)\.[^\s]*",
        f"{title}\n{body}",
    )
    title_major = bool(version_jump and version_jump.group(1) != version_jump.group(2))
    dep_files_changed = any(is_manifest_or_lock(path) for path in changed)
    return metadata_major or (dep_files_changed and title_major)


def compute_tier(
    policy: dict[str, Any],
    changed: list[str],
    pr_data: dict[str, Any] | None,
    *,
    baseline_ok: str,
    run_heavy: bool,
    base_ref: str,
    branch_mode: bool,
) -> tuple[int, list[str]]:
    """Return (tier, reasons). Tier only goes up; every trigger appends a reason."""
    reasons: list[str] = []
    tier = 0

    def require_tier(minimum: int, reason: str) -> None:
        nonlocal tier
        if minimum > tier:
            tier = minimum
        reasons.append(reason)

    # --- Tier 2: deepest CI (Heavy) — checked first -------------------------
    tier2_paths = policy.get("tier_2_paths") or []
    if any_path_matches(tier2_paths, changed):
        require_tier(2, "Tier-2 paths changed (Heavy required).")

    promotion = policy.get("promotion") or {}
    if (
        promotion.get("version_to_main_is_tier_2", True)
        and pr_data
        and (pr_data.get("base") or {}).get("ref") == "main"
        and any(is_version_file(p) for p in changed)
    ):
        require_tier(2, "VERSION change on PR targeting main (Heavy required).")

    # --- Tier 1: Extended CI (integration / SME) ----------------------------
    if any_path_matches(policy.get("locked_paths") or [], changed):
        require_tier(1, "Locked paths changed (Extended required).")
    if any_path_matches(policy.get("sme_owned_paths") or [], changed):
        require_tier(1, "SME-owned paths changed (Extended required).")
    if baseline_ok == "false":
        require_tier(1, "Baseline marker signal failed (Extended required).")

    dependabot_cfg = policy.get("dependabot") or {}
    if (
        dependabot_cfg.get("major_bump_is_tier_1", True)
        and pr_data
        and dependabot_major_bump(pr_data, changed)
    ):
        require_tier(1, "Dependabot semver-major bump (Extended required).")

    # Manual Heavy: only upgrades 1→2, never skips Extended from tier 0.
    if run_heavy:
        if tier >= 1:
            require_tier(2, "Heavy requested via workflow_dispatch (tier 2).")
        else:
            reasons.append("Heavy requested but ignored (tier 0 — no Extended path yet).")

    if branch_mode:
        reasons.insert(0, f"Branch mode: comparing '{base_ref}' to selected head.")

    if tier == 0:
        reasons.append("Tier 0: baseline checks only.")

    return tier, reasons


def write_github_outputs(tier: int, reasons: list[str]) -> None:
    # Expose tier + human/json reasons to later jobs in the same workflow.
    reasons = reasons or ["No tier reason available."]
    out_path = os.environ["GITHUB_OUTPUT"]
    with open(out_path, "a", encoding="utf-8") as out:
        out.write(f"tier={tier}\n")
        out.write(f"reasons={' | '.join(reasons)}\n")
        out.write(f"reasons_json={json.dumps(reasons, ensure_ascii=True)}\n")


def main() -> int:
    # Inputs come from ci.yml tier-decision job env vars.
    repo = os.environ["REPO"]
    pr_number_raw = os.environ.get("PR_NUMBER", "").strip()
    pr_number = int(pr_number_raw) if pr_number_raw.isdigit() else None
    head_sha = os.environ["HEAD_SHA"]
    base_sha = os.environ["BASE_SHA"]
    base_ref = os.environ.get("BASE_REF", "base")
    baseline_ok = os.environ.get("BASELINE_OK", "")
    run_heavy = os.environ.get("RUN_HEAVY", "false").lower() == "true"

    env = os.environ.copy()
    env["GH_TOKEN"] = os.environ["GH_TOKEN"]

    try:
        policy = load_policy(repo, base_sha, env)
    except Exception as err:
        # Fail safe: if policy is unreadable, run Extended rather than skip checks.
        write_github_outputs(
            1,
            [
                f"Policy fetch failed on '{base_ref}' ({base_sha}): {err}",
                "Fallback tier 1 for safety.",
            ],
        )
        return 0

    changed, pr_data = list_changed_files(repo, pr_number, base_sha, head_sha, env)
    tier, reasons = compute_tier(
        policy,
        changed,
        pr_data,
        baseline_ok=baseline_ok,
        run_heavy=run_heavy,
        base_ref=base_ref,
        branch_mode=pr_number is None,
    )
    write_github_outputs(tier, reasons)
    print(f"Tier {tier}: {' | '.join(reasons)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 European Space Agency (ESA) - ACRI-ST
# SPDX-License-Identifier: Apache-2.0
# pre-commit commit-msg: require DCO Signed-off-by trailer.
set -euo pipefail
msg_file="${1:?commit message file required}"
if ! grep -qE '^Signed-off-by: ' "$msg_file"; then
  echo 'error: commit message must include a Signed-off-by line (use: git commit -s)' >&2
  echo '  Or enable always: git config format.signoff true' >&2
  exit 1
fi

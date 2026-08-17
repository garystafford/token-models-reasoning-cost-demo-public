#!/usr/bin/env bash

set -u

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS="${RUNS:-1}"

if [[ ! "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "RUNS must be a positive integer; received: $RUNS" >&2
  exit 2
fi

cd "$PROJECT_ROOT" || exit 1

modules=(
  benchmarks.operations.benchmark_anthropic
  benchmarks.operations.benchmark_openai
  benchmarks.resilience.benchmark_anthropic
  benchmarks.resilience.benchmark_openai
  benchmarks.scientific.benchmark_anthropic
  benchmarks.scientific.benchmark_openai
)

failed=()

for ((run = 1; run <= RUNS; run++)); do
  printf '\nStarting benchmark cycle %d of %d\n' "$run" "$RUNS"

  for module in "${modules[@]}"; do
    printf '\n[%s] Starting %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$module"

    if "$PYTHON_BIN" -u -m "$module"; then
      printf '[%s] Completed %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$module"
    else
      status=$?
      printf '[%s] FAILED %s (exit code %d)\n' \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$module" "$status" >&2
      failed+=("$module, cycle $run, exit code $status")
    fi
  done
done

if ((${#failed[@]} > 0)); then
  printf '\nBenchmark failures:\n' >&2
  printf ' - %s\n' "${failed[@]}" >&2
  exit 1
fi

printf '\nAll benchmark cycles completed successfully.\n'

---
name: aisbench-gsm8k-test
description: Run a simple AISBench GSM8K accuracy test against a vLLM API model on a remote Docker container. Use when the user asks to run ais_bench with vllm_api_general_chat and gsm8k_gen_0_shot_cot_chat_prompt, inspect the editable ais_bench_benchmark install, optionally adjust max_out_len or reduce the GSM8K test.jsonl sample count, and report the summary accuracy.
---

# AISBench GSM8K Test

Run the AISBench GSM8K generation accuracy path for the editable
`ais_bench_benchmark` checkout.

Default target:

- host: `192.168.13.157`
- container: `zyj_aisbench`
- package: `ais_bench_benchmark`
- editable root: discovered from `pip show ais_bench_benchmark`
- model config: `<editable-root>/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py`
- dataset: `<editable-root>/ais_bench/datasets/gsm8k/test.jsonl`
- command: `ais_bench --models vllm_api_general_chat --dataset gsm8k_gen_0_shot_cot_chat_prompt`
- run cwd: `/workspace`
- container shell: `noninteractive` (`bash -lc`)

## Rules

- First check `pip show ais_bench_benchmark` and use `Editable project location`
  as the source tree. Do not assume the package lives under site-packages.
- Do not edit the model config or dataset unless the user asks for a change, or
  the current files are missing the requested setting/sample shape.
- Before modifying the remote model config or dataset, create a timestamped
  backup beside the original file.
- This skill runs AISBench only. It does not launch or restart the vLLM API
  server that `vllm_api_general_chat` calls.
- The GSM8K dataset may be intentionally truncated. Report the detected line
  count with the final accuracy so the user knows the denominator.

## Quick Start

From the workspace root:

```bash
python3 .agents/skills/aisbench-gsm8k-test/scripts/run_aisbench_gsm8k.py
```

Run against the known container and keep the current config/dataset unchanged:

```bash
python3 .agents/skills/aisbench-gsm8k-test/scripts/run_aisbench_gsm8k.py \
  --host 192.168.13.157 \
  --container zyj_aisbench
```

Inspect the editable install and dataset line count without running AISBench:

```bash
python3 .agents/skills/aisbench-gsm8k-test/scripts/run_aisbench_gsm8k.py \
  --inspect-only
```

Temporarily prepare a small 16-sample GSM8K run:

```bash
python3 .agents/skills/aisbench-gsm8k-test/scripts/run_aisbench_gsm8k.py \
  --limit-samples 16
```

Set the request output limit before running:

```bash
python3 .agents/skills/aisbench-gsm8k-test/scripts/run_aisbench_gsm8k.py \
  --max-out-len 512
```

The script writes progress to stderr and one final JSON object to stdout.
Use `--container-shell interactive` only if the noninteractive shell cannot
find the configured AISBench environment.

## Workflow

1. Resolve the editable install:

```bash
ssh <host> "docker exec <container> bash -lc 'python3 -m pip show ais_bench_benchmark'"
```

Use the `Editable project location` field. For the current remote this is
normally `/home/zyj/scripts/benchmark`.

2. Confirm paths:

```text
<editable-root>/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
<editable-root>/ais_bench/datasets/gsm8k/test.jsonl
```

3. If requested, patch `max_out_len` in the model config. If requested, reduce
   `test.jsonl` to the first `N` samples. Back up both files first.

4. Run:

```bash
ais_bench --models vllm_api_general_chat --dataset gsm8k_gen_0_shot_cot_chat_prompt
```

5. Parse the summary path from AISBench output, read the generated markdown
   summary, and report:

- editable project location
- model config path
- dataset path and line count
- summary markdown path
- GSM8K accuracy

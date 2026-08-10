---
name: aisbench-test
description: Run AISBench GSM8K or GPQA accuracy tests against a vLLM API model on a remote Docker container. Use when the user asks to run ais_bench with vllm_api_general_chat and either gsm8k_gen_0_shot_cot_chat_prompt or gpqa_gen_0_shot_cot_chat_prompt, inspect the editable ais_bench_benchmark install, optionally adjust max_out_len or reduce the selected dataset sample count, and report summary accuracy.
---

# AISBench Test

Run an AISBench generation accuracy test from the editable
`ais_bench_benchmark` checkout. GSM8K is the default; select GPQA explicitly.

Default target:

- host: `192.168.13.157`
- container: `zyj_aisbench`
- package: `ais_bench_benchmark`
- editable root: discovered from `pip show ais_bench_benchmark`
- model config: `<editable-root>/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py`
- GSM8K dataset: `<editable-root>/ais_bench/datasets/gsm8k/test.jsonl`
- GPQA dataset: `<editable-root>/ais_bench/datasets/gpqa/gpqa_diamond.csv`
- GSM8K AISBench ID: `gsm8k_gen_0_shot_cot_chat_prompt`
- GPQA AISBench ID: `gpqa_gen_0_shot_cot_chat_prompt`
- run cwd: `/workspace`
- container shell: `noninteractive` (`bash -lc`)

## Rules

- The root/main agent must delegate the remote AISBench run and result evidence
  according to `../remote-toolbox/references/subagent-verification.md`. An
  already-delegated subagent executes directly without recursive delegation.
- First check `pip show ais_bench_benchmark` and use `Editable project location`
  as the source tree. Do not assume the package lives under site-packages.
- Do not edit the model config or dataset unless the user asks for a change, or
  the current files are missing the requested setting/sample shape.
- Before modifying the remote model config or dataset, create a timestamped
  backup beside the original file.
- This skill runs AISBench only. It does not launch or restart the vLLM API
  server that `vllm_api_general_chat` calls.
- A dataset may be intentionally truncated. Report the selected dataset and
  detected sample count with the final accuracy so the user knows the denominator.

## Quick Start

From the workspace root:

```bash
python3 .agents/skills/aisbench-test/scripts/run_aisbench.py
```

Run against the known container and keep the current config/dataset unchanged:

```bash
python3 .agents/skills/aisbench-test/scripts/run_aisbench.py \
  --host 192.168.13.157 \
  --container zyj_aisbench
```

Run GPQA:

```bash
python3 .agents/skills/aisbench-test/scripts/run_aisbench.py \
  --dataset gpqa
```

Inspect the editable install and dataset sample count without running AISBench:

```bash
python3 .agents/skills/aisbench-test/scripts/run_aisbench.py \
  --inspect-only
```

Temporarily prepare a small 16-sample run of the selected dataset:

```bash
python3 .agents/skills/aisbench-test/scripts/run_aisbench.py \
  --limit-samples 16
```

Set the request output limit before running:

```bash
python3 .agents/skills/aisbench-test/scripts/run_aisbench.py \
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
<editable-root>/ais_bench/datasets/gpqa/gpqa_diamond.csv
```

3. If requested, patch `max_out_len` in the model config. If requested, reduce
   the selected dataset to its first `N` samples. Preserve the GPQA CSV header
   and back up every modified file first.

4. Run the selected dataset:

```bash
# GSM8K
ais_bench --models vllm_api_general_chat --dataset gsm8k_gen_0_shot_cot_chat_prompt

# GPQA
ais_bench --models vllm_api_general_chat --dataset gpqa_gen_0_shot_cot_chat_prompt
```

5. Parse the summary path from AISBench output, read the generated markdown
   summary, and report:

- editable project location
- model config path
- selected dataset, AISBench dataset ID, path, and sample count
- summary markdown path
- accuracy

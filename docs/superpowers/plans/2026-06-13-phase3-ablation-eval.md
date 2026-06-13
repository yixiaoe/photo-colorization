# Phase 3 Ablation Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add isolated Phase 3 prompt-control evaluation and short loss ablation runners without changing the main training path.

**Architecture:** Keep `train_phase3.py` unchanged. Add one evaluation script that loads existing checkpoints and writes metrics/images, plus one shell runner that launches 3-epoch rank and outside sweeps under unique experiment names.

**Tech Stack:** Python, PyTorch, PIL, existing Phase 3 dataset/model utilities, Bash.

---

### Task 1: Prompt-Control Metric Helpers

**Files:**
- Create: `code/tests/phase3/test_prompt_control_eval.py`
- Create: `code/scripts/eval_phase3_prompt_control.py`

- [ ] Write failing tests for checkpoint summarization and TV metrics.
- [ ] Run `cd code && pytest tests/phase3/test_prompt_control_eval.py -q` and verify import failure.
- [ ] Implement minimal helper functions in `scripts/eval_phase3_prompt_control.py`.
- [ ] Run the test and verify it passes.

### Task 2: Evaluation CLI

**Files:**
- Modify: `code/scripts/eval_phase3_prompt_control.py`

- [ ] Add CLI args for records, checkpoints, model name, output directory, sample limit, and rank margin.
- [ ] Reuse `CocoColorObjectDataset`, `TextColorModel`, and existing color utilities.
- [ ] Write `metrics.csv`, `summary.json`, and prompt comparison images per checkpoint.

### Task 3: Short Ablation Runner

**Files:**
- Create: `code/scripts/run_phase3_loss_ablation.sh`

- [ ] Add `rank` mode for `lambda_rank=10,50,100` with `lambda_outside=0.2`.
- [ ] Add `outside <rank>` mode for `lambda_outside=1,5,10` with the selected rank.
- [ ] Use unique experiment names and run the evaluation CLI after each short training run.
- [ ] Validate with `bash -n code/scripts/run_phase3_loss_ablation.sh`.

### Task 4: Upload

**Files:**
- Upload modified/new scripts and tests to `/root/autodl-tmp/photo-colorization/code` on the remote host.

- [ ] Run local tests and syntax checks.
- [ ] Copy files to the remote host.
- [ ] Run remote syntax/import checks without starting training.

# Weco × Talkie HumanEval

Optimizing [Talkie-1930-13b](https://huggingface.co/talkie-lm/talkie-1930-13b-base)'s HumanEval performance via [Weco](https://weco.ai) harness optimization.

**Thesis:** Talkie-1930-13b has never seen code during pre-training (pre-1931 data), yet it can solve simple HumanEval problems via in-context learning. By optimizing the *harness* (ICL example selection, prompt format, output parsing) with Weco's tree search, we can unlock more of this latent capability — without changing model weights.

## Setup

### Prerequisites
- Python >= 3.11
- CUDA GPU with >= 28 GB VRAM
- [Weco CLI](https://github.com/WecoAI/weco-cli)
- [Talkie](https://github.com/talkie-lm/talkie)

### Install

```bash
git clone https://github.com/DTennant/weco-talkie-humaneval.git
cd weco-talkie-humaneval
pip install -r requirements.txt

# Install Talkie
pip install git+https://github.com/talkie-lm/talkie.git

# Install Weco CLI
pip install weco
```

### Download the model (first time)

```bash
python -c "from talkie import download_model; download_model('talkie-1930-13b-base')"
```

## Usage

### 1. Run baseline evaluation (no optimization)

```bash
python evaluate.py
```

This runs Talkie-1930-13b on HumanEval with the default harness and prints `pass_at_1`, `pass_at_10`, and `pass_at_100` (the primary metric from the Talkie blog, Figure 3).

### 2. Optimize the harness with Weco

```bash
weco run \
  --source harness.py \
  --eval-command "python evaluate.py" \
  --metric pass_at_100 \
  --goal maximize \
  --steps 50 \
  --additional-instructions "Optimize the ICL example selection strategy, prompt template, and output parsing to maximize HumanEval pass@100 for a vintage language model that has never seen code. The model is talkie-1930-13b-base. Do NOT modify the model or the evaluation logic — only optimize the harness (how we prompt the model). Key constraints: the model has 13B params, pre-1931 training data only, and limited context window."
```

### 3. Check results on the dashboard

After the run, check [dashboard.weco.ai](https://dashboard.weco.ai) for the optimization trajectory.

## Project Structure

```
├── harness.py         # The harness that Weco optimizes (ICL selection, prompt template, parsing)
├── evaluate.py        # Evaluation script — runs harness on HumanEval, prints metrics
├── requirements.txt   # Python dependencies
└── README.md
```

## How It Works

1. **`harness.py`** defines how we prompt Talkie for each HumanEval problem:
   - Selects which HumanEval problems to use as ICL (in-context learning) examples
   - Formats the prompt with examples + the target problem
   - Parses the model's output to extract the generated code

2. **`evaluate.py`** loads all 164 HumanEval problems, calls the harness for each one, executes the generated code, and computes pass@k metrics.

3. **Weco** iterates on `harness.py` — trying different ICL selection strategies, prompt formats, number of examples, chain-of-thought templates, etc. — guided by the pass@1 metric from `evaluate.py`.

## References

- [Talkie blog post](https://talkie-lm.com/introducing-talkie) — Figure 3 shows Talkie's HumanEval baseline
- [HumanEval](https://github.com/openai/human-eval) — OpenAI's code generation benchmark
- [Weco CLI](https://github.com/WecoAI/weco-cli) — AI-powered code optimization

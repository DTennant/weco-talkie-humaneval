"""
Harness for prompting Talkie-1930-13b-it on HumanEval problems.

This file is what Weco optimizes. It controls:
  1. ICL example selection — which solved problems to show the model
  2. Prompt template — how examples and the target problem are formatted
  3. Output parsing — how to extract code from the model's generation
  4. Generation parameters — temperature, max_tokens, etc.

Uses the Talkie IT model's chat API (model.chat with Message objects).
Baseline: single-turn chat (one user message with ICL examples + target).

Do NOT put evaluation logic here. That lives in evaluate.py.
"""

import re
import random
from typing import Optional
from datasets import load_dataset
from talkie import Message

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (Weco will explore variations of these)
# ──────────────────────────────────────────────────────────────────────────────

NUM_ICL_EXAMPLES = 3
TEMPERATURE = 0.8
MAX_TOKENS = 512

# ICL example selection strategy: "random", "fixed", "similar_length"
ICL_SELECTION_STRATEGY = "random"

# Seed for reproducible ICL selection when strategy is "random"
ICL_SEED = 42

# Fixed ICL example indices (used when strategy is "fixed")
# These are HumanEval task_ids that have simple, short solutions
FIXED_ICL_INDICES = [
    "HumanEval/0",   # has_close_elements
    "HumanEval/1",   # separate_paren_groups
    "HumanEval/2",   # truncate_number
]


# ──────────────────────────────────────────────────────────────────────────────
# ICL Example Pool
# ──────────────────────────────────────────────────────────────────────────────

def _load_humaneval():
    """Load HumanEval dataset."""
    ds = load_dataset("openai/openai_humaneval", split="test")
    return {row["task_id"]: row for row in ds}


_HUMANEVAL_CACHE = None

def get_humaneval():
    global _HUMANEVAL_CACHE
    if _HUMANEVAL_CACHE is None:
        _HUMANEVAL_CACHE = _load_humaneval()
    return _HUMANEVAL_CACHE


def select_icl_examples(target_task_id: str, n: int = NUM_ICL_EXAMPLES) -> list[dict]:
    """
    Select ICL examples to show the model before the target problem.
    
    Returns a list of dicts with keys: prompt, canonical_solution, task_id
    """
    humaneval = get_humaneval()
    
    # Exclude the target problem from the pool
    pool = {k: v for k, v in humaneval.items() if k != target_task_id}
    
    if ICL_SELECTION_STRATEGY == "fixed":
        indices = [idx for idx in FIXED_ICL_INDICES if idx != target_task_id]
        examples = [pool[idx] for idx in indices[:n] if idx in pool]
        
    elif ICL_SELECTION_STRATEGY == "similar_length":
        target_len = len(humaneval[target_task_id]["prompt"])
        sorted_pool = sorted(pool.values(), key=lambda x: abs(len(x["prompt"]) - target_len))
        examples = sorted_pool[:n]
        
    else:
        # Random selection (default)
        rng = random.Random(ICL_SEED)
        examples = rng.sample(list(pool.values()), min(n, len(pool)))
    
    return examples


# ──────────────────────────────────────────────────────────────────────────────
# Prompt Template
# ──────────────────────────────────────────────────────────────────────────────

def format_icl_example(example: dict) -> str:
    """Format a single ICL example (prompt + solution)."""
    return f"{example['prompt']}{example['canonical_solution']}"


def build_messages(target_task: dict, icl_examples: list[dict]) -> list[Message]:
    """
    Build the chat messages for the IT model.
    
    Baseline: single user turn containing ICL examples + target problem.
    The model responds as assistant with the completion.
    
    Chat template: <|user|>{content}<|end|><|assistant|>
    """
    # Build user message content
    parts = []
    
    # Instruction
    parts.append(
        "Below are Python function implementations. "
        "Complete the last function following the same pattern.\n\n"
    )
    
    # ICL examples (complete functions)
    for ex in icl_examples:
        parts.append(format_icl_example(ex))
        parts.append("\n\n")
    
    # Target problem (just the prompt/signature + docstring, model completes it)
    parts.append(target_task["prompt"])
    
    user_content = "".join(parts)
    
    messages = [
        Message(role="user", content=user_content),
    ]
    
    return messages


# ──────────────────────────────────────────────────────────────────────────────
# Output Parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_completion(raw_output: str) -> str:
    """
    Extract the function body from the model's raw output.
    
    The model generates the continuation of the function.
    We stop at markers that indicate the function is done.
    """
    completion = raw_output
    
    # Stop at the next function definition or class definition
    stop_sequences = ["\ndef ", "\nclass ", "\n# ", "\nif __name__"]
    for stop in stop_sequences:
        idx = completion.find(stop)
        if idx != -1:
            completion = completion[:idx]
    
    # Stop at triple backticks
    idx = completion.find("```")
    if idx != -1:
        completion = completion[:idx]
    
    return completion


# ──────────────────────────────────────────────────────────────────────────────
# Main Interface (called by evaluate.py)
# ──────────────────────────────────────────────────────────────────────────────

def generate_completion(
    model,
    task_id: str,
    num_samples: int = 1,
) -> list[str]:
    """
    Generate code completions for a HumanEval task using Talkie IT chat API.
    
    Uses model.chat() with Message objects — single turn (one user message).
    
    Args:
        model: A Talkie model instance (talkie-1930-13b-it)
        task_id: HumanEval task ID (e.g., "HumanEval/0")
        num_samples: Number of completions to generate (for pass@k)
    
    Returns:
        List of completion strings (just the function body)
    """
    humaneval = get_humaneval()
    target_task = humaneval[task_id]
    
    # Select ICL examples
    icl_examples = select_icl_examples(task_id)
    
    # Build chat messages
    messages = build_messages(target_task, icl_examples)
    
    # Generate completions
    completions = []
    for _ in range(num_samples):
        result = model.chat(
            messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        
        completion = parse_completion(result.text)
        completions.append(completion)
    
    return completions

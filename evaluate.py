"""
Evaluation script for Talkie-1930-13b on HumanEval.

This script:
  1. Loads the Talkie model
  2. Runs the harness on each HumanEval problem
  3. Executes the generated code in a sandbox
  4. Computes and prints pass@1 and pass@10

Weco reads the printed metrics to guide optimization of harness.py.
Do NOT modify this file — Weco only optimizes harness.py.
"""

import argparse
import json
import os
import sys
import itertools
import numpy as np
from typing import Optional

from datasets import load_dataset
from human_eval.evaluation import estimate_pass_at_k
from human_eval.execution import check_correctness

from harness import generate_completion


def evaluate(
    model_name: str = "talkie-1930-13b-base",
    num_samples_per_task: int = 10,
    timeout: float = 10.0,
    max_problems: Optional[int] = None,
):
    """
    Run HumanEval evaluation.
    
    Args:
        model_name: Which Talkie model to use
        num_samples_per_task: Number of completions per problem (for pass@k)
        timeout: Seconds to allow each code execution
        max_problems: If set, only evaluate this many problems (for faster iteration)
    """
    # Load model
    print(f"Loading model: {model_name}", file=sys.stderr)
    from talkie import Talkie
    model = Talkie(model_name)
    
    # Load HumanEval
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = list(ds)
    
    if max_problems is not None:
        problems = problems[:max_problems]
    
    print(f"Evaluating {len(problems)} problems, {num_samples_per_task} samples each", file=sys.stderr)
    
    # Generate completions for each problem
    results = []
    num_correct = np.zeros(len(problems))
    
    for i, problem in enumerate(problems):
        task_id = problem["task_id"]
        print(f"  [{i+1}/{len(problems)}] {task_id}", file=sys.stderr)
        
        # Generate completions using the harness
        completions = generate_completion(
            model=model,
            task_id=task_id,
            num_samples=num_samples_per_task,
        )
        
        # Check correctness of each completion
        correct = 0
        for j, completion in enumerate(completions):
            # Build the full program: prompt + completion + test cases
            full_code = problem["prompt"] + completion + "\n" + problem["test"]
            
            result = check_correctness(
                problem={"task_id": task_id, "prompt": problem["prompt"],
                         "canonical_solution": problem["canonical_solution"],
                         "test": problem["test"], "entry_point": problem["entry_point"],
                         "completion": completion},
                timeout=timeout,
                completion_id=j,
            )
            
            if result["passed"]:
                correct += 1
        
        num_correct[i] = correct
        
        if correct > 0:
            print(f"    ✓ {correct}/{num_samples_per_task} passed", file=sys.stderr)
    
    # Compute pass@k
    total = np.array([num_samples_per_task] * len(problems))
    
    pass_at_1 = estimate_pass_at_k(total, num_correct, 1).mean()
    
    if num_samples_per_task >= 10:
        pass_at_10 = estimate_pass_at_k(total, num_correct, 10).mean()
    else:
        pass_at_10 = 0.0
    
    num_solved = (num_correct > 0).sum()
    
    # Print metrics for Weco to read
    print(f"pass_at_1: {pass_at_1:.4f}")
    print(f"pass_at_10: {pass_at_10:.4f}")
    print(f"problems_solved: {num_solved}/{len(problems)}")
    
    # Also print to stderr for human readability
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Results: pass@1={pass_at_1:.4f}, pass@10={pass_at_10:.4f}", file=sys.stderr)
    print(f"Solved: {num_solved}/{len(problems)} problems", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    
    return pass_at_1, pass_at_10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Talkie on HumanEval")
    parser.add_argument("--model", default="talkie-1930-13b-base", help="Model name")
    parser.add_argument("--samples", type=int, default=10, help="Samples per problem")
    parser.add_argument("--timeout", type=float, default=10.0, help="Execution timeout (s)")
    parser.add_argument("--max-problems", type=int, default=None, help="Max problems to eval (for debugging)")
    
    args = parser.parse_args()
    
    evaluate(
        model_name=args.model,
        num_samples_per_task=args.samples,
        timeout=args.timeout,
        max_problems=args.max_problems,
    )

import json
import os
import sys

# Qualitative Scale Behavioral Anchors (Task 3B requirement)
RUBRIC = """
================================================================================
QUALITATIVE EVALUATION RUBRIC (1 - 3 Scale)
================================================================================
1. Factual Correctness
   (1) Incorrect/Hallucinated : Makes false claims contrary to quantum history.
   (2) Partially Correct      : Mixes truth with minor inaccuracies or omissions.
   (3) Fully Correct          : Exclusively factual according to known history.

2. Completeness
   (1) Incomplete             : Fails to answer the core question or is missing key parts.
   (2) Satisfactory           : Answers the main question but lacks detail.
   (3) Comprehensive          : Fully addresses the question and provides nuanced context.

3. Coherence
   (1) Confusing              : Disjointed phrasing, grammatically poor, or illogical.
   (2) Readable               : Understandable but slightly awkward or repetitive.
   (3) Fluent                 : Professional, encyclopedic tone with excellent flow.

4. Grounding (Does it stick to the source?)
   (1) Un-grounded            : Heavily relies on outside knowledge not in the text.
   (2) Somewhat Grounded      : Mostly uses text, but brings in some external context.
   (3) Perfectly Grounded     : Strictly derives the answer ONLY from the source text.
================================================================================
"""

def print_evaluation_screen(q_num: int, total: int, question: str, expected: str, generated: str):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(RUBRIC)
    print(f"--- Question {q_num} of {total} ---")
    print(f"\n[Q]: {question}")
    print(f"\n[EXPECTED]:\n{expected}")
    print(f"\n[GENERATED]:\n{generated}")
    print("\n" + "-" * 80)
    print("Please input scores (1-3) for the following dimensions:")

def get_score(prompt_text: str):
    while True:
        val = input(prompt_text).strip().lower()
        if val in ['1', '2', '3']:
            return int(val)
        elif val == 'n':
            return None
        print("    [!] Invalid input. Please enter 1, 2, 3, or 'n' to skip.")

def run_cli():
    print("Loading Ground Truth and generating answers...")
    # Normally we would load generated answers. Since this is an interactive script,
    # we simulate the generation output or load from a pre-generated file.
    
    # We will assume a file `data/generated_results.json` exists for offline grading.
    # If not, we will just use the expected answer as a placeholder for UI mockup.
    res_path = os.path.join("data", "generated_results.json")
    if not os.path.exists(res_path):
        print(f"Warning: {res_path} not found. Using placeholder generated text for UI demonstration.")
        with open(os.path.join("data", "ground_truth.json"), "r") as f:
            data = json.load(f)
            for item in data:
                item["generated_answer"] = item["expected_answer"] + " (Generated Mock)"
    else:
        with open(res_path, "r") as f:
            data = json.load(f)
            
    scores = []
    
    for idx, item in enumerate(data, start=1):
        print_evaluation_screen(
            idx, 
            len(data), 
            item["question"], 
            item["expected_answer"], 
            item.get("generated_answer", "No generation found.")
        )
        
        try:
            fc = get_score("  Factual Correctness (1-3) [n to skip]: ")
            comp = get_score("  Completeness        (1-3) [n to skip]: ")
            coh = get_score("  Coherence           (1-3) [n to skip]: ")
            grnd = get_score("  Grounding           (1-3) [n to skip]: ")
            
            # Simple Empty Table formatting for terminal
            def fmt(v): return " " if v is None else str(v)
            print("\n  +-----------------------+-------+")
            print(f"  | Factual Correctness   |   {fmt(fc)}   |")
            print(f"  | Completeness          |   {fmt(comp)}   |")
            print(f"  | Coherence             |   {fmt(coh)}   |")
            print(f"  | Grounding             |   {fmt(grnd)}   |")
            print("  +-----------------------+-------+")
            
            input("\nPress Enter to continue to the next question...")
            
            scores.append({
                "question": item["question"],
                "scores": {
                    "factual_correctness": fc,
                    "completeness": comp,
                    "coherence": coh,
                    "grounding": grnd
                }
            })
        except KeyboardInterrupt:
            print("\nEvaluation gracefully aborted by user.")
            sys.exit(0)
            
    # Save the human evaluation results
    out_path = os.path.join("data", "human_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(scores, f, indent=4)
    print(f"\nEvaluation complete! Results saved to {out_path}")

if __name__ == "__main__":
    run_cli()

"""Main agent script for the Mathematical Modeling Agent.

This script implements a sophisticated workflow:
1.  Problem Extraction: It first reads a PDF and asks the LLM to identify and list out the distinct problems.
2.  Sequential Solving: It then tackles each problem one by one.
3.  Iterative Refinement: For each problem, it uses a multi-round Model -> Analyze -> Correct loop with acceptance and rejection criteria to robustly refine the solution.
"""

import os
import sys
import argparse
import glob
import re
import fitz  # PyMuPDF
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Import the prompts from our custom module
from prompts import MODELER_SYSTEM_PROMPT, ANALYZER_SYSTEM_PROMPT, CORRECTION_PROMPT
from cache_utils import setup_cache, get_cache_key, get_cached_data, set_cached_data, get_file_cache_key

# --- CONFIGURATION & SETUP ---

# Load environment variables from .env file
load_dotenv()

# Fetch API configuration from environment
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
BASE_URL = os.getenv("BASE_URL", "https://aihubmix.com/gemini")

LOGS_DIR = "run_logs"
PLOTS_DIR = os.path.join(LOGS_DIR, "plots")

_log_file = None
original_print = print

def log_print(*args, **kwargs):
    original_print(*args, **kwargs)
    if _log_file:
        message = ' '.join(str(arg) for arg in args)
        _log_file.write(message + '\n')
        _log_file.flush()

print = log_print

# --- HELPER FUNCTIONS ---

def setup_logging(log_file_path):
    global _log_file
    if _log_file: # Close previous log file if any
        _log_file.close()
    try:
        # Ensure the logs directory exists
        os.makedirs(LOGS_DIR, exist_ok=True)
        full_log_path = os.path.join(LOGS_DIR, log_file_path)
        _log_file = open(full_log_path, 'w', encoding='utf-8')
        original_print(f"Logging to file: {full_log_path}")
    except IOError as e:
        original_print(f"Error opening log file {full_log_path}: {e}")
        sys.exit(1)

def close_logging():
    global _log_file
    if _log_file:
        _log_file.close()
        _log_file = None

def read_file_content(filepath):
    cache_key = get_file_cache_key(filepath)
    if cache_key and (cached_content := get_cached_data(cache_key)) is not None:
        print(f"Cache hit for file content: {filepath}")
        return cached_content

    print(f"Cache miss for file content: {filepath}")
    try:
        if os.path.splitext(filepath)[1].lower() == '.pdf':
            doc = fitz.open(filepath)
            text = "".join(page.get_text() for page in doc)
            doc.close()
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        if cache_key:
            set_cached_data(cache_key, text)
        return text
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

def get_attachment_preview(filepath, num_lines=5):
    cache_key = get_file_cache_key(filepath)
    if cache_key and (cached_preview := get_cached_data(cache_key)) is not None:
        print(f"Cache hit for attachment preview: {filepath}")
        return cached_preview

    print(f"Cache miss for attachment preview: {filepath}")
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.csv':
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [next(f) for _ in range(num_lines)]
            preview = "File Preview (first 5 rows):\n" + "".join(lines)
        elif ext == '.xlsx':
            preview = "File Type: Excel (.xlsx)\nPreview: Not available. Please use pandas.read_excel() to load."
        else:
            preview = f"File Type: {ext}\nPreview: Not supported."
    except StopIteration:
        preview = "File Preview:\n" + read_file_content(filepath)
    except Exception as e:
        preview = f"Error reading attachment file '{filepath}': {e}"

    if cache_key:
        set_cached_data(cache_key, preview)
    return preview

def generate_with_thinking(system_prompt: str, user_prompt: str, thinking_log_path: str):
    if not API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set. Please create a .env file and add your API key.")
        sys.exit(1)

    os.makedirs(LOGS_DIR, exist_ok=True)
    full_thinking_log_path = os.path.join(LOGS_DIR, thinking_log_path)

    cache_key = get_cache_key(f"{system_prompt}{user_prompt}")
    if (cached_response := get_cached_data(cache_key)) is not None:
        print(f"Cache hit for API request. Skipping generation. See log: {full_thinking_log_path}")
        with open(full_thinking_log_path, "w", encoding="utf-8") as f_log:
             f_log.write("-- Cached Response --\n\n" + cached_response)
        return cached_response

    print(f"Cache miss for API request. Generating... See thinking log: {full_thinking_log_path}")
    client = genai.Client(api_key=API_KEY, http_options={"base_url": BASE_URL})
    full_prompt = f"{system_prompt}\n\n--- USER REQUEST ---\n\n{user_prompt}"
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=full_prompt)])]
    config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=8192),
    )

    full_response = ""
    try:
        with open(full_thinking_log_path, "w", encoding="utf-8") as f_log:
            f_log.write(f"-- Thinking Process for {os.path.basename(full_thinking_log_path)} --\n\n")
            stream = client.models.generate_content_stream(model=MODEL_NAME, contents=contents, config=config)
            for chunk in stream:
                if chunk.candidates:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'thought') and part.thought:
                            f_log.write(part.text)
                        elif part.text:
                            full_response += part.text
    except Exception as e:
        print(f"An error occurred during Gemini API call: {e}")
        return None

    set_cached_data(cache_key, full_response)
    return full_response

def extract_problems(full_text: str) -> list[str]:
    print("\n--- Stage 1: Extracting Individual Problems from PDF Text ---")
    extraction_system_prompt = "You are an expert at parsing academic and contest documents. Read the following text and identify the distinct problems or questions, often labeled as '问题一', '问题二', etc. List each problem's full, verbatim text. Use a clear separator '###---PROBLEM-SEPARATOR---###' between each distinct problem."
    extraction_user_prompt = full_text
    extraction_thinking_log = "thinking_log_extraction.txt"

    extracted_text = generate_with_thinking(extraction_system_prompt, extraction_user_prompt, extraction_thinking_log)

    if not extracted_text:
        print("Failed to extract problems. Exiting.")
        return []

    problems = extracted_text.split("###---PROBLEM-SEPARATOR---###")
    cleaned_problems = [p.strip() for p in problems if p.strip()]
    print(f"Successfully extracted {len(cleaned_problems)} problems.")
    return cleaned_problems

def parse_verdict(analysis_text: str) -> str:
    """Parses the final verdict from the analyzer's response."""
    verdict_match = re.search(r"FINAL VERDICT: \[(.*?)\]", analysis_text)
    if verdict_match:
        return verdict_match.group(1)
    return "Unknown"

def run_solution_workflow_for_problem(problem_text: str, attachments_info: str, problem_index: int, previous_solutions: list[str]):
    setup_logging(f"run_log_problem_{problem_index + 1}.txt")
    print(f"\n{'='*25} Solving Problem {problem_index + 1} of {len(previous_solutions)+1} {'='*25}")
    print(f"Problem Statement:\n{problem_text}")

    # --- ITERATION CONTROL PARAMETERS ---
    max_iterations = 10  # Rejection criteria
    consecutive_passes_needed = 3 # Acceptance criteria
    # -------------------------------------

    iteration_count = 0
    consecutive_passes = 0
    current_solution = None
    analysis = "Initial analysis before first modeling attempt."

    # Initial modeling context
    context = "".join(f"### Solution to Problem {i+1}\n{sol}\n\n" for i, sol in enumerate(previous_solutions))
    initial_user_prompt = f"{context}### Attachments Information\n{attachments_info}\n\n### Current Problem to Solve:\n{problem_text}"

    while iteration_count < max_iterations:
        iteration_count += 1
        print(f"\n{'-'*20} Iteration {iteration_count} {'-'*20}")

        # --- Step 1: Modeling or Correction ---
        if iteration_count == 1:
            print(f"--- Step 1.{iteration_count}: Contacting Modeler Agent for Initial Solution ---")
            thinking_log_path = f"thinking_log_problem_{problem_index+1}_modeler_{iteration_count}.txt"
            current_solution = generate_with_thinking(MODELER_SYSTEM_PROMPT, initial_user_prompt, thinking_log_path)
        else:
            print(f"--- Step 1.{iteration_count}: Contacting Modeler Agent for Correction ---")
            correction_user_prompt = f"### Previous Solution:\n{current_solution}\n\n### Critique from last iteration:\n{analysis}"
            thinking_log_path = f"thinking_log_problem_{problem_index+1}_modeler_{iteration_count}.txt"
            current_solution = generate_with_thinking(CORRECTION_PROMPT, correction_user_prompt, thinking_log_path)

        if not current_solution:
            print(f"Modeler Agent failed in iteration {iteration_count}. Aborting this problem.")
            return None
        
        print(f"--- Modeler Agent's Solution (Iteration {iteration_count}) ---\n{current_solution}\n---------------------------------")

        # --- Step 2: Analysis of the new solution ---
        print(f"\n--- Step 2.{iteration_count}: Contacting Analyzer Agent for Verification ---")
        analyzer_user_prompt = f"""Please analyze the following solution for the specific problem: '{problem_text[:100]}...'\n\n---\n{current_solution}\n---"""
        thinking_log_path = f"thinking_log_problem_{problem_index+1}_analyzer_{iteration_count}.txt"
        analysis = generate_with_thinking(ANALYZER_SYSTEM_PROMPT, analyzer_user_prompt, thinking_log_path)
        
        if not analysis:
            print(f"Analyzer Agent failed in iteration {iteration_count}. Aborting this problem.")
            return None
            
        print(f"""--- Analyzer Agent's Critique (Iteration {iteration_count}) ---\n{analysis}
----------------------------------""")

        # --- Step 3: Check Verdict and Control Loop ---
        verdict = parse_verdict(analysis)
        print(f"Analyzer Verdict: {verdict}")

        if verdict == "Excellent":
            consecutive_passes += 1
            print(f"Solution passed validation. Consecutive passes: {consecutive_passes}/{consecutive_passes_needed}")
            if consecutive_passes >= consecutive_passes_needed:
                print(f"\nSolution for Problem {problem_index + 1} has been accepted after {consecutive_passes} consecutive passes.")
                return current_solution # SUCCESS EXIT
        else:
            # Any verdict other than Excellent means issues were found, so reset the pass counter.
            print("Issues found. Resetting consecutive pass counter and continuing to next correction iteration.")
            consecutive_passes = 0

    # If the loop finishes, it means we hit the max_iterations limit
    print(f"\nFailed to produce a satisfactory solution for Problem {problem_index + 1} after {max_iterations} iterations. Aborting.")
    return None # FAILURE EXIT

# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    setup_cache()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Math Modeling Agent - v3.0 with Iterative Refinement")
    parser.add_argument('--dir', '-d', type=str, required=True, help='Path to the problem directory.')
    args = parser.parse_args()

    problem_dir = args.dir
    if not os.path.isdir(problem_dir):
        print(f"Error: Directory not found at '{problem_dir}'")
        sys.exit(1)

    pdf_files = glob.glob(os.path.join(problem_dir, '*.pdf'))
    if not pdf_files:
        print(f"Error: No PDF file found in '{problem_dir}'")
        sys.exit(1)
    problem_pdf_path = pdf_files[0]

    full_text = read_file_content(problem_pdf_path)
    problems_to_solve = extract_problems(full_text)

    if not problems_to_solve:
        sys.exit(1)

    all_files = glob.glob(os.path.join(problem_dir, '*.*'))
    attachment_paths = [f for f in all_files if os.path.abspath(f) != os.path.abspath(problem_pdf_path)]
    attachments_info = ""
    if attachment_paths:
        attachments_info = "\n*** Attachment Files Information ***\n"
        for i, file_path in enumerate(attachment_paths):
            attachments_info += f"\n--- Attachment {i+1} ---\nFile Path: {os.path.abspath(file_path)}\n{get_attachment_preview(file_path)}\n"

    final_solutions = []
    for i, problem_text in enumerate(problems_to_solve):
        solution_code = run_solution_workflow_for_problem(problem_text, attachments_info, i, final_solutions)
        if solution_code:
            final_solutions.append(solution_code)
        else:
            print(f"Could not generate a final solution for Problem {i+1}. Stopping.")
            break
    
    print("\n\n" + "#"*30 + " FINAL SOLUTIONS " + "#"*30)
    for i, sol in enumerate(final_solutions):
        print(f"\n--- Final Solution for Problem {i+1} ---\n{sol}")

    close_logging()
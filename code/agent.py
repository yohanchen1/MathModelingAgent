"""
Main agent script for the Mathematical Modeling Agent.

This script orchestrates the workflow of solving a mathematical modeling problem
by using a Modeler Agent and an Analyzer Agent.
"""

import os
import sys
import argparse
import json
from dotenv import load_dotenv
from openai import OpenAI

# Import the prompts from our custom module
from prompts import MODELER_SYSTEM_PROMPT, ANALYZER_SYSTEM_PROMPT, CORRECTION_PROMPT

# --- CONFIGURATION & SETUP ---

# Load environment variables from .env file
load_dotenv()

# Fetch API configuration from environment
API_KEY = os.getenv("GEMINI_API_KEY") # Using GEMINI_API_KEY for clarity
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-pro-latest")
API_URL = os.getenv("API_URL", "https://generativelanguage.googleapis.com/v1beta")

# Global logger
_log_file = None
original_print = print

def log_print(*args, **kwargs):
    """Custom print to also write to a log file if specified."""
    original_print(*args, **kwargs)
    if _log_file:
        message = ' '.join(str(arg) for arg in args)
        _log_file.write(message + '\n')
        _log_file.flush()

print = log_print

# --- HELPER FUNCTIONS ---

def setup_logging(log_file_path):
    """Initializes the log file."""
    global _log_file
    try:
        _log_file = open(log_file_path, 'w', encoding='utf-8')
    except IOError as e:
        original_print(f"Error opening log file {log_file_path}: {e}")
        sys.exit(1)

def close_logging():
    """Closes the log file."""
    if _log_file:
        _log_file.close()

def read_file_content(filepath):
    """Reads and returns the content of a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Problem file not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

def send_api_request(system_prompt, user_prompt):
    """Sends a request to the configured API and returns the text response."""
    if not API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    try:
        client = OpenAI(api_key=API_KEY, base_url=API_URL)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7, # Allowing for some creativity in modeling
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n--- Error during API request ---")
        print(f"{e}")
        print("---------------------------------")
        return None

# --- CORE AGENT WORKFLOW ---

def run_math_modeling_agent(problem_statement):
    """Runs the main Modeler -> Analyzer workflow."""

    # Step 1: Run the Modeler Agent
    print("--- Step 1: Contacting Modeler Agent ---")
    print(f"System Prompt: MODELER_SYSTEM_PROMPT")
    print(f"User Prompt: {problem_statement[:200]}...")
    
    model_solution = send_api_request(MODELER_SYSTEM_PROMPT, problem_statement)
    
    if not model_solution:
        print("\nModeler Agent failed to produce a solution. Exiting.")
        return

    print("\n--- Modeler Agent's Solution ---")
    print(model_solution)
    print("---------------------------------")

    # Step 2: Run the Analyzer Agent
    print("\n--- Step 2: Contacting Analyzer Agent ---")
    print(f"System Prompt: ANALYZER_SYSTEM_PROMPT")
    analyzer_prompt = f"Please analyze the following mathematical modeling solution:\n\n---\"{model_solution}"""
    print(f"User Prompt: Analyzing the modeler's solution...")

    analysis = send_api_request(ANALYZER_SYSTEM_PROMPT, analyzer_prompt)

    if not analysis:
        print("\nAnalyzer Agent failed to produce an analysis. Exiting.")
        return

    print("\n--- Analyzer Agent's Critique ---")
    print(analysis)
    print("----------------------------------")
    
    # Future step: Implement the correction loop using CORRECTION_PROMPT
    print("\nWorkflow complete. Future work could implement a correction loop.")


# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mathematical Modeling Agent")
    parser.add_argument('problem_file', 
                        help='Path to the problem statement file.')
    parser.add_argument('--log', '-l', type=str, 
                        help='Path to save the log file.')

    args = parser.parse_args()

    if args.log:
        setup_logging(args.log)
        print(f"Logging to file: {args.log}")

    problem_statement = read_file_content(args.problem_file)
    run_math_modeling_agent(problem_statement)

    close_logging()
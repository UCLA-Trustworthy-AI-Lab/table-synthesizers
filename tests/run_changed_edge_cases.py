#!/usr/bin/env python3
"""
Run edge case tests only for models that have been modified locally or against the target branch.

Usage:
  python tests/run_changed_edge_cases.py               # Compares against 'main'
  python tests/run_changed_edge_cases.py HEAD          # Only unstaged/staged changes
  python tests/run_changed_edge_cases.py origin/main   # Compares against origin/main
"""

import sys
import subprocess

def get_changed_models(target_branch="main"):
    try:
        # Get list of changed files
        cmd = ["git", "diff", "--name-only", target_branch]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        
        # Parse model names from 'src/stg/<ModelName>/...'
        models = set()
        for line in output.splitlines():
            parts = line.split('/')
            if len(parts) >= 3 and parts[0] == 'src' and parts[1] == 'stg':
                models.add(parts[2])
                
        return sorted(list(models))
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e.output}")
        sys.exit(1)

def main():
    target_branch = sys.argv[1] if len(sys.argv) > 1 else "main"
    print(f"🔍 Detecting changed models compared to {target_branch}...")
    
    models = get_changed_models(target_branch)
    
    if not models:
        print("✅ No models have been changed in src/stg/! Skipping tests.")
        sys.exit(0)
        
    print(f"📦 Changed models detected: {', '.join(models)}")
    
    # We build a regex phrase for pytest -k (e.g., 'CTGAN or TabDDPM')
    k_filter = " or ".join(models)
    
    cmd = ["pytest", "tests/integration/test_edge_cases.py", "-k", k_filter, "-v"]
    print(f"🚀 Running command: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("\n❌ Tests failed for one or more changed models.")
        sys.exit(1)
        
if __name__ == "__main__":
    main()

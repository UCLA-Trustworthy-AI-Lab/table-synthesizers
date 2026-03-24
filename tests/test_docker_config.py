
import subprocess
import os
import sys

MAIN_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../main.py'))

def run_and_check(args, env, expected_stderr_snippets, unexpected_stderr_snippets=None):
    result = subprocess.run([sys.executable, MAIN_SCRIPT] + args, env=env, capture_output=True, text=True)
    try:
        if expected_stderr_snippets:
            # We generally expect non-zero for these failure tests, but checking strings is more precise
            pass 
        
        for snippet in expected_stderr_snippets or []:
            assert snippet in result.stderr, f"Expected '{snippet}' in stderr, got:\n{result.stderr}"
        
        for snippet in unexpected_stderr_snippets or []:
            assert snippet not in result.stderr, f"Did NOT expect '{snippet}' in stderr, got:\n{result.stderr}"
            
        return result
    except AssertionError as e:
        print(f"Subprocess output:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        raise e

def test_missing_all():
     """Test that it fails with one of the expected messages."""
     env = os.environ.copy()
     env.pop('SYNTHESIZER_CONFIG_PATH', None)
     env.pop('TRANSFORMED_DATA_PATH', None)
     env.pop('SYNTHESIZER_OUTPUT_PATH', None)
     
     # Just check loosely
     run_and_check([], env, [], []) # This won't assert anything yet
     
     result = subprocess.run([sys.executable, MAIN_SCRIPT], env=env, capture_output=True, text=True)
     if result.returncode == 0:
         print(f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
         raise AssertionError("Expected command to fail")
     
     found = any(x in result.stderr for x in ["Environment variables SYNTHESIZER_CONFIG_PATH", "Config path must be provided", "must be set"])
     if not found:
         print(f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
         raise AssertionError("Missing expected error message in stderr")

def test_env_vars_fallback():
    """Test that it picks up environment variables."""
    env = os.environ.copy()
    env['SYNTHESIZER_CONFIG_PATH'] = 'dummy_config.json'
    env['TRANSFORMED_DATA_PATH'] = 'dummy_data_path'
    env['SYNTHESIZER_OUTPUT_PATH'] = 'dummy_output_path'
    
    # We expect verify failure (FileNotFound), NOT arg failure
    run_and_check([], env, ["FileNotFoundError", "Config file not found"], ["Config path must be provided"])

def test_args_priority():
    """Test that CLI args override env vars."""
    env = os.environ.copy()
    env['SYNTHESIZER_CONFIG_PATH'] = 'env_config.json'
    
    run_and_check([
        '--config-path', 'cli_config.json',
        '--transformed-data-path', 'cli_data',
        '--synthesizer-output-path', 'cli_output'
    ], env, ["Config file not found at cli_config.json"], ["env_config.json"])

if __name__ == "__main__":
    try:
        test_missing_all()
        print("test_missing_all PASSED")
        test_env_vars_fallback()
        print("test_env_vars_fallback PASSED")
        test_args_priority()
        print("test_args_priority PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"TEST ERROR: {e}")
        exit(1)

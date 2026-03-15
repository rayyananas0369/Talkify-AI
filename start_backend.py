import subprocess
import os
import sys

# Set up environment
os.environ['PYTHONPATH'] = '.'

print("Starting backend with logging...")
log_file = 'backend_debug_log.txt'
with open(log_file, 'w') as f:
    try:
        # Use simple 'python' as it was verified to work in previous steps
        process = subprocess.run(['python', '-m', 'backend.main'], 
                               stdout=f, stderr=subprocess.STDOUT, check=True)
        print("Backend exited normally.")
    except subprocess.CalledProcessError as e:
        print(f"Backend CRASHED with exit code {e.returncode}")
    except Exception as e:
        print(f"Unexpected error: {e}")

print(f"Check {log_file} for details.")

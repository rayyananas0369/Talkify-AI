import subprocess
import os
import sys

# Set up environment
os.environ['PYTHONPATH'] = '.'

# Run backend
with open('backend_log.txt', 'w') as f:
    try:
        subprocess.run([r'C:\Users\Delta Infosys\AppData\Local\Programs\Python\Python310\python.exe', '-m', 'backend.main'], 
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    except Exception as e:
        print(f"Error: {e}")

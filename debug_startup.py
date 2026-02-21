
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

print("DEBUG: Starting Imports...")
try:
    from backend.inference.sign_inference import SignInference
    print("DEBUG: SignInference Imported.")
except Exception as e:
    print(f"DEBUG: Failed to import SignInference: {e}")
    sys.exit(1)

try:
    from backend.inference.lip_inference import LipInference
    print("DEBUG: LipInference Imported.")
except Exception as e:
    print(f"DEBUG: Failed to import LipInference: {e}")
    sys.exit(1)

print("DEBUG: Initializing SignInference...")
try:
    sign_engine = SignInference()
    print("DEBUG: SignInference Initialized.")
except Exception as e:
    print(f"DEBUG: Failed to init SignInference: {e}")
    sys.exit(1)

print("DEBUG: Initializing LipInference...")
try:
    lip_engine = LipInference()
    print("DEBUG: LipInference Initialized.")
except Exception as e:
    print(f"DEBUG: Failed to init LipInference: {e}")
    sys.exit(1)

print("DEBUG: All Initialized Successfully.")

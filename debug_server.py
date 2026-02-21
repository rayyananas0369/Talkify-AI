from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

print("DEBUG: Importing Inference Engines...")
try:
    from backend.inference.sign_inference import SignInference
    from backend.inference.lip_inference import LipInference
    print("DEBUG: Imports successful.")
except Exception as e:
    print(f"DEBUG: Import failed: {e}")

print("DEBUG: Initializing Engines...")
sign_engine = SignInference()
# lip_engine = LipInference()
print("DEBUG: SignInference Initialized.")

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "Debug Server Running"}

if __name__ == "__main__":
    print("DEBUG: Starting Uvicorn...")
    uvicorn.run(app, host="127.0.0.1", port=8002)

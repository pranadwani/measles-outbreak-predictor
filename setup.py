"""
Run this once before starting the app:
    python setup.py

Fetches live JHU data, generates historical data, and trains both ML models.
Re-run any time you want to refresh with the latest data.
"""
import subprocess
import sys

print("Step 1/2: Fetching measles data (JHU live + CDC historical)...")
subprocess.run([sys.executable, "src/fetch_data.py"], check=True)

print("\nStep 2/2: Training ML models...")
subprocess.run([sys.executable, "src/train.py"], check=True)

print("\nSetup complete. Run: python app.py")

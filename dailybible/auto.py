#!/usr/bin/env python3
import subprocess
import glob
import os
import sys

def main():
    # Run topic.py to generate plan JSON
    print("Generating video plan...")
    try:
        subprocess.run([sys.executable, "topic.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running topic.py: {e}")
        return

    # Locate the most recent plan file
    plan_files = glob.glob("video_plan_*.json")
    if not plan_files:
        print("No video_plan_*.json file found. Exiting.")
        return
    plan_file = max(plan_files, key=os.path.getmtime)
    print(f"Using plan file: {plan_file}")

    # Run the app workflow with the plan
    print("Running app.py workflow...")
    try:
        subprocess.run([sys.executable, "app.py", "--plan", plan_file], check=True)
        print("Workflow complete!")
    except subprocess.CalledProcessError as e:
        print(f"Error running app.py: {e}")

if __name__ == "__main__":
    main()

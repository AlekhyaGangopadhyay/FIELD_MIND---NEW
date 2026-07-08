import os
import sys
import subprocess

def run_script(script_path, interpreter):
    print(f"\n==================================================")
    print(f"Running: {script_path}")
    print(f"==================================================")
    
    # Run the script with the specific interpreter to avoid Windows Store alias issues
    result = subprocess.run([interpreter, script_path], capture_output=True, text=True)
    
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error executing {script_path}:")
        print(result.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Get python interpreter path
    interpreter = sys.executable
    if not interpreter:
        interpreter = r"C:\Users\Student\AppData\Local\Python\bin\python.exe"
        
    print(f"Using Python interpreter: {interpreter}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define script paths
    preprocess_script = os.path.join(base_dir, "src", "preprocess.py")
    train_script = os.path.join(base_dir, "src", "train.py")
    evaluate_script = os.path.join(base_dir, "src", "evaluate.py")
    
    # Execute stages in order
    print("Starting FIELD-MIND Optimized Environmental Model Pipeline...")
    
    run_script(preprocess_script, interpreter)
    run_script(train_script, interpreter)
    run_script(evaluate_script, interpreter)
    
    print("\n==================================================")
    print("Pipeline Execution Completed Successfully!")
    print("==================================================")

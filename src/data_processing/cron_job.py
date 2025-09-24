import schedule
import time
import subprocess
import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON = "python"
SCRIPT = os.path.join(PROJECT_DIR, "src", "data_processing", "db_writer.py")

def job():
    print("Running db_writer...")
    subprocess.run([PYTHON, SCRIPT], cwd=PROJECT_DIR)

schedule.every(5).minutes.do(job)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(1)

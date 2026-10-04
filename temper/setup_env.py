import os
import sys
import subprocess

PROJECT_DIR = r"F:\URCA_PROJECTS"
TMP_DIR = os.path.join(PROJECT_DIR, ".tmp")
CACHE_DIR = os.path.join(PROJECT_DIR, ".pip_cache")

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Set temporary and cache directories exclusively on F:
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR
os.environ["TMPDIR"] = TMP_DIR

print(f"[*] Verified Python Executable: {sys.executable}")
print(f"[*] Staging Directory (TEMP): {TMP_DIR}")
print(f"[*] Pip Cache Directory: {CACHE_DIR}")

def run_pip(args):
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--cache-dir", CACHE_DIR,
    ] + args
    print(f"\n[>] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, env=os.environ)

if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "all"

    if action in ("torch", "all"):
        print("\n--- Installing PyTorch (CUDA 12.6) ---")
        run_pip([
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu126"
        ])

    if action in ("libs", "all"):
        print("\n--- Installing AI & Data Science Libraries ---")
        run_pip([
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "ipykernel",
            "tqdm",
            "tensorboard"
        ])

    print("\n[+] All requested packages installed successfully into .venv on F:!")

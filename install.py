# install.py — creates .venv and installs requirements on Windows/macOS/Linux
import os, sys, subprocess, venv

VENV = ".venv"
REQ = "requirements.txt"

def venv_python():
    if os.name == "nt":
        return os.path.join(VENV, "Scripts", "python.exe")
    return os.path.join(VENV, "bin", "python")

def main():
    # 1) Create venv if missing
    if not os.path.isdir(VENV):
        print(f"Creating virtual environment: {VENV} ...")
        venv.EnvBuilder(with_pip=True).create(VENV)
    else:
        print(f"Found existing {VENV}")

    py = venv_python()
    if not os.path.exists(py):
        sys.exit(" Could not find venv Python (creation failed).")

    # 2) Upgrade pip tooling
    print("Upgrading pip, wheel, setuptools ...")
    subprocess.check_call([py, "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"])

    # 3) Install dependencies
    if os.path.exists(REQ):
        print(f"Installing dependencies from {REQ} ...")
        subprocess.check_call([py, "-m", "pip", "install", "-r", REQ])
    else:
        print(f"⚠️ {REQ} not found; skipping dependency install.")

    # 4) How to activate / run
    if os.name == "nt":
        activate = r".\.venv\Scripts\Activate.ps1"
        run = r".\.venv\Scripts\python.exe"
    else:
        activate = "source .venv/bin/activate"
        run = ".venv/bin/python"
    print("\nEnvironment ready.")
    print(f"Activate: {activate}")
    print(f"Run app: {run} main.py")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"Install step failed with exit code {e.returncode}.")

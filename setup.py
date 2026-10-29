import subprocess
import sys

# pip package name -> import name
REQUIRED_PACKAGES = {
    "flask": "flask",
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "joblib": "joblib",
    "tqdm": "tqdm",
}

def install_and_verify():
    print("Checking and installing dependencies...")
    for package, import_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
            print(f"✓ '{package}' is already installed.")
        except ImportError:
            print(f"Installing '{package}'...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    install_and_verify()
    print("\nAll dependencies are ready! You can now run 'python training_model.py' and 'python app.py'.")
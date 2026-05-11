"""Auto-install missing pip packages listed in requirements.txt."""
import subprocess
import sys
import os


def ensure_requirements():
    req_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'requirements.txt')
    if not os.path.isfile(req_file):
        return

    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', req_file,
             '--quiet', '--disable-pip-version-check'],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print('[deps] installed:', result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f'[deps] warning: pip install failed:\n{e.stderr}')

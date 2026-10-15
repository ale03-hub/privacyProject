# Runs the whole pipeline (or a single step: `python main.py 02`)
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    scripts = sorted(glob.glob(os.path.join(ROOT, "scripts", "0*.py")))
    if len(sys.argv) > 1:
        # select by number ("02") or by name fragment ("baseline")
        scripts = [s for s in scripts if any(a in os.path.basename(s) for a in sys.argv[1:])]
        if not scripts:
            sys.exit(f"no script matches {sys.argv[1:]}")

    for path in scripts:
        print(f"\n{'=' * 70}\n{os.path.basename(path)}\n{'=' * 70}", flush=True)
        subprocess.run([sys.executable, path], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

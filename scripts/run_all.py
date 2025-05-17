"""Run the full ICML 2025 experimental suite in sequence."""
import subprocess, yaml, pathlib, argparse, sys

ROOT = pathlib.Path(__file__).resolve().parent
CONFIGS = ROOT.parent / "configs"

def main():
    cfgs = CONFIGS.glob("*.yaml")
    for cfg in cfgs:
        print(f"[RNF] ▶ Running experiment {cfg.name}")
        subprocess.check_call([sys.executable, "scripts/experiment.py", "--config", str(cfg)])

if __name__ == "__main__":
    main()

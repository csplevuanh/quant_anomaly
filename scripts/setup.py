"""Download dataset & checkpoints."""
import argparse, os, subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if args.all:
        print("[RNF] (TODO) Implement dataset & checkpoint download.")
    else:
        print("Use --all to download everything.")

if __name__ == "__main__":
    main()

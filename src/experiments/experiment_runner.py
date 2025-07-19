import argparse, subprocess, time, sys

def run_dummy(frames: int):
    print(f"Running dummy experiment for {frames} frames")
    for i in range(frames):
        time.sleep(0.01)  # placeholder latency
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--profile", type=str, default="5G")
    args = parser.parse_args()

    print(f"Applying TC profile {args.profile}")
    subprocess.run([sys.executable, "tc/apply_tc.py", args.profile])
    run_dummy(args.frames)

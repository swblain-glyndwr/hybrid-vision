import subprocess, sys
profile = sys.argv[1] if len(sys.argv) > 1 else "5G"
cmd = ["sudo", "bash", "tc/netem_config.sh"]
subprocess.check_call(cmd)
print(f"Applied TC profile {profile}")

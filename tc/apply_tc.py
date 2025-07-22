"""
tc/apply_tc.py
Apply a traffic-control (netem) profile on the host.

"""

from __future__ import annotations
import os, subprocess, sys

def _inside_container() -> bool:
    """Return True if running in Docker/LXC."""
    return os.path.exists("/.dockerenv") or \
           any("docker" in p for p in open("/proc/1/cgroup", "r"))

def apply_profile(profile: str = "5G") -> None:
    """
    profile: name understood by tc/netem_config.sh
    """
    if _inside_container():
        print(f"[TC] Skipped applying profile '{profile}' inside container")
        return

    script = os.path.join(os.path.dirname(__file__), "netem_config.sh")
    cmd = ["sudo", "bash", script, profile]
    print(f"[TC] Applying {profile} on host …")
    subprocess.check_call(cmd)
    print(f"[TC] Applied profile {profile}")

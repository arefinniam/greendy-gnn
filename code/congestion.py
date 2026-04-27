#!/usr/bin/env python3
"""Congestion driver: applies time-varying tc netem delays (15-25 ms) to
non-coordinator cluster nodes over the DGL RPC port.

Node IPs are read from the path passed via --cong_ips (one IP per line).
The first entry in ip_config.txt is the coordinator and is NOT congested;
pass the remaining P-1 IPs to this driver.
"""

import argparse, os, re, subprocess, sys, time

DGL_PORT = 30050


def ssh(ip, cmd, timeout=15):
    r = subprocess.run(
        f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {ip} '{cmd}'",
        shell=True, capture_output=True, timeout=timeout, text=True)
    return r.returncode == 0, r.stdout + r.stderr


def apply_delay(ip, iface, ms):
    if ms <= 0:
        return remove_delay(ip, iface)
    ssh(ip, f"sudo tc qdisc del dev {iface} root 2>/dev/null; "
            f"sudo tc qdisc add dev {iface} root handle 1: prio && "
            f"sudo tc filter add dev {iface} parent 1: protocol ip u32 "
            f"match ip sport {DGL_PORT} 0xffff flowid 1:1 && "
            f"sudo tc qdisc add dev {iface} parent 1:1 netem delay {ms}ms")


def remove_delay(ip, iface):
    ssh(ip, f"sudo tc qdisc del dev {iface} root 2>/dev/null || true")


def clear_all(ips, iface):
    for ip in ips:
        remove_delay(ip, iface)


def verify_clean(ips, iface):
    ok = True
    for ip in ips:
        _, out = ssh(ip, f"tc qdisc show dev {iface}")
        clean = "netem" not in out
        print(f"  {ip}: {'clean' if clean else 'DIRTY'}")
        ok = ok and clean
    return ok


def realistic_pattern(n_epochs, ips):
    """Epochs 0-2 clean warmup, epochs 3..n-2 cycle the patterns below,
    final epoch forced clean. Each pattern names a subset of ips by index."""
    events = [(0, {})]
    if n_epochs <= 4 or not ips:
        events.append((n_epochs - 1, {}))
        return events

    # Index into `ips`; delay in ms.
    patterns = [
        {0: 20},
        {len(ips) - 1: 25},
        {1 % len(ips): 15, 0: 20},
        {},
        {len(ips) - 1: 15, 1 % len(ips): 20},
        {0: 25},
        {1 % len(ips): 20},
    ]
    for i, ep in enumerate(range(3, n_epochs - 1)):
        idx_delays = patterns[i % len(patterns)]
        events.append((ep, {ips[k]: v for k, v in idx_delays.items()}))
    events.append((n_epochs - 1, {}))
    return events


def get_epoch(log_file):
    if not os.path.exists(log_file):
        return -1
    try:
        with open(log_file) as f:
            matches = re.findall(
                r"Part 0 Ep(?:och)?\s*(\d+).*(?:Time|:\s*\d+\.\d+s)", f.read())
        return max(int(m) for m in matches) if matches else -1
    except Exception:
        return -1


def run(log_file, n_epochs, ips, iface, timeout=3600):
    events = realistic_pattern(n_epochs, ips)
    print("=== Congestion driver ===")
    for ep, d in events:
        print(f"  Epoch {ep}: {d or 'CLEAN'}")
    clear_all(ips, iface)
    verify_clean(ips, iface)

    applied, state = set(), {}
    t0 = time.time()
    while time.time() - t0 < timeout:
        epoch = get_epoch(log_file)
        for eep, delays in events:
            if eep in applied or epoch < eep:
                continue
            for ip in list(state):
                if ip not in delays:
                    remove_delay(ip, iface)
                    del state[ip]
            for ip, ms in delays.items():
                apply_delay(ip, iface, ms)
                state[ip] = ms
            if not delays:
                for ip in list(state):
                    remove_delay(ip, iface)
                state.clear()
            applied.add(eep)
            print(f"[epoch {epoch}] -> {delays or 'CLEAN'}")
        if epoch >= n_epochs - 1:
            break
        time.sleep(1.5)

    print("\n=== Cleanup ===")
    clear_all(ips, iface)
    time.sleep(1)
    if not verify_clean(ips, iface):
        clear_all(ips, iface)
        time.sleep(2)
        verify_clean(ips, iface)
    print("Done.")


def load_ips(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_file", default="",
                    help="Training log to watch for epoch progression")
    ap.add_argument("--total_epochs", type=int, default=10)
    ap.add_argument("--cong_ips", default="",
                    help="Path to file with one IP per line (non-coordinator nodes)")
    ap.add_argument("--iface", default="eno1",
                    help="Network interface on each node (default: eno1)")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--cleanup", action="store_true",
                    help="Remove any existing netem rules and exit")
    a = ap.parse_args()

    ips = load_ips(a.cong_ips) if a.cong_ips else []

    if a.cleanup:
        print("=== Cleanup ===")
        clear_all(ips, a.iface)
        time.sleep(1)
        verify_clean(ips, a.iface)
    elif a.log_file and ips:
        run(a.log_file, a.total_epochs, ips, a.iface, a.timeout)
    else:
        print("Usage: congestion.py --log_file <path> --cong_ips <path> [--iface eno1]")
        print("       congestion.py --cleanup --cong_ips <path>")
        sys.exit(1)

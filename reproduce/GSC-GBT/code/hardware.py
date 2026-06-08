import os
import subprocess


# ── CPU ───────────────────────────────────────────────────────────────────────

CPU_TOTAL = os.cpu_count() or 1

# Reserve cores for the OS and other processes.
# Rule: at least 4 cores, or 20% of total — whichever is larger.
# This prevents the training job from starving the system under heavy load.
_CPU_RESERVE = max(4, CPU_TOTAL // 5)
CPU_WORKERS  = max(1, CPU_TOTAL - _CPU_RESERVE)


# ── GPU ───────────────────────────────────────────────────────────────────────

def _detect_gpus() -> list[dict]:
    """Query available GPUs via nvidia-smi. Returns an empty list if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    gpus.append({
                        "id":      int(parts[0]),
                        "name":    parts[1],
                        "vram_mb": int(parts[2]),
                        "free_mb": int(parts[3]),
                    })
                except ValueError:
                    pass
        return gpus
    except Exception:
        return []


GPUS             = _detect_gpus()
N_GPUS_AVAILABLE = len(GPUS)
GPU_IDS          = [g["id"] for g in GPUS]


# ── RAM ───────────────────────────────────────────────────────────────────────

def _detect_ram_gb() -> float | None:
    """Return total system RAM in GB. Returns None if detection fails."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except ImportError:
        pass
    try:
        # Linux fallback via /proc/meminfo
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return None


RAM_GB = _detect_ram_gb()


# ── Summary ───────────────────────────────────────────────────────────────────

def summary() -> str:
    """One-line hardware description for startup logging."""
    ram_str = f"{RAM_GB:.1f} GB" if RAM_GB is not None else "unknown"

    if GPUS:
        gpu_parts = [
            f"GPU{g['id']} {g['name']} ({g['vram_mb']} MB total, {g['free_mb']} MB free)"
            for g in GPUS
        ]
        gpu_str = " | ".join(gpu_parts)
    else:
        gpu_str = "none detected"

    return (
        f"CPU: {CPU_TOTAL} cores total, "
        f"{CPU_WORKERS} allocated ({_CPU_RESERVE} reserved for OS)  |  "
        f"RAM: {ram_str}  |  "
        f"GPU: {gpu_str}"
    )

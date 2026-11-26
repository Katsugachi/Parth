import glob
import json
import os
import subprocess
import sys
import time

COMMIT = "__PARTH_COMMIT__"
BRANCH = "devin/1795610261-parth-pipeline"
REPO = "https://github.com/Katsugachi/Parth.git"
OUT_DIR = "/kaggle/working/parth"


def run(cmd, **kwargs):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kwargs)


def stream_training(micro_batch, grad_accum):
    cmd = [
        sys.executable,
        "-m",
        "parth.train_tpu",
        "--total-tokens",
        "__PARTH_TOTAL_TOKENS__",
        "--max-hours",
        "__PARTH_MAX_HOURS__",
        "--micro-batch",
        str(micro_batch),
        "--grad-accum",
        str(grad_accum),
        "--download-workers",
        "12",
        "--require-tpu",
    ]
    log_path = f"/kaggle/working/train_mb{micro_batch}.log"
    print("+", " ".join(cmd), flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = proc.wait()
    with open(log_path, errors="replace") as handle:
        text = handle.read()
    return rc, text


def main():
    print("START", time.time(), flush=True)
    os.chdir("/kaggle/working")
    print("python", sys.version, flush=True)
    print("disk", subprocess.check_output(["df", "-h", "/kaggle/working", "/tmp"], text=True), flush=True)

    if not os.path.exists("Parth/.git"):
        run(["git", "clone", REPO], check=True)
    os.chdir("Parth")
    run(["git", "fetch", "origin", COMMIT], check=False)
    run(["git", "checkout", COMMIT], check=True)
    print("commit", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), flush=True)

    env_install = os.environ.copy()
    env_install["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True, env=env_install)
    run([sys.executable, "-m", "pip", "show", "jax", "jaxlib", "libtpu"], check=False)

    os.environ["PARTH_OUT"] = OUT_DIR
    os.environ["PARTH_DATA_CACHE"] = "/tmp/parth_data_cache"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    rc, log = stream_training(__PARTH_MICRO_BATCH__, __PARTH_GRAD_ACCUM__)
    if rc and "TPU training requested" not in log and not glob.glob(OUT_DIR + "/parth_step_*.json"):
        if "RESOURCE_EXHAUSTED" in log or "Out of memory" in log or "out of memory" in log:
            print("micro-batch 2 exhausted device memory; retrying micro-batch 1", flush=True)
            rc, log = stream_training(1, __PARTH_FALLBACK_GRAD_ACCUM__)
    status = {
        "returncode": rc,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoints": sorted(glob.glob(OUT_DIR + "/*")),
        "finished": time.time(),
    }
    with open("/kaggle/working/parth_tpu_status.json", "w") as handle:
        json.dump(status, handle, indent=2)
    print("STATUS", json.dumps(status, indent=2), flush=True)
    if rc:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()

"""Probe the allocated CUDA device before importing or loading the VLA model."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import socket


def probe(driver):
    def call(name, *args):
        code = getattr(driver, name)(*args)
        if code:
            raise RuntimeError("{} failed with CUDA driver status {}".format(name, code))

    call("cuInit", 0)
    count = ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    if count.value != 1:
        raise RuntimeError("Expected one visible allocated GPU, found {}".format(count.value))
    device, memory = ctypes.c_int(), ctypes.c_size_t()
    name = ctypes.create_string_buffer(256)
    call("cuDeviceGet", ctypes.byref(device), 0)
    call("cuDeviceGetName", name, len(name), device)
    call("cuDeviceTotalMem_v2", ctypes.byref(memory), device)
    gpu = dict(name=name.value.decode(), total_memory_bytes=memory.value, visible_device_count=count.value)
    if "H100" not in gpu["name"] or memory.value < 70 * 2**30:
        raise RuntimeError("This run requires a full H100 with at least 70 GiB visible: " + str(gpu))
    return gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run this probe only inside the allocated Slurm GPU job")
    record = dict(passed=False, hostname=socket.gethostname(), environment={
        key: os.environ.get(key) for key in ["SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
            "SLURM_GPUS_ON_NODE", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"]})
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        record.update(probe(driver), passed=True)
    except Exception as error:
        record["error"] = str(error)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if not record["passed"]:
        raise SystemExit("Allocated CUDA device is unusable; exit before model loading")


if __name__ == "__main__":
    main()

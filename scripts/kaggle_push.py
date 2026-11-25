"""Push a Parth training/probe notebook to Kaggle using the token-only MCP endpoint.

The public CLI currently drops custom TPU-VM docker images when enable_tpu=true;
this script sends the image explicitly through Kaggle's MCP save_notebook API.
"""

import argparse
import json
import os
import requests
import subprocess

MCP = "https://www.kaggle.com/mcp"
TPUVM_IMAGE = "gcr.io/kaggle-private-byod/python-tpuvm@sha256:02e61fa3973e04ecbf03a3ba9c7f7bcf58f7684662c2029bd8033aea3b249ebd"


def mcp_call(token: str, name: str, request: dict):
    response = requests.post(
        MCP,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": "Bearer " + token.strip(),
        },
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": {"request": request}}},
        timeout=180,
    )
    response.raise_for_status()
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    if not events or events[0].get("result", {}).get("isError"):
        raise RuntimeError(response.text[:2000])
    return events[0]["result"]


def notebook_source(commit: str, total_tokens: int, max_hours: float, micro_batch: int, grad_accum: int, extra_args: str = "") -> str:
    code = f'''import os, subprocess, sys, time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.chdir("/kaggle/working")
if not os.path.exists("Parth/.git"):
    subprocess.run(["git", "clone", "--depth", "50", "https://github.com/Katsugachi/Parth.git"], check=True)
os.chdir("Parth")
subprocess.run(["git", "fetch", "--depth", "50", "origin", "{commit}"], check=False)
subprocess.run(["git", "checkout", "{commit}"], check=True)
print("commit", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
cmd = [sys.executable, "-m", "parth.train_tpu", "--total-tokens", "{total_tokens}", "--max-hours", "{max_hours}", "--micro-batch", "{micro_batch}", "--grad-accum", "{grad_accum}"]
extra = "{extra_args}".split()
cmd.extend([x for x in extra if x])
print("running", cmd, flush=True)
subprocess.run(cmd, check=True)
'''
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code, "id": "train"}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "kaggle": {"accelerator": "tpuV5e8", "dataSources": [], "isInternetEnabled": True, "language": "python", "sourceType": "notebook", "isGpuEnabled": False},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", default="katsugachi/parth-pretrain")
    parser.add_argument("--title", default="Parth pretrain")
    parser.add_argument("--commit", default=None)
    parser.add_argument("--total-tokens", type=int, default=30_000_000)
    parser.add_argument("--max-hours", type=float, default=0.7)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--timeout", type=int, default=39600)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    token = os.environ.get("KAGGLE_TOKEN") or os.environ.get("KAGGLE_JSON")
    if not token:
        raise RuntimeError("set KAGGLE_TOKEN")
    commit = args.commit or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    request = {
        "slug": args.slug,
        "newTitle": args.title,
        "text": notebook_source(commit, args.total_tokens, args.max_hours, args.micro_batch, args.grad_accum, args.extra_args),
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": args.private,
        "enableGpu": False,
        "enableTpu": True,
        "enableInternet": True,
        "machineShape": "TpuV5E8",
        "dockerImage": TPUVM_IMAGE,
        "sessionTimeoutSeconds": args.timeout,
    }
    print(json.dumps(mcp_call(token, "save_notebook", request), indent=2)[:2000])


if __name__ == "__main__":
    main()

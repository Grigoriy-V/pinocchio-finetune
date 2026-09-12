"""The fine-tuned model as an OpenAI-compatible endpoint: vLLM on Modal.

What the harness takes back is a model set — an endpoint and a served
name. This App serves `/vol/runs/<run>/merged` from the training Volume
under the name `gemma-4-12b-tuned`, behind Modal's proxy auth, so the
harness's `config.toml` gains:

    [model.sets.tuned]
    endpoint = "https://grigoriy-v--pinocchio-tune-serve-server-serve.modal.run/v1"
    name = "gemma-4-12b-tuned"
    auth_style = "modal_proxy"

The vLLM flags are the ones the harness validated for Gemma 4 in its own
`deploy/modal/model_app.py` (tool parser, reasoning parser, multimodal
limits); the pinned vLLM/transformers pair is the same. No memory snapshot
in this first version: the endpoint exists to be measured for an hour, not
to answer a person at any moment.

    modal deploy modal_apps/serve_app.py                    # the tuned model
    SERVE_TARGET=base modal deploy modal_apps/serve_app.py  # the untuned base, `[model.sets.base]`
    SERVE_TARGET=dpo modal deploy modal_apps/serve_app.py   # the DPO run merged, `[model.sets.dpo]`

Both are gates: the GPU starts on the first request.
"""

from __future__ import annotations

import modal

import os

VOLUME = "pinocchio-tune"
VOL = "/vol"
RUN = "v1-r16"
BASE_SNAPSHOT = f"{VOL}/hf/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"

# `SERVE_TARGET=base` deploys a second App that serves the untuned bf16 base
# from the same Volume, on the same card and vLLM: the "before" side of the
# measurement, so that the only difference between the two endpoints is
# the weights (the harness's own Gemma endpoint is a QAT int4 checkpoint on
# another workspace). Read at deploy time, on the client.
TARGET = os.environ.get("SERVE_TARGET", "tuned")
DPO_RUN = "dpo-v1"
if TARGET == "base":
    APP_NAME = "pinocchio-tune-serve-base"
    MODEL_PATH = BASE_SNAPSHOT
    SERVED_NAME = "gemma-4-12b-base"
elif TARGET == "dpo":
    # The DPO adapter merged (item 1, 2026-09-12): a third App, `[model.sets.dpo]`.
    APP_NAME = "pinocchio-tune-serve-dpo"
    MODEL_PATH = f"{VOL}/runs/{DPO_RUN}/merged"
    SERVED_NAME = "gemma-4-12b-dpo"
else:
    APP_NAME = "pinocchio-tune-serve"
    MODEL_PATH = f"{VOL}/runs/{RUN}/merged"
    SERVED_NAME = "gemma-4-12b-tuned"

VLLM_VERSION = "0.26.0"
TRANSFORMERS_VERSION = "5.14.1"
VLLM_PORT = 8000
MINUTES = 60
MAX_MODEL_LEN = 32768
GPU = "L40S"
# 60 s, not the 2 s minimum: inside one turn the harness runs tools between
# two model requests — a venv and a pip install take 20–30 s — and a window
# shorter than that would stop the container mid-turn and pay a cold start
# for the next request. Idle after a run costs cents (the human, 2026-09-12).
SCALEDOWN_WINDOW = 60
MM_LIMITS = {"image": 4, "audio": 1}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME)
vllm_cache = modal.Volume.from_name("pinocchio-tune-vllm-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        f"vllm[audio]=={VLLM_VERSION}",
        f"transformers=={TRANSFORMERS_VERSION}",
    )
    .env(
        {
            # Baked into the image: the container has no SERVE_TARGET of its
            # own, and read at import there the base App served the tuned
            # weights under the tuned name (404 on the base name, 2026-09-12).
            "SERVE_TARGET": TARGET,
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "VLLM_HOST_IP": "127.0.0.1",
            "NCCL_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
        }
    )
)

with image.imports():
    import requests


def _wait_ready(process, timeout: float) -> None:
    import time

    deadline = time.monotonic() + timeout
    while True:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"vLLM exited with code {code}; see the container's logs")
        try:
            if requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError(f"vLLM not healthy after {timeout:.0f}s")
        time.sleep(2)


@app.cls(
    image=image,
    gpu=GPU,
    volumes={VOL: volume, "/root/.cache/vllm": vllm_cache},
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=0,
    max_containers=1,
    timeout=15 * MINUTES,
)
@modal.concurrent(max_inputs=8)
class Server:
    @modal.enter()
    def start(self) -> None:
        import json
        import subprocess

        command = [
            "vllm",
            "serve",
            MODEL_PATH,
            "--served-model-name",
            SERVED_NAME,
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--gpu-memory-utilization",
            "0.90",
            "--limit-mm-per-prompt",
            json.dumps(MM_LIMITS),
            "--enable-auto-tool-choice",
            "--enable-prompt-tokens-details",
            "--tool-call-parser",
            "gemma4",
            "--reasoning-parser",
            "gemma4",
            # No CUDA-graph capture: a minute or two less at every cold start,
            # slower tokens. This endpoint is measured for what it answers,
            # not how fast, and it boots for every measuring run.
            "--enforce-eager",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
        ]
        print(*command, flush=True)
        self.process = subprocess.Popen(command)
        _wait_ready(self.process, 10 * MINUTES)
        vllm_cache.commit()

    @modal.web_server(port=VLLM_PORT, startup_timeout=12 * MINUTES, requires_proxy_auth=True)
    def serve(self) -> None:
        pass

    @modal.exit()
    def stop(self) -> None:
        process = getattr(self, "process", None)
        if process is not None:
            process.terminate()

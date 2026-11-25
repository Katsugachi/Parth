# Kaggle interactive TPU automation

Batch `kernels push` TPU runs provision the correct `python-tpuvm` image but do not attach `/dev/accel*`; JAX and torch-xla fall back to CPU. Kaggle interactive sessions expose a authenticated Jupyter proxy through the long-running operation response and can be automated without browser login.

Critical API details:

- Use `https://api.kaggle.com/v1/kernels.KernelsApiService/CreateKernelSession` with snake_case fields: `slug`, `language`, `kernel_type`, `docker_image`, `machine_shape`, `enable_internet`. CamelCase is silently ineffective here.
- Poll `https://api.kaggle.com/v1/common.OperationsService/GetOperation` with the returned operation `name`.
- The done response contains `jupyterUrl` and `tokenizedJupyterUrl`. Use `jupyterUrl` directly as the Jupyter base; `/api/kernels`, `/api/sessions`, `/api/status` work there. Do not print or commit these URLs.
- Correct TPU image observed on a public TPU notebook: `gcr.io/kaggle-private-byod/python-tpuvm@sha256:02e61fa3973e04ecbf03a3ba9c7f7bcf58f7684662c2029bd8033aea3b249ebd`.
- Execute code through `wss://.../api/kernels/{kernel_id}/channels` using standard Jupyter `execute_request` messages. Helper: `/home/ubuntu/work/jupyter_exec.py`; session helper: `/home/ubuntu/work/kaggle_interactive.py`.
- If a session exists for the kernel, cancel it through `kernels.KernelsApiService/CancelKernelSession` using the numeric id embedded as `/k/{id}/` in `jupyterUrl`.
- `/kaggle/working` is about 21 GB. Use `/tmp/parth_data_cache` for the large parquet cache.

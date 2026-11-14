# strata-models-client

What an app imports to use the strata-models service on the same machine. Standard library only, so it installs into any venv:

```
pip install "strata-models-client @ git+https://github.com/jburnhams/strata-models#subdirectory=client"
```

```python
import strata_models_client as sm

if sm.available():  # the service is running
    faces = sm.faces("/path/a.jpg", app="bright-horizons")["faces"]
    text = sm.transcribe("/path/talk.wav", app="strata-360", language="en")
    page = sm.ocr("/path/page.png", app="strata-scan", pdf="/path/page.pdf")  # lines, words, boxes; and a searchable PDF page

with sm.gpu_lease(
    "strata-stills", vram_gb=4, note="arena build"
):  # waits its turn, then the GPU is yours until the block ends
    run_my_own_cuda_code()
```

- `run(role, params, app=...)` is the general form; `faces`, `transcribe`, `speak` and `ocr` are shortcuts. Failures raise `JobFailed`; no service raises `Unavailable`, so an app can fall back to loading the model itself.
- Every call takes `priority=`: `"high"` for a job a person or an app is waiting on, `"normal"` (the default) for background work, `"low"` for work that can wait behind everything else (evals use it). A running job is never stopped for a higher one.
- `gpu_lease` sends heartbeats from a thread while the block runs. With no service it yields `None` and the block runs anyway (pass `required=True` to raise instead), so an app keeps working where strata-models is not installed.
- The service URL is `STRATA_MODELS_URL`, default `http://127.0.0.1:8366`.

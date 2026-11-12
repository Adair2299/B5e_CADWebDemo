from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import Predictor
from executor import Executor
from cad_schema import parse_command

import os
import tempfile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from dxf_export import export_dxf

import time


ROOT = Path(__file__).resolve().parent

app = FastAPI(title="Transformer CAD Controller")

app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")

predictor = Predictor(str(ROOT / "cad_model.pt"))
executor = Executor()


class TextInput(BaseModel):
    text: str


class CommandInput(BaseModel):
    command: dict


@app.get("/health")
def health():
    return {"status": "ok", "model": "cad_model.pt"}


@app.post("/predict")
def predict(data: TextInput):
    try:
        start = time.perf_counter()
        raw = predictor.predict(data.text)
        inference_ms = (time.perf_counter() - start) * 1000

        command = parse_command(raw)
        print(f"Inference time: {inference_ms:.1f} ms", flush=True)
        return {"command": command}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/execute")
def execute(data: CommandInput):
    try:
        result = executor.execute(data.command)
        return {
            "result": result,
            "objects": list(executor.objects.values())
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
def get_state():
    return {"objects": list(executor.objects.values())}



@app.get("/export/dxf")
def download_dxf():
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tmp:
        path = tmp.name
    try:
        export_dxf(executor.objects, path)
        return FileResponse(
            path,
            filename="cad_drawing.dxf",
            media_type="application/dxf",
            background=BackgroundTask(os.remove, path)
        )
    except Exception:
        os.remove(path)
        raise

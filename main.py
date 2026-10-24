from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import Predictor
from executor import Executor
from cad_schema import parse_command

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
        raw = predictor.predict(data.text)
        command = parse_command(raw)
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

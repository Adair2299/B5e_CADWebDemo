
const form = document.getElementById("command-form");
const input = document.getElementById("command-input");
const predictBtn = document.getElementById("predict-btn");
const executeBtn = document.getElementById("execute-btn");
const predictionOutput = document.getElementById("prediction-output");
const executionResult = document.getElementById("execution-result");
const historyPanel = document.getElementById("history");
const objectCount = document.getElementById("object-count");
const canvas = document.getElementById("cad-canvas");
const ctx = canvas.getContext("2d");

let pendingCommand = null;
let objects = [];

async function api(path, payload = null) {
    const response = await fetch(path, {
        method: payload === null ? "GET" : "POST",
        headers: { "Content-Type": "application/json" },
        body: payload === null ? undefined : JSON.stringify(payload)
    });

    const data = await response.json();
    if (!response.ok) {
        throw new Error(typeof data.detail === "string"
            ? data.detail : JSON.stringify(data.detail || data));
    }
    return data;
}

function drawCanvas() {
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    // Determine bounds from the current geometry.
    const points = [{ x: 0, y: 0 }];

    for (const obj of objects) {
        if (obj.type === "line") {
            points.push({ x: obj.x1, y: obj.y1 });
            points.push({ x: obj.x2, y: obj.y2 });
        } else if (obj.type === "circle") {
            const x = obj.x ?? obj.cx ?? 0;
            const y = obj.y ?? obj.cy ?? 0;
            points.push({ x: x - obj.radius, y: y - obj.radius });
            points.push({ x: x + obj.radius, y: y + obj.radius });
        } else if (obj.type === "rectangle") {
            points.push({ x: obj.x, y: obj.y });
            points.push({ x: obj.x + obj.width, y: obj.y + obj.height });
        } else if (obj.type === "point") {
            points.push({ x: obj.x, y: obj.y });
        }
    }

    const xs = points.map(p => p.x);
    const ys = points.map(p => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    const rangeX = Math.max(10, maxX - minX);
    const rangeY = Math.max(10, maxY - minY);
    const scale = Math.min((width - 100) / rangeX, (height - 100) / rangeY);

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    function screen(x, y) {
        return {
            x: width / 2 + (x - centerX) * scale,
            y: height / 2 - (y - centerY) * scale
        };
    }

    // Grid.
    ctx.strokeStyle = "#edf0f5";
    ctx.lineWidth = 1;
    const gridStep = Math.max(1, Math.ceil(40 / scale));

    const left = centerX - width / (2 * scale);
    const right = centerX + width / (2 * scale);
    const bottom = centerY - height / (2 * scale);
    const top = centerY + height / (2 * scale);

    for (let x = Math.ceil(left / gridStep) * gridStep; x <= right; x += gridStep) {
        const p = screen(x, 0);
        ctx.beginPath();
        ctx.moveTo(p.x, 0);
        ctx.lineTo(p.x, height);
        ctx.stroke();
    }

    for (let y = Math.ceil(bottom / gridStep) * gridStep; y <= top; y += gridStep) {
        const p = screen(0, y);
        ctx.beginPath();
        ctx.moveTo(0, p.y);
        ctx.lineTo(width, p.y);
        ctx.stroke();
    }

    // Axes.
    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 1;
    const origin = screen(0, 0);
    ctx.beginPath();
    ctx.moveTo(origin.x, 0);
    ctx.lineTo(origin.x, height);
    ctx.moveTo(0, origin.y);
    ctx.lineTo(width, origin.y);
    ctx.stroke();

    // Objects.
    for (const obj of objects) {
        ctx.strokeStyle = obj.color || "#2563eb";
        ctx.fillStyle = obj.color || "#2563eb";
        ctx.lineWidth = Math.max(1, obj.width || 1);

        let labelPoint = null;

        if (obj.type === "line") {
            const a = screen(obj.x1, obj.y1);
            const b = screen(obj.x2, obj.y2);
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
            labelPoint = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };

        } else if (obj.type === "circle") {
            const p = screen(obj.x ?? obj.cx ?? 0, obj.y ?? obj.cy ?? 0);
            ctx.beginPath();
            ctx.arc(p.x, p.y, obj.radius * scale, 0, Math.PI * 2);
            ctx.stroke();
            labelPoint = p;

        } else if (obj.type === "rectangle") {
            const a = screen(obj.x, obj.y);
            ctx.strokeRect(a.x, a.y - obj.height * scale,
                obj.width * scale, obj.height * scale);
            labelPoint = a;

        } else if (obj.type === "point") {
            const p = screen(obj.x, obj.y);
            ctx.beginPath();
            ctx.arc(p.x, 4, 0, Math.PI * 2);
            ctx.fill();
            labelPoint = p;
        }

        if (labelPoint !== null) {
            ctx.fillStyle = "#334155";
            ctx.font = "12px Arial";
            ctx.fillText(`#${obj.id}`, labelPoint.x + 8, labelPoint.y - 8);
        }
    }

    objectCount.textContent = `${objects.length} Objects`;
}

async function refreshState() {
    const state = await api("/state");
    objects = state.objects || [];
    drawCanvas();
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!input.value.trim()) return;

    pendingCommand = null;
    executeBtn.disabled = true;
    predictBtn.disabled = true;
    predictionOutput.textContent = "Generating command...";

    try {
        const data = await api("/predict", { text: input.value.trim() });
        pendingCommand = data.command;
        predictionOutput.value = JSON.stringify(pendingCommand, null, 2);
        executeBtn.disabled = false;
    } catch (error) {
        predictionOutput.textContent = `Prediction error: ${error.message}`;
    } finally {
        predictBtn.disabled = false;
    }
});

executeBtn.addEventListener("click", async () => {
    if (!pendingCommand) return;

    let editedCommand;

    try {
        editedCommand = JSON.parse(predictionOutput.value);
    } catch (error) {
        executionResult.textContent = "Invalid JSON: " + error.message;
        return;
    }

    if (!editedCommand || typeof editedCommand !== "object"
        || Array.isArray(editedCommand)
        || typeof editedCommand.command !== "string") {
        executionResult.textContent = "Invalid CAD command.";
        return;
    }

    executeBtn.disabled = true;

    try {
        const data = await api("/execute", { command: editedCommand });
        executionResult.textContent = typeof data.result === "string"
            ? data.result : JSON.stringify(data.result);

        const entry = document.createElement("div");
        entry.textContent = `${input.value.trim()} → ${executionResult.textContent}`;
        historyPanel.prepend(entry);

        objects = data.objects || [];
        drawCanvas();
        pendingCommand = null;
    } catch (error) {
        executionResult.textContent = `Execution error: ${error.message}`;
        executeBtn.disabled = false;
    }
});

document.getElementById("undo-btn").addEventListener("click", async () => {
    try {
        await api("/execute", { command: { command: "UNDO" } });
        await refreshState();
        executionResult.textContent = "Undo completed.";
    } catch (error) {
        executionResult.textContent = error.message;
    }
});

document.getElementById("reset-btn").addEventListener("click", async () => {
    if (!window.confirm("Clear the entire canvas?")) return;

    try {
        await api("/execute", { command: { command: "CLEAR_CANVAS" } });
        await refreshState();
        executionResult.textContent = "Canvas cleared.";
    } catch (error) {
        executionResult.textContent = error.message;
    }
});

refreshState().catch(error => {
    executionResult.textContent = `Connection error: ${error.message}`;
});


document.getElementById("export-dxf-btn").addEventListener("click", () => {
    window.location.href = "/export/dxf";
});


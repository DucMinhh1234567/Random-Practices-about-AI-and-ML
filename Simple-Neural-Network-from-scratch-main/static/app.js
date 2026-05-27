const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

const resultEl = document.getElementById("result");
const barsEl = document.getElementById("bars");
const errEl = document.getElementById("err");

const clearBtn = document.getElementById("clear");
const predictBtn = document.getElementById("predict");

function initBars() {
  barsEl.innerHTML = "";
  for (let i = 0; i < 10; i++) {
    const row = document.createElement("div");
    row.className = "bar";
    row.innerHTML = `
      <div style="font-weight:700">${i}</div>
      <div class="track"><div class="fill" data-fill="${i}"></div></div>
      <div class="pct" data-pct="${i}">0.0000%</div>
    `;
    barsEl.appendChild(row);
  }
}

function setBars(proba) {
  for (let i = 0; i < 10; i++) {
    const p = proba[i] ?? 0;
    const fill = barsEl.querySelector(`[data-fill="${i}"]`);
    const pct = barsEl.querySelector(`[data-pct="${i}"]`);
    if (fill) fill.style.width = `${Math.max(0, Math.min(1, p)) * 100}%`;
    if (pct) pct.textContent = `${(p * 100).toFixed(4)}%`;
  }
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

initBars();
clearCanvas();

let isDrawing = false;

function getPos(evt) {
  const rect = canvas.getBoundingClientRect();
  const x = (evt.clientX - rect.left) * (canvas.width / rect.width);
  const y = (evt.clientY - rect.top) * (canvas.height / rect.height);
  return { x, y };
}

function drawAt(x, y) {
  // Draw on the 28x28 grid directly
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.arc(x, y, 1.6, 0, Math.PI * 2);
  ctx.fill();
}

canvas.addEventListener("pointerdown", (e) => {
  isDrawing = true;
  canvas.setPointerCapture(e.pointerId);
  const { x, y } = getPos(e);
  drawAt(x, y);
});

canvas.addEventListener("pointermove", (e) => {
  if (!isDrawing) return;
  const { x, y } = getPos(e);
  drawAt(x, y);
});

canvas.addEventListener("pointerup", () => {
  isDrawing = false;
});
canvas.addEventListener("pointercancel", () => {
  isDrawing = false;
});

clearBtn.addEventListener("click", () => {
  errEl.textContent = "";
  resultEl.textContent = "Prediction: —";
  setBars(new Array(10).fill(0));
  clearCanvas();
});

function canvasToPixels784() {
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = img.data;
  const pixels = new Array(28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    // since we draw white on black, use red channel (all equal)
    pixels[i] = data[i * 4];
  }
  return pixels;
}

predictBtn.addEventListener("click", async () => {
  errEl.textContent = "";
  resultEl.textContent = "Prediction: …";

  try {
    const pixels = canvasToPixels784();
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pixels }),
    });
    const json = await res.json();
    if (!res.ok) {
      throw new Error(json?.error || `HTTP ${res.status}`);
    }

    resultEl.textContent = `Prediction: ${json.pred}`;
    const proba = json.proba || [];
    setBars(proba);
  } catch (e) {
    resultEl.textContent = "Prediction: —";
    errEl.textContent = String(e?.message || e);
  }
});


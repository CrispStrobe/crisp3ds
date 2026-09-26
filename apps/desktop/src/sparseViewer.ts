import type { SparsePoint } from "./sparseReport";

export function mountSparseViewer(canvas: HTMLCanvasElement, points: SparsePoint[]): () => void {
  const context = canvas.getContext("2d");
  if (!context || !points.length) return () => {};
  const mins = [Infinity, Infinity, Infinity];
  const maxs = [-Infinity, -Infinity, -Infinity];
  for (const point of points) for (let axis = 0; axis < 3; axis++) {
    mins[axis] = Math.min(mins[axis], point.positionMm[axis]);
    maxs[axis] = Math.max(maxs[axis], point.positionMm[axis]);
  }
  const center = mins.map((min, axis) => (min + maxs[axis]) / 2);
  const extent = Math.max(1, ...maxs.map((max, axis) => max - mins[axis]));
  const sampleStep = Math.max(1, Math.ceil(points.length / 20_000));
  let yaw = -0.5;
  let pitch = -0.35;
  let zoom = 1;
  let pointer: { x: number; y: number } | null = null;

  function draw() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    const pixelWidth = Math.round(width * ratio);
    const pixelHeight = Math.round(height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) { canvas.width = pixelWidth; canvas.height = pixelHeight; }
    context!.setTransform(ratio, 0, 0, ratio, 0, 0);
    context!.fillStyle = "#10202a";
    context!.fillRect(0, 0, width, height);
    const scale = Math.min(width, height) * 0.66 * zoom / extent;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const project = (position: number[]) => {
      const x = position[0] - center[0], y = position[1] - center[1], z = position[2] - center[2];
      const rx = cy * x + sy * z;
      const rz = -sy * x + cy * z;
      return { x: width / 2 + rx * scale, y: height / 2 + (cp * y - sp * rz) * scale, depth: sp * y + cp * rz };
    };
    const origin = project([0, 0, 0]);
    const axisLength = Math.max(extent * 0.25, 1);
    const axes: [number[], string, string][] = [
      [[axisLength, 0, 0], "#f28c82", "X"], [[0, axisLength, 0], "#9ce0a8", "Y"], [[0, 0, axisLength], "#91baff", "Z"],
    ];
    context!.lineWidth = 1.5;
    context!.font = "11px system-ui";
    for (const [position, color, label] of axes) {
      const end = project(position);
      context!.strokeStyle = color;
      context!.beginPath(); context!.moveTo(origin.x, origin.y); context!.lineTo(end.x, end.y); context!.stroke();
      context!.fillStyle = color;
      context!.fillText(label, end.x + 4, end.y - 4);
    }
    const projected: { x: number; y: number; depth: number; error: number }[] = [];
    for (let index = 0; index < points.length; index += sampleStep) {
      const point = points[index];
      projected.push({ ...project(point.positionMm), error: point.maxReprojectionErrorPx });
    }
    projected.sort((a, b) => a.depth - b.depth);
    for (const point of projected) {
      context!.fillStyle = point.error > 1 ? "#ffc989" : "#80dcdd";
      context!.fillRect(point.x - 1, point.y - 1, 2, 2);
    }
    context!.fillStyle = "#b3c4cb";
    context!.fillText(`${projected.length.toLocaleString()} shown · drag to orbit · scroll to zoom`, 14, height - 14);
  }

  canvas.addEventListener("pointerdown", (event) => { pointer = { x: event.clientX, y: event.clientY }; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => {
    if (!pointer) return;
    yaw += (event.clientX - pointer.x) * 0.01;
    pitch = Math.max(-1.5, Math.min(1.5, pitch + (event.clientY - pointer.y) * 0.01));
    pointer = { x: event.clientX, y: event.clientY };
    draw();
  });
  canvas.addEventListener("pointerup", () => { pointer = null; });
  canvas.addEventListener("pointercancel", () => { pointer = null; });
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); zoom = Math.max(0.2, Math.min(8, zoom * Math.exp(-event.deltaY * 0.001))); draw(); }, { passive: false });
  const observer = new ResizeObserver(draw);
  observer.observe(canvas);
  draw();
  return () => observer.disconnect();
}

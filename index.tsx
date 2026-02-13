
import React, { useState, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';

// --- Types ---
type Point = { x: number; y: number };

type ProcessedImage = {
  id: string;
  originalSrc: string;
  count: number;
  processedSrc?: string;
};

type FlaskData = {
  id: string;
  name: string;
  images: ProcessedImage[];
};

// --- Math & Homography ---
const solveLinearSystem = (A: number[][], B: number[]): number[] => {
  const n = A.length;
  for (let i = 0; i < n; i++) A[i].push(B[i]);
  for (let i = 0; i < n; i++) {
    let maxEl = Math.abs(A[i][i]), maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A[k][i]) > maxEl) { maxEl = Math.abs(A[k][i]); maxRow = k; }
    }
    [A[maxRow], A[i]] = [A[i], A[maxRow]];
    for (let k = i + 1; k < n; k++) {
      const c = -A[k][i] / A[i][i];
      for (let j = i; j < n + 1; j++) A[k][j] = i === j ? 0 : A[k][j] + c * A[i][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) sum += A[i][j] * x[j];
    x[i] = (A[i][n] - sum) / A[i][i];
  }
  return x;
};

const getHomographyMatrix = (src: Point[], dst: Point[]) => {
  const A: number[][] = [], B: number[] = [];
  for (let i = 0; i < 4; i++) {
    const s = src[i], d = dst[i];
    A.push([s.x, s.y, 1, 0, 0, 0, -s.x * d.x, -s.y * d.x]);
    A.push([0, 0, 0, s.x, s.y, 1, -s.x * d.y, -s.y * d.y]);
    B.push(d.x, d.y);
  }
  const h = solveLinearSystem(A, B);
  h.push(1);
  return h;
};

const warpPerspective = (ctx: CanvasRenderingContext2D, img: HTMLImageElement, corners: Point[], width: number, height: number) => {
  const srcPixels = corners.map(p => ({ x: p.x * img.naturalWidth, y: p.y * img.naturalHeight }));
  const dstPixels = [{ x: 0, y: 0 }, { x: width, y: 0 }, { x: width, y: height }, { x: 0, y: height }];
  const H = getHomographyMatrix(dstPixels, srcPixels);
  const imgData = ctx.createImageData(width, height);
  const data = imgData.data;

  const tmp = document.createElement('canvas');
  tmp.width = img.naturalWidth; tmp.height = img.naturalHeight;
  const tc = tmp.getContext('2d')!;
  tc.drawImage(img, 0, 0);
  const srcData = tc.getImageData(0, 0, tmp.width, tmp.height).data;
  const srcW = tmp.width, srcH = tmp.height;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const d = H[6] * x + H[7] * y + H[8];
      const u = Math.floor((H[0] * x + H[1] * y + H[2]) / d);
      const v = Math.floor((H[3] * x + H[4] * y + H[5]) / d);
      const di = (y * width + x) * 4;
      if (u >= 0 && u < srcW && v >= 0 && v < srcH) {
        const si = (v * srcW + u) * 4;
        data[di] = srcData[si]; data[di + 1] = srcData[si + 1]; data[di + 2] = srcData[si + 2]; data[di + 3] = 255;
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
};

// --- Helpers ---
const generateId = () => Math.random().toString(36).substr(2, 9);

const saveImage = (dataUrl: string, filename: string) => {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  a.click();
};

// --- CV Pipeline ---
const processImage = (
  imgElement: HTMLImageElement, corners: Point[], threshold: number,
  invert: boolean, showMask: boolean, minCellSize: number, circularityThresh: number
): { count: number; dataUrl: string } => {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return { count: 0, dataUrl: '' };

  const size = 600;
  canvas.width = size; canvas.height = size;
  warpPerspective(ctx, imgElement, corners, size, size);

  const imageData = ctx.getImageData(0, 0, size, size);
  const data = imageData.data;

  // Grayscale
  const gray = new Uint8Array(size * size);
  for (let i = 0; i < size * size; i++)
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];

  // Threshold
  let binary = new Uint8Array(size * size);
  for (let i = 0; i < size * size; i++)
    binary[i] = (invert ? gray[i] < threshold : gray[i] > threshold) ? 1 : 0;

  // Grid line removal (projection)
  const rowSum = new Int32Array(size), colSum = new Int32Array(size);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++)
      if (binary[y * size + x]) { rowSum[y]++; colSum[x]++; }

  const lineThreshold = size * 0.60;
  for (let y = 0; y < size; y++)
    if (rowSum[y] > lineThreshold) for (let x = 0; x < size; x++) binary[y * size + x] = 0;
  for (let x = 0; x < size; x++)
    if (colSum[x] > lineThreshold) for (let y = 0; y < size; y++) binary[y * size + x] = 0;

  // Morphological opening
  const temp = new Uint8Array(size * size);
  for (let y = 1; y < size - 1; y++)
    for (let x = 1; x < size - 1; x++) {
      const i = y * size + x;
      if (binary[i] && binary[i - 1] && binary[i + 1] && binary[i - size] && binary[i + size]) temp[i] = 1;
    }
  const finalBinary = new Uint8Array(size * size);
  for (let y = 1; y < size - 1; y++)
    for (let x = 1; x < size - 1; x++) {
      const i = y * size + x;
      if (temp[i]) { finalBinary[i] = 1; finalBinary[i - 1] = 1; finalBinary[i + 1] = 1; finalBinary[i - size] = 1; finalBinary[i + size] = 1; }
    }

  // Blob analysis
  const visited = new Uint8Array(size * size);
  const blobs: { area: number; perimeter: number; cx: number; cy: number; minX: number; maxX: number; minY: number; maxY: number }[] = [];
  const dx = [1, -1, 0, 0], dy = [0, 0, 1, -1];

  for (let i = 0; i < size * size; i++) {
    if (!finalBinary[i] || visited[i]) continue;
    const stack = [i]; visited[i] = 1;
    let area = 0, perimeter = 0, sumX = 0, sumY = 0;
    let minX = size, maxX = 0, minY = size, maxY = 0;

    while (stack.length) {
      const curr = stack.pop()!;
      const cx = curr % size, cy = Math.floor(curr / size);
      area++; sumX += cx; sumY += cy;
      if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;

      let neighbors = 0;
      for (let k = 0; k < 4; k++) {
        const nx = cx + dx[k], ny = cy + dy[k], ni = ny * size + nx;
        if (nx >= 0 && nx < size && ny >= 0 && ny < size && finalBinary[ni]) {
          neighbors++;
          if (!visited[ni]) { visited[ni] = 1; stack.push(ni); }
        }
      }
      if (neighbors < 4) perimeter++;
    }
    if (area >= minCellSize)
      blobs.push({ area, perimeter, cx: sumX / area, cy: sumY / area, minX, maxX, minY, maxY });
  }

  // Median area of circular blobs
  const validBlobs = blobs.filter(b => (4 * Math.PI * b.area) / (b.perimeter ** 2) > circularityThresh);
  validBlobs.sort((a, b) => a.area - b.area);
  let medianArea = validBlobs.length
    ? validBlobs[Math.floor(validBlobs.length / 2)].area
    : blobs.length ? blobs.reduce((s, b) => s + b.area, 0) / blobs.length : minCellSize;
  medianArea = Math.max(medianArea, minCellSize);

  let totalCount = 0;

  // Render
  if (showMask) {
    const maskImg = ctx.createImageData(size, size);
    const cellPixels = new Uint8Array(size * size);
    const visited2 = new Uint8Array(size * size);
    for (let i = 0; i < size * size; i++) {
      if (!finalBinary[i] || visited2[i]) continue;
      const stack2 = [i]; visited2[i] = 1;
      const pixels: number[] = [];
      let area2 = 0, perim2 = 0, bMinX = size, bMaxX = 0, bMinY = size, bMaxY = 0;
      while (stack2.length) {
        const c = stack2.pop()!;
        const px = c % size, py = Math.floor(c / size);
        pixels.push(c); area2++;
        if (px < bMinX) bMinX = px; if (px > bMaxX) bMaxX = px;
        if (py < bMinY) bMinY = py; if (py > bMaxY) bMaxY = py;
        let nn = 0;
        for (let k = 0; k < 4; k++) {
          const nx2 = px + dx[k], ny2 = py + dy[k], ni = ny2 * size + nx2;
          if (nx2 >= 0 && nx2 < size && ny2 >= 0 && ny2 < size && finalBinary[ni]) {
            nn++;
            if (!visited2[ni]) { visited2[ni] = 1; stack2.push(ni); }
          }
        }
        if (nn < 4) perim2++;
      }
      if (area2 >= minCellSize) {
        const circ2 = (4 * Math.PI * area2) / (perim2 ** 2);
        const bw = bMaxX - bMinX + 1, bh = bMaxY - bMinY + 1;
        if (circ2 >= circularityThresh * 0.5 && Math.max(bw, bh) / Math.min(bw, bh) <= 4.0)
          for (const p of pixels) cellPixels[p] = 1;
      }
    }
    for (let i = 0; i < size * size; i++) {
      const v = cellPixels[i] ? 255 : finalBinary[i] ? 60 : 0;
      maskImg.data[i * 4] = v; maskImg.data[i * 4 + 1] = v; maskImg.data[i * 4 + 2] = v; maskImg.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(maskImg, 0, 0);
  } else {
    ctx.putImageData(imageData, 0, 0);
  }

  blobs.forEach(b => {
    const circ = (4 * Math.PI * b.area) / (b.perimeter ** 2);
    if (circ < circularityThresh * 0.5) return;
    const w = b.maxX - b.minX + 1, h = b.maxY - b.minY + 1;
    if (Math.max(w, h) / Math.min(w, h) > 4.0) return;

    const n = b.area > medianArea * 1.5 ? Math.round(b.area / medianArea) : 1;
    totalCount += n;

    if (!showMask) {
      ctx.beginPath();
      if (n === 1) {
        ctx.strokeStyle = '#00ff00'; ctx.lineWidth = 2;
        ctx.arc(b.cx, b.cy, Math.sqrt(b.area / Math.PI) + 2, 0, 2 * Math.PI); ctx.stroke();
      } else {
        ctx.strokeStyle = '#00ffff'; ctx.lineWidth = 2;
        ctx.rect(b.minX, b.minY, b.maxX - b.minX, b.maxY - b.minY); ctx.stroke();
        ctx.fillStyle = '#fff'; ctx.font = 'bold 16px monospace';
        ctx.shadowColor = 'black'; ctx.shadowBlur = 4;
        ctx.fillText(n.toString(), b.cx - 6, b.cy + 6); ctx.shadowBlur = 0;
      }
    }
  });

  return { count: totalCount, dataUrl: canvas.toDataURL() };
};

// --- Components ---

const FlaskManager = ({ flasks, activeFlaskId, onAddFlask, onSelectFlask, onUpdateFlask }: {
  flasks: FlaskData[]; activeFlaskId: string | null;
  onAddFlask: () => void; onSelectFlask: (id: string) => void; onUpdateFlask: (id: string, data: Partial<FlaskData>) => void;
}) => {
  const activeFlask = flasks.find(f => f.id === activeFlaskId);
  return (
    <div className="card">
      <div className="flex justify-between mb-4">
        <div className="flex">
          <h2 className="font-bold text-xl">Flasks</h2>
          <span className="text-sm text-gray" style={{ alignSelf: 'center' }}>| Manage your cultures</span>
        </div>
        <button onClick={onAddFlask} className="primary">+ New Flask</button>
      </div>
      {flasks.length === 0 ? (
        <p className="text-gray text-center py-8 bg-panel rounded">No flasks active. Create one to begin.</p>
      ) : (
        <div className="flex" style={{ overflowX: 'auto', paddingBottom: '10px' }}>
          {flasks.map(flask => (
            <button key={flask.id} onClick={() => onSelectFlask(flask.id)}
              className={`secondary ${flask.id === activeFlaskId ? 'active' : ''}`} style={{ minWidth: '140px' }}>
              {flask.name}
            </button>
          ))}
        </div>
      )}
      {activeFlask && (
        <div className="mt-4 bg-panel p-4 rounded border-dashed">
          <label>Flask Name</label>
          <input type="text" value={activeFlask.name} onChange={e => onUpdateFlask(activeFlask.id, { name: e.target.value })} />
        </div>
      )}
    </div>
  );
};

const ImageProcessor = ({ flask, onUpdateCount }: { flask: FlaskData; onUpdateCount: (imgId: string, count: number, processedSrc: string) => void }) => {
  const [selectedImgIndex, setSelectedImgIndex] = useState(0);
  const [corners, setCorners] = useState<Point[]>([{ x: 0.2, y: 0.2 }, { x: 0.8, y: 0.2 }, { x: 0.8, y: 0.8 }, { x: 0.2, y: 0.8 }]);
  const [clickMode, setClickMode] = useState(false);
  const [clickStep, setClickStep] = useState(0);
  const [threshold, setThreshold] = useState(180);
  const [invert, setInvert] = useState(false);
  const [showMask, setShowMask] = useState(false);
  const [minCellSize, setMinCellSize] = useState(20);
  const [circularity, setCircularity] = useState(0.4);
  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const activeImg = flask.images[selectedImgIndex];

  useEffect(() => {
    setCorners([{ x: 0.2, y: 0.2 }, { x: 0.8, y: 0.2 }, { x: 0.8, y: 0.8 }, { x: 0.2, y: 0.8 }]);
  }, [selectedImgIndex]);

  const runProcessing = () => {
    if (imgRef.current && activeImg) {
      const result = processImage(imgRef.current, corners, threshold, invert, showMask, minCellSize, circularity);
      onUpdateCount(activeImg.id, result.count, result.dataUrl);
    }
  };

  const handleMouseDown = (e: React.MouseEvent, index: number) => {
    if (clickMode) return;
    e.preventDefault(); e.stopPropagation();
    setDragIndex(index);
  };

  const handleContainerClick = (e: React.MouseEvent) => {
    if (clickMode && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;
      setCorners(prev => { const next = [...prev]; next[clickStep] = { x, y }; return next; });
      if (clickStep === 3) { setClickMode(false); setClickStep(0); } else setClickStep(s => s + 1);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (dragIndex !== null && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      setCorners(prev => { const next = [...prev]; next[dragIndex] = { x, y }; return next; });
    }
  };

  const handleMouseUp = () => setDragIndex(null);
  const getPointsStr = () => corners.map(p => `${p.x * 100},${p.y * 100}`).join(' ');

  return (
    <div className="card">
      <div className="flex justify-between mb-4">
        <h3 className="font-bold text-lg">Cell Counter</h3>
      </div>
      {flask.images.length === 0 ? (
        <div className="text-center py-10 border-2 border-dashed border-gray-300 rounded bg-panel">
          <p className="mb-4 text-gray">Upload 4 microscope images to begin counting</p>
        </div>
      ) : (
        <div className="grid grid-2" style={{ gap: '24px' }}>
          {/* Left: Editor & Controls */}
          <div>
            <div className="flex mb-2 justify-between">
              <div className="text-sm text-gray font-mono">{clickMode ? `SET CORNER ${clickStep + 1}/4` : 'ROI SELECTION'}</div>
              <button className="icon-btn text-xs" onClick={() => { setClickMode(true); setClickStep(0); }}>{clickMode ? 'CANCEL' : 'MANUAL SET'}</button>
            </div>
            <div className="canvas-container" ref={containerRef}
              onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp} onClick={handleContainerClick}
              style={{ cursor: clickMode ? 'crosshair' : 'default', background: '#111' }}>
              {activeImg && (
                <>
                  <img ref={imgRef} src={activeImg.originalSrc} style={{ pointerEvents: 'none', userSelect: 'none' }} />
                  <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                    <polygon points={getPointsStr()} fill="rgba(14,165,233,0.2)" stroke="var(--accent)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
                  </svg>
                  {corners.map((p, i) => (
                    <div key={i} className="resize-handle"
                      style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%`, backgroundColor: clickMode && i === clickStep ? '#fff' : 'var(--accent)' }}
                      onMouseDown={e => handleMouseDown(e, i)} />
                  ))}
                </>
              )}
            </div>
            <div className="grid grid-2 mt-4 gap-4">
              <div>
                <label>Threshold ({threshold})</label>
                <input type="range" min="150" max="220" value={threshold} onChange={e => setThreshold(parseInt(e.target.value))} />
                <p className="text-xs text-gray" style={{ marginTop: 2 }}>↑ Higher = stricter. ↓ Lower = more detections.</p>
              </div>
              <div>
                <label>Circularity ({circularity.toFixed(2)})</label>
                <input type="range" min="0.1" max="1.0" step="0.05" value={circularity} onChange={e => setCircularity(parseFloat(e.target.value))} />
              </div>
              <div>
                <label>Min Cell Size ({minCellSize}px)</label>
                <input type="range" min="5" max="100" value={minCellSize} onChange={e => setMinCellSize(parseInt(e.target.value))} />
              </div>
              <div className="flex flex-col justify-end">
                <label className="flex items-center text-sm cursor-pointer mb-2">
                  <input type="checkbox" checked={showMask} onChange={e => setShowMask(e.target.checked)} style={{ marginRight: '8px' }} />
                  Show Mask (B&W)
                </label>
                <label className="flex items-center text-sm cursor-pointer">
                  <input type="checkbox" checked={invert} onChange={e => setInvert(e.target.checked)} style={{ marginRight: '8px' }} />
                  Invert (Dark Cells)
                </label>
              </div>
            </div>
            <div className="mt-4">
              <button className="primary w-full" onClick={runProcessing} style={{ padding: '12px' }}>PROCESS & COUNT</button>
            </div>
          </div>

          {/* Right: Results & Thumbnails */}
          <div className="flex-col" style={{ height: '100%' }}>
            <h4 className="font-bold mb-2">Analysis Result</h4>
            <div className="canvas-container bg-panel" style={{ flexGrow: 1, maxHeight: '450px', border: '2px solid var(--border)' }}>
              {activeImg.processedSrc ? (
                <img src={activeImg.processedSrc} style={{ imageRendering: 'pixelated', maxHeight: '100%' }} />
              ) : (
                <div className="text-gray text-center p-4">
                  <p className="font-mono text-sm">NO DATA</p>
                  <p className="text-xs">Click Process to analyze ROI</p>
                </div>
              )}
            </div>
            <div className="p-4 bg-panel border rounded mb-2">
              <div className="flex justify-between items-center mb-2">
                <label className="font-bold">DETECTED COUNT</label>
                <span className="text-3xl font-bold font-mono text-primary">{activeImg.count}</span>
              </div>
              <div className="flex" style={{ gap: '8px' }}>
                <div style={{ flex: 1 }}>
                  <label>Manual Correction</label>
                  <input type="number" value={activeImg?.count || 0}
                    onChange={e => onUpdateCount(activeImg.id, parseInt(e.target.value) || 0, activeImg.processedSrc || activeImg.originalSrc)} />
                </div>
                <button className="primary" disabled={!activeImg.processedSrc}
                  style={{ alignSelf: 'flex-end', whiteSpace: 'nowrap' }}
                  onClick={() => activeImg.processedSrc && saveImage(activeImg.processedSrc, `${flask.name}_image${selectedImgIndex + 1}.png`)}>
                  💾 Save Image
                </button>
              </div>
            </div>
            <div>
              <h4 className="font-bold mb-2 text-xs text-gray">IMAGES IN FLASK</h4>
              <div className="flex" style={{ overflowX: 'auto', paddingBottom: '5px' }}>
                {flask.images.map((img, idx) => (
                  <div key={img.id} onClick={() => setSelectedImgIndex(idx)}
                    style={{ minWidth: '60px', width: '60px', cursor: 'pointer', border: idx === selectedImgIndex ? '2px solid var(--primary)' : '1px solid transparent', opacity: idx === selectedImgIndex ? 1 : 0.6 }}>
                    <img src={img.processedSrc || img.originalSrc} style={{ width: '100%', height: '50px', objectFit: 'cover', display: 'block', borderRadius: '4px' }} />
                    <div className="text-center text-xs font-mono font-bold mt-1">{img.count}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// --- App ---
const App = () => {
  const [flasks, setFlasks] = useState<FlaskData[]>([]);
  const [activeFlaskId, setActiveFlaskId] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    document.body.classList.toggle('dark-mode', darkMode);
  }, [darkMode]);

  const activeFlask = flasks.find(f => f.id === activeFlaskId);

  const addFlask = () => {
    const f: FlaskData = { id: generateId(), name: `Flask ${flasks.length + 1}`, images: [] };
    setFlasks([...flasks, f]);
    setActiveFlaskId(f.id);
  };

  const updateFlask = (id: string, data: Partial<FlaskData>) => {
    setFlasks(flasks.map(f => f.id === id ? { ...f, ...data } : f));
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!activeFlaskId || !e.target.files) return;
    const newImages = Array.from(e.target.files).map(file => ({
      id: generateId(), originalSrc: URL.createObjectURL(file as Blob), count: 0,
    }));
    const flask = flasks.find(f => f.id === activeFlaskId);
    if (flask) updateFlask(activeFlaskId, { images: [...flask.images, ...newImages] });
  };

  const updateImageCount = (imgId: string, count: number, processedSrc: string) => {
    if (!activeFlaskId) return;
    const flask = flasks.find(f => f.id === activeFlaskId);
    if (!flask) return;
    updateFlask(activeFlaskId, { images: flask.images.map(img => img.id === imgId ? { ...img, count, processedSrc } : img) });
  };

  return (
    <div className="container">
      <div className="flex justify-between mb-6">
        <h1 className="text-2xl font-bold text-primary">🔬 Hemocytometer Assistant</h1>
        <button className="secondary" onClick={() => setDarkMode(!darkMode)} style={{ fontSize: '0.8rem', padding: '6px 12px' }}>
          {darkMode ? '☀️ Light' : '🌙 Dark'}
        </button>
      </div>

      <FlaskManager flasks={flasks} activeFlaskId={activeFlaskId} onAddFlask={addFlask} onSelectFlask={setActiveFlaskId} onUpdateFlask={updateFlask} />

      {activeFlask && (
        <>
          <div className="flex justify-between items-center mb-4 card" style={{ padding: '12px 24px' }}>
            <div>
              <span className="font-bold text-sm">IMAGE SET: {activeFlask.name}</span>
              <span className="block text-xs text-gray">{activeFlask.images.length} images loaded</span>
            </div>
            <label className="upload-btn">
              UPLOAD IMAGES
              <input type="file" multiple accept="image/*" onChange={handleImageUpload} className="hidden" />
            </label>
          </div>
          <ImageProcessor flask={activeFlask} onUpdateCount={updateImageCount} />
        </>
      )}
    </div>
  );
};

createRoot(document.getElementById('root')!).render(<App />);

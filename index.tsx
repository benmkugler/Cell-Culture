
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { createRoot } from 'react-dom/client';

// --- Constants (Aligned with CSV) ---
const CONSTANTS = {
  FACTOR_HEMOCYTOMETER: 10000,
  T25_MIN_REQ: 0.7 * 10 ** 6,
  T75_MIN_REQ: 2.1 * 10 ** 6,
  T25_MEDIA_VOL: 5.0,
  T75_MEDIA_VOL: 12.0,
  CRYO_SUGGESTED_PER_TUBE: 1.0, // Default to 1M, but user can change
  CRYO_MAX_VOL_ML: 1.3, // CSV says 1.3
};

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
  passage: number;
  currentVol: number;
  dilutionFactor: number;
  images: ProcessedImage[];
};

// --- Math & Homography Helpers ---
const solveLinearSystem = (A: number[][], B: number[]): number[] => {
  const n = A.length;
  for (let i = 0; i < n; i++) A[i].push(B[i]);
  for (let i = 0; i < n; i++) {
    let maxEl = Math.abs(A[i][i]);
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A[k][i]) > maxEl) {
        maxEl = Math.abs(A[k][i]);
        maxRow = k;
      }
    }
    let tmp = A[maxRow];
    A[maxRow] = A[i];
    A[i] = tmp;
    for (let k = i + 1; k < n; k++) {
      const c = -A[k][i] / A[i][i];
      for (let j = i; j < n + 1; j++) {
        if (i === j) A[k][j] = 0;
        else A[k][j] += c * A[i][j];
      }
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i > -1; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) {
      sum += A[i][j] * x[j];
    }
    x[i] = (A[i][n] - sum) / A[i][i];
  }
  return x;
};

const getHomographyMatrix = (src: Point[], dst: Point[]) => {
  const A: number[][] = [];
  const B: number[] = [];
  for (let i = 0; i < 4; i++) {
    const s = src[i];
    const d = dst[i];
    A.push([s.x, s.y, 1, 0, 0, 0, -s.x * d.x, -s.y * d.x]);
    A.push([0, 0, 0, s.x, s.y, 1, -s.x * d.y, -s.y * d.y]);
    B.push(d.x);
    B.push(d.y);
  }
  const h = solveLinearSystem(A, B);
  h.push(1); 
  return h;
};

const warpPerspective = (
  ctx: CanvasRenderingContext2D, 
  img: HTMLImageElement, 
  corners: Point[], 
  width: number, 
  height: number
) => {
  const srcPixels = corners.map(p => ({ x: p.x * img.naturalWidth, y: p.y * img.naturalHeight }));
  const dstPixels = [
    { x: 0, y: 0 },
    { x: width, y: 0 },
    { x: width, y: height },
    { x: 0, y: height }
  ];
  const H = getHomographyMatrix(dstPixels, srcPixels);
  const imgData = ctx.createImageData(width, height);
  const data = imgData.data;

  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = img.naturalWidth;
  tempCanvas.height = img.naturalHeight;
  const tempCtx = tempCanvas.getContext('2d');
  if (!tempCtx) return;
  tempCtx.drawImage(img, 0, 0);
  const srcData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height).data;
  const srcW = tempCanvas.width;
  const srcH = tempCanvas.height;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const d = H[6] * x + H[7] * y + H[8];
      const u = (H[0] * x + H[1] * y + H[2]) / d;
      const v = (H[3] * x + H[4] * y + H[5]) / d;
      const srcX = Math.floor(u);
      const srcY = Math.floor(v);
      const dstIdx = (y * width + x) * 4;

      if (srcX >= 0 && srcX < srcW && srcY >= 0 && srcY < srcH) {
        const srcIdx = (srcY * srcW + srcX) * 4;
        data[dstIdx] = srcData[srcIdx];
        data[dstIdx + 1] = srcData[srcIdx + 1];
        data[dstIdx + 2] = srcData[srcIdx + 2];
        data[dstIdx + 3] = 255;
      } else {
        data[dstIdx + 3] = 0;
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
};

// --- Helper Functions ---
const safeDiv = (n: number, d: number) => (d === 0 ? 0 : n / d);
const generateId = () => Math.random().toString(36).substr(2, 9);

// --- CV ALGORITHM: Grid Removal & Robust Counting ---
const processImage = (
  imgElement: HTMLImageElement,
  corners: Point[], 
  threshold: number,
  invert: boolean,
  showMask: boolean,
  minCellSize: number,
  circularityThresh: number
): { count: number; dataUrl: string } => {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return { count: 0, dataUrl: '' };

  const size = 600; // Standard processing size
  canvas.width = size;
  canvas.height = size;
  
  warpPerspective(ctx, imgElement, corners, size, size);

  const imageData = ctx.getImageData(0, 0, size, size);
  const data = imageData.data;

  // 1. Grayscale
  const gray = new Uint8Array(size * size);
  for (let i = 0; i < size * size; i++) {
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  }

  // 2. Thresholding
  let binary = new Uint8Array(size * size);
  for (let i = 0; i < size * size; i++) {
    const val = gray[i];
    const isHit = invert ? (val < threshold) : (val > threshold);
    binary[i] = isHit ? 1 : 0;
  }

  // 3. Grid Line Removal (Projection Method)
  // Grid lines create peaks in row/col sums.
  const rowSum = new Int32Array(size).fill(0);
  const colSum = new Int32Array(size).fill(0);

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (binary[y * size + x]) {
        rowSum[y]++;
        colSum[x]++;
      }
    }
  }

  // Identify lines: if > 60% of pixels in a row/col are active, it's likely a line
  const lineThreshold = size * 0.60; 
  
  for (let y = 0; y < size; y++) {
    if (rowSum[y] > lineThreshold) {
      // Mask out entire row
      for (let x = 0; x < size; x++) binary[y * size + x] = 0;
    }
  }
  for (let x = 0; x < size; x++) {
    if (colSum[x] > lineThreshold) {
      // Mask out entire col
      for (let y = 0; y < size; y++) binary[y * size + x] = 0;
    }
  }

  // 4. Morphological Opening (Erode -> Dilate) to remove noise/dust
  const temp = new Uint8Array(size * size);
  // Erode
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const idx = y * size + x;
      if (binary[idx]) {
         if (binary[idx-1] && binary[idx+1] && binary[idx-size] && binary[idx+size]) {
           temp[idx] = 1;
         }
      }
    }
  }
  // Dilate
  const finalBinary = new Uint8Array(size * size);
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const idx = y * size + x;
      if (temp[idx]) {
        finalBinary[idx] = 1;
        finalBinary[idx-1] = 1;
        finalBinary[idx+1] = 1;
        finalBinary[idx-size] = 1;
        finalBinary[idx+size] = 1;
      }
    }
  }

  // 5. Blob Analysis
  const visited = new Uint8Array(size * size);
  const blobs: { area: number, perimeter: number, cx: number, cy: number, minX: number, maxX: number, minY: number, maxY: number }[] = [];
  
  // Directions for perimeter (4-connectivity)
  const dx = [1, -1, 0, 0];
  const dy = [0, 0, 1, -1];

  for (let i = 0; i < size * size; i++) {
    if (finalBinary[i] && !visited[i]) {
      let stack = [i];
      visited[i] = 1;
      let area = 0;
      let perimeter = 0;
      let sumX = 0, sumY = 0;
      let minX = size, maxX = 0, minY = size, maxY = 0;

      while(stack.length > 0) {
        const curr = stack.pop()!;
        const cx = curr % size;
        const cy = Math.floor(curr / size);
        
        area++;
        sumX += cx;
        sumY += cy;
        if(cx < minX) minX = cx;
        if(cx > maxX) maxX = cx;
        if(cy < minY) minY = cy;
        if(cy > maxY) maxY = cy;

        // Check 4 neighbors
        let neighborsCount = 0;
        for(let k=0; k<4; k++) {
          const nx = cx + dx[k];
          const ny = cy + dy[k];
          const nIdx = ny * size + nx;
          if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
             if (finalBinary[nIdx]) {
               neighborsCount++;
               if (!visited[nIdx]) {
                 visited[nIdx] = 1;
                 stack.push(nIdx);
               }
             }
          }
        }
        // If not surrounded by 4 neighbors, it's an edge pixel
        if (neighborsCount < 4) perimeter++;
      }
      
      if (area >= minCellSize) {
        blobs.push({
          area, 
          perimeter, 
          cx: sumX / area, 
          cy: sumY / area, 
          minX, maxX, minY, maxY 
        });
      }
    }
  }

  // 6. Counting with Cluster Splitting
  // Determine Median Area of "Circular" cells
  const validBlobs = blobs.filter(b => {
    // Circularity = 4 * PI * Area / Perimeter^2. Circle = 1.0, Square = 0.785
    const circ = (4 * Math.PI * b.area) / (b.perimeter * b.perimeter);
    return circ > circularityThresh;
  });

  // Calculate Median Area of high-quality single cells
  validBlobs.sort((a,b) => a.area - b.area);
  let medianArea = 0;
  if (validBlobs.length > 0) {
    medianArea = validBlobs[Math.floor(validBlobs.length / 2)].area;
  } else if (blobs.length > 0) {
    medianArea = blobs.reduce((sum, b) => sum + b.area, 0) / blobs.length;
  }
  medianArea = Math.max(medianArea, minCellSize);

  let totalCount = 0;

  // Render Result
  if (!showMask) {
    ctx.putImageData(imageData, 0, 0); // Original
  } else {
    // B&W mask: black background, white binary, bright cyan for detected cell regions
    const maskImg = ctx.createImageData(size, size);
    // Build a set of pixels belonging to valid blobs (post-filter) for highlighting
    const cellPixels = new Uint8Array(size * size);
    // Re-label valid blobs via second flood-fill pass over finalBinary using visited2
    const visited2 = new Uint8Array(size * size);
    for (let i = 0; i < size * size; i++) {
      if (finalBinary[i] && !visited2[i]) {
        const stack2 = [i];
        visited2[i] = 1;
        const pixels: number[] = [];
        let area2 = 0, perim2 = 0;
        let bMinX = size, bMaxX = 0, bMinY = size, bMaxY = 0;
        while (stack2.length > 0) {
          const c = stack2.pop()!;
          const px = c % size, py = Math.floor(c / size);
          pixels.push(c);
          area2++;
          if (px < bMinX) bMinX = px;
          if (px > bMaxX) bMaxX = px;
          if (py < bMinY) bMinY = py;
          if (py > bMaxY) bMaxY = py;
          let nn = 0;
          for (let k = 0; k < 4; k++) {
            const nx2 = px + dx[k], ny2 = py + dy[k];
            const ni = ny2 * size + nx2;
            if (nx2 >= 0 && nx2 < size && ny2 >= 0 && ny2 < size && finalBinary[ni]) {
              nn++;
              if (!visited2[ni]) { visited2[ni] = 1; stack2.push(ni); }
            }
          }
          if (nn < 4) perim2++;
        }
        if (area2 >= minCellSize) {
          const circ2 = (4 * Math.PI * area2) / (perim2 * perim2);
          const bw = bMaxX - bMinX + 1, bh = bMaxY - bMinY + 1;
          const ar2 = Math.max(bw, bh) / Math.min(bw, bh);
          if (circ2 >= circularityThresh * 0.5 && ar2 <= 4.0) {
            for (const p of pixels) cellPixels[p] = 1;
          }
        }
      }
    }
    for (let i = 0; i < size * size; i++) {
      if (cellPixels[i]) {
        // Detected cell: bright white
        maskImg.data[i * 4] = 255;
        maskImg.data[i * 4 + 1] = 255;
        maskImg.data[i * 4 + 2] = 255;
      } else if (finalBinary[i]) {
        // Binary foreground but not a valid cell: dark gray
        maskImg.data[i * 4] = 60;
        maskImg.data[i * 4 + 1] = 60;
        maskImg.data[i * 4 + 2] = 60;
      } else {
        // Background: black
        maskImg.data[i * 4] = 0;
        maskImg.data[i * 4 + 1] = 0;
        maskImg.data[i * 4 + 2] = 0;
      }
      maskImg.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(maskImg, 0, 0);
  }

  blobs.forEach(b => {
    // Recalculate circularity
    const circ = (4 * Math.PI * b.area) / (b.perimeter * b.perimeter);
    
    // Strict garbage filter (lines that weren't caught, or irregular dust)
    if (circ < circularityThresh * 0.5) return; 

    // Safeguard: Filter out line bits based on Aspect Ratio
    // Cells are round (AR ~ 1), clusters might be 2:1 or 3:1.
    // Grid line bits are typically long and thin (AR > 4).
    const w = b.maxX - b.minX + 1;
    const h = b.maxY - b.minY + 1;
    const aspectRatio = Math.max(w, h) / Math.min(w, h);
    
    if (aspectRatio > 4.0) return;

    let n = 1;
    // Split clusters
    if (b.area > medianArea * 1.5) {
      n = Math.round(b.area / medianArea);
    }
    totalCount += n;

    // Draw Overlays
    if (!showMask) {
      ctx.beginPath();
      const radius = Math.sqrt(b.area / Math.PI);
      
      if (n === 1) {
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.arc(b.cx, b.cy, radius + 2, 0, 2*Math.PI);
        ctx.stroke();
      } else {
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 2;
        ctx.rect(b.minX, b.minY, b.maxX - b.minX, b.maxY - b.minY);
        ctx.stroke();
        
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 16px monospace';
        ctx.shadowColor = 'black';
        ctx.shadowBlur = 4;
        ctx.fillText(n.toString(), b.cx - 6, b.cy + 6);
        ctx.shadowBlur = 0;
      }
    }
  });

  return { count: totalCount, dataUrl: canvas.toDataURL() };
};

// --- Components ---

const FlaskManager = ({ 
  flasks, 
  activeFlaskId, 
  onAddFlask, 
  onSelectFlask, 
  onUpdateFlask 
}: {
  flasks: FlaskData[],
  activeFlaskId: string | null,
  onAddFlask: () => void,
  onSelectFlask: (id: string) => void,
  onUpdateFlask: (id: string, data: Partial<FlaskData>) => void
}) => {
  const activeFlask = flasks.find(f => f.id === activeFlaskId);

  return (
    <div className="card">
      <div className="flex justify-between mb-4">
        <div className="flex">
            <h2 className="font-bold text-xl">Flasks</h2>
            <span className="text-sm text-gray" style={{alignSelf:'center'}}>| Manage your cultures</span>
        </div>
        <button onClick={onAddFlask} className="primary">+ New Flask</button>
      </div>

      {flasks.length === 0 ? (
        <p className="text-gray text-center py-8 bg-panel rounded">No flasks active. Create one to begin.</p>
      ) : (
        <div className="flex" style={{ overflowX: 'auto', paddingBottom: '10px' }}>
          {flasks.map(flask => (
            <button
              key={flask.id}
              onClick={() => onSelectFlask(flask.id)}
              className={`secondary ${flask.id === activeFlaskId ? 'active' : ''}`}
              style={{ minWidth: '140px' }}
            >
              {flask.name}
              <div className="text-xs text-gray font-normal mt-1">P{flask.passage}</div>
            </button>
          ))}
        </div>
      )}

      {activeFlask && (
        <div className="grid grid-4 mt-4 bg-panel p-4 rounded border-dashed">
          <div>
            <label>Flask Name</label>
            <input 
              type="text" 
              value={activeFlask.name} 
              onChange={(e) => onUpdateFlask(activeFlask.id, { name: e.target.value })} 
            />
          </div>
          <div>
            <label>Passage #</label>
            <input 
              type="number" 
              value={activeFlask.passage} 
              onChange={(e) => onUpdateFlask(activeFlask.id, { passage: parseInt(e.target.value) || 0 })} 
            />
          </div>
          <div>
            <label>Current Vol (mL)</label>
            <input 
              type="number" 
              value={activeFlask.currentVol} 
              step="0.1"
              onChange={(e) => onUpdateFlask(activeFlask.id, { currentVol: parseFloat(e.target.value) || 0 })} 
            />
          </div>
          <div>
            <label>Dilution Factor</label>
            <input 
              type="number" 
              value={activeFlask.dilutionFactor} 
              step="0.1"
              onChange={(e) => onUpdateFlask(activeFlask.id, { dilutionFactor: parseFloat(e.target.value) || 1 })} 
            />
          </div>
        </div>
      )}
    </div>
  );
};

const ImageProcessor = ({ flask, onUpdateCount }: { flask: FlaskData, onUpdateCount: (imgId: string, count: number, processedSrc: string) => void }) => {
  const [selectedImgIndex, setSelectedImgIndex] = useState(0);
  
  const [corners, setCorners] = useState<Point[]>([
      {x: 0.2, y: 0.2}, {x: 0.8, y: 0.2}, {x: 0.8, y: 0.8}, {x: 0.2, y: 0.8}
  ]);
  const [clickMode, setClickMode] = useState(false);
  const [clickStep, setClickStep] = useState(0);

  // CV Parameters
  const [threshold, setThreshold] = useState(180); 
  const [invert, setInvert] = useState(false);
  const [showMask, setShowMask] = useState(false);
  const [minCellSize, setMinCellSize] = useState(20); 
  const [circularity, setCircularity] = useState(0.4); 

  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dragIndex, setDragIndex] = useState<number | null>(null);

  const activeImg = flask.images[selectedImgIndex];

  // Reset corners when switching images (optional, could persist)
  useEffect(() => {
    setCorners([{x: 0.2, y: 0.2}, {x: 0.8, y: 0.2}, {x: 0.8, y: 0.8}, {x: 0.2, y: 0.8}]);
  }, [selectedImgIndex]);

  const runProcessing = () => {
    if (imgRef.current && activeImg) {
      const result = processImage(imgRef.current, corners, threshold, invert, showMask, minCellSize, circularity);
      onUpdateCount(activeImg.id, result.count, result.dataUrl);
    }
  };

  // --- ROI Interactions ---
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
          
          setCorners(prev => {
              const next = [...prev];
              next[clickStep] = {x, y};
              return next;
          });
          if (clickStep === 3) { setClickMode(false); setClickStep(0); } 
          else { setClickStep(s => s + 1); }
      }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (dragIndex !== null && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
      setCorners(prev => {
          const next = [...prev];
          next[dragIndex] = {x, y};
          return next;
      });
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
            {/* Left Column: Editor & Controls */}
            <div>
                <div className="flex mb-2 justify-between">
                     <div className="text-sm text-gray font-mono">
                        {clickMode 
                            ? `SET CORNER ${clickStep + 1}/4` 
                            : "ROI SELECTION"}
                     </div>
                     <button className="icon-btn text-xs" onClick={() => { setClickMode(true); setClickStep(0); }}>
                        {clickMode ? "CANCEL" : "MANUAL SET"}
                     </button>
                </div>

                <div 
                    className="canvas-container" 
                    ref={containerRef}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    onClick={handleContainerClick}
                    style={{ cursor: clickMode ? 'crosshair' : 'default', background: '#111' }}
                >
                    {activeImg && (
                        <>
                            <img ref={imgRef} src={activeImg.originalSrc} style={{ pointerEvents: 'none', userSelect: 'none' }} />
                            <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                                <polygon points={getPointsStr()} fill="rgba(14, 165, 233, 0.2)" stroke="var(--accent)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
                            </svg>
                            {corners.map((p, i) => (
                                <div 
                                    key={i}
                                    className="resize-handle"
                                    style={{
                                        left: `calc(${p.x * 100}%)`,
                                        top: `calc(${p.y * 100}%)`,
                                        backgroundColor: clickMode && i === clickStep ? '#fff' : 'var(--accent)',
                                    }}
                                    onMouseDown={(e) => handleMouseDown(e, i)}
                                />
                            ))}
                        </>
                    )}
                </div>

                <div className="grid grid-2 mt-4 gap-4">
                    <div>
                        <label>Threshold ({threshold})</label>
                        <input type="range" min="150" max="220" value={threshold} onChange={(e) => setThreshold(parseInt(e.target.value))} />
                        <p className="text-xs text-gray" style={{marginTop: 2}}>↑ Higher = fewer detections (stricter). ↓ Lower = more detections (may include noise).</p>
                    </div>
                     <div>
                        <label>Circularity ({circularity.toFixed(2)})</label>
                        <input type="range" min="0.1" max="1.0" step="0.05" value={circularity} onChange={(e) => setCircularity(parseFloat(e.target.value))} />
                    </div>
                     <div>
                        <label>Min Cell Size ({minCellSize}px)</label>
                        <input type="range" min="5" max="100" value={minCellSize} onChange={(e) => setMinCellSize(parseInt(e.target.value))} />
                    </div>
                    <div className="flex flex-col justify-end">
                        <label className="flex items-center text-sm cursor-pointer mb-2">
                            <input type="checkbox" checked={showMask} onChange={e => setShowMask(e.target.checked)} style={{marginRight: '8px'}}/>
                            Show Mask (B&W)
                        </label>
                        <label className="flex items-center text-sm cursor-pointer">
                            <input type="checkbox" checked={invert} onChange={e => setInvert(e.target.checked)} style={{marginRight: '8px'}}/>
                            Invert (Dark Cells)
                        </label>
                    </div>
                </div>
                
                <div className="mt-4">
                    <button className="primary w-full" onClick={runProcessing} style={{padding:'12px'}}>
                        PROCESS & COUNT
                    </button>
                </div>
            </div>

            {/* Right Column: Results & Thumbnails */}
            <div className="flex-col" style={{height: '100%'}}>
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
                    <label>Manual Correction</label>
                    <input 
                        type="number" 
                        value={activeImg?.count || 0} 
                        onChange={(e) => onUpdateCount(activeImg.id, parseInt(e.target.value) || 0, activeImg.processedSrc || activeImg.originalSrc)}
                    />
                </div>

                <div>
                    <h4 className="font-bold mb-2 text-xs text-gray">IMAGES IN FLASK</h4>
                    <div className="flex" style={{ overflowX: 'auto', paddingBottom: '5px' }}>
                        {flask.images.map((img, idx) => (
                            <div 
                                key={img.id} 
                                onClick={() => setSelectedImgIndex(idx)}
                                style={{ 
                                    minWidth: '60px', 
                                    width: '60px',
                                    cursor: 'pointer',
                                    border: idx === selectedImgIndex ? '2px solid var(--primary)' : '1px solid transparent',
                                    opacity: idx === selectedImgIndex ? 1 : 0.6
                                }}
                            >
                                <img 
                                    src={img.processedSrc || img.originalSrc} 
                                    style={{ width: '100%', height: '50px', objectFit: 'cover', display: 'block', borderRadius: '4px' }} 
                                />
                                <div className="text-center text-xs font-mono font-bold mt-1">
                                    {img.count}
                                </div>
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


// --- Main Application ---
const App = () => {
  const [flasks, setFlasks] = useState<FlaskData[]>([]);
  const [activeFlaskId, setActiveFlaskId] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    if (darkMode) document.body.classList.add('dark-mode');
    else document.body.classList.remove('dark-mode');
  }, [darkMode]);

  const activeFlask = flasks.find(f => f.id === activeFlaskId);
  
  const metrics = useMemo(() => {
    if (!activeFlask) return null;
    const counts = activeFlask.images.map(img => img.count);
    const n = counts.length;
    if (n === 0) return { avg: 0, sd: 0, sem: 0, conc: 0, total: 0 };

    const avg = counts.reduce((a, b) => a + b, 0) / n;
    const variance = counts.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / (n > 1 ? n - 1 : 1);
    const sd = Math.sqrt(variance);
    const sem = sd / Math.sqrt(n);
    
    // Concentration = Avg * DilutionFactor * 10^4
    const conc = avg * activeFlask.dilutionFactor * CONSTANTS.FACTOR_HEMOCYTOMETER;
    const total = conc * activeFlask.currentVol;

    return { avg, sd, sem, conc, total };
  }, [activeFlask]);

  const addFlask = () => {
    const newFlask: FlaskData = {
      id: generateId(),
      name: `Flask ${flasks.length + 1}`,
      passage: 1,
      currentVol: 10,
      dilutionFactor: 2,
      images: []
    };
    setFlasks([...flasks, newFlask]);
    setActiveFlaskId(newFlask.id);
  };

  const updateFlask = (id: string, data: Partial<FlaskData>) => {
    setFlasks(flasks.map(f => f.id === id ? { ...f, ...data } : f));
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (activeFlaskId && e.target.files) {
      const newImages = Array.from(e.target.files).map(file => ({
        id: generateId(),
        originalSrc: URL.createObjectURL(file as Blob),
        count: 0
      }));
      const flask = flasks.find(f => f.id === activeFlaskId);
      if (flask) {
        updateFlask(activeFlaskId, { images: [...flask.images, ...newImages] });
      }
    }
  };

  const updateImageCount = (imgId: string, count: number, processedSrc: string) => {
    if (!activeFlaskId) return;
    const flask = flasks.find(f => f.id === activeFlaskId);
    if (!flask) return;
    const updatedImages = flask.images.map(img => img.id === imgId ? { ...img, count, processedSrc } : img);
    updateFlask(activeFlaskId, { images: updatedImages });
  };

  // ── Inline Section Components (rendered inside unified analytics card) ──

  const PassagingSection = ({ flask, metrics: m }: { flask: FlaskData; metrics: { conc: number; total: number } }) => {
    const [numFlasks, setNumFlasks] = useState(7);
    const flaskTypes: { key: string; label: string; minReq: number; mediaVol: number; seedRange: string }[] = [
      { key: 'T12.5', label: 'T12.5', minReq: 0.25e6, mediaVol: 2.0, seedRange: '2–3' },
      { key: 'T25', label: 'T25', minReq: 0.7e6, mediaVol: 5.0, seedRange: '3–5' },
      { key: 'T75', label: 'T75', minReq: 2.1e6, mediaVol: 12.0, seedRange: '8–12' },
    ];
    // Per-flask suspension volumes
    const suspPerFlask = flaskTypes.map(ft => ({
      ...ft,
      suspUl: safeDiv(ft.minReq, m.conc) * 1e6, // µL
      seedDensity: ft.minReq / 1e6,
      confT25: (ft.minReq / 3e6 * 100),
      confT75: (ft.minReq / 9e6 * 100),
    }));
    const totalSuspUsed = suspPerFlask.reduce((s, f) => s + f.suspUl * (numFlasks > 0 ? 1 : 0), 0) * numFlasks / flaskTypes.length;
    const volLeft = flask.currentVol * 1000 - totalSuspUsed;

    return (
      <div style={{padding: '20px 24px', borderBottom: '1px solid var(--border)'}}>
        <h3 className="font-bold" style={{marginBottom: 8, fontSize: '0.95rem'}}>Passaging — Flasks</h3>
        <div style={{display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12}}>
          <label style={{margin: 0, whiteSpace: 'nowrap'}}>Number of new flasks per type</label>
          <input type="number" min={1} value={numFlasks} onChange={e => setNumFlasks(parseInt(e.target.value) || 1)} style={{width: 70}} />
        </div>
        <div style={{overflowX: 'auto'}}>
          <table style={{width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem'}}>
            <thead>
              <tr style={{borderBottom: '2px solid var(--border)'}}>
                <th style={{textAlign: 'left', padding: '6px 8px', color: 'var(--text-sub)'}}>Flask Type</th>
                {Array.from({length: numFlasks}, (_, i) => (
                  <th key={i} style={{textAlign: 'right', padding: '6px 8px', color: 'var(--text-sub)'}}>Flask {i + 1}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {/* Rows for each flask type */}
              {flaskTypes.map(ft => {
                const suspUl = safeDiv(ft.minReq, m.conc) * 1e6;
                return (
                  <React.Fragment key={ft.key}>
                    <tr style={{borderBottom: '1px solid var(--border)'}}>
                      <td style={{padding: '4px 8px', fontWeight: 600}}>{ft.label}</td>
                      {Array.from({length: numFlasks}, (_, i) => (
                        <td key={i} className="font-mono text-right" style={{padding: '4px 8px'}}>{ft.label}</td>
                      ))}
                    </tr>
                    <tr><td className="text-gray" style={{padding: '2px 8px', fontSize: '0.78rem'}}>Passage</td>
                      {Array.from({length: numFlasks}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '2px 8px'}}>{flask.passage + 1}</td>)}</tr>
                    <tr><td className="text-gray" style={{padding: '2px 8px', fontSize: '0.78rem'}}>Cell Suspension (µL)</td>
                      {Array.from({length: numFlasks}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '2px 8px'}}>{suspUl.toFixed(1)}</td>)}</tr>
                    <tr><td className="text-gray" style={{padding: '2px 8px', fontSize: '0.78rem'}}>Seeding Density (×10⁶)</td>
                      {Array.from({length: numFlasks}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '2px 8px'}}>{(ft.minReq / 1e6).toFixed(2)}</td>)}</tr>
                    <tr><td className="text-gray" style={{padding: '2px 8px', fontSize: '0.78rem'}}>Media (mL)</td>
                      {Array.from({length: numFlasks}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '2px 8px'}}>{ft.mediaVol}</td>)}</tr>
                    <tr style={{borderBottom: '1px solid var(--border)'}}><td className="text-gray" style={{padding: '2px 8px', fontSize: '0.78rem'}}>Seed Range (×10⁶)</td>
                      {Array.from({length: numFlasks}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '2px 8px'}}>{ft.seedRange}</td>)}</tr>
                  </React.Fragment>
                );
              })}
              <tr><td className="text-gray" style={{padding: '4px 8px'}}>% Confluency T25</td>
                {Array.from({length: numFlasks}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '4px 8px'}}>{(m.total / numFlasks / 3e6 * 100).toFixed(1)}%</td>)}</tr>
              <tr style={{borderBottom: '1px solid var(--border)'}}><td className="text-gray" style={{padding: '4px 8px'}}>% Confluency T75</td>
                {Array.from({length: numFlasks}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '4px 8px'}}>{(m.total / numFlasks / 9e6 * 100).toFixed(1)}%</td>)}</tr>
              <tr><td style={{padding: '4px 8px', fontWeight: 600}}>Vol Remaining (µL)</td>
                <td className="font-mono text-right" colSpan={numFlasks} style={{padding: '4px 8px', color: volLeft < 0 ? '#ef4444' : '#10b981', fontWeight: 700}}>
                  {volLeft.toFixed(1)}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const DilutionSection = ({ flask, metrics: m, type }: { flask: FlaskData; metrics: { conc: number }; type: '96' | '24' }) => {
    const is96 = type === '96';
    const title = is96 ? 'Dilutions for a 96-Well Plate' : 'Dilutions for a 24-Transwell Plate';
    const unitLabel = is96 ? 'well' : 'transwell';
    const [numCols, setNumCols] = useState(is96 ? 7 : 7);
    const [minPerWell, setMinPerWell] = useState(is96 ? 0.01 : 0.1);
    const [volPerWellUl, setVolPerWellUl] = useState(is96 ? 100 : 500);
    const [numWells, setNumWells] = useState(is96 ? 96 : 24);
    const [dilFactor, setDilFactor] = useState(1);

    const suspUlPerWell = safeDiv(minPerWell * 1e6, m.conc) * 1e6;
    const mediaUlPerWell = volPerWellUl - suspUlPerWell;
    const totalVolPerWell = volPerWellUl;
    const concPerWell = safeDiv(minPerWell * 1e6, volPerWellUl) * 1e3; // 10^6 cells/mL
    const totalSusp = suspUlPerWell * numWells;
    const status = totalSusp > flask.currentVol * 1000 ? '⚠ INSUFFICIENT' : '✓ OK';

    return (
      <div style={{padding: '20px 24px', borderBottom: '1px solid var(--border)'}}>
        <h3 className="font-bold" style={{marginBottom: 8, fontSize: '0.95rem'}}>{title}</h3>
        <div style={{display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12}}>
          <div><label>Min per {unitLabel} (×10⁶)</label><input type="number" step="0.001" value={minPerWell} onChange={e => setMinPerWell(parseFloat(e.target.value) || 0)} style={{width: 100}} /></div>
          <div><label>Vol per {unitLabel} (µL)</label><input type="number" value={volPerWellUl} onChange={e => setVolPerWellUl(parseInt(e.target.value) || 1)} style={{width: 90}} /></div>
          <div><label>Number of {unitLabel}s</label><input type="number" value={numWells} onChange={e => setNumWells(parseInt(e.target.value) || 1)} style={{width: 80}} /></div>
          <div><label>Dilution Factor</label><input type="number" step="0.1" value={dilFactor} onChange={e => setDilFactor(parseFloat(e.target.value) || 1)} style={{width: 70}} /></div>
        </div>
        <div style={{overflowX: 'auto'}}>
          <table style={{width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem'}}>
            <thead>
              <tr style={{borderBottom: '2px solid var(--border)'}}>
                <th style={{textAlign: 'left', padding: '6px 8px', color: 'var(--text-sub)'}}>Parameter</th>
                {Array.from({length: numCols}, (_, i) => (
                  <th key={i} style={{textAlign: 'right', padding: '6px 8px', color: 'var(--text-sub)'}}>{is96 ? `Col ${i+1}` : `Well ${i+1}`}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr><td className="text-gray" style={{padding: '4px 8px'}}>Cell Suspension (µL)</td>
                {Array.from({length: numCols}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '4px 8px'}}>{suspUlPerWell.toFixed(1)}</td>)}</tr>
              <tr><td className="text-gray" style={{padding: '4px 8px'}}>Media (µL)</td>
                {Array.from({length: numCols}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '4px 8px'}}>{mediaUlPerWell.toFixed(1)}</td>)}</tr>
              <tr><td className="text-gray" style={{padding: '4px 8px'}}>Total Volume (µL)</td>
                {Array.from({length: numCols}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '4px 8px'}}>{totalVolPerWell}</td>)}</tr>
              <tr><td className="text-gray" style={{padding: '4px 8px'}}>Conc (×10⁶ cells/mL)</td>
                {Array.from({length: numCols}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '4px 8px'}}>{(minPerWell).toFixed(4)}</td>)}</tr>
              <tr style={{borderTop: '1px solid var(--border)'}}>
                <td style={{padding: '4px 8px', fontWeight: 600}}>Status</td>
                <td colSpan={numCols} className="text-right font-bold" style={{padding: '4px 8px', color: status.startsWith('⚠') ? '#ef4444' : '#10b981'}}>{status}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const CryoSection = ({ flask, metrics: m }: { flask: FlaskData; metrics: { conc: number; total: number } }) => {
    const [numTubes, setNumTubes] = useState(7);
    const suggestedPerTube = 10; // ×10⁶ cells per spreadsheet
    const maxVolTube = 1.3; // mL, not counting DMSO

    const suspUlPerTube = safeDiv(suggestedPerTube * 1e6, m.conc) * 1e6;
    const maxSeedVol = safeDiv(maxVolTube * m.conc, suggestedPerTube * 1e6) * 1e6; // µL
    const seedDensity = suggestedPerTube;
    const dmsoPerTube = maxVolTube * 0.1 * 1000; // µL (10% of total)
    const mediaPerTube = maxVolTube * 1000 - suspUlPerTube - dmsoPerTube;
    const totalVolPerTube = suspUlPerTube + dmsoPerTube + Math.max(0, mediaPerTube);
    const concPerTube = safeDiv(suggestedPerTube, totalVolPerTube / 1000);
    const totalSusp = suspUlPerTube * numTubes;
    const status = totalSusp > flask.currentVol * 1000 ? '⚠ INSUFFICIENT' : '✓ OK';

    return (
      <div style={{padding: '20px 24px', borderBottom: '1px solid var(--border)'}}>
        <h3 className="font-bold" style={{marginBottom: 8, fontSize: '0.95rem'}}>Cryotubes</h3>
        <div style={{display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 8}}>
          <div className="metric" style={{padding: '8px 12px', flex: '0 0 auto'}}>
            <div className="metric-label" style={{marginTop: 0, fontSize: '0.7rem'}}>Suggested/tube (×10⁶)</div>
            <div className="metric-val" style={{fontSize: '1rem'}}>{suggestedPerTube}</div>
          </div>
          <div className="metric" style={{padding: '8px 12px', flex: '0 0 auto'}}>
            <div className="metric-label" style={{marginTop: 0, fontSize: '0.7rem'}}>Max Vol/tube (mL)</div>
            <div className="metric-val" style={{fontSize: '1rem'}}>{maxVolTube}</div>
          </div>
          <div className="metric" style={{padding: '8px 12px', flex: '0 0 auto'}}>
            <div className="metric-label" style={{marginTop: 0, fontSize: '0.7rem'}}>Max Seed Vol (µL)</div>
            <div className="metric-val" style={{fontSize: '1rem'}}>{maxSeedVol.toFixed(0)}</div>
          </div>
          <div style={{display: 'flex', alignItems: 'center', gap: 8}}>
            <label style={{margin: 0, whiteSpace: 'nowrap'}}>Tubes</label>
            <input type="number" min={1} value={numTubes} onChange={e => setNumTubes(parseInt(e.target.value) || 1)} style={{width: 70}} />
          </div>
        </div>
        <div style={{overflowX: 'auto'}}>
          <table style={{width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem'}}>
            <thead>
              <tr style={{borderBottom: '2px solid var(--border)'}}>
                <th style={{textAlign: 'left', padding: '6px 8px', color: 'var(--text-sub)'}}>Parameter</th>
                {Array.from({length: numTubes}, (_, i) => (
                  <th key={i} style={{textAlign: 'right', padding: '6px 8px', color: 'var(--text-sub)'}}>Tube {i + 1}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr><td className="text-gray" style={{padding: '4px 8px'}}>New Passage</td>
                {Array.from({length: numTubes}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '4px 8px'}}>{flask.passage + 1}</td>)}</tr>
              <tr><td className="text-gray" style={{padding: '4px 8px'}}>Cell Suspension (µL)</td>
                {Array.from({length: numTubes}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '4px 8px'}}>{suspUlPerTube.toFixed(1)}</td>)}</tr>
              <tr><td className="text-gray" style={{padding: '4px 8px'}}>Seeding Density (×10⁶)</td>
                {Array.from({length: numTubes}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '4px 8px'}}>{seedDensity}</td>)}</tr>
              <tr><td className="text-gray" style={{padding: '4px 8px'}}>DMSO (µL)</td>
                {Array.from({length: numTubes}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '4px 8px'}}>{dmsoPerTube.toFixed(0)}</td>)}</tr>
              <tr><td className="text-gray" style={{padding: '4px 8px'}}>Media (µL)</td>
                {Array.from({length: numTubes}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '4px 8px'}}>{Math.max(0, mediaPerTube).toFixed(0)}</td>)}</tr>
              <tr><td className="text-gray" style={{padding: '4px 8px'}}>Total Volume (µL)</td>
                {Array.from({length: numTubes}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '4px 8px'}}>{totalVolPerTube.toFixed(0)}</td>)}</tr>
              <tr><td className="text-gray" style={{padding: '4px 8px'}}>Conc (×10⁶/mL)</td>
                {Array.from({length: numTubes}, (_, i) => <td key={i} className="font-mono text-right" style={{padding: '4px 8px'}}>{concPerTube.toFixed(2)}</td>)}</tr>
              <tr style={{borderTop: '1px solid var(--border)'}}>
                <td style={{padding: '4px 8px', fontWeight: 600}}>Status</td>
                <td colSpan={numTubes} className="text-right font-bold" style={{padding: '4px 8px', color: status.startsWith('⚠') ? '#ef4444' : '#10b981'}}>{status}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  return (
    <div className="container">
      <div className="flex justify-between mb-6">
        <h1 className="text-2xl font-bold text-primary">🔬 Hemocytometer Assistant</h1>
        <button 
            className="secondary" 
            onClick={() => setDarkMode(!darkMode)}
            style={{ fontSize: '0.8rem', padding: '6px 12px' }}
        >
            {darkMode ? '☀️ Light' : '🌙 Dark'}
        </button>
      </div>
      
      <FlaskManager 
        flasks={flasks} 
        activeFlaskId={activeFlaskId} 
        onAddFlask={addFlask}
        onSelectFlask={setActiveFlaskId}
        onUpdateFlask={updateFlask}
      />

      {activeFlask && metrics && (
        <>
            {/* ═══ ANALYTICS DASHBOARD — matches spreadsheet ═══ */}
            <div className="card" style={{padding: 0, overflow: 'hidden'}}>
              {/* ── Section 1: Cell Count Summary ── */}
              <div style={{padding: '20px 24px', borderBottom: '1px solid var(--border)'}}>
                <h3 className="font-bold" style={{marginBottom: 12, fontSize: '0.95rem'}}>Cell Count Summary</h3>
                <div className="grid grid-4" style={{gap: 12}}>
                  {activeFlask.images.slice(0, 4).map((img, i) => (
                    <div key={img.id} className="metric" style={{padding: 10}}>
                      <div className="metric-label" style={{marginTop: 0, marginBottom: 4}}>Q{i + 1}</div>
                      <div className="metric-val" style={{fontSize: '1.2rem'}}>{img.count}</div>
                    </div>
                  ))}
                  {[...Array(Math.max(0, 4 - activeFlask.images.length))].map((_, i) => (
                    <div key={`empty-${i}`} className="metric" style={{padding: 10, opacity: 0.4}}>
                      <div className="metric-label" style={{marginTop: 0, marginBottom: 4}}>Q{activeFlask.images.length + i + 1}</div>
                      <div className="metric-val" style={{fontSize: '1.2rem'}}>—</div>
                    </div>
                  ))}
                </div>
                <div className="grid" style={{gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr', gap: 12, marginTop: 12}}>
                  <div className="metric" style={{padding: 10}}>
                    <div className="metric-label" style={{marginTop: 0, marginBottom: 4}}>Average</div>
                    <div className="metric-val" style={{fontSize: '1.1rem'}}>{metrics.avg.toFixed(1)}</div>
                  </div>
                  <div className="metric" style={{padding: 10}}>
                    <div className="metric-label" style={{marginTop: 0, marginBottom: 4}}>Conc (×10⁶/mL)</div>
                    <div className="metric-val" style={{fontSize: '1.1rem'}}>{(metrics.conc / 1e6).toFixed(4)}</div>
                  </div>
                  <div className="metric" style={{padding: 10}}>
                    <div className="metric-label" style={{marginTop: 0, marginBottom: 4}}>Vol Resuspension (µL)</div>
                    <div className="metric-val" style={{fontSize: '1.1rem'}}>{(activeFlask.currentVol * 1000).toFixed(0)}</div>
                  </div>
                  <div className="metric" style={{padding: 10}}>
                    <div className="metric-label" style={{marginTop: 0, marginBottom: 4}}>Density (×10⁶ cells)</div>
                    <div className="metric-val" style={{fontSize: '1.1rem'}}>{(metrics.total / 1e6).toFixed(4)}</div>
                  </div>
                  <div className="metric" style={{padding: 10}}>
                    <div className="metric-label" style={{marginTop: 0, marginBottom: 4}}>SEM</div>
                    <div className="metric-val" style={{fontSize: '1.1rem'}}>± {metrics.sem.toFixed(2)}</div>
                  </div>
                </div>
              </div>

              {/* ── Section 2: Passaging ── */}
              <PassagingSection flask={activeFlask} metrics={metrics} />

              {/* ── Section 3: 96-Well Plate ── */}
              <DilutionSection flask={activeFlask} metrics={metrics} type="96" />

              {/* ── Section 4: Cryotubes ── */}
              <CryoSection flask={activeFlask} metrics={metrics} />

              {/* ── Section 5: 24-Well Transwell ── */}
              <DilutionSection flask={activeFlask} metrics={metrics} type="24" />
            </div>

            {/* ═══ CELL COUNTER (images) ═══ */}
            <div className="flex justify-between items-center mb-4 card" style={{padding: '12px 24px'}}>
                <div>
                    <span className="font-bold text-sm">IMAGE SET: {activeFlask.name}</span>
                    <span className="block text-xs text-gray">{activeFlask.images.length} images loaded</span>
                </div>
                <label className="primary cursor-pointer" style={{ display: 'inline-block', padding: '8px 16px', borderRadius: '4px', color: 'white' }}>
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

const root = createRoot(document.getElementById('root')!);
root.render(<App />);

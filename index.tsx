
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
    // Green mask
    const maskImg = ctx.createImageData(size, size);
    for(let i=0; i<size*size; i++) {
       if (finalBinary[i]) {
         maskImg.data[i*4] = 0;   // R
         maskImg.data[i*4+1] = 255; // G
         maskImg.data[i*4+2] = 0;   // B
         maskImg.data[i*4+3] = 100; // A
       } else {
         maskImg.data[i*4+3] = 0;
       }
    }
    // Blend with original (grayscale version for contrast)
    ctx.putImageData(imageData, 0, 0);
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = size; tempCanvas.height = size;
    tempCanvas.getContext('2d')?.putImageData(maskImg, 0, 0);
    ctx.drawImage(tempCanvas, 0, 0);
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
  const [threshold, setThreshold] = useState(120); 
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
                        <input type="range" min="0" max="255" value={threshold} onChange={(e) => setThreshold(parseInt(e.target.value))} />
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
                            Show Mask
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
  const [activeTab, setActiveTab] = useState('counter');
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

  const PassagingTab = () => {
    const [targetType, setTargetType] = useState('T75');
    const [numFlasks, setNumFlasks] = useState(1);

    if (!activeFlask || !metrics) return <p className="text-center p-4">Please select a flask.</p>;

    const minReq = targetType === 'T25' ? CONSTANTS.T25_MIN_REQ : CONSTANTS.T75_MIN_REQ;
    const mediaVol = targetType === 'T25' ? CONSTANTS.T25_MEDIA_VOL : CONSTANTS.T75_MEDIA_VOL;
    
    const cellsNeeded = minReq * numFlasks;
    const volSuspNeeded = safeDiv(cellsNeeded, metrics.conc);
    const volRemaining = activeFlask.currentVol - volSuspNeeded;
    
    const maxCap = targetType === 'T25' ? 3e6 : 9e6; // Approximate max for confluency calc
    const confluency = (minReq / maxCap) * 100;

    return (
      <div className="card">
        <h3 className="font-bold mb-4 border-b pb-2">Passaging Calculator</h3>
        <div className="grid grid-2">
            <div>
                <label>Target Flask Type</label>
                <select value={targetType} onChange={e => setTargetType(e.target.value)}>
                    <option value="T25">T25</option>
                    <option value="T75">T75</option>
                </select>
                <label className="mt-4">Number of Flasks</label>
                <input type="number" min="1" value={numFlasks} onChange={e => setNumFlasks(parseInt(e.target.value))} />
            </div>
            <div className="metric items-start" style={{ textAlign: 'left' }}>
                <div className="flex justify-between w-full border-b pb-1 mb-2">
                    <span className="text-gray text-sm">Suspension Needed</span>
                    <span className="font-bold font-mono">{(volSuspNeeded * 1000).toFixed(1)} µL</span>
                </div>
                 <div className="flex justify-between w-full border-b pb-1 mb-2">
                    <span className="text-gray text-sm">Media per Flask</span>
                    <span className="font-bold font-mono">{mediaVol} mL</span>
                </div>
                 <div className="flex justify-between w-full border-b pb-1 mb-2">
                    <span className="text-gray text-sm">Est. Seeding Confluency</span>
                    <span className="font-bold font-mono">~{confluency.toFixed(1)}%</span>
                </div>
                <div className="mt-4 text-center w-full">
                    <p className={`font-bold ${volRemaining < 0 ? "text-danger" : "text-success"}`}>
                        {volRemaining < 0 ? `Short by ${Math.abs(volRemaining).toFixed(2)} mL` : `Remaining Vol: ${volRemaining.toFixed(2)} mL`}
                    </p>
                </div>
            </div>
        </div>
      </div>
    );
  };

  const DilutionTab = ({ type }: { type: '96' | '24' }) => {
    const [wantedConcM, setWantedConcM] = useState(type === '96' ? 0.1 : 0.5);
    const [wells, setWells] = useState(type === '96' ? 96 : 24);
    const [volPerWell, setVolPerWell] = useState(type === '96' ? 100 : 500);

    if (!metrics) return <p>Please select a flask.</p>;

    const wantedConc = wantedConcM * 10 ** 6;
    const deadVolBuffer = 1.05; // 5% overage
    const totalVolUl = (wells * volPerWell) * deadVolBuffer;
    
    // C1V1 = C2V2 => V1 = (C2*V2)/C1
    const suspUl = safeDiv(wantedConc * totalVolUl, metrics.conc);
    const mediaUl = totalVolUl - suspUl;

    return (
      <div className="card">
        <h3 className="font-bold mb-4 border-b pb-2">{type}-Well Plate Dilutions</h3>
        <div className="grid grid-2">
            <div>
                <label>Wanted Conc. (M cells/mL)</label>
                <input type="number" step="0.01" value={wantedConcM} onChange={e => setWantedConcM(parseFloat(e.target.value))} />
                
                <label className="mt-4">Number of Wells/Inserts</label>
                <input type="number" value={wells} onChange={e => setWells(parseInt(e.target.value))} />

                <label className="mt-4">Vol per Well (µL)</label>
                <input type="number" value={volPerWell} onChange={e => setVolPerWell(parseInt(e.target.value))} />
            </div>
            <div className="metric" style={{ textAlign: 'left' }}>
                <p className="font-bold mb-2 text-sm text-gray">RECIPE (Total + 5% Dead Vol)</p>
                <div className="flex justify-between border-b py-2">
                    <span>Cell Suspension</span>
                    <span className="font-bold font-mono">{suspUl.toFixed(1)} µL</span>
                </div>
                <div className="flex justify-between border-b py-2">
                    <span>Media</span>
                    <span className="font-bold font-mono">{mediaUl.toFixed(1)} µL</span>
                </div>
                <div className="flex justify-between py-2 text-primary">
                    <span>Total Volume</span>
                    <span className="font-bold font-mono">{(totalVolUl/1000).toFixed(2)} mL</span>
                </div>
                {suspUl > (activeFlask?.currentVol || 0) * 1000 && (
                    <p className="text-danger text-sm mt-4 font-bold text-center">⚠️ INSUFFICIENT SUSPENSION</p>
                )}
            </div>
        </div>
      </div>
    );
  };

  const CryoTab = () => {
    const [targetM, setTargetM] = useState(CONSTANTS.CRYO_SUGGESTED_PER_TUBE); 
    const [tubes, setTubes] = useState(1);

    if (!metrics) return <p>Please select a flask.</p>;

    const targetPerTube = targetM * 10 ** 6;
    const maxTubes = Math.floor(safeDiv(metrics.total, targetPerTube));
    
    const cellsReq = tubes * targetPerTube;
    const volToSpin = safeDiv(cellsReq, metrics.conc);
    
    const finalVol = tubes * 1.0; 
    const dmsoVol = finalVol * 0.1;
    const resuspVol = finalVol * 0.9;

    return (
      <div className="card">
        <h3 className="font-bold mb-4 border-b pb-2">Cryopreservation</h3>
        <div className="grid grid-2">
            <div>
                <label>Target Cells / Tube (Millions)</label>
                <input type="number" step="0.1" value={targetM} onChange={e => setTargetM(parseFloat(e.target.value))} />
                
                <p className="text-xs text-gray mt-1">Max possible tubes: {maxTubes}</p>

                <label className="mt-4">Tubes to Make</label>
                <input type="number" max={maxTubes} value={tubes} onChange={e => setTubes(parseInt(e.target.value))} />
            </div>
            <div className="metric" style={{ textAlign: 'left' }}>
                <p className="font-bold text-sm text-gray mb-2">PROTOCOL STEP-BY-STEP</p>
                <ol className="text-sm pl-4" style={{ lineHeight: '1.8' }}>
                    <li>Spin down <strong>{volToSpin.toFixed(2)} mL</strong> of suspension.</li>
                    <li>Aspirate supernatant.</li>
                    <li>Resuspend pellet in <strong>{resuspVol.toFixed(2)} mL</strong> Media/FBS.</li>
                    <li>Add <strong>{dmsoVol.toFixed(2)} mL</strong> (10%) DMSO.</li>
                    <li>Aliquot <strong>1 mL</strong> into {tubes} cryotubes.</li>
                </ol>
            </div>
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
            {darkMode ? "☀️ Light Mode" : "🌙 Dark Mode"}
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
            <div className="grid grid-4 mb-6">
                <div className="metric">
                    <div className="metric-val">{metrics.avg.toFixed(1)}</div>
                    <div className="metric-label">Avg Count</div>
                </div>
                <div className="metric">
                    <div className="metric-val">{(metrics.conc / 1e6).toFixed(2)} M</div>
                    <div className="metric-label">Conc (cells/mL)</div>
                </div>
                <div className="metric">
                    <div className="metric-val">{(metrics.total / 1e6).toFixed(2)} M</div>
                    <div className="metric-label">Total Cells</div>
                </div>
                <div className="metric">
                    <div className="metric-val text-gray">± {metrics.sem.toFixed(2)}</div>
                    <div className="metric-label">SEM</div>
                </div>
            </div>

            <div className="tabs">
                {['counter', 'passaging', '96well', '24well', 'cryo'].map(t => (
                    <button 
                        key={t}
                        className={`tab-btn ${activeTab === t ? 'active' : ''}`}
                        onClick={() => setActiveTab(t)}
                    >
                        {t === 'counter' ? '📷 Counter' : 
                         t === 'passaging' ? '🧪 Passaging' :
                         t === '96well' ? '🧫 96-Well' :
                         t === '24well' ? '⚪ 24-Well' : '❄️ Cryo'}
                    </button>
                ))}
            </div>

            {activeTab === 'counter' && (
                <>
                    <div className="flex justify-between items-center mb-4 card p-4 items-center">
                        <div>
                            <span className="font-bold text-sm">IMAGE SET FOR: {activeFlask.name}</span>
                            <span className="block text-xs text-gray">{activeFlask.images.length} images loaded</span>
                        </div>
                        <label className="primary cursor-pointer" style={{ display: 'inline-block', padding: '8px 16px', borderRadius: '4px', color: 'white' }}>
                            UPLOAD MICROSCOPE IMAGES
                            <input type="file" multiple accept="image/*" onChange={handleImageUpload} className="hidden" />
                        </label>
                    </div>
                    <ImageProcessor flask={activeFlask} onUpdateCount={updateImageCount} />
                </>
            )}
            {activeTab === 'passaging' && <PassagingTab />}
            {activeTab === '96well' && <DilutionTab type="96" />}
            {activeTab === '24well' && <DilutionTab type="24" />}
            {activeTab === 'cryo' && <CryoTab />}
        </>
      )}
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);

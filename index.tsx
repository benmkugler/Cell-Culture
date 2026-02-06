
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { createRoot } from 'react-dom/client';

// --- Constants ---
const CONSTANTS = {
  FACTOR_HEMOCYTOMETER: 10000,
  T25_MIN_REQ: 0.7 * 10 ** 6,
  T75_MIN_REQ: 2.1 * 10 ** 6,
  T25_MEDIA_VOL: 5.0,
  T75_MEDIA_VOL: 12.0,
  CRYO_MAX_VOL_ML: 1.0, 
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

const autoDetectCorners = (img: HTMLImageElement): Point[] => {
    const w = 512;
    const h = 512;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    
    // Default fallback
    const defaults = [
        {x: 0.2, y: 0.2}, {x: 0.8, y: 0.2},
        {x: 0.8, y: 0.8}, {x: 0.2, y: 0.8}
    ];

    if (!ctx) return defaults;
    
    try {
        ctx.drawImage(img, 0, 0, w, h);
        const imageData = ctx.getImageData(0, 0, w, h);
        const data = imageData.data;
        const gray = new Uint8Array(w * h);

        // Convert to grayscale
        for (let i = 0; i < w * h; i++) {
            gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
        }

        // Compute edge projections
        const rowScores = new Float32Array(h);
        const colScores = new Float32Array(w);

        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                const idx = y * w + x;
                const score = Math.abs(gray[idx - 1] - gray[idx + 1]) + Math.abs(gray[idx - w] - gray[idx + w]);
                if (score > 30) {
                    rowScores[y]++;
                    colScores[x]++;
                }
            }
        }

        const findBounds = (scores: Float32Array, len: number) => {
            let max = 0;
            for (let i = 0; i < len; i++) if (scores[i] > max) max = scores[i];
            
            if (max < 10) return { start: 0.2, end: 0.8 }; // No strong edges

            const thresh = max * 0.15;
            let start = 0, end = len - 1;

            // Find start
            while (start < len && scores[start] < thresh) start++;
            // Find end
            while (end > 0 && scores[end] < thresh) end--;

            if (end <= start + 20) return { start: 0.2, end: 0.8 }; // Too small

            return { start: start / len, end: end / len };
        };

        const xB = findBounds(colScores, w);
        const yB = findBounds(rowScores, h);

        return [
            { x: xB.start, y: yB.start },
            { x: xB.end, y: yB.start },
            { x: xB.end, y: yB.end },
            { x: xB.start, y: yB.end }
        ];

    } catch (e) {
        console.error("Auto detection failed", e);
        return defaults;
    }
};

// --- Computer Vision Logic ---
const processImage = (
  imgElement: HTMLImageElement,
  corners: Point[], 
  threshold: number,
  invert: boolean,
  showMask: boolean,
  minCellSize: number,
  erosionSteps: number
): { count: number; dataUrl: string } => {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return { count: 0, dataUrl: '' };

  // Use a fixed high resolution for processing to ensure small cells aren't lost
  const processSize = 800; 
  canvas.width = processSize;
  canvas.height = processSize;
  
  warpPerspective(ctx, imgElement, corners, processSize, processSize);

  const imageData = ctx.getImageData(0, 0, processSize, processSize);
  const data = imageData.data;

  // 1. Thresholding
  let binary = new Uint8Array(processSize * processSize);
  const margin = Math.floor(processSize * 0.01); // Minimal margin

  for (let i = 0; i < processSize * processSize; i++) {
      const y = Math.floor(i / processSize);
      const x = i % processSize;
      
      if (x < margin || x > processSize - margin || y < margin || y > processSize - margin) {
          binary[i] = 0;
          continue;
      }

      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      // Luminance
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;

      const isHit = invert ? (lum < threshold) : (lum > threshold);
      binary[i] = isHit ? 1 : 0;
  }

  // 2. Erosion (Variable Steps)
  // Erosion shrinks white areas. This separates touching cells.
  // We do NOT dilate back, because we want to count the "cores" (seeds).
  for (let step = 0; step < erosionSteps; step++) {
    const eroded = new Uint8Array(processSize * processSize);
    for (let y = 1; y < processSize - 1; y++) {
        for (let x = 1; x < processSize - 1; x++) {
            const idx = y * processSize + x;
            if (binary[idx] === 1) {
                // Keep only if all 4 neighbors are 1
                if (binary[idx - 1] && binary[idx + 1] && binary[idx - processSize] && binary[idx + processSize]) {
                    eroded[idx] = 1;
                } else {
                    eroded[idx] = 0;
                }
            }
        }
    }
    binary = eroded;
  }

  // 3. Blob Detection
  const blobs: { area: number, cx: number, cy: number, aspectRatio: number, minX: number, maxX: number, minY: number, maxY: number }[] = [];
  const visited = new Uint8Array(processSize * processSize); 
  const maxBlobSize = processSize * processSize * 0.10; // 10% of screen is huge.

  for (let y = 0; y < processSize; y++) {
    for (let x = 0; x < processSize; x++) {
      const idx = y * processSize + x;
      if (binary[idx] === 1 && visited[idx] === 0) {
        // Flood Fill
        let stack = [idx];
        visited[idx] = 1;
        
        let pixelCount = 0;
        let sumX = 0;
        let sumY = 0;
        let minX = x, maxX = x, minY = y, maxY = y;

        while (stack.length > 0) {
          const curr = stack.pop()!;
          const cx = curr % processSize;
          const cy = Math.floor(curr / processSize);
          pixelCount++;
          sumX += cx;
          sumY += cy;
          
          if (cx < minX) minX = cx;
          if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy;
          if (cy > maxY) maxY = cy;

          const neighbors = [curr - 1, curr + 1, curr - processSize, curr + processSize];
          for (let n of neighbors) {
            if (n >= 0 && n < binary.length && binary[n] === 1 && visited[n] === 0) {
               const nx = n % processSize;
               if (Math.abs(nx - cx) > 1) continue;
               visited[n] = 1;
               stack.push(n);
            }
          }
        }

        if (pixelCount > 0) {
            const width = maxX - minX + 1;
            const height = maxY - minY + 1;
            const aspectRatio = width > height ? width / height : height / width;
            
            // Filter noise immediately
            if (pixelCount >= minCellSize && pixelCount < maxBlobSize) {
                blobs.push({
                    area: pixelCount,
                    cx: sumX / pixelCount,
                    cy: sumY / pixelCount,
                    aspectRatio,
                    minX, maxX, minY, maxY
                });
            }
        }
      }
    }
  }

  // 4. Cluster Logic
  // First, find the "Typical Cell Size" from the most circular single cells
  const singles = blobs.filter(b => b.aspectRatio < 1.5); // Strict circles
  singles.sort((a, b) => a.area - b.area);
  
  let typicalArea = 0;
  if (singles.length > 0) {
      // Median
      typicalArea = singles[Math.floor(singles.length / 2)].area;
  } else if (blobs.length > 0) {
      // Fallback: Average of all valid blobs
      typicalArea = blobs.reduce((acc, b) => acc + b.area, 0) / blobs.length;
  }
  // Safety floor
  typicalArea = Math.max(typicalArea, minCellSize);

  let totalCount = 0;

  // Visuals
  if (!showMask) {
    ctx.putImageData(imageData, 0, 0); // Restore Original
    ctx.lineWidth = 2;
    ctx.font = "bold 16px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
  } else {
     // Draw Mask
     const maskImg = ctx.createImageData(processSize, processSize);
     for(let i=0; i<binary.length; i++) {
        const val = binary[i] * 255;
        maskImg.data[i*4] = val; 
        maskImg.data[i*4+1] = val; 
        maskImg.data[i*4+2] = val; 
        maskImg.data[i*4+3] = 255;
     }
     ctx.putImageData(maskImg, 0, 0);
  }

  blobs.forEach(b => {
      // 5. Line Filter
      // Grid lines are long (High Aspect Ratio) OR span the image width/height
      const isLine = b.aspectRatio > 4.0 || (b.maxX - b.minX > processSize * 0.8) || (b.maxY - b.minY > processSize * 0.8);
      
      if (isLine) return; 

      // 6. Count Estimation
      // If it's a "single", count is 1. If it's a cluster, estimate area.
      let n = 1;
      
      // Only split if it's significantly larger than typical (e.g. > 1.6x)
      if (b.area > typicalArea * 1.6) {
          n = Math.round(b.area / typicalArea);
      }
      
      totalCount += n;

      if (!showMask) {
          const radius = (b.maxX - b.minX) / 2;
          
          if (n === 1) {
              // Single Cell
              ctx.strokeStyle = '#22c55e'; // Green
              ctx.beginPath();
              ctx.arc(b.cx, b.cy, radius + 2, 0, 2 * Math.PI);
              ctx.stroke();
          } else {
              // Cluster
              ctx.strokeStyle = '#3b82f6'; // Blue
              ctx.beginPath();
              ctx.rect(b.minX, b.minY, b.maxX - b.minX, b.maxY - b.minY);
              ctx.stroke();
              
              // Draw text number
              ctx.fillStyle = '#fff';
              ctx.shadowColor="black";
              ctx.shadowBlur=4;
              ctx.fillText(n.toString(), b.cx, b.cy);
              ctx.shadowBlur=0;
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
        <h2 className="font-bold text-xl">Flask Configuration</h2>
        <button onClick={onAddFlask} className="primary">+ New Flask</button>
      </div>

      {flasks.length === 0 ? (
        <p className="text-gray text-center py-4">No flasks yet. Create one to start.</p>
      ) : (
        <div className="flex" style={{ overflowX: 'auto', paddingBottom: '10px' }}>
          {flasks.map(flask => (
            <button
              key={flask.id}
              onClick={() => onSelectFlask(flask.id)}
              className={`secondary ${flask.id === activeFlaskId ? 'active' : ''}`}
              style={{ 
                border: flask.id === activeFlaskId ? '2px solid var(--primary)' : '1px solid var(--border)',
                minWidth: '120px'
              }}
            >
              {flask.name}
            </button>
          ))}
        </div>
      )}

      {activeFlask && (
        <div className="grid grid-4 mt-4">
          <div>
            <label className="text-sm text-gray">Flask Name</label>
            <input 
              type="text" 
              value={activeFlask.name} 
              onChange={(e) => onUpdateFlask(activeFlask.id, { name: e.target.value })} 
            />
          </div>
          <div>
            <label className="text-sm text-gray">Passage #</label>
            <input 
              type="number" 
              value={activeFlask.passage} 
              onChange={(e) => onUpdateFlask(activeFlask.id, { passage: parseInt(e.target.value) || 0 })} 
            />
          </div>
          <div>
            <label className="text-sm text-gray">Current Vol (mL)</label>
            <input 
              type="number" 
              value={activeFlask.currentVol} 
              step="0.1"
              onChange={(e) => onUpdateFlask(activeFlask.id, { currentVol: parseFloat(e.target.value) || 0 })} 
            />
          </div>
          <div>
            <label className="text-sm text-gray">Dilution Factor</label>
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
      {x: 0.2, y: 0.2},
      {x: 0.8, y: 0.2},
      {x: 0.8, y: 0.8},
      {x: 0.2, y: 0.8}
  ]);
  const [clickMode, setClickMode] = useState(false);
  const [clickStep, setClickStep] = useState(0);

  const [threshold, setThreshold] = useState(120); 
  const [invert, setInvert] = useState(false);
  const [showMask, setShowMask] = useState(false);
  const [minCellSize, setMinCellSize] = useState(15); 
  const [erosionSteps, setErosionSteps] = useState(2); // Start with medium separation

  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [dragIndex, setDragIndex] = useState<number | null>(null);

  const activeImg = flask.images[selectedImgIndex];

  useEffect(() => {
    setCorners([
      {x: 0.2, y: 0.2},
      {x: 0.8, y: 0.2},
      {x: 0.8, y: 0.8},
      {x: 0.2, y: 0.8}
    ]);
  }, [selectedImgIndex]);

  const handleImageLoad = () => {
    if (imgRef.current) {
        setCorners(autoDetectCorners(imgRef.current));
    }
  };

  const runProcessing = () => {
    if (imgRef.current && activeImg) {
      const result = processImage(imgRef.current, corners, threshold, invert, showMask, minCellSize, erosionSteps);
      onUpdateCount(activeImg.id, result.count, result.dataUrl);
    }
  };

  const handleMouseDown = (e: React.MouseEvent, index: number) => {
    if (clickMode) return;
    e.preventDefault();
    e.stopPropagation();
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

          if (clickStep === 3) {
              setClickMode(false);
              setClickStep(0);
          } else {
              setClickStep(s => s + 1);
          }
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

  const handleMouseUp = () => {
    setDragIndex(null);
  };

  const getPointsStr = () => {
      return corners.map(p => `${p.x * 100},${p.y * 100}`).join(' ');
  };

  return (
    <div className="card">
      <div className="flex justify-between mb-4">
        <h3 className="font-bold text-lg">Cell Counter</h3>
      </div>

      {flask.images.length === 0 ? (
          <div className="text-center py-10 border-2 border-dashed border-gray-300 rounded">
            <p className="mb-4">Upload 4 microscope images</p>
             <p className="text-sm text-gray">Use the "Upload Images" button in the tab bar</p>
          </div>
      ) : (
        <div className="grid grid-2" style={{ gap: '20px' }}>
            {/* Left Column: Editor */}
            <div>
                <div className="flex mb-2 justify-between">
                     <div className="text-sm text-gray">
                        {clickMode 
                            ? `Click corner ${clickStep + 1}/4 (TL -> TR -> BR -> BL)` 
                            : "1. Match Red Grid to Hemocytometer"}
                     </div>
                     <button 
                        className="secondary" 
                        style={{ fontSize: '0.8rem', padding: '4px 8px' }}
                        onClick={() => {
                            setClickMode(true);
                            setClickStep(0);
                        }}
                     >
                        {clickMode ? "Cancel" : "Set Manually"}
                     </button>
                </div>

                <div 
                    className="canvas-container" 
                    ref={containerRef}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    onClick={handleContainerClick}
                    style={{ 
                        cursor: clickMode ? 'crosshair' : 'default',
                        height: 'auto', 
                        background: 'transparent'
                    }}
                >
                    {activeImg && (
                        <>
                            <img 
                                ref={imgRef}
                                onLoad={handleImageLoad}
                                src={activeImg.originalSrc} 
                                style={{ 
                                    width: '100%', 
                                    height: 'auto', 
                                    display: 'block',
                                    pointerEvents: 'none', 
                                    userSelect: 'none' 
                                }} 
                            />
                            
                            <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                                <polygon points={getPointsStr()} fill="rgba(255,0,0,0.1)" stroke="red" strokeWidth="2" vectorEffect="non-scaling-stroke" />
                            </svg>

                            {corners.map((p, i) => (
                                <div 
                                    key={i}
                                    className="resize-handle"
                                    style={{
                                        left: `calc(${p.x * 100}% - 5px)`,
                                        top: `calc(${p.y * 100}% - 5px)`,
                                        cursor: 'move',
                                        backgroundColor: clickMode && i === clickStep ? '#2563eb' : '#ef4444',
                                        zIndex: 10,
                                        width: '12px', height: '12px', borderRadius: '50%'
                                    }}
                                    onMouseDown={(e) => handleMouseDown(e, i)}
                                />
                            ))}
                        </>
                    )}
                </div>

                <div className="grid grid-2 mt-4 gap-4">
                    <div>
                        <label className="text-sm font-bold block mb-1">Threshold ({threshold})</label>
                        <input 
                            type="range" 
                            min="0" max="255" 
                            value={threshold} 
                            onChange={(e) => setThreshold(parseInt(e.target.value))} 
                        />
                    </div>
                     <div>
                        <label className="text-sm font-bold block mb-1">Separation Force ({erosionSteps})</label>
                        <input 
                            type="range" 
                            min="0" max="10" 
                            value={erosionSteps} 
                            onChange={(e) => setErosionSteps(parseInt(e.target.value))} 
                        />
                    </div>
                     <div>
                        <label className="text-sm font-bold block mb-1">Min Cell Size ({minCellSize}px)</label>
                        <input 
                            type="range" 
                            min="1" max="100" 
                            value={minCellSize} 
                            onChange={(e) => setMinCellSize(parseInt(e.target.value))} 
                        />
                    </div>
                    <div className="flex flex-col justify-end">
                        <label className="flex items-center text-sm cursor-pointer mb-2">
                            <input type="checkbox" checked={showMask} onChange={e => setShowMask(e.target.checked)} className="mr-2"/>
                            Show Mask
                        </label>
                        <label className="flex items-center text-sm cursor-pointer">
                            <input type="checkbox" checked={invert} onChange={e => setInvert(e.target.checked)} className="mr-2"/>
                            Dark Cells
                        </label>
                    </div>
                </div>
                
                <div className="flex mt-4">
                    <button className="primary w-full" onClick={runProcessing}>2. Process / Count</button>
                </div>
            </div>

            {/* Right Column: Results & Thumbnails */}
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                <h4 className="font-bold mb-2">Processed Result</h4>
                <div 
                    className="canvas-container" 
                    style={{ 
                        background: '#000', 
                        flexGrow: 1, 
                        minHeight: '300px', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        marginBottom: '10px'
                    }}
                >
                    {activeImg.processedSrc ? (
                        <img 
                            src={activeImg.processedSrc} 
                            style={{ 
                                width: '100%', 
                                height: 'auto', 
                                maxHeight: '400px', 
                                objectFit: 'contain', 
                                imageRendering: 'pixelated' 
                            }} 
                        />
                    ) : (
                        <div className="text-gray text-center">
                            <p>No result yet.</p>
                            <p className="text-sm">Click "Process / Count"</p>
                        </div>
                    )}
                </div>

                {/* Manual Override & Count Display */}
                <div className="p-3 bg-white border rounded mb-4">
                    <div className="flex justify-between items-center mb-2">
                        <label className="text-sm text-gray font-bold">Detected Count:</label>
                        <span className="text-2xl font-bold text-primary">{activeImg.count}</span>
                    </div>
                    <label className="text-xs text-gray block mb-1">Manual Override</label>
                    <input 
                        type="number" 
                        value={activeImg?.count || 0} 
                        onChange={(e) => onUpdateCount(activeImg.id, parseInt(e.target.value) || 0, activeImg.processedSrc || activeImg.originalSrc)}
                    />
                </div>

                {/* Thumbnails Strip */}
                <div>
                    <h4 className="font-bold mb-2 text-sm">All Images</h4>
                    <div className="flex" style={{ overflowX: 'auto', paddingBottom: '5px' }}>
                        {flask.images.map((img, idx) => (
                            <div 
                                key={img.id} 
                                onClick={() => setSelectedImgIndex(idx)}
                                style={{ 
                                    minWidth: '70px', 
                                    width: '70px',
                                    cursor: 'pointer',
                                    border: idx === selectedImgIndex ? '2px solid var(--primary)' : '1px solid #eee',
                                    borderRadius: '4px',
                                    opacity: idx === selectedImgIndex ? 1 : 0.7
                                }}
                            >
                                <img 
                                    src={img.processedSrc || img.originalSrc} 
                                    style={{ width: '100%', height: '50px', objectFit: 'cover', display: 'block' }} 
                                />
                                <div className="text-center text-xs bg-gray-100 py-1 font-bold">
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

  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== 'undefined') {
        return localStorage.getItem('theme') === 'dark' || 
               (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    }
    return false;
  });

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
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

    if (!activeFlask || !metrics) return <p>Please select a flask.</p>;

    const minReq = targetType === 'T25' ? CONSTANTS.T25_MIN_REQ : CONSTANTS.T75_MIN_REQ;
    const mediaVol = targetType === 'T25' ? CONSTANTS.T25_MEDIA_VOL : CONSTANTS.T75_MEDIA_VOL;
    
    const cellsNeeded = minReq * numFlasks;
    const volSuspNeeded = safeDiv(cellsNeeded, metrics.conc);
    const volRemaining = activeFlask.currentVol - volSuspNeeded;
    
    const maxCap = targetType === 'T25' ? 3e6 : 9e6;
    const confluency = (minReq / maxCap) * 100;

    return (
      <div className="card">
        <h3 className="font-bold mb-4">Passaging Calculator</h3>
        <div className="grid grid-2">
            <div>
                <label className="text-sm">Target Flask Type</label>
                <select value={targetType} onChange={e => setTargetType(e.target.value)}>
                    <option value="T25">T25</option>
                    <option value="T75">T75</option>
                </select>
                <label className="text-sm mt-2 block">Number of Flasks</label>
                <input type="number" min="1" value={numFlasks} onChange={e => setNumFlasks(parseInt(e.target.value))} />
            </div>
            <div className="metric" style={{ textAlign: 'left' }}>
                <p><strong>Cell Suspension Needed:</strong> {(volSuspNeeded * 1000).toFixed(1)} µL</p>
                <p><strong>Media per Flask:</strong> {mediaVol} mL</p>
                <p><strong>Seeding Confluency:</strong> ~{confluency.toFixed(1)}%</p>
                <hr className="my-2"/>
                <p className={volRemaining < 0 ? "text-red-500 font-bold" : "text-green-600"}>
                    {volRemaining < 0 ? `Short by ${Math.abs(volRemaining).toFixed(2)} mL` : `Remaining Vol: ${volRemaining.toFixed(2)} mL`}
                </p>
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
    const deadVolBuffer = type === '96' ? 1.05 : 1.10;
    const totalVolUl = (wells * volPerWell) * deadVolBuffer;
    
    const suspUl = safeDiv(wantedConc * totalVolUl, metrics.conc);
    const mediaUl = totalVolUl - suspUl;

    return (
      <div className="card">
        <h3 className="font-bold mb-4">{type}-Well Plate Dilutions</h3>
        <div className="grid grid-2">
            <div>
                <label className="text-sm">Wanted Conc. (M cells/mL)</label>
                <input type="number" step="0.01" value={wantedConcM} onChange={e => setWantedConcM(parseFloat(e.target.value))} />
                
                <label className="text-sm mt-2 block">Number of Wells/Inserts</label>
                <input type="number" value={wells} onChange={e => setWells(parseInt(e.target.value))} />

                <label className="text-sm mt-2 block">Vol per Well (µL)</label>
                <input type="number" value={volPerWell} onChange={e => setVolPerWell(parseInt(e.target.value))} />
            </div>
            <div className="metric" style={{ textAlign: 'left' }}>
                <p className="font-bold mb-2">Recipe (inc. {(deadVolBuffer - 1)*100}% dead vol)</p>
                <div className="flex justify-between border-b py-1">
                    <span>Cell Suspension:</span>
                    <span className="font-bold">{suspUl.toFixed(1)} µL</span>
                </div>
                <div className="flex justify-between border-b py-1">
                    <span>Media:</span>
                    <span className="font-bold">{mediaUl.toFixed(1)} µL</span>
                </div>
                <div className="flex justify-between py-1 text-primary">
                    <span>Total Volume:</span>
                    <span className="font-bold">{(totalVolUl/1000).toFixed(2)} mL</span>
                </div>
                {suspUl > (activeFlask?.currentVol || 0) * 1000 && (
                    <p className="text-red-500 text-sm mt-2 font-bold">⚠️ Insufficient Suspension!</p>
                )}
            </div>
        </div>
      </div>
    );
  };

  const CryoTab = () => {
    const [targetM, setTargetM] = useState(1.0); 
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
        <h3 className="font-bold mb-4">Cryopreservation</h3>
        <div className="grid grid-2">
            <div>
                <label className="text-sm">Target Cells / Tube (Millions)</label>
                <input type="number" step="0.1" value={targetM} onChange={e => setTargetM(parseFloat(e.target.value))} />
                
                <p className="text-sm text-gray mt-1">Max possible tubes: {maxTubes}</p>

                <label className="text-sm mt-2 block">Tubes to Make</label>
                <input type="number" max={maxTubes} value={tubes} onChange={e => setTubes(parseInt(e.target.value))} />
            </div>
            <div className="metric" style={{ textAlign: 'left' }}>
                <p className="font-bold">Protocol Step-by-Step:</p>
                <ol className="text-sm pl-4" style={{ lineHeight: '1.6' }}>
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
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-primary">🔬 Hemocytometer Assistant</h1>
         <button onClick={() => setDarkMode(!darkMode)} className="secondary" style={{ padding: '4px 8px', fontSize: '1.2rem' }}>
            {darkMode ? '☀️' : '🌙'}
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
                        {t === 'counter' ? '📷 Counter & Grid' : 
                         t === 'passaging' ? '🧪 Passaging' :
                         t === '96well' ? '🧫 96-Well' :
                         t === '24well' ? '⚪ 24-Well' : '❄️ Cryo'}
                    </button>
                ))}
            </div>

            {activeTab === 'counter' && (
                <>
                    <div className="flex justify-between items-center mb-4">
                        <label className="primary cursor-pointer" style={{ display: 'inline-block', padding: '8px 16px', borderRadius: '6px', color: 'white' }}>
                            Upload Images
                            <input type="file" multiple accept="image/*" onChange={handleImageUpload} className="hidden" />
                        </label>
                        <span className="text-sm text-gray">{activeFlask.images.length} images loaded</span>
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

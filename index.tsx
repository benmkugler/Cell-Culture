import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import type { Point, FlaskData, PipelineParams, PipelineStats } from './src/cv/types';
import { DEFAULT_PARAMS } from './src/cv/types';
import { processImage } from './src/cv/pipeline';

// --- Helpers ---

const generateId = (): string =>
  typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2, 11);

const saveImage = (dataUrl: string, filename: string, count?: number): void => {
  if (count == null) {
    const a = document.createElement('a');
    a.href = dataUrl;
    a.download = filename;
    a.click();
    return;
  }
  const img = new Image();
  img.onload = () => {
    const barHeight = 44;
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height + barHeight;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0);
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, img.height, canvas.width, barHeight);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 20px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`Total Cells: ${count}`, canvas.width / 2, img.height + barHeight / 2);
    const a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = filename;
    a.click();
  };
  img.src = dataUrl;
};

// --- Components ---

const FlaskManager: React.FC<{
  flasks: FlaskData[];
  activeFlaskId: string | null;
  onAddFlask: () => void;
  onSelectFlask: (id: string) => void;
  onUpdateFlask: (id: string, data: Partial<FlaskData>) => void;
  onDeleteFlask: (id: string) => void;
}> = ({ flasks, activeFlaskId, onAddFlask, onSelectFlask, onUpdateFlask, onDeleteFlask }) => {
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
            <button
              key={flask.id}
              onClick={() => onSelectFlask(flask.id)}
              className={`secondary ${flask.id === activeFlaskId ? 'active' : ''}`}
              style={{ minWidth: '140px' }}
            >
              {flask.name}
            </button>
          ))}
        </div>
      )}
      {activeFlask && (
        <div className="mt-4 bg-panel p-4 rounded border-dashed">
          <div className="flex justify-between">
            <div style={{ flex: 1 }}>
              <label>Flask Name</label>
              <input
                type="text"
                value={activeFlask.name}
                onChange={e => onUpdateFlask(activeFlask.id, { name: e.target.value })}
              />
            </div>
            <button
              className="secondary"
              style={{ alignSelf: 'flex-end', color: '#ef4444', borderColor: '#ef4444' }}
              onClick={() => onDeleteFlask(activeFlask.id)}
            >
              Delete Flask
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

const DEFAULT_CORNERS: Point[] = [
  { x: 0.2, y: 0.2 }, { x: 0.8, y: 0.2 },
  { x: 0.8, y: 0.8 }, { x: 0.2, y: 0.8 },
];

const ImageProcessor: React.FC<{
  flask: FlaskData;
  onUpdateCount: (imgId: string, count: number, processedSrc: string) => void;
}> = ({ flask, onUpdateCount }) => {
  const [selectedImgIndex, setSelectedImgIndex] = useState(0);
  const [corners, setCorners] = useState<Point[]>(DEFAULT_CORNERS);
  const [clickMode, setClickMode] = useState(false);
  const [clickStep, setClickStep] = useState(0);
  const [params, setParams] = useState<PipelineParams>({ ...DEFAULT_PARAMS });
  const [lastOtsu, setLastOtsu] = useState<number | null>(null);
  const [lastStats, setLastStats] = useState<PipelineStats | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [processing, setProcessing] = useState(false);

  const activeImg = flask.images[selectedImgIndex];

  useEffect(() => {
    setCorners([...DEFAULT_CORNERS]);
    setLastOtsu(null);
  }, [selectedImgIndex]);

  useEffect(() => {
    if (selectedImgIndex >= flask.images.length) {
      setSelectedImgIndex(Math.max(0, flask.images.length - 1));
    }
  }, [flask.images.length, selectedImgIndex]);

  const updateParam = useCallback(<K extends keyof PipelineParams>(key: K, value: PipelineParams[K]) => {
    setParams(prev => ({ ...prev, [key]: value }));
  }, []);

  const runProcessing = useCallback(() => {
    if (!imgRef.current || !activeImg) return;
    setProcessing(true);
    requestAnimationFrame(() => {
      try {
        const result = processImage(imgRef.current!, corners, params);
        onUpdateCount(activeImg.id, result.count, result.dataUrl);
        setLastOtsu(result.otsuThreshold);
        setLastStats(result.stats);
      } catch (err) {
        console.error('Processing failed:', err);
      } finally {
        setProcessing(false);
      }
    });
  }, [activeImg, corners, params, onUpdateCount]);

  const handleMouseDown = (e: React.MouseEvent, index: number) => {
    if (clickMode) return;
    e.preventDefault();
    e.stopPropagation();
    setDragIndex(index);
  };

  const handleContainerClick = (e: React.MouseEvent) => {
    if (!clickMode || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    setCorners(prev => {
      const next = [...prev];
      next[clickStep] = { x, y };
      return next;
    });
    if (clickStep === 3) {
      setClickMode(false);
      setClickStep(0);
    } else {
      setClickStep(s => s + 1);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (dragIndex === null || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    setCorners(prev => {
      const next = [...prev];
      next[dragIndex] = { x, y };
      return next;
    });
  };

  const handleMouseUp = () => setDragIndex(null);
  const getPointsStr = () => corners.map(p => `${p.x * 100},${p.y * 100}`).join(' ');

  if (flask.images.length === 0) {
    return (
      <div className="card">
        <div className="text-center py-10 border-2 border-dashed border-gray-300 rounded bg-panel">
          <p className="mb-4 text-gray">Upload microscope images to begin counting</p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="flex justify-between mb-4">
        <h3 className="font-bold text-lg">Cell Counter</h3>
        {lastOtsu !== null && (
          <span className="text-xs text-gray font-mono" style={{ alignSelf: 'center' }}>
            Otsu threshold: {lastOtsu}
          </span>
        )}
      </div>

      <div className="grid grid-2" style={{ gap: '24px' }}>
        {/* Left: Editor & Controls */}
        <div>
          <div className="flex mb-2 justify-between">
            <div className="text-sm text-gray font-mono">
              {clickMode ? `SET CORNER ${clickStep + 1}/4` : 'ROI SELECTION'}
            </div>
            <button
              className="icon-btn text-xs"
              onClick={() => { setClickMode(!clickMode); setClickStep(0); }}
            >
              {clickMode ? 'CANCEL' : 'MANUAL SET'}
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
                <img
                  ref={imgRef}
                  src={activeImg.originalSrc}
                  crossOrigin="anonymous"
                  style={{ pointerEvents: 'none', userSelect: 'none' }}
                />
                <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                  <polygon
                    points={getPointsStr()}
                    fill="rgba(14,165,233,0.2)"
                    stroke="var(--accent)"
                    strokeWidth="2"
                    vectorEffect="non-scaling-stroke"
                  />
                </svg>
                {corners.map((p, i) => (
                  <div
                    key={i}
                    className="resize-handle"
                    style={{
                      left: `${p.x * 100}%`,
                      top: `${p.y * 100}%`,
                      backgroundColor: clickMode && i === clickStep ? '#fff' : 'var(--accent)',
                    }}
                    onMouseDown={e => handleMouseDown(e, i)}
                  />
                ))}
              </>
            )}
          </div>

          {/* Parameter Controls */}
          <div className="grid grid-2 mt-4 gap-4">
            <div>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={params.autoThreshold}
                  onChange={e => updateParam('autoThreshold', e.target.checked)}
                  style={{ marginRight: '8px' }}
                />
                Auto Threshold (Otsu)
              </label>
              {!params.autoThreshold && (
                <>
                  <label style={{ marginTop: 8 }}>Threshold ({params.threshold})</label>
                  <input
                    type="range" min="50" max="240"
                    value={params.threshold}
                    onChange={e => updateParam('threshold', parseInt(e.target.value))}
                  />
                </>
              )}
            </div>
            <div>
              <label>Circularity ({params.circularityThresh.toFixed(2)})</label>
              <input
                type="range" min="0.1" max="1.0" step="0.05"
                value={params.circularityThresh}
                onChange={e => updateParam('circularityThresh', parseFloat(e.target.value))}
              />
            </div>
            <div>
              <label>Min Cell Size ({params.minCellSize}px)</label>
              <input
                type="range" min="5" max="200"
                value={params.minCellSize}
                onChange={e => updateParam('minCellSize', parseInt(e.target.value))}
              />
            </div>
            <div>
              <label>Max Cell Size ({params.maxCellSize}px)</label>
              <input
                type="range" min="200" max="10000" step="100"
                value={params.maxCellSize}
                onChange={e => updateParam('maxCellSize', parseInt(e.target.value))}
              />
            </div>
            <div>
              <label>Blur σ ({params.gaussianSigma.toFixed(1)})</label>
              <input
                type="range" min="0" max="4" step="0.1"
                value={params.gaussianSigma}
                onChange={e => updateParam('gaussianSigma', parseFloat(e.target.value))}
              />
            </div>
            <div>
              <label>Morph Radius ({params.morphRadius})</label>
              <input
                type="range" min="0" max="4"
                value={params.morphRadius}
                onChange={e => updateParam('morphRadius', parseInt(e.target.value))}
              />
            </div>
            <div>
              <label>Cluster Split ({params.clusterSplitRatio.toFixed(1)}x)</label>
              <input
                type="range" min="1.2" max="3.0" step="0.1"
                value={params.clusterSplitRatio}
                onChange={e => updateParam('clusterSplitRatio', parseFloat(e.target.value))}
              />
            </div>
            <div>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={params.adaptiveIllumination}
                  onChange={e => updateParam('adaptiveIllumination', e.target.checked)}
                  style={{ marginRight: '8px' }}
                />
                Adaptive Illumination
              </label>
              {params.adaptiveIllumination && (
                <>
                  <label style={{ marginTop: 8 }}>Block Size ({params.illuminationBlockSize})</label>
                  <input
                    type="range" min="11" max="151" step="2"
                    value={params.illuminationBlockSize}
                    onChange={e => updateParam('illuminationBlockSize', parseInt(e.target.value))}
                  />
                </>
              )}
            </div>
            <div>
              <label>Median Filter ({params.medianFilterRadius}px)</label>
              <input
                type="range" min="0" max="3"
                value={params.medianFilterRadius}
                onChange={e => updateParam('medianFilterRadius', parseInt(e.target.value))}
              />
            </div>
            <div>
              <label>Min Solidity ({params.minSolidity.toFixed(2)})</label>
              <input
                type="range" min="0.1" max="1.0" step="0.05"
                value={params.minSolidity}
                onChange={e => updateParam('minSolidity', parseFloat(e.target.value))}
              />
            </div>
            <div>
              <label>Max Eccentricity ({params.maxEccentricity.toFixed(2)})</label>
              <input
                type="range" min="0.5" max="1.0" step="0.01"
                value={params.maxEccentricity}
                onChange={e => updateParam('maxEccentricity', parseFloat(e.target.value))}
              />
            </div>
            <div>
              <label>Max Intensity σ ({params.maxIntensityStdDev})</label>
              <input
                type="range" min="10" max="150"
                value={params.maxIntensityStdDev}
                onChange={e => updateParam('maxIntensityStdDev', parseInt(e.target.value))}
              />
            </div>
            <div>
              <label className="flex items-center text-sm cursor-pointer mb-2">
                <input
                  type="checkbox"
                  checked={params.use8Connected}
                  onChange={e => updateParam('use8Connected', e.target.checked)}
                  style={{ marginRight: '8px' }}
                />
                8-Connected Fill
              </label>
              <label className="flex items-center text-sm cursor-pointer mb-2">
                <input
                  type="checkbox"
                  checked={params.enableCLAHE}
                  onChange={e => updateParam('enableCLAHE', e.target.checked)}
                  style={{ marginRight: '8px' }}
                />
                CLAHE Enhancement
              </label>
              {params.enableCLAHE && (
                <>
                  <label style={{ marginTop: 4 }}>Clip Limit ({params.claheClipLimit.toFixed(1)})</label>
                  <input
                    type="range" min="0.5" max="6" step="0.1"
                    value={params.claheClipLimit}
                    onChange={e => updateParam('claheClipLimit', parseFloat(e.target.value))}
                  />
                </>
              )}
            </div>
            <div>
              <label>Min Agreement ({params.minAgreement}/3)</label>
              <input
                type="range" min="1" max="3" step="1"
                value={params.minAgreement}
                onChange={e => updateParam('minAgreement', parseInt(e.target.value))}
              />
            </div>
            <div className="flex flex-col justify-end">
              <label className="flex items-center text-sm cursor-pointer mb-2">
                <input
                  type="checkbox"
                  checked={params.showMask}
                  onChange={e => updateParam('showMask', e.target.checked)}
                  style={{ marginRight: '8px' }}
                />
                Show Mask (B&W)
              </label>
              <label className="flex items-center text-sm cursor-pointer mb-2">
                <input
                  type="checkbox"
                  checked={params.invert}
                  onChange={e => updateParam('invert', e.target.checked)}
                  style={{ marginRight: '8px' }}
                />
                Invert (Dark Cells)
              </label>
              <label className="flex items-center text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={params.excludeBorder}
                  onChange={e => updateParam('excludeBorder', e.target.checked)}
                  style={{ marginRight: '8px' }}
                />
                Exclude Border Cells
              </label>
            </div>
          </div>

          <div className="mt-4">
            <button
              className="primary w-full"
              onClick={runProcessing}
              disabled={processing}
              style={{ padding: '12px' }}
            >
              {processing ? 'PROCESSING…' : 'PROCESS & COUNT'}
            </button>
          </div>
        </div>

        {/* Right: Results & Thumbnails */}
        <div className="flex-col" style={{ height: '100%' }}>
          <h4 className="font-bold mb-2">Analysis Result</h4>
          <div
            className="canvas-container bg-panel"
            style={{ flexGrow: 1, maxHeight: '450px', border: '2px solid var(--border)' }}
          >
            {activeImg.processedSrc ? (
              <img
                src={activeImg.processedSrc}
                style={{ imageRendering: 'pixelated', maxHeight: '100%' }}
                alt="Processed result"
              />
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
            {lastStats && (
              <div className="text-xs font-mono text-gray mb-2" style={{ lineHeight: '1.6' }}>
                <div>95% CI: [{lastStats.ci95[0]}, {lastStats.ci95[1]}]</div>
                <div>Confidence: {(lastStats.meanConfidence * 100).toFixed(0)}%</div>
                <div>CV(area): {(lastStats.cvCellArea * 100).toFixed(1)}%</div>
                <div>Median ⌀: {lastStats.medianDiameter.toFixed(1)}px &nbsp;
                  [{lastStats.diameterRange[0].toFixed(1)}–{lastStats.diameterRange[1].toFixed(1)}]</div>
                <div>Mean area: {lastStats.meanCellArea.toFixed(0)}px² ± {lastStats.stdCellArea.toFixed(0)}</div>
              </div>
            )}
            <div className="flex" style={{ gap: '8px' }}>
              <div style={{ flex: 1 }}>
                <label>Manual Correction</label>
                <input
                  type="number"
                  value={activeImg.count}
                  min={0}
                  onChange={e => onUpdateCount(
                    activeImg.id,
                    Math.max(0, parseInt(e.target.value) || 0),
                    activeImg.processedSrc || activeImg.originalSrc,
                  )}
                />
              </div>
              <button
                className="primary"
                disabled={!activeImg.processedSrc}
                style={{ alignSelf: 'flex-end', whiteSpace: 'nowrap' }}
                onClick={() => activeImg.processedSrc && saveImage(
                  activeImg.processedSrc,
                  `${flask.name}_image${selectedImgIndex + 1}.png`,
                  activeImg.count,
                )}
              >
                💾 Save Image
              </button>
            </div>
          </div>

          <div>
            <h4 className="font-bold mb-2 text-xs text-gray">IMAGES IN FLASK</h4>
            <div className="flex" style={{ overflowX: 'auto', paddingBottom: '5px' }}>
              {flask.images.map((img, idx) => (
                <div
                  key={img.id}
                  onClick={() => setSelectedImgIndex(idx)}
                  style={{
                    minWidth: '60px', width: '60px', cursor: 'pointer',
                    border: idx === selectedImgIndex ? '2px solid var(--primary)' : '1px solid transparent',
                    opacity: idx === selectedImgIndex ? 1 : 0.6,
                  }}
                >
                  <img
                    src={img.processedSrc || img.originalSrc}
                    style={{ width: '100%', height: '50px', objectFit: 'cover', display: 'block', borderRadius: '4px' }}
                    alt={`Image ${idx + 1}`}
                  />
                  <div className="text-center text-xs font-mono font-bold mt-1">{img.count}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// --- App ---

const App: React.FC = () => {
  const [flasks, setFlasks] = useState<FlaskData[]>([]);
  const [activeFlaskId, setActiveFlaskId] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    document.body.classList.toggle('dark-mode', darkMode);
  }, [darkMode]);

  const activeFlask = flasks.find(f => f.id === activeFlaskId);

  const addFlask = useCallback(() => {
    setFlasks(prev => {
      const f: FlaskData = { id: generateId(), name: `Flask ${prev.length + 1}`, images: [] };
      setActiveFlaskId(f.id);
      return [...prev, f];
    });
  }, []);

  const deleteFlask = useCallback((id: string) => {
    setFlasks(prev => prev.filter(f => f.id !== id));
    setActiveFlaskId(prev => prev === id ? null : prev);
  }, []);

  const updateFlask = useCallback((id: string, data: Partial<FlaskData>) => {
    setFlasks(prev => prev.map(f => f.id === id ? { ...f, ...data } : f));
  }, []);

  const handleImageUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (!activeFlaskId || !e.target.files) return;
    const newImages = Array.from(e.target.files).map((file: File) => ({
      id: generateId(),
      originalSrc: URL.createObjectURL(file),
      count: 0,
    }));
    setFlasks(prev => prev.map(f =>
      f.id === activeFlaskId ? { ...f, images: [...f.images, ...newImages] } : f
    ));
    e.target.value = '';
  }, [activeFlaskId]);

  const updateImageCount = useCallback((imgId: string, count: number, processedSrc: string) => {
    if (!activeFlaskId) return;
    setFlasks(prev => prev.map(f =>
      f.id === activeFlaskId
        ? { ...f, images: f.images.map(img => img.id === imgId ? { ...img, count, processedSrc } : img) }
        : f
    ));
  }, [activeFlaskId]);

  return (
    <div className="container">
      <div className="flex justify-between mb-6">
        <h1 className="text-2xl font-bold text-primary">🔬 Hemocytometer Assistant</h1>
        <button
          className="secondary"
          onClick={() => setDarkMode(d => !d)}
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
        onDeleteFlask={deleteFlask}
      />

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

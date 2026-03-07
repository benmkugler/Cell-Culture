import type { Point, PipelineParams, PipelineResult, PipelineStats, BlobStats, Blob } from './types';
import { warpPerspective } from './homography';
import {
  toGrayscale, otsuThreshold, gaussianBlur,
  binaryThreshold, removeGridLines, morphOpen, morphClose,
  adaptiveIlluminationCorrection, medianFilter,
  detectGridLines, buildGridZone, clahe,
} from './imageops';
import { extractBlobs, computeMedianCellArea } from './blobs';
import { ensembleClassifyBlobs, watershedSeparation } from './counting';

const WARP_SIZE = 600;

/**
 * Get source image pixel data from an HTMLImageElement.
 */
const getImageData = (img: HTMLImageElement): { data: Uint8ClampedArray; w: number; h: number } => {
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  const id = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return { data: id.data, w: canvas.width, h: canvas.height };
};

/**
 * Color palette for individual cells within clusters.
 * High-contrast, colorblind-friendly.
 */
const CELL_PALETTE = [
  '#ff6b6b', '#4ecdc4', '#ffe66d', '#a8e6cf',
  '#ff8b94', '#b8d4e3', '#f7dc6f', '#82e0aa',
];

/**
 * Render detected blobs onto a canvas context as overlays.
 * Single cells: green circle + center dot.
 * Clusters: cyan bbox, each cell marked with colored circle + crosshair,
 * labeled with count, agreement shown via opacity.
 */
const renderOverlay = (
  ctx: CanvasRenderingContext2D,
  blobs: Blob[],
  showMask: boolean,
  binaryData: Uint8Array,
  size: number,
): void => {
  if (showMask) {
    const maskImg = ctx.createImageData(size, size);
    const md = maskImg.data;
    const cellPixels = new Uint8Array(size * size);
    for (const b of blobs) {
      for (const p of b.pixels) cellPixels[p] = 1;
    }
    for (let i = 0; i < size * size; i++) {
      const v = cellPixels[i] ? 255 : binaryData[i] ? 50 : 0;
      md[i * 4] = v;
      md[i * 4 + 1] = v;
      md[i * 4 + 2] = v;
      md[i * 4 + 3] = 255;
    }
    ctx.putImageData(maskImg, 0, 0);
  }

  for (const b of blobs) {
    if (b.cellCount === 1) {
      // ── Single cell: green circle + small center dot ──
      const loc = b.cellLocations[0];
      const cx = loc?.x ?? b.cx;
      const cy = loc?.y ?? b.cy;
      const r = loc?.radius ?? Math.sqrt(b.area / Math.PI);

      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, r + 2, 0, 2 * Math.PI);
      ctx.stroke();

      ctx.fillStyle = '#00ff00';
      ctx.beginPath();
      ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI);
      ctx.fill();
    } else {
      // ── Cluster: bbox + individual cell markers ──
      const { minX, minY, maxX, maxY } = b.bbox;
      const pad = 3;

      // Semi-transparent cyan bbox
      ctx.strokeStyle = 'rgba(0, 255, 255, 0.7)';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.rect(minX - pad, minY - pad, maxX - minX + 2 * pad, maxY - minY + 2 * pad);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw each identified cell location
      for (let i = 0; i < b.cellLocations.length; i++) {
        const loc = b.cellLocations[i];
        const color = CELL_PALETTE[i % CELL_PALETTE.length];
        const alpha = 0.4 + 0.2 * loc.agreement; // 0.6 / 0.8 / 1.0

        ctx.save();
        ctx.globalAlpha = alpha;

        // Cell circle
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(loc.x, loc.y, Math.max(3, loc.radius), 0, 2 * Math.PI);
        ctx.stroke();

        // Crosshair at center
        const ch = 5;
        ctx.beginPath();
        ctx.moveTo(loc.x - ch, loc.y);
        ctx.lineTo(loc.x + ch, loc.y);
        ctx.moveTo(loc.x, loc.y - ch);
        ctx.lineTo(loc.x, loc.y + ch);
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.restore();
      }

      // Count label with shadow
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 14px monospace';
      ctx.shadowColor = '#000';
      ctx.shadowBlur = 4;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(b.cellCount.toString(), b.cx, b.cy);
      ctx.shadowBlur = 0;
    }
  }
};

/**
 * Compute statistical output from classified blobs.
 * Provides CI, CV, per-blob confidence — compliant with ISO cell counting standards.
 */
export const computePipelineStats = (blobs: Blob[], totalCount: number): PipelineStats => {
  // Collect single-cell areas for distribution analysis
  const singleCellAreas = blobs
    .filter(b => b.cellCount === 1)
    .map(b => b.area);

  const meanCellArea = singleCellAreas.length > 0
    ? singleCellAreas.reduce((s, a) => s + a, 0) / singleCellAreas.length
    : 0;

  const stdCellArea = singleCellAreas.length > 1
    ? Math.sqrt(
        singleCellAreas.reduce((s, a) => s + (a - meanCellArea) ** 2, 0) /
        (singleCellAreas.length - 1),
      )
    : 0;

  const cvCellArea = meanCellArea > 0 ? stdCellArea / meanCellArea : 0;

  // 95% CI using Poisson approximation (standard for cell counting):
  //   count ± 1.96 * sqrt(count)
  const margin = 1.96 * Math.sqrt(Math.max(1, totalCount));
  const ci95: [number, number] = [
    Math.max(0, Math.round(totalCount - margin)),
    Math.round(totalCount + margin),
  ];

  // Per-blob statistics
  const perBlob: BlobStats[] = blobs.map((b, i) => {
    const maxAgreement = b.cellLocations.length > 0
      ? Math.max(...b.cellLocations.map(l => l.agreement))
      : 0;
    const meanAgreement = b.cellLocations.length > 0
      ? b.cellLocations.reduce((s, l) => s + l.agreement, 0) / b.cellLocations.length
      : 0;
    return {
      blobIndex: i,
      area: b.area,
      circularity: b.circularity,
      solidity: b.solidity,
      eccentricity: b.eccentricity,
      cellCount: b.cellCount,
      confidence: meanAgreement / 3,
      agreement: maxAgreement,
    };
  });

  const meanConfidence = perBlob.length > 0
    ? perBlob.reduce((s, b) => s + b.confidence, 0) / perBlob.length
    : 0;

  // Diameter statistics
  const diameters = singleCellAreas.map(a => 2 * Math.sqrt(a / Math.PI)).sort((a, b) => a - b);
  const medianDiameter = diameters.length > 0
    ? diameters[Math.floor(diameters.length / 2)]
    : 0;
  const diameterRange: [number, number] = diameters.length > 0
    ? [diameters[0], diameters[diameters.length - 1]]
    : [0, 0];

  return {
    totalCount,
    meanCellArea,
    stdCellArea,
    cvCellArea,
    ci95,
    meanConfidence,
    perBlob,
    medianDiameter,
    diameterRange,
  };
};

/**
 * Full processing pipeline.
 * Pure computational core with rendering side-effects isolated to renderOverlay.
 */
export const processImage = (
  imgElement: HTMLImageElement,
  corners: Point[],
  params: PipelineParams,
): PipelineResult => {
  const size = WARP_SIZE;
  const { data: srcData, w: srcW, h: srcH } = getImageData(imgElement);

  // 1. Perspective warp
  const warped = warpPerspective(srcData, srcW, srcH, corners, size, size);

  // 2. Grayscale
  let gray = toGrayscale(warped.data, size * size);

  // 2a. Adaptive illumination correction (uneven lighting / ripples)
  if (params.adaptiveIllumination) {
    gray = adaptiveIlluminationCorrection(gray, size, size, params.illuminationBlockSize);
  }

  // 2b. CLAHE (local contrast enhancement)
  if (params.enableCLAHE) {
    gray = clahe(gray, size, size, params.claheTileSize, params.claheClipLimit);
  }

  // 2c. Median filter (salt-and-pepper noise / small debris)
  if (params.medianFilterRadius > 0) {
    gray = medianFilter(gray, size, size, params.medianFilterRadius);
  }

  // 3. Gaussian blur (noise suppression)
  const blurred = gaussianBlur(gray, size, size, params.gaussianSigma);

  // 4. Threshold (Otsu or manual)
  const otsu = otsuThreshold(blurred);
  const thresh = params.autoThreshold ? otsu : params.threshold;
  let binary = binaryThreshold(blurred, thresh, params.invert);

  // 5. Grid line detection & removal
  const { rows: gridRows, cols: gridCols } = detectGridLines(binary, size, size);
  binary = removeGridLines(binary, size, size);
  const gridZone = buildGridZone(size, size, gridRows, gridCols, 4);

  // 5a. Morphological close to reconnect cells split by grid erasure
  if (gridRows.size > 0 || gridCols.size > 0) {
    binary = morphClose(binary, size, size, 1);
  }

  // 6. Morphological opening (remove noise)
  binary = morphOpen(binary, size, size, params.morphRadius);

  // 7. First-pass blob extraction (for cell-size estimation)
  const rawBlobs1 = extractBlobs(
    binary, size, size,
    params.minCellSize, params.maxCellSize,
    params.excludeBorder, gray, params.use8Connected,
  );
  const medianArea1 = computeMedianCellArea(rawBlobs1, params.circularityThresh, params.minCellSize);
  const cellRadius1 = Math.sqrt(medianArea1 / Math.PI);

  // 7a. Watershed separation of touching cells
  const separated = watershedSeparation(
    binary, size, size, cellRadius1 * 0.7, Math.max(3.0, cellRadius1 * 0.4),
  );

  // 7b. Second-pass blob extraction (from separated mask)
  const rawBlobs = extractBlobs(
    separated, size, size,
    params.minCellSize, params.maxCellSize,
    params.excludeBorder, gray, params.use8Connected,
  );

  // 8. Compute reference cell area
  const medianArea = computeMedianCellArea(rawBlobs, params.circularityThresh, params.minCellSize);

  // 9. Classify and count (ensemble: DT watershed + concavity + LoG)
  const classifiedBlobs = ensembleClassifyBlobs(
    rawBlobs, medianArea,
    params.circularityThresh, params.clusterSplitRatio,
    binary, gray, size, size,
    params.minSolidity, params.maxEccentricity, params.maxIntensityStdDev,
    params.minAgreement,
  );

  // 9a. Filter phantom blobs near grid lines
  const filteredBlobs = classifiedBlobs.filter(b => {
    if (b.pixels.length === 0) return true;
    let inZone = 0;
    for (const p of b.pixels) if (gridZone[p]) inZone++;
    return inZone / b.pixels.length < 0.4;
  });

  const totalCount = filteredBlobs.reduce((sum, b) => sum + b.cellCount, 0);

  // 10. Render output
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  if (!params.showMask) {
    ctx.putImageData(warped, 0, 0);
  }

  renderOverlay(ctx, filteredBlobs, params.showMask, binary, size);

  const stats = computePipelineStats(filteredBlobs, totalCount);

  return {
    count: totalCount,
    blobs: filteredBlobs,
    dataUrl: canvas.toDataURL('image/png'),
    otsuThreshold: otsu,
    stats,
  };
};

/**
 * Process a raw grayscale buffer (for testing without DOM/canvas).
 * Returns count, classified blobs, and statistical analysis.
 */
export const processGrayscaleBuffer = (
  gray: Uint8Array,
  w: number,
  h: number,
  params: Omit<PipelineParams, 'showMask'>,
): { count: number; blobs: Blob[]; otsuThreshold: number; stats: PipelineStats } => {
  // Pre-processing for degraded images
  let processed = gray;
  if (params.adaptiveIllumination) {
    processed = adaptiveIlluminationCorrection(processed, w, h, params.illuminationBlockSize);
  }
  if (params.enableCLAHE) {
    processed = clahe(processed, w, h, params.claheTileSize, params.claheClipLimit);
  }
  if (params.medianFilterRadius > 0) {
    processed = medianFilter(processed, w, h, params.medianFilterRadius);
  }

  const blurred = gaussianBlur(processed, w, h, params.gaussianSigma);
  const otsu = otsuThreshold(blurred);
  const thresh = params.autoThreshold ? otsu : params.threshold;
  let binary = binaryThreshold(blurred, thresh, params.invert);
  const { rows: gridRows, cols: gridCols } = detectGridLines(binary, w, h);
  binary = removeGridLines(binary, w, h);
  const gridZone = buildGridZone(w, h, gridRows, gridCols, 4);
  // Morphological close to reconnect cells split by grid erasure
  if (gridRows.size > 0 || gridCols.size > 0) {
    binary = morphClose(binary, w, h, 1);
  }
  binary = morphOpen(binary, w, h, params.morphRadius);
  const rawBlobs1 = extractBlobs(binary, w, h, params.minCellSize, params.maxCellSize, params.excludeBorder, processed, params.use8Connected);
  const medianArea1 = computeMedianCellArea(rawBlobs1, params.circularityThresh, params.minCellSize);
  const cellRadius1 = Math.sqrt(medianArea1 / Math.PI);
  const separated = watershedSeparation(binary, w, h, cellRadius1 * 0.7, Math.max(3.0, cellRadius1 * 0.4));
  const rawBlobs = extractBlobs(separated, w, h, params.minCellSize, params.maxCellSize, params.excludeBorder, processed, params.use8Connected);
  const medianArea = computeMedianCellArea(rawBlobs, params.circularityThresh, params.minCellSize);
  const classifiedBlobs = ensembleClassifyBlobs(
    rawBlobs, medianArea,
    params.circularityThresh, params.clusterSplitRatio,
    binary, processed, w, h,
    params.minSolidity, params.maxEccentricity, params.maxIntensityStdDev,
    params.minAgreement,
  );
  const filteredBlobs = classifiedBlobs.filter(b => {
    if (b.pixels.length === 0) return true;
    let inZone = 0;
    for (const p of b.pixels) if (gridZone[p]) inZone++;
    return inZone / b.pixels.length < 0.4;
  });
  const count = filteredBlobs.reduce((sum, b) => sum + b.cellCount, 0);
  const stats = computePipelineStats(filteredBlobs, count);
  return { count, blobs: filteredBlobs, otsuThreshold: otsu, stats };
};

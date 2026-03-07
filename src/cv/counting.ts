/**
 * counting.ts — Multi-strategy ensemble cell counter.
 *
 * Three orthogonal strategies vote on cluster cell counts:
 *   A. Distance-transform watershed — finds local maxima in the EDT,
 *      each maximum ≈ one cell nucleus.
 *   B. Concavity-point splitting — detects constriction points on the
 *      contour where touching cells create concavities.
 *   C. Laplacian-of-Gaussian (LoG) blob detection — scale-space
 *      extrema on the raw grayscale, threshold-free.
 *
 * Final blob.cellCount = median of the 3 estimates, with the area-ratio
 * method as a tiebreaker when strategies disagree by > 2x.
 */

import type { Blob, CellLocation } from './types';

// ─── Strategy A: Distance-Transform Watershed ────────────────────────

/**
 * Exact Euclidean distance transform (squared) via Felzenszwalb & Huttenlocher.
 * Returns Float64Array of squared distances from each foreground pixel
 * to the nearest background pixel.
 */
export const distanceTransformSq = (
  binary: Uint8Array, w: number, h: number,
): Float64Array => {
  const INF = 1e20;
  const dt = new Float64Array(w * h);

  // Initialize: 0 for foreground, INF for background
  for (let i = 0; i < w * h; i++) {
    dt[i] = binary[i] ? 0 : INF;
  }

  // 1D distance transform helper (operates in-place on a slice)
  const dt1d = (f: Float64Array, n: number) => {
    const d = new Float64Array(n);
    const v = new Int32Array(n); // envelope indices
    const z = new Float64Array(n + 1); // envelope boundaries
    let k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;

    for (let q = 1; q < n; q++) {
      let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      while (s <= z[k]) {
        k--;
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
      }
      k++;
      v[k] = q;
      z[k] = s;
      z[k + 1] = INF;
    }

    k = 0;
    for (let q = 0; q < n; q++) {
      while (z[k + 1] < q) k++;
      d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
    }
    for (let q = 0; q < n; q++) f[q] = d[q];
  };

  // Transform columns
  const col = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) col[y] = dt[y * w + x];
    dt1d(col, h);
    for (let y = 0; y < h; y++) dt[y * w + x] = col[y];
  }

  // Transform rows
  const row = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) row[x] = dt[y * w + x];
    dt1d(row, w);
    for (let x = 0; x < w; x++) dt[y * w + x] = row[x];
  }

  return dt;
};

/**
 * Marker-controlled watershed separation of touching cells.
 * Finds DT peaks in the binary mask, grows each marker via BFS, and
 * draws 1 px boundaries where different watershed regions meet.
 * Single-peak regions are left untouched.
 */
export const watershedSeparation = (
  binary: Uint8Array, w: number, h: number,
  minPeakDist: number, minPeakVal = 3.0,
): Uint8Array => {
  const inv = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) inv[i] = binary[i] ? 0 : 1;
  const dtSq = distanceTransformSq(inv, w, h);

  const dt = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) dt[i] = binary[i] ? Math.sqrt(dtSq[i]) : 0;

  // Local maxima detection
  const r = Math.max(2, Math.round(minPeakDist));
  const rawPeaks: { idx: number; val: number }[] = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const val = dt[y * w + x];
      if (val < minPeakVal) continue;
      let isMax = true;
      for (let dy = -r; dy <= r && isMax; dy++) {
        for (let dx = -r; dx <= r && isMax; dx++) {
          if (dx === 0 && dy === 0) continue;
          const ny = y + dy, nx = x + dx;
          if (ny >= 0 && ny < h && nx >= 0 && nx < w && dt[ny * w + nx] > val) isMax = false;
        }
      }
      if (isMax) rawPeaks.push({ idx: y * w + x, val });
    }
  }

  // Greedy NMS
  rawPeaks.sort((a, b) => b.val - a.val);
  const peaks: typeof rawPeaks = [];
  for (const p of rawPeaks) {
    const px = p.idx % w, py = (p.idx - px) / w;
    let tooClose = false;
    for (const q of peaks) {
      const qx = q.idx % w, qy = (q.idx - qx) / w;
      if (Math.sqrt((px - qx) ** 2 + (py - qy) ** 2) < minPeakDist) { tooClose = true; break; }
    }
    if (!tooClose) peaks.push(p);
  }

  if (peaks.length <= 1) return binary;

  // BFS from markers — assign each foreground pixel to nearest peak
  const labels = new Int32Array(w * h).fill(-1);
  const queue: number[] = [];
  for (let i = 0; i < peaks.length; i++) {
    labels[peaks[i].idx] = i;
    queue.push(peaks[i].idx);
  }

  const dx4 = [1, -1, 0, 0];
  const dy4 = [0, 0, 1, -1];
  let head = 0;
  while (head < queue.length) {
    const idx = queue[head++];
    const x = idx % w, y = (idx - x) / w;
    for (let k = 0; k < 4; k++) {
      const nx = x + dx4[k], ny = y + dy4[k];
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const ni = ny * w + nx;
      if (binary[ni] && labels[ni] < 0) {
        labels[ni] = labels[idx];
        queue.push(ni);
      }
    }
  }

  // Draw 1 px boundaries where different labels meet
  const out = new Uint8Array(binary);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (!binary[idx] || labels[idx] < 0) continue;
      for (let k = 0; k < 4; k++) {
        const nx = x + dx4[k], ny = y + dy4[k];
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const ni = ny * w + nx;
        if (binary[ni] && labels[ni] >= 0 && labels[ni] !== labels[idx]) {
          out[idx] = 0;
          break;
        }
      }
    }
  }

  return out;
};

/**
 * Locate cell centers via distance-transform peak detection.
 * Each local maximum in the EDT corresponds to approximately one cell center.
 * Returns CellLocation[] in image-space coordinates.
 */
export const locateByDistanceTransform = (
  binary: Uint8Array, w: number, h: number,
  blob: Blob, minDist: number,
): CellLocation[] => {
  const { minX, maxX, minY, maxY } = blob.bbox;
  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;

  const patch = new Uint8Array(bw * bh);
  const pixelSet = new Set(blob.pixels);
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (pixelSet.has(y * w + x)) {
        patch[(y - minY) * bw + (x - minX)] = 1;
      }
    }
  }

  const inverted = new Uint8Array(bw * bh);
  for (let i = 0; i < bw * bh; i++) inverted[i] = patch[i] ? 0 : 1;

  const dtSq = distanceTransformSq(inverted, bw, bh);

  const dt = new Float64Array(bw * bh);
  for (let i = 0; i < bw * bh; i++) {
    dt[i] = patch[i] ? Math.sqrt(dtSq[i]) : 0;
  }

  const r = Math.max(2, Math.round(minDist));
  const peaks: { x: number; y: number; val: number }[] = [];

  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      const val = dt[y * bw + x];
      if (val < 1.5) continue;

      let isMax = true;
      for (let dy = -r; dy <= r && isMax; dy++) {
        for (let dx = -r; dx <= r && isMax; dx++) {
          if (dx === 0 && dy === 0) continue;
          const ny = y + dy, nx = x + dx;
          if (ny >= 0 && ny < bh && nx >= 0 && nx < bw) {
            if (dt[ny * bw + nx] > val) isMax = false;
          }
        }
      }
      if (isMax) peaks.push({ x, y, val });
    }
  }

  peaks.sort((a, b) => b.val - a.val);
  const kept: typeof peaks = [];
  for (const p of peaks) {
    let tooClose = false;
    for (const q of kept) {
      const d = Math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2);
      if (d < minDist) { tooClose = true; break; }
    }
    if (!tooClose) kept.push(p);
  }

  if (kept.length === 0) {
    return [{ x: blob.cx, y: blob.cy, radius: Math.sqrt(blob.area / Math.PI), agreement: 0 }];
  }
  return kept.map(p => ({
    x: p.x + minX,
    y: p.y + minY,
    radius: Math.max(1, p.val),
    agreement: 0,
  }));
};

/** Count-only wrapper for backward compatibility. */
export const countByDistanceTransform = (
  binary: Uint8Array, w: number, h: number,
  blob: Blob, minDist: number,
): number => locateByDistanceTransform(binary, w, h, blob, minDist).length;

// ─── Strategy B: Contour Concavity Splitting ─────────────────────────

/**
 * Extract ordered contour (boundary trace) of a blob using Moore
 * neighbour tracing (8-connected).
 * Returns array of (x, y) pairs in order around the contour.
 */
const traceContour = (
  pixels: Set<number>, w: number, minX: number, minY: number,
  maxX: number, maxY: number,
): [number, number][] => {
  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;

  // Build local grid
  const grid = new Uint8Array(bw * bh);
  for (const p of pixels) {
    const x = (p % w) - minX;
    const y = Math.floor(p / w) - minY;
    if (x >= 0 && x < bw && y >= 0 && y < bh) {
      grid[y * bw + x] = 1;
    }
  }

  // Find topmost-leftmost boundary pixel
  let startX = -1, startY = -1;
  outer:
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      if (grid[y * bw + x]) {
        startX = x; startY = y;
        break outer;
      }
    }
  }
  if (startX < 0) return [];

  // Moore neighbour tracing (clockwise)
  const dx8 = [1, 1, 0, -1, -1, -1, 0, 1];
  const dy8 = [0, 1, 1, 1, 0, -1, -1, -1];

  const contour: [number, number][] = [];
  let cx = startX, cy = startY;
  let dir = 7; // start looking from the left

  const maxIter = bw * bh * 4;
  for (let iter = 0; iter < maxIter; iter++) {
    contour.push([cx + minX, cy + minY]);

    let found = false;
    // Check 8 neighbors starting from (dir+1)%8
    const startDir = (dir + 5) % 8; // backtrack direction + 1
    for (let i = 0; i < 8; i++) {
      const d = (startDir + i) % 8;
      const nx = cx + dx8[d];
      const ny = cy + dy8[d];
      if (nx >= 0 && nx < bw && ny >= 0 && ny < bh && grid[ny * bw + nx]) {
        cx = nx;
        cy = ny;
        dir = d;
        found = true;
        break;
      }
    }

    if (!found) break;
    if (cx === startX && cy === startY && contour.length > 2) break;
  }

  return contour;
};

/**
 * Locate cell centers via contour concavity analysis.
 * Finds constriction points between touching cells and estimates
 * individual cell centers using farthest-first k-means partitioning.
 */
export const locateByConcavity = (
  blob: Blob, w: number, kNeighbour = 8,
): CellLocation[] => {
  const defaultLoc: CellLocation = {
    x: blob.cx, y: blob.cy,
    radius: Math.sqrt(blob.area / Math.PI),
    agreement: 0,
  };
  if (blob.area < 30) return [defaultLoc];

  const pixelSet = new Set(blob.pixels);
  const { minX, maxX, minY, maxY } = blob.bbox;
  const contour = traceContour(pixelSet, w, minX, minY, maxX, maxY);

  if (contour.length < kNeighbour * 3) return [defaultLoc];

  const n = contour.length;
  const k = Math.min(kNeighbour, Math.floor(n / 4));

  // Compute curvature at each contour point
  const curvatures: number[] = [];
  for (let i = 0; i < n; i++) {
    const prev = contour[(i - k + n) % n];
    const curr = contour[i];
    const next = contour[(i + k) % n];

    const v1x = prev[0] - curr[0];
    const v1y = prev[1] - curr[1];
    const v2x = next[0] - curr[0];
    const v2y = next[1] - curr[1];

    const cross = v1x * v2y - v1y * v2x;
    const dot = v1x * v2x + v1y * v2y;
    const angle = Math.atan2(cross, dot);

    curvatures.push(angle);
  }

  const concavityThresh = Math.PI * 0.22;
  const concavePoints: number[] = [];

  for (let i = 0; i < n; i++) {
    if (curvatures[i] > concavityThresh) {
      const prev = curvatures[(i - 1 + n) % n];
      const next = curvatures[(i + 1) % n];
      if (curvatures[i] >= prev && curvatures[i] >= next) {
        concavePoints.push(i);
      }
    }
  }

  const mergeRadius = Math.max(3, k);
  const mergedConcavities: number[] = [];
  for (const p of concavePoints) {
    let tooClose = false;
    for (const q of mergedConcavities) {
      const arcDist = Math.min(Math.abs(p - q), n - Math.abs(p - q));
      if (arcDist < mergeRadius) { tooClose = true; break; }
    }
    if (!tooClose) mergedConcavities.push(p);
  }

  const numCells = Math.floor(mergedConcavities.length / 2) + 1;
  if (numCells <= 1) return [defaultLoc];

  // Farthest-first k-means to locate cell centers
  const coords: [number, number][] = blob.pixels.map(p => [p % w, Math.floor(p / w)]);

  // Seed 1: pixel farthest from blob centroid
  const seeds: [number, number][] = [];
  let maxD = 0, bestI = 0;
  for (let i = 0; i < coords.length; i++) {
    const d = (coords[i][0] - blob.cx) ** 2 + (coords[i][1] - blob.cy) ** 2;
    if (d > maxD) { maxD = d; bestI = i; }
  }
  seeds.push(coords[bestI]);

  // Subsequent seeds: farthest from any existing seed
  while (seeds.length < numCells) {
    let best = 0, bestDist = 0;
    for (let i = 0; i < coords.length; i++) {
      let minSeedD = Infinity;
      for (const s of seeds) {
        const d = (coords[i][0] - s[0]) ** 2 + (coords[i][1] - s[1]) ** 2;
        if (d < minSeedD) minSeedD = d;
      }
      if (minSeedD > bestDist) { bestDist = minSeedD; best = i; }
    }
    seeds.push(coords[best]);
  }

  // Lloyd's iterations
  let centroids = seeds.map(s => [...s] as [number, number]);
  for (let iter = 0; iter < 5; iter++) {
    const sums: [number, number][] = Array.from({ length: numCells }, () => [0, 0]);
    const counts = new Float64Array(numCells);
    for (let i = 0; i < coords.length; i++) {
      let minD = Infinity, minJ = 0;
      for (let j = 0; j < centroids.length; j++) {
        const d = (coords[i][0] - centroids[j][0]) ** 2 + (coords[i][1] - centroids[j][1]) ** 2;
        if (d < minD) { minD = d; minJ = j; }
      }
      sums[minJ][0] += coords[i][0];
      sums[minJ][1] += coords[i][1];
      counts[minJ]++;
    }
    for (let j = 0; j < numCells; j++) {
      if (counts[j] > 0) {
        centroids[j] = [sums[j][0] / counts[j], sums[j][1] / counts[j]];
      }
    }
  }

  const cellRadius = Math.sqrt(blob.area / (numCells * Math.PI));
  return centroids.map(c => ({ x: c[0], y: c[1], radius: cellRadius, agreement: 0 }));
};

/** Count-only wrapper for backward compatibility. */
export const countByConcavity = (
  blob: Blob, w: number, kNeighbour = 8,
): number => locateByConcavity(blob, w, kNeighbour).length;

// ─── Strategy C: Laplacian-of-Gaussian (LoG) ────────────────────────

/**
 * LoG filter at a given sigma.
 * Returns the scale-normalized LoG response (negative = dark blob).
 */
const logFilter = (
  gray: Uint8Array, w: number, h: number, sigma: number,
): Float64Array => {
  const r = Math.ceil(sigma * 3);
  const size = 2 * r + 1;

  // Build 2D LoG kernel
  const kernel = new Float64Array(size * size);
  const s2 = sigma * sigma;
  const s4 = s2 * s2;
  let sum = 0;
  for (let ky = -r; ky <= r; ky++) {
    for (let kx = -r; kx <= r; kx++) {
      const rr = kx * kx + ky * ky;
      const val = -(1 / (Math.PI * s4)) * (1 - rr / (2 * s2)) * Math.exp(-rr / (2 * s2));
      kernel[(ky + r) * size + (kx + r)] = val;
      sum += val;
    }
  }
  // Zero-mean normalization
  const correction = sum / (size * size);
  for (let i = 0; i < size * size; i++) kernel[i] -= correction;

  // Convolve
  const response = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let val = 0;
      for (let ky = -r; ky <= r; ky++) {
        for (let kx = -r; kx <= r; kx++) {
          const sy = Math.min(h - 1, Math.max(0, y + ky));
          const sx = Math.min(w - 1, Math.max(0, x + kx));
          val += gray[sy * w + sx] * kernel[(ky + r) * size + (kx + r)];
        }
      }
      // Scale-normalize: multiply by sigma^2
      response[y * w + x] = val * s2;
    }
  }

  return response;
};

/**
 * Locate cell centers via multi-scale Laplacian-of-Gaussian detection.
 * Returns CellLocation[] in image-space coordinates.
 */
export const locateByLoG = (
  gray: Uint8Array, w: number, h: number,
  blob: Blob, sigmaRange: [number, number], numScales = 5,
): CellLocation[] => {
  const { minX, maxX, minY, maxY } = blob.bbox;
  const pad = 3;
  const rx0 = Math.max(0, minX - pad);
  const ry0 = Math.max(0, minY - pad);
  const rx1 = Math.min(w - 1, maxX + pad);
  const ry1 = Math.min(h - 1, maxY + pad);
  const rw = rx1 - rx0 + 1;
  const rh = ry1 - ry0 + 1;

  const fallback: CellLocation = {
    x: blob.cx, y: blob.cy,
    radius: Math.sqrt(blob.area / Math.PI),
    agreement: 0,
  };
  if (rw < 5 || rh < 5) return [fallback];

  const patch = new Uint8Array(rw * rh);
  for (let y = ry0; y <= ry1; y++) {
    for (let x = rx0; x <= rx1; x++) {
      patch[(y - ry0) * rw + (x - rx0)] = gray[y * w + x];
    }
  }

  const [sMin, sMax] = sigmaRange;
  const scaleStep = numScales > 1 ? (sMax - sMin) / (numScales - 1) : 0;

  const maxResponse = new Float64Array(rw * rh).fill(Infinity);
  const bestScale = new Float64Array(rw * rh);

  for (let si = 0; si < numScales; si++) {
    const sigma = sMin + si * scaleStep;
    const resp = logFilter(patch, rw, rh, sigma);
    for (let i = 0; i < rw * rh; i++) {
      if (resp[i] < maxResponse[i]) {
        maxResponse[i] = resp[i];
        bestScale[i] = sigma;
      }
    }
  }

  const pixelSet = new Set(blob.pixels);
  const inBlob = new Uint8Array(rw * rh);
  for (let y = ry0; y <= ry1; y++) {
    for (let x = rx0; x <= rx1; x++) {
      if (pixelSet.has(y * w + x)) {
        inBlob[(y - ry0) * rw + (x - rx0)] = 1;
      }
    }
  }

  const blobResponses: number[] = [];
  for (let i = 0; i < rw * rh; i++) {
    if (inBlob[i]) blobResponses.push(maxResponse[i]);
  }
  if (blobResponses.length === 0) return [fallback];
  blobResponses.sort((a, b) => a - b);
  const threshold = blobResponses[Math.floor(blobResponses.length * 0.15)];

  const detections: { x: number; y: number; sigma: number; val: number }[] = [];
  const nmsRadius = Math.max(2, Math.round(sMin));

  for (let y = 1; y < rh - 1; y++) {
    for (let x = 1; x < rw - 1; x++) {
      if (!inBlob[y * rw + x]) continue;
      const val = maxResponse[y * rw + x];
      if (val > threshold) continue;

      let isMin = true;
      for (let dy = -1; dy <= 1 && isMin; dy++) {
        for (let dx = -1; dx <= 1 && isMin; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (maxResponse[(y + dy) * rw + (x + dx)] < val) isMin = false;
        }
      }
      if (isMin) {
        detections.push({ x, y, sigma: bestScale[y * rw + x], val });
      }
    }
  }

  detections.sort((a, b) => a.val - b.val);
  const kept: typeof detections = [];
  for (const d of detections) {
    let tooClose = false;
    for (const q of kept) {
      const dist = Math.sqrt((d.x - q.x) ** 2 + (d.y - q.y) ** 2);
      if (dist < nmsRadius) { tooClose = true; break; }
    }
    if (!tooClose) kept.push(d);
  }

  if (kept.length === 0) return [fallback];
  return kept.map(d => ({
    x: d.x + rx0,
    y: d.y + ry0,
    radius: d.sigma * Math.SQRT2,
    agreement: 0,
  }));
};

/** Count-only wrapper for backward compatibility. */
export const countByLoG = (
  gray: Uint8Array, w: number, h: number,
  blob: Blob, sigmaRange: [number, number], numScales = 5,
): number => locateByLoG(gray, w, h, blob, sigmaRange, numScales).length;

// ─── Ensemble Voter ──────────────────────────────────────────────────

export type CountingStrategy = 'distance-transform' | 'concavity' | 'log';

export interface EnsembleResult {
  /** Final estimated cell count for this blob. */
  cellCount: number;
  /** Individual strategy estimates. */
  votes: Record<CountingStrategy, number>;
  /** Merged cell locations with agreement scores. */
  locations: CellLocation[];
  /** Confidence: 1.0 if all agree, lower if they diverge. */
  confidence: number;
  /** Agreement-weighted count: sum of (agreement/3) per location. */
  weightedCount: number;
}

// ─── Spatial Location Merger ─────────────────────────────────────────

type TaggedLoc = CellLocation & { strategy: number };

/**
 * Merge cell locations from multiple strategies via greedy spatial clustering.
 * Each output location records how many strategies agreed (1–3).
 */
export const mergeLocations = (
  dtLocs: CellLocation[],
  concavLocs: CellLocation[],
  logLocs: CellLocation[],
  mergeRadius: number,
): CellLocation[] => {
  const all: TaggedLoc[] = [
    ...dtLocs.map(l => ({ ...l, strategy: 0 })),
    ...concavLocs.map(l => ({ ...l, strategy: 1 })),
    ...logLocs.map(l => ({ ...l, strategy: 2 })),
  ];
  if (all.length === 0) return [];

  // Process DT peaks first (most geometrically grounded), then LoG, then concavity
  const priority = [0, 2, 1];
  all.sort((a, b) => priority.indexOf(a.strategy) - priority.indexOf(b.strategy));

  const used = new Set<number>();
  const clusters: TaggedLoc[][] = [];

  for (let i = 0; i < all.length; i++) {
    if (used.has(i)) continue;

    const cluster: TaggedLoc[] = [all[i]];
    const strategies = new Set([all[i].strategy]);
    used.add(i);

    // Find nearest unassigned detection from each other strategy
    for (const s of [0, 1, 2]) {
      if (strategies.has(s)) continue;

      let bestJ = -1, bestDist = Infinity;
      const cx = cluster.reduce((sum, c) => sum + c.x, 0) / cluster.length;
      const cy = cluster.reduce((sum, c) => sum + c.y, 0) / cluster.length;

      for (let j = 0; j < all.length; j++) {
        if (used.has(j) || all[j].strategy !== s) continue;
        const dist = Math.sqrt((all[j].x - cx) ** 2 + (all[j].y - cy) ** 2);
        if (dist < mergeRadius && dist < bestDist) {
          bestDist = dist;
          bestJ = j;
        }
      }

      if (bestJ >= 0) {
        cluster.push(all[bestJ]);
        strategies.add(s);
        used.add(bestJ);
      }
    }

    clusters.push(cluster);
  }

  return clusters.map(cluster => {
    const x = cluster.reduce((s, c) => s + c.x, 0) / cluster.length;
    const y = cluster.reduce((s, c) => s + c.y, 0) / cluster.length;
    const radius = cluster.reduce((s, c) => s + c.radius, 0) / cluster.length;
    const strategies = new Set(cluster.map(c => c.strategy));
    return { x, y, radius, agreement: strategies.size };
  });
};

/**
 * Run the full ensemble on a single blob: locate + merge + vote.
 */
export const ensembleLocate = (
  blob: Blob,
  binary: Uint8Array,
  gray: Uint8Array,
  w: number,
  h: number,
  medianArea: number,
  clusterSplitRatio: number,
): EnsembleResult => {
  const cellRadius = Math.sqrt(medianArea / Math.PI);

  // Short-circuit: clearly a single cell (small AND circular)
  if (blob.area <= medianArea * clusterSplitRatio && blob.circularity > 0.70) {
    const loc: CellLocation = { x: blob.cx, y: blob.cy, radius: cellRadius, agreement: 3 };
    return {
      cellCount: 1,
      votes: { 'distance-transform': 1, concavity: 1, log: 1 },
      locations: [loc],
      confidence: 1.0,
      weightedCount: 1.0,
    };
  }

  const areaEstimate = Math.max(1, Math.round(blob.area / medianArea));
  const minDist = cellRadius * 0.55;

  // Strategy A: Distance Transform
  const dtLocs = locateByDistanceTransform(binary, w, h, blob, minDist);

  // Strategy B: Concavity
  const kNeighbour = Math.max(5, Math.round(cellRadius * 0.6));
  const concavLocs = locateByConcavity(blob, w, kNeighbour);

  // Strategy C: LoG
  const sigmaMin = cellRadius * 0.4;
  const sigmaMax = cellRadius * 1.2;
  const logLocs = locateByLoG(gray, w, h, blob, [sigmaMin, sigmaMax], 5);

  const dtCount = dtLocs.length;
  const concavityCount = concavLocs.length;
  const logCount = logLocs.length;

  // Merge all locations spatially
  const mergeRadius = cellRadius * 1.2;
  const merged = mergeLocations(dtLocs, concavLocs, logLocs, mergeRadius);

  // Count from merged locations
  const votes = [dtCount, concavityCount, logCount];
  votes.sort((a, b) => a - b);

  let finalCount = votes[1]; // median of 3

  const spread = votes[2] / Math.max(1, votes[0]);
  if (spread > 2.5) {
    const geoMean = Math.cbrt(dtCount * concavityCount * logCount);
    finalCount = Math.round(0.6 * geoMean + 0.4 * areaEstimate);
  }

  finalCount = Math.max(1, Math.min(finalCount, areaEstimate * 2));

  // Reconcile: if merged location count differs from finalCount,
  // trust the merged locations when agreement is strong
  const highAgreement = merged.filter(l => l.agreement >= 2);
  if (highAgreement.length >= finalCount) {
    // Strong spatial agreement: trust location count
    finalCount = merged.length;
  } else if (merged.length > finalCount) {
    // Too many detections: keep the top ones by agreement
    merged.sort((a, b) => b.agreement - a.agreement);
    merged.splice(finalCount);
  }

  // Ensure at least finalCount locations (pad with k-means if needed)
  while (merged.length < finalCount) {
    merged.push({ x: blob.cx, y: blob.cy, radius: cellRadius, agreement: 0 });
  }

  finalCount = Math.max(1, Math.min(finalCount, areaEstimate * 2));

  const mean = (dtCount + concavityCount + logCount) / 3;
  const stddev = Math.sqrt(
    ((dtCount - mean) ** 2 + (concavityCount - mean) ** 2 + (logCount - mean) ** 2) / 3,
  );
  const confidence = mean > 0 ? Math.max(0, 1 - stddev / mean) : 0;

  // Agreement-weighted count: each location contributes agreement/3
  const locs = merged.slice(0, finalCount);
  const weightedCount = locs.reduce((s, l) => s + l.agreement / 3, 0);

  return {
    cellCount: finalCount,
    votes: {
      'distance-transform': dtCount,
      concavity: concavityCount,
      log: logCount,
    },
    locations: locs,
    confidence,
    weightedCount,
  };
};

/** Simplified count-only interface (backward compat). */
export const ensembleCount = (
  blob: Blob,
  binary: Uint8Array,
  gray: Uint8Array,
  w: number,
  h: number,
  medianArea: number,
  clusterSplitRatio: number,
): Omit<EnsembleResult, 'locations'> => {
  const { cellCount, votes, confidence, weightedCount } = ensembleLocate(
    blob, binary, gray, w, h, medianArea, clusterSplitRatio,
  );
  return { cellCount, votes, confidence, weightedCount };
};

/**
 * Run the ensemble counter on all blobs, replacing cellCount assignments.
 * Sets both cellCount and cellLocations on each blob.
 * Optionally filters cell locations by minimum agreement threshold.
 */
export const ensembleClassifyBlobs = (
  blobs: Blob[],
  medianArea: number,
  circularityThresh: number,
  clusterSplitRatio: number,
  binary: Uint8Array,
  gray: Uint8Array,
  w: number,
  h: number,
  minSolidity = 0.5,
  maxEccentricity = 0.92,
  maxIntensityStdDev = 80,
  minAgreement = 1,
): Blob[] => {
  const minCirc = circularityThresh * 0.4;
  const maxAspect = 4.0;

  return blobs
    .filter(b => {
      if (b.circularity < minCirc) return false;
      const bw = b.bbox.maxX - b.bbox.minX + 1;
      const bh = b.bbox.maxY - b.bbox.minY + 1;
      if (Math.max(bw, bh) / Math.max(1, Math.min(bw, bh)) > maxAspect) return false;
      if (b.solidity < minSolidity) return false;
      if (b.eccentricity > maxEccentricity) return false;
      if (maxIntensityStdDev < Infinity && b.intensityStdDev > maxIntensityStdDev) return false;
      return true;
    })
    .map(b => {
      const result = ensembleLocate(
        b, binary, gray, w, h, medianArea, clusterSplitRatio,
      );
      // Filter locations by minimum agreement
      const filtered = minAgreement > 1
        ? result.locations.filter(l => l.agreement >= minAgreement)
        : result.locations;
      const count = filtered.length > 0 ? filtered.length : result.cellCount;
      return { ...b, cellCount: count, cellLocations: filtered.length > 0 ? filtered : result.locations };
    });
};

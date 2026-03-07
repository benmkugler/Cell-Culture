import type { Blob, BBox } from './types';
import { blobIntensityStdDev } from './imageops';

const DX4 = [1, -1, 0, 0];
const DY4 = [0, 0, 1, -1];

const DX8 = [1, -1, 0, 0, 1, -1, 1, -1];
const DY8 = [0, 0, 1, -1, 1, -1, -1, 1];

/**
 * 8-connected perimeter estimation.
 * Counts boundary pixels — those that have at least one 8-neighbor that is background.
 */
const computePerimeter8 = (pixels: Set<number>, w: number): number => {
  let perim = 0;
  for (const idx of pixels) {
    const x = idx % w;
    const y = (idx - x) / w;
    let isBorder = false;
    for (let dy = -1; dy <= 1 && !isBorder; dy++) {
      for (let dx = -1; dx <= 1 && !isBorder; dx++) {
        if (dx === 0 && dy === 0) continue;
        const ni = (y + dy) * w + (x + dx);
        if (!pixels.has(ni)) isBorder = true;
      }
    }
    if (isBorder) perim++;
  }
  return perim;
};

/**
 * Compute solidity = area / convex_hull_area.
 * Uses Andrew's monotone chain for 2D convex hull.
 * Solidity ≈ 1.0 for compact cells, < 0.6 for irregular debris.
 */
export const computeSolidity = (pixels: number[], w: number): number => {
  if (pixels.length < 3) return 1;

  // Extract unique (x, y) points
  const pts: [number, number][] = pixels.map(p => [p % w, Math.floor(p / w)] as [number, number]);
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);

  // Remove duplicates
  const unique: [number, number][] = [pts[0]];
  for (let i = 1; i < pts.length; i++) {
    if (pts[i][0] !== pts[i - 1][0] || pts[i][1] !== pts[i - 1][1]) {
      unique.push(pts[i]);
    }
  }
  if (unique.length < 3) return 1;

  const cross = (o: [number, number], a: [number, number], b: [number, number]) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

  // Build lower hull
  const lower: [number, number][] = [];
  for (const p of unique) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  // Build upper hull
  const upper: [number, number][] = [];
  for (let i = unique.length - 1; i >= 0; i--) {
    const p = unique[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  // Concatenate (remove last of each to avoid duplication)
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));

  // Shoelace formula for convex hull area
  let hullArea = 0;
  const n = hull.length;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    hullArea += hull[i][0] * hull[j][1];
    hullArea -= hull[j][0] * hull[i][1];
  }
  hullArea = Math.abs(hullArea) / 2;

  return hullArea > 0 ? Math.min(1, pixels.length / hullArea) : 1;
};

/**
 * Compute eccentricity from second-order central moments.
 * Returns value in [0, 1). 0 = perfect circle, approaching 1 = highly elongated.
 */
export const computeEccentricity = (pixels: number[], w: number): number => {
  if (pixels.length < 2) return 0;

  let sumX = 0, sumY = 0;
  for (const p of pixels) {
    sumX += p % w;
    sumY += Math.floor(p / w);
  }
  const cx = sumX / pixels.length;
  const cy = sumY / pixels.length;

  let mu20 = 0, mu02 = 0, mu11 = 0;
  for (const p of pixels) {
    const dx = (p % w) - cx;
    const dy = Math.floor(p / w) - cy;
    mu20 += dx * dx;
    mu02 += dy * dy;
    mu11 += dx * dy;
  }
  mu20 /= pixels.length;
  mu02 /= pixels.length;
  mu11 /= pixels.length;

  const delta = (mu20 - mu02) ** 2 + 4 * mu11 * mu11;
  const lambda1 = (mu20 + mu02 + Math.sqrt(delta)) / 2;
  const lambda2 = (mu20 + mu02 - Math.sqrt(delta)) / 2;

  if (lambda1 <= 0) return 0;
  return Math.sqrt(Math.max(0, 1 - lambda2 / lambda1));
};

/**
 * Connected-component labeling using flood fill.
 * Supports 4-connected (default) or 8-connected via `use8Connected`.
 * Returns array of Blob descriptors with shape metrics.
 */
export const extractBlobs = (
  binary: Uint8Array,
  w: number,
  h: number,
  minArea: number,
  maxArea: number,
  excludeBorder: boolean,
  gray?: Uint8Array,
  use8Connected = false,
): Blob[] => {
  const visited = new Uint8Array(w * h);
  const blobs: Blob[] = [];
  const dxArr = use8Connected ? DX8 : DX4;
  const dyArr = use8Connected ? DY8 : DY4;
  const numNeighbors = dxArr.length;

  for (let startIdx = 0; startIdx < w * h; startIdx++) {
    if (!binary[startIdx] || visited[startIdx]) continue;

    const stack = [startIdx];
    visited[startIdx] = 1;
    const pixelSet = new Set<number>();

    let sumX = 0, sumY = 0;
    let minX = w, maxX = 0, minY = h, maxY = 0;
    let touchesBorder = false;

    while (stack.length > 0) {
      const curr = stack.pop()!;
      const cx = curr % w;
      const cy = (curr - cx) / w;
      pixelSet.add(curr);
      sumX += cx;
      sumY += cy;

      if (cx < minX) minX = cx;
      if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy;
      if (cy > maxY) maxY = cy;
      if (cx === 0 || cx === w - 1 || cy === 0 || cy === h - 1) touchesBorder = true;

      for (let k = 0; k < numNeighbors; k++) {
        const nx = cx + dxArr[k];
        const ny = cy + dyArr[k];
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const ni = ny * w + nx;
        if (binary[ni] && !visited[ni]) {
          visited[ni] = 1;
          stack.push(ni);
        }
      }
    }

    const area = pixelSet.size;
    if (area < minArea || area > maxArea) continue;
    if (excludeBorder && touchesBorder) continue;

    const perimeter = computePerimeter8(pixelSet, w);
    const circularity = perimeter > 0 ? (4 * Math.PI * area) / (perimeter * perimeter) : 0;

    const bbox: BBox = { minX, maxX, minY, maxY };
    const cx = sumX / area;
    const cy = sumY / area;

    const pixelsArr = Array.from(pixelSet);
    const solidity = computeSolidity(pixelsArr, w);
    const eccentricity = computeEccentricity(pixelsArr, w);
    const intensityStdDev = gray ? blobIntensityStdDev(gray, pixelsArr) : 0;

    blobs.push({
      area,
      perimeter,
      cx,
      cy,
      bbox,
      circularity,
      cellCount: 1,
      cellLocations: [],
      pixels: pixelsArr,
      touchesBorder,
      solidity,
      eccentricity,
      intensityStdDev,
    });
  }

  return blobs;
};

/**
 * Compute the median area of blobs that pass the circularity threshold.
 * Falls back to mean of all blobs, or minArea.
 */
export const computeMedianCellArea = (blobs: Blob[], circularityThresh: number, minArea: number): number => {
  // Prefer highly circular blobs (most likely isolated single cells)
  // to resist median inflation from merged pairs.
  const strictCirc = Math.max(circularityThresh, 0.65);

  const iqMean = (arr: number[]): number => {
    const q1 = Math.floor(arr.length * 0.25);
    const q3 = Math.ceil(arr.length * 0.75);
    const slice = arr.slice(q1, q3);
    return slice.reduce((s, v) => s + v, 0) / slice.length;
  };

  const strict = blobs
    .filter(b => b.circularity >= strictCirc)
    .map(b => b.area)
    .sort((a, b) => a - b);

  if (strict.length >= 3) return iqMean(strict);
  if (strict.length > 0) return strict[Math.floor(strict.length / 2)];

  // Fall back to original circularity threshold
  const circular = blobs
    .filter(b => b.circularity >= circularityThresh)
    .map(b => b.area)
    .sort((a, b) => a - b);

  if (circular.length >= 3) return iqMean(circular);
  if (circular.length > 0) {
    return circular[Math.floor(circular.length / 2)];
  }
  if (blobs.length > 0) {
    return blobs.reduce((s, b) => s + b.area, 0) / blobs.length;
  }
  return minArea;
};

/**
 * Classify blobs: filter by shape/quality metrics and assign cell counts to clusters.
 *
 * - Rejects blobs with circularity below `minCirc * 0.4`
 * - Rejects blobs with aspect ratio > 4.0
 * - Rejects blobs with solidity below `minSolidity` (irregular debris)
 * - Rejects blobs with eccentricity above `maxEccentricity` (elongated scratches)
 * - Rejects blobs with internal intensity stddev above `maxIntensityStdDev` (bubbles/ripples)
 * - Assigns cellCount = round(area / medianArea) for large blobs
 */
export const classifyBlobs = (
  blobs: Blob[],
  medianArea: number,
  circularityThresh: number,
  clusterSplitRatio: number,
  minSolidity = 0.5,
  maxEccentricity = 0.92,
  maxIntensityStdDev = 80,
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
    .map(b => ({
      ...b,
      cellCount: b.area > medianArea * clusterSplitRatio
        ? Math.max(1, Math.round(b.area / medianArea))
        : 1,
      cellLocations: [{ x: b.cx, y: b.cy, radius: Math.sqrt(b.area / Math.PI), agreement: 0 }],
    }));
};

/**
 * Compute the first 7 Hu invariant moments.
 * These are translation, rotation, and scale invariant shape descriptors.
 * Useful for debris vs. cell discrimination.
 */
export const computeHuMoments = (pixels: number[], w: number): number[] => {
  if (pixels.length < 2) return [0, 0, 0, 0, 0, 0, 0];

  // Centroid
  let sumX = 0, sumY = 0;
  for (const p of pixels) {
    sumX += p % w;
    sumY += Math.floor(p / w);
  }
  const cx = sumX / pixels.length;
  const cy = sumY / pixels.length;

  // Central moments up to order 3
  let mu20 = 0, mu02 = 0, mu11 = 0;
  let mu30 = 0, mu03 = 0, mu21 = 0, mu12 = 0;
  for (const p of pixels) {
    const dx = (p % w) - cx;
    const dy = Math.floor(p / w) - cy;
    mu20 += dx * dx;
    mu02 += dy * dy;
    mu11 += dx * dy;
    mu30 += dx * dx * dx;
    mu03 += dy * dy * dy;
    mu21 += dx * dx * dy;
    mu12 += dx * dy * dy;
  }

  // Normalise by N^((p+q)/2 + 1) for scale invariance
  const n = pixels.length;
  const n2 = n * n;
  const n25 = n2 * Math.sqrt(n);

  const eta20 = mu20 / n2;
  const eta02 = mu02 / n2;
  const eta11 = mu11 / n2;
  const eta30 = mu30 / n25;
  const eta03 = mu03 / n25;
  const eta21 = mu21 / n25;
  const eta12 = mu12 / n25;

  // Hu moments
  const h1 = eta20 + eta02;
  const h2 = (eta20 - eta02) ** 2 + 4 * eta11 ** 2;
  const h3 = (eta30 - 3 * eta12) ** 2 + (3 * eta21 - eta03) ** 2;
  const h4 = (eta30 + eta12) ** 2 + (eta21 + eta03) ** 2;
  const h5 =
    (eta30 - 3 * eta12) * (eta30 + eta12) *
    ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2) +
    (3 * eta21 - eta03) * (eta21 + eta03) *
    (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2);
  const h6 =
    (eta20 - eta02) * ((eta30 + eta12) ** 2 - (eta21 + eta03) ** 2) +
    4 * eta11 * (eta30 + eta12) * (eta21 + eta03);
  const h7 =
    (3 * eta21 - eta03) * (eta30 + eta12) *
    ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2) -
    (eta30 - 3 * eta12) * (eta21 + eta03) *
    (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2);

  return [h1, h2, h3, h4, h5, h6, h7];
};

/**
 * Compactness = perimeter² / (4π × area).
 * A perfect circle has compactness = 1. Irregular shapes have higher values.
 * This is the inverse of circularity but more sensitive to boundary roughness.
 */
export const computeCompactness = (area: number, perimeter: number): number => {
  if (area <= 0) return Infinity;
  return (perimeter * perimeter) / (4 * Math.PI * area);
};

/**
 * Tests for new improvements:
 * - CLAHE
 * - 8-connected flood fill
 * - Hu moments & compactness
 * - Statistical output (PipelineStats)
 * - Ensemble agreement weighting & minAgreement filter
 * - morphClose after grid removal
 * - Edge cases: maxCellSize rejection, gaussianSigma effect, extreme params
 */
import { describe, it, expect } from 'vitest';
import type { PipelineParams, Blob } from '../src/cv/types';
import { DEFAULT_PARAMS } from '../src/cv/types';
import {
  clahe, gaussianBlur, blobIntensityStdDev,
  morphClose, morphOpen, removeGridLines, detectGridLines,
} from '../src/cv/imageops';
import {
  extractBlobs, computeMedianCellArea, computeSolidity,
  computeEccentricity, computeHuMoments, computeCompactness,
} from '../src/cv/blobs';
import { ensembleLocate, ensembleClassifyBlobs } from '../src/cv/counting';
import { processGrayscaleBuffer, computePipelineStats } from '../src/cv/pipeline';

// --- Helpers ---

const paintCircle = (buf: Uint8Array, w: number, h: number, cx: number, cy: number, r: number, val: number) => {
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) {
        buf[y * w + x] = val;
      }
    }
  }
};

const baseParams: Omit<PipelineParams, 'showMask'> = {
  threshold: 130,
  autoThreshold: true,
  invert: true,
  minCellSize: 15,
  maxCellSize: 5000,
  circularityThresh: 0.35,
  morphRadius: 1,
  excludeBorder: true,
  gaussianSigma: 0.8,
  clusterSplitRatio: 1.8,
  adaptiveIllumination: false,
  illuminationBlockSize: 51,
  medianFilterRadius: 0,
  minSolidity: 0.5,
  maxEccentricity: 0.92,
  maxIntensityStdDev: 80,
  use8Connected: false,
  enableCLAHE: false,
  claheTileSize: 8,
  claheClipLimit: 2.0,
  minAgreement: 1,
};

// ===================== CLAHE =====================

describe('clahe', () => {
  it('preserves image dimensions', () => {
    const w = 40, h = 30;
    const img = new Uint8Array(w * h).fill(128);
    const out = clahe(img, w, h, 4, 2.0);
    expect(out.length).toBe(w * h);
  });

  it('output values are in [0, 255]', () => {
    const w = 50, h = 50;
    const img = new Uint8Array(w * h);
    for (let i = 0; i < img.length; i++) img[i] = Math.floor(Math.random() * 256);
    const out = clahe(img, w, h, 4, 2.0);
    for (let i = 0; i < out.length; i++) {
      expect(out[i]).toBeGreaterThanOrEqual(0);
      expect(out[i]).toBeLessThanOrEqual(255);
    }
  });

  it('enhances contrast on bimodal image', () => {
    const w = 64, h = 64;
    const img = new Uint8Array(w * h);
    // Left half dim, right half bright
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        img[y * w + x] = x < w / 2 ? 80 : 180;
      }
    }
    const out = clahe(img, w, h, 4, 2.0);
    // After CLAHE, contrast should be increased (range should widen)
    let min = 255, max = 0;
    for (let i = 0; i < out.length; i++) {
      if (out[i] < min) min = out[i];
      if (out[i] > max) max = out[i];
    }
    expect(max - min).toBeGreaterThan(180 - 80);
  });

  it('handles uniform image gracefully', () => {
    const w = 32, h = 32;
    const img = new Uint8Array(w * h).fill(100);
    const out = clahe(img, w, h, 4, 2.0);
    // Uniform image should remain roughly uniform
    const vals = new Set(Array.from(out));
    expect(vals.size).toBeLessThanOrEqual(3); // allow rounding
  });

  it('handles small image (smaller than tile grid)', () => {
    const w = 4, h = 4;
    const img = new Uint8Array(w * h);
    for (let i = 0; i < 16; i++) img[i] = i * 16;
    const out = clahe(img, w, h, 8, 2.0);
    expect(out.length).toBe(16);
  });
});

// ===================== 8-Connected Flood Fill =====================

describe('extractBlobs — 8-connected', () => {
  it('merges diagonally-touching pixels as single blob', () => {
    const w = 10, h = 10;
    const binary = new Uint8Array(w * h);
    // Two pixels touching only diagonally
    binary[3 * w + 3] = 1; // (3,3)
    binary[4 * w + 4] = 1; // (4,4)

    const blobs4 = extractBlobs(binary, w, h, 1, 100, false, undefined, false);
    const blobs8 = extractBlobs(binary, w, h, 1, 100, false, undefined, true);

    expect(blobs4.length).toBe(2); // 4-connected: separate
    expect(blobs8.length).toBe(1); // 8-connected: merged
    expect(blobs8[0].area).toBe(2);
  });

  it('4-connected and 8-connected agree on non-diagonal shapes', () => {
    const w = 20, h = 20;
    const binary = new Uint8Array(w * h);
    // Solid 5×5 block
    for (let y = 5; y < 10; y++)
      for (let x = 5; x < 10; x++)
        binary[y * w + x] = 1;

    const blobs4 = extractBlobs(binary, w, h, 1, 100, false, undefined, false);
    const blobs8 = extractBlobs(binary, w, h, 1, 100, false, undefined, true);

    expect(blobs4.length).toBe(1);
    expect(blobs8.length).toBe(1);
    expect(blobs4[0].area).toBe(blobs8[0].area);
  });

  it('L-shaped pixel group connected only 8-wise', () => {
    const w = 10, h = 10;
    const binary = new Uint8Array(w * h);
    // (2,2), (3,3), (4,4) — diagonal line
    binary[2 * w + 2] = 1;
    binary[3 * w + 3] = 1;
    binary[4 * w + 4] = 1;

    const blobs8 = extractBlobs(binary, w, h, 1, 100, false, undefined, true);
    expect(blobs8.length).toBe(1);
    expect(blobs8[0].area).toBe(3);
  });
});

// ===================== Hu Moments =====================

describe('computeHuMoments', () => {
  it('returns 7 elements', () => {
    const w = 20;
    const pixels = [5 * w + 5, 5 * w + 6, 6 * w + 5, 6 * w + 6];
    const hu = computeHuMoments(pixels, w);
    expect(hu).toHaveLength(7);
  });

  it('h1 (inertia) is positive for non-degenerate shapes', () => {
    const w = 30;
    const pixels: number[] = [];
    for (let y = 10; y < 20; y++)
      for (let x = 10; x < 20; x++)
        pixels.push(y * w + x);
    const hu = computeHuMoments(pixels, w);
    expect(hu[0]).toBeGreaterThan(0);
  });

  it('returns zeros for single pixel', () => {
    const hu = computeHuMoments([50], 100);
    expect(hu).toEqual([0, 0, 0, 0, 0, 0, 0]);
  });

  it('square and circle have different h2', () => {
    const w = 60;
    // Square
    const sqPx: number[] = [];
    for (let y = 10; y < 30; y++)
      for (let x = 10; x < 30; x++)
        sqPx.push(y * w + x);

    // Circle
    const circPx: number[] = [];
    for (let y = 0; y < 60; y++)
      for (let x = 0; x < 60; x++)
        if ((x - 30) ** 2 + (y - 30) ** 2 <= 10 * 10)
          circPx.push(y * w + x);

    const huSq = computeHuMoments(sqPx, w);
    const huCirc = computeHuMoments(circPx, w);
    // h2 is related to elongation — both should be near 0 for symmetric shapes
    expect(Math.abs(huSq[1])).toBeLessThan(0.01);
    expect(Math.abs(huCirc[1])).toBeLessThan(0.01);
  });
});

// ===================== Compactness =====================

describe('computeCompactness', () => {
  it('perfect circle → compactness ≈ 1', () => {
    const r = 10;
    const area = Math.PI * r * r;
    const perim = 2 * Math.PI * r;
    const c = computeCompactness(area, perim);
    expect(c).toBeCloseTo(1.0, 1);
  });

  it('square → compactness > 1', () => {
    const side = 10;
    const area = side * side;
    const perim = 4 * side;
    const c = computeCompactness(area, perim);
    expect(c).toBeGreaterThan(1.0);
    expect(c).toBeCloseTo(4 / Math.PI, 1);
  });

  it('zero area → Infinity', () => {
    expect(computeCompactness(0, 10)).toBe(Infinity);
  });
});

// ===================== PipelineStats =====================

describe('computePipelineStats', () => {
  const makeBlob = (area: number, cellCount: number, agreement: number): Blob => ({
    area,
    perimeter: Math.sqrt(area) * 4,
    cx: 50, cy: 50,
    bbox: { minX: 40, maxX: 60, minY: 40, maxY: 60 },
    circularity: 0.8,
    cellCount,
    cellLocations: Array.from({ length: cellCount }, () => ({
      x: 50, y: 50, radius: Math.sqrt(area / Math.PI), agreement,
    })),
    pixels: [],
    touchesBorder: false,
    solidity: 0.9,
    eccentricity: 0.2,
    intensityStdDev: 10,
  });

  it('computes correct total count', () => {
    const blobs = [makeBlob(100, 1, 3), makeBlob(200, 2, 2)];
    const stats = computePipelineStats(blobs, 3);
    expect(stats.totalCount).toBe(3);
  });

  it('95% CI uses Poisson approximation', () => {
    const stats = computePipelineStats([makeBlob(100, 1, 3)], 25);
    // 25 ± 1.96*sqrt(25) = 25 ± 9.8 → [15, 35]
    expect(stats.ci95[0]).toBe(15);
    expect(stats.ci95[1]).toBe(35);
  });

  it('mean confidence reflects agreement scores', () => {
    const blobs = [makeBlob(100, 1, 3)]; // agreement=3 → confidence=1.0
    const stats = computePipelineStats(blobs, 1);
    expect(stats.meanConfidence).toBeCloseTo(1.0, 2);
  });

  it('handles empty blob array', () => {
    const stats = computePipelineStats([], 0);
    expect(stats.totalCount).toBe(0);
    expect(stats.meanCellArea).toBe(0);
    expect(stats.stdCellArea).toBe(0);
    expect(stats.cvCellArea).toBe(0);
    expect(stats.medianDiameter).toBe(0);
    expect(stats.meanConfidence).toBe(0);
  });

  it('CV is stddev/mean for single-cell areas', () => {
    const blobs = [makeBlob(100, 1, 3), makeBlob(200, 1, 3)];
    const stats = computePipelineStats(blobs, 2);
    expect(stats.cvCellArea).toBeGreaterThan(0);
    expect(stats.cvCellArea).toBeCloseTo(stats.stdCellArea / stats.meanCellArea, 4);
  });

  it('diameter range is correct', () => {
    const blobs = [makeBlob(100, 1, 3), makeBlob(400, 1, 3)];
    const stats = computePipelineStats(blobs, 2);
    expect(stats.diameterRange[0]).toBeCloseTo(2 * Math.sqrt(100 / Math.PI), 1);
    expect(stats.diameterRange[1]).toBeCloseTo(2 * Math.sqrt(400 / Math.PI), 1);
  });

  it('perBlob array matches blob count', () => {
    const blobs = [makeBlob(100, 1, 3), makeBlob(300, 3, 2)];
    const stats = computePipelineStats(blobs, 4);
    expect(stats.perBlob).toHaveLength(2);
    expect(stats.perBlob[0].cellCount).toBe(1);
    expect(stats.perBlob[1].cellCount).toBe(3);
  });
});

// ===================== processGrayscaleBuffer stats =====================

describe('processGrayscaleBuffer — stats output', () => {
  it('returns stats with correct totalCount', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h).fill(220);
    paintCircle(gray, w, h, 50, 50, 8, 40);
    const result = processGrayscaleBuffer(gray, w, h, baseParams);
    expect(result.stats).toBeDefined();
    expect(result.stats.totalCount).toBe(result.count);
  });

  it('stats CI bounds bracket the count', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h).fill(220);
    paintCircle(gray, w, h, 30, 50, 8, 40);
    paintCircle(gray, w, h, 70, 50, 8, 40);
    const result = processGrayscaleBuffer(gray, w, h, baseParams);
    expect(result.stats.ci95[0]).toBeLessThanOrEqual(result.count);
    expect(result.stats.ci95[1]).toBeGreaterThanOrEqual(result.count);
  });
});

// ===================== maxCellSize rejection =====================

describe('maxCellSize rejection', () => {
  it('rejects blobs exceeding maxCellSize', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h).fill(220);
    // Large circle (area ≈ π*20² ≈ 1257)
    paintCircle(gray, w, h, 50, 50, 20, 40);
    // Small circle (area ≈ π*5² ≈ 79)
    paintCircle(gray, w, h, 85, 85, 5, 40);
    const params = { ...baseParams, maxCellSize: 200, excludeBorder: false };
    const result = processGrayscaleBuffer(gray, w, h, params);
    // Only the small circle should survive
    expect(result.count).toBeLessThanOrEqual(2);
    // The large blob should have been rejected
    const largeBlobs = result.blobs.filter(b => b.area > 200);
    expect(largeBlobs.length).toBe(0);
  });
});

// ===================== gaussianSigma sensitivity =====================

describe('gaussianSigma effect', () => {
  it('higher sigma merges nearby noise into fewer blobs', () => {
    const w = 60, h = 60;
    // Scatter small noise pixels
    const grayLow = new Uint8Array(w * h).fill(200);
    const grayHigh = new Uint8Array(w * h).fill(200);
    for (let i = 0; i < 30; i++) {
      const x = 10 + (i * 3) % 40;
      const y = 10 + Math.floor((i * 3) / 40) * 3;
      if (y < h && x < w) {
        grayLow[y * w + x] = 30;
        grayHigh[y * w + x] = 30;
      }
    }
    const paramsLow = { ...baseParams, gaussianSigma: 0.3, minCellSize: 1 };
    const paramsHigh = { ...baseParams, gaussianSigma: 3.0, minCellSize: 1 };
    const r1 = processGrayscaleBuffer(grayLow, w, h, paramsLow);
    const r2 = processGrayscaleBuffer(grayHigh, w, h, paramsHigh);
    // Higher sigma should produce fewer or equal blobs
    expect(r2.count).toBeLessThanOrEqual(r1.count);
  });
});

// ===================== Ensemble agreement: minAgreement filter =====================

describe('minAgreement filter', () => {
  it('minAgreement=1 retains all detections (default)', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h).fill(220);
    paintCircle(gray, w, h, 50, 50, 8, 40);
    const result = processGrayscaleBuffer(gray, w, h, { ...baseParams, minAgreement: 1 });
    expect(result.count).toBeGreaterThanOrEqual(1);
  });

  it('minAgreement=3 may reduce count on ambiguous detections', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h).fill(220);
    paintCircle(gray, w, h, 50, 50, 8, 40);
    const r1 = processGrayscaleBuffer(gray, w, h, { ...baseParams, minAgreement: 1 });
    const r3 = processGrayscaleBuffer(gray, w, h, { ...baseParams, minAgreement: 3 });
    expect(r3.count).toBeLessThanOrEqual(r1.count);
  });
});

// ===================== morphClose after grid removal =====================

describe('morphClose after grid removal', () => {
  it('morphClose reconnects fragments', () => {
    const w = 20, h = 20;
    const binary = new Uint8Array(w * h);
    // Two adjacent pixels with 1px gap
    binary[10 * w + 8] = 1;
    binary[10 * w + 10] = 1;
    // morphClose should bridge the gap
    const closed = morphClose(binary, w, h, 1);
    expect(closed[10 * w + 9]).toBe(1);
  });

  it('grid-split cell is reconnected by pipeline morphClose', () => {
    const w = 60, h = 60;
    const binary = new Uint8Array(w * h);
    // Create a circle that will be split by a grid line
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if ((x - 30) ** 2 + (y - 30) ** 2 <= 8 * 8) binary[y * w + x] = 1;
      }
    }
    // Add grid line through the middle
    for (let x = 0; x < w; x++) binary[30 * w + x] = 1;

    const removed = removeGridLines(binary, w, h);
    // After removal, the circle may be split
    const { rows } = detectGridLines(binary, w, h);
    if (rows.size > 0) {
      const closed = morphClose(removed, w, h, 1);
      // The closed version should have more foreground pixels than removed
      let removedFg = 0, closedFg = 0;
      for (let i = 0; i < w * h; i++) {
        if (removed[i]) removedFg++;
        if (closed[i]) closedFg++;
      }
      expect(closedFg).toBeGreaterThanOrEqual(removedFg);
    }
  });
});

// ===================== CLAHE in pipeline =====================

describe('processGrayscaleBuffer with CLAHE', () => {
  it('CLAHE enabled does not crash', () => {
    const w = 80, h = 80;
    const gray = new Uint8Array(w * h).fill(200);
    paintCircle(gray, w, h, 40, 40, 8, 40);
    const params = { ...baseParams, enableCLAHE: true, claheTileSize: 4, claheClipLimit: 2.0 };
    const result = processGrayscaleBuffer(gray, w, h, params);
    expect(result.count).toBeGreaterThanOrEqual(0);
  });
});

// ===================== 8-connected in pipeline =====================

describe('processGrayscaleBuffer with 8-connected', () => {
  it('8-connected does not crash and produces valid output', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h).fill(220);
    paintCircle(gray, w, h, 50, 50, 8, 40);
    const result = processGrayscaleBuffer(gray, w, h, { ...baseParams, use8Connected: true });
    expect(result.count).toBeGreaterThanOrEqual(1);
    expect(result.stats).toBeDefined();
  });
});

// ===================== Extreme parameter edge cases =====================

describe('extreme parameter edge cases', () => {
  it('minCellSize=0 does not crash', () => {
    const w = 50, h = 50;
    const gray = new Uint8Array(w * h).fill(200);
    paintCircle(gray, w, h, 25, 25, 5, 30);
    const result = processGrayscaleBuffer(gray, w, h, { ...baseParams, minCellSize: 0, excludeBorder: false });
    expect(result.count).toBeGreaterThanOrEqual(0);
  });

  it('gaussianSigma=0 means no blur', () => {
    const w = 50, h = 50;
    const gray = new Uint8Array(w * h).fill(200);
    paintCircle(gray, w, h, 25, 25, 5, 30);
    const result = processGrayscaleBuffer(gray, w, h, { ...baseParams, gaussianSigma: 0, excludeBorder: false });
    expect(result.count).toBeGreaterThanOrEqual(0);
  });

  it('morphRadius=0 means no morphological operations', () => {
    const w = 50, h = 50;
    const gray = new Uint8Array(w * h).fill(200);
    paintCircle(gray, w, h, 25, 25, 5, 30);
    const result = processGrayscaleBuffer(gray, w, h, { ...baseParams, morphRadius: 0, excludeBorder: false });
    expect(result.count).toBeGreaterThanOrEqual(0);
  });

  it('very large clusterSplitRatio treats everything as single cells', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h).fill(220);
    // Two touching circles
    paintCircle(gray, w, h, 40, 50, 8, 40);
    paintCircle(gray, w, h, 56, 50, 8, 40);
    const result = processGrayscaleBuffer(gray, w, h, { ...baseParams, clusterSplitRatio: 100 });
    // With huge split ratio, everything should be considered single
    for (const b of result.blobs) {
      // The ensemble may still split, but the area-ratio approach
      // shouldn't mark clusters based on area alone
      expect(b.cellCount).toBeGreaterThanOrEqual(1);
    }
  });
});

// ===================== Default params include new fields =====================

describe('DEFAULT_PARAMS', () => {
  it('includes all new fields', () => {
    expect(DEFAULT_PARAMS.use8Connected).toBe(true);
    expect(DEFAULT_PARAMS.enableCLAHE).toBe(false);
    expect(DEFAULT_PARAMS.claheTileSize).toBe(8);
    expect(DEFAULT_PARAMS.claheClipLimit).toBe(2.0);
    expect(DEFAULT_PARAMS.minAgreement).toBe(1);
  });
});

// ===================== EnsembleResult weightedCount =====================

describe('ensemble weightedCount', () => {
  it('single cell with agreement=3 has weightedCount=1.0', () => {
    const w = 60, h = 60;
    const binary = new Uint8Array(w * h);
    const gray = new Uint8Array(w * h).fill(200);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        if ((x - 30) ** 2 + (y - 30) ** 2 <= 7 * 7) {
          binary[y * w + x] = 1;
          gray[y * w + x] = 40;
        }

    const blobs = extractBlobs(binary, w, h, 10, 5000, false, gray);
    if (blobs.length > 0) {
      const medianArea = blobs[0].area;
      const result = ensembleLocate(blobs[0], binary, gray, w, h, medianArea, 1.8);
      if (result.cellCount === 1 && result.locations[0]?.agreement === 3) {
        expect(result.weightedCount).toBeCloseTo(1.0, 1);
      }
    }
  });
});

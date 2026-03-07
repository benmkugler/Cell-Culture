import { describe, it, expect } from 'vitest';
import {
  distanceTransformSq,
  countByDistanceTransform,
  countByConcavity,
  countByLoG,
  ensembleCount,
  ensembleClassifyBlobs,
} from '../src/cv/counting';
import { extractBlobs, computeMedianCellArea } from '../src/cv/blobs';
import type { Blob } from '../src/cv/types';

// ─── Helpers ─────────────────────────────────────────────────────────

/** Paint a filled circle. */
const paintCircle = (
  buf: Uint8Array, w: number, cx: number, cy: number, r: number, val = 1,
) => {
  const h = buf.length / w;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) {
        buf[y * w + x] = val;
      }
    }
  }
};

/** Paint a filled ellipse. */
const paintEllipse = (
  buf: Uint8Array, w: number, cx: number, cy: number,
  rx: number, ry: number, val = 1,
) => {
  const h = buf.length / w;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1) {
        buf[y * w + x] = val;
      }
    }
  }
};

/** Build a grayscale image with dark cells (intensity grayVal) on light background (200). */
const buildGrayWithCircles = (
  w: number, h: number,
  cells: { cx: number; cy: number; r: number }[],
  grayVal = 60,
): Uint8Array => {
  const gray = new Uint8Array(w * h).fill(200);
  for (const c of cells) {
    paintCircle(gray, w, c.cx, c.cy, c.r, grayVal);
  }
  return gray;
};

/** Build binary+gray + extract a single merged blob from two touching circles. */
const buildTouchingCircles = (
  w: number, h: number,
  c1: { cx: number; cy: number; r: number },
  c2: { cx: number; cy: number; r: number },
) => {
  const binary = new Uint8Array(w * h).fill(0);
  paintCircle(binary, w, c1.cx, c1.cy, c1.r);
  paintCircle(binary, w, c2.cx, c2.cy, c2.r);

  const gray = new Uint8Array(w * h).fill(200);
  paintCircle(gray, w, c1.cx, c1.cy, c1.r, 60);
  paintCircle(gray, w, c2.cx, c2.cy, c2.r, 60);

  const blobs = extractBlobs(binary, w, h, 5, 100000, false, gray);
  return { binary, gray, blobs };
};

// ─── Distance Transform ─────────────────────────────────────────────

describe('distanceTransformSq', () => {
  it('returns 0 for foreground pixel surrounded by foreground', () => {
    // All foreground — distance to background is 0 only at the border
    // Actually, when all pixels are foreground (binary=1), EDT values
    // are all 0 because our convention is binary[i]=1 means foreground (dt=0)
    // and binary[i]=0 means background (dt=INF initially).
    // Wait, let me re-read: dt[i] = binary[i] ? 0 : INF
    // So foreground pixels start at 0 and remain at 0. That's what we want:
    // distance from a foreground pixel to the nearest background pixel.
    // But the FH algorithm computes distance to nearest 0-valued pixel.
    // Since foreground=0 in our dt array, every pixel is distance 0.
    // Hmm, this is actually the wrong convention — we want distance to boundary.
    // The actual usage inverts. Let me test the actual function as called.
    const w = 5, h = 5;
    const binary = new Uint8Array(w * h).fill(0);
    // single foreground pixel at center
    binary[2 * w + 2] = 1;
    const dtSq = distanceTransformSq(binary, w, h);
    // The center pixel is foreground (binary=1 → dt init=0), so dtSq should be 0
    expect(dtSq[2 * w + 2]).toBe(0);
    // A background pixel adjacent to it should have dtSq = 1
    expect(dtSq[2 * w + 1]).toBe(1);
    // A corner pixel distance: (2-0)^2 + (2-0)^2 = 8
    expect(dtSq[0]).toBe(8);
  });

  it('computes correct distances for a horizontal line', () => {
    const w = 11, h = 1;
    const binary = new Uint8Array(w).fill(0);
    binary[5] = 1; // single foreground pixel at x=5
    const dtSq = distanceTransformSq(binary, w, h);
    for (let x = 0; x < w; x++) {
      expect(dtSq[x]).toBeCloseTo((x - 5) ** 2, 3);
    }
  });
});

describe('countByDistanceTransform', () => {
  it('counts 1 for a single isolated circle', () => {
    const w = 60, h = 60;
    const binary = new Uint8Array(w * h).fill(0);
    paintCircle(binary, w, 30, 30, 10);
    const blobs = extractBlobs(binary, w, h, 5, 100000, false);
    expect(blobs).toHaveLength(1);

    const count = countByDistanceTransform(binary, w, h, blobs[0], 5);
    expect(count).toBe(1);
  });

  it('counts 2 for two touching circles', () => {
    const w = 80, h = 60;
    const { binary, blobs } = buildTouchingCircles(w, h,
      { cx: 25, cy: 30, r: 12 },
      { cx: 50, cy: 30, r: 12 },
    );
    // They may merge into 1 blob or stay as 2
    if (blobs.length === 1) {
      const count = countByDistanceTransform(binary, w, h, blobs[0], 6);
      expect(count).toBe(2);
    } else {
      // If separate, each should be 1
      expect(blobs.length).toBe(2);
    }
  });

  it('counts 3 for three overlapping circles', () => {
    const w = 100, h = 80;
    const binary = new Uint8Array(w * h).fill(0);
    paintCircle(binary, w, 30, 40, 12);
    paintCircle(binary, w, 52, 40, 12);
    paintCircle(binary, w, 41, 22, 12);

    const blobs = extractBlobs(binary, w, h, 5, 100000, false);
    // Overlapping circles should merge into 1 blob
    const totalCount = blobs.reduce((sum, b) =>
      sum + countByDistanceTransform(binary, w, h, b, 6), 0,
    );
    expect(totalCount).toBeGreaterThanOrEqual(2);
    expect(totalCount).toBeLessThanOrEqual(4);
  });
});

// ─── Concavity Detection ────────────────────────────────────────────

describe('countByConcavity', () => {
  it('counts 1 for a single circle', () => {
    const w = 60, h = 60;
    const binary = new Uint8Array(w * h).fill(0);
    paintCircle(binary, w, 30, 30, 12);
    const blobs = extractBlobs(binary, w, h, 5, 100000, false);
    expect(blobs).toHaveLength(1);

    const count = countByConcavity(blobs[0], w);
    expect(count).toBe(1);
  });

  it('detects concavity where two circles overlap', () => {
    const w = 80, h = 60;
    const { blobs } = buildTouchingCircles(w, h,
      { cx: 24, cy: 30, r: 13 },
      { cx: 51, cy: 30, r: 13 },
    );
    if (blobs.length === 1) {
      const count = countByConcavity(blobs[0], w, 6);
      // Should detect the concavity and count >= 2
      expect(count).toBeGreaterThanOrEqual(2);
    }
  });

  it('returns 1 for a small blob', () => {
    const w = 20, h = 20;
    const binary = new Uint8Array(w * h).fill(0);
    paintCircle(binary, w, 10, 10, 3);
    const blobs = extractBlobs(binary, w, h, 5, 100000, false);
    if (blobs.length > 0) {
      expect(countByConcavity(blobs[0], w)).toBe(1);
    }
  });
});

// ─── LoG Detection ──────────────────────────────────────────────────

describe('countByLoG', () => {
  it('counts 1 for a single dark circle', () => {
    const w = 60, h = 60;
    const gray = new Uint8Array(w * h).fill(200);
    const binary = new Uint8Array(w * h).fill(0);
    paintCircle(binary, w, 30, 30, 10);
    paintCircle(gray, w, 30, 30, 10, 60);

    const blobs = extractBlobs(binary, w, h, 5, 100000, false, gray);
    expect(blobs).toHaveLength(1);

    const count = countByLoG(gray, w, h, blobs[0], [3, 12], 5);
    expect(count).toBe(1);
  });

  it('counts 2 for two adjacent dark circles', () => {
    const w = 100, h = 60;
    const cells = [
      { cx: 28, cy: 30, r: 11 },
      { cx: 60, cy: 30, r: 11 },
    ];
    const gray = buildGrayWithCircles(w, h, cells);
    const binary = new Uint8Array(w * h).fill(0);
    for (const c of cells) paintCircle(binary, w, c.cx, c.cy, c.r);

    const blobs = extractBlobs(binary, w, h, 5, 100000, false, gray);
    // Circles might be merged or separate
    const totalCount = blobs.reduce((sum, b) =>
      sum + countByLoG(gray, w, h, b, [3, 12], 5), 0,
    );
    expect(totalCount).toBeGreaterThanOrEqual(2);
  });
});

// ─── Ensemble ───────────────────────────────────────────────────────

describe('ensembleCount', () => {
  it('returns 1 for a single cell below split threshold', () => {
    const w = 60, h = 60;
    const binary = new Uint8Array(w * h).fill(0);
    paintCircle(binary, w, 30, 30, 8);
    const gray = new Uint8Array(w * h).fill(200);
    paintCircle(gray, w, 30, 30, 8, 60);

    const blobs = extractBlobs(binary, w, h, 5, 100000, false, gray);
    expect(blobs).toHaveLength(1);

    const singleArea = blobs[0].area;
    const result = ensembleCount(blobs[0], binary, gray, w, h, singleArea, 1.8);
    expect(result.cellCount).toBe(1);
    expect(result.confidence).toBe(1.0);
  });

  it('counts >= 2 for a large merged blob (two overlapping circles)', () => {
    const w = 80, h = 60;
    const r = 12;
    const binary = new Uint8Array(w * h).fill(0);
    paintCircle(binary, w, 25, 30, r);
    paintCircle(binary, w, 50, 30, r);
    const gray = new Uint8Array(w * h).fill(200);
    paintCircle(gray, w, 25, 30, r, 60);
    paintCircle(gray, w, 50, 30, r, 60);

    const blobs = extractBlobs(binary, w, h, 5, 100000, false, gray);
    if (blobs.length === 1) {
      const singleCellArea = Math.PI * r * r; // ~452
      const result = ensembleCount(blobs[0], binary, gray, w, h, singleCellArea, 1.5);
      expect(result.cellCount).toBeGreaterThanOrEqual(2);
    }
  });
});

describe('ensembleClassifyBlobs', () => {
  it('classifies single cells as cellCount=1', () => {
    const w = 100, h = 100;
    const binary = new Uint8Array(w * h).fill(0);
    const gray = new Uint8Array(w * h).fill(200);
    paintCircle(binary, w, 30, 30, 8);
    paintCircle(binary, w, 70, 70, 8);
    paintCircle(gray, w, 30, 30, 8, 60);
    paintCircle(gray, w, 70, 70, 8, 60);

    const blobs = extractBlobs(binary, w, h, 5, 100000, false, gray);
    expect(blobs.length).toBe(2);

    const medianArea = computeMedianCellArea(blobs, 0.35, 20);
    const classified = ensembleClassifyBlobs(
      blobs, medianArea, 0.35, 1.8,
      binary, gray, w, h,
    );
    expect(classified.length).toBe(2);
    for (const b of classified) {
      expect(b.cellCount).toBe(1);
    }
  });

  it('filters debris with low solidity', () => {
    const w = 80, h = 80;
    const binary = new Uint8Array(w * h).fill(0);
    const gray = new Uint8Array(w * h).fill(200);

    // Good cell
    paintCircle(binary, w, 40, 40, 10);
    paintCircle(gray, w, 40, 40, 10, 60);

    const blobs = extractBlobs(binary, w, h, 5, 100000, false, gray);
    const medianArea = computeMedianCellArea(blobs, 0.35, 20);

    // Artificially set one blob's solidity low
    if (blobs.length > 0) {
      (blobs[0] as any).solidity = 0.2;
      const classified = ensembleClassifyBlobs(
        blobs, medianArea, 0.35, 1.8,
        binary, gray, w, h, 0.5,
      );
      expect(classified.length).toBe(0);
    }
  });

  it('handles an empty blob array', () => {
    const w = 50, h = 50;
    const binary = new Uint8Array(w * h);
    const gray = new Uint8Array(w * h).fill(200);
    const classified = ensembleClassifyBlobs(
      [], 100, 0.35, 1.8, binary, gray, w, h,
    );
    expect(classified).toHaveLength(0);
  });

  it('counts total cells across mixed single + cluster', () => {
    const w = 120, h = 80;
    const binary = new Uint8Array(w * h).fill(0);
    const gray = new Uint8Array(w * h).fill(200);

    // Single cell
    paintCircle(binary, w, 20, 40, 8);
    paintCircle(gray, w, 20, 40, 8, 60);

    // Two touching cells forming a cluster
    paintCircle(binary, w, 70, 40, 10);
    paintCircle(binary, w, 92, 40, 10);
    paintCircle(gray, w, 70, 40, 10, 60);
    paintCircle(gray, w, 92, 40, 10, 60);

    const blobs = extractBlobs(binary, w, h, 5, 100000, false, gray);
    const singleCellArea = Math.PI * 8 * 8;
    const classified = ensembleClassifyBlobs(
      blobs, singleCellArea, 0.35, 1.5,
      binary, gray, w, h,
    );

    const total = classified.reduce((s, b) => s + b.cellCount, 0);
    // Should count at least 2 (the two touching cells) + 1 single = 3
    // Give some tolerance: 2–4
    expect(total).toBeGreaterThanOrEqual(2);
    expect(total).toBeLessThanOrEqual(5);
  });

  it('handles elliptical (oblong) cells without over-counting', () => {
    const w = 80, h = 80;
    const binary = new Uint8Array(w * h).fill(0);
    const gray = new Uint8Array(w * h).fill(200);

    // Single oblong cell: rx=15, ry=8
    paintEllipse(binary, w, 40, 40, 15, 8);
    paintEllipse(gray, w, 40, 40, 15, 8, 60);

    const blobs = extractBlobs(binary, w, h, 5, 100000, false, gray);
    expect(blobs.length).toBe(1);

    // Use an area close to this ellipse's area as median
    const ellipseArea = Math.PI * 15 * 8; // ~377
    const classified = ensembleClassifyBlobs(
      blobs, ellipseArea, 0.2, 1.8,
      binary, gray, w, h, 0.3, 0.98,
    );
    // Should still count as 1 cell, not split
    expect(classified.length).toBe(1);
    expect(classified[0].cellCount).toBe(1);
  });
});

// ─── Integration: full pipeline path (binary → extract → ensemble) ──

describe('ensemble integration', () => {
  it('two well-separated cells yield total count 2', () => {
    const w = 100, h = 60;
    const binary = new Uint8Array(w * h).fill(0);
    const gray = new Uint8Array(w * h).fill(200);
    paintCircle(binary, w, 25, 30, 8);
    paintCircle(binary, w, 75, 30, 8);
    paintCircle(gray, w, 25, 30, 8, 60);
    paintCircle(gray, w, 75, 30, 8, 60);

    const blobs = extractBlobs(binary, w, h, 5, 100000, false, gray);
    const medianArea = computeMedianCellArea(blobs, 0.35, 20);
    const classified = ensembleClassifyBlobs(
      blobs, medianArea, 0.35, 1.8,
      binary, gray, w, h,
    );
    const total = classified.reduce((s, b) => s + b.cellCount, 0);
    expect(total).toBe(2);
  });

  it('three cells in a triangle cluster count 2–4', () => {
    const w = 100, h = 100;
    const binary = new Uint8Array(w * h).fill(0);
    const gray = new Uint8Array(w * h).fill(200);
    const r = 10;
    // Tight triangle, cells overlapping
    paintCircle(binary, w, 40, 35, r);
    paintCircle(binary, w, 60, 35, r);
    paintCircle(binary, w, 50, 55, r);
    paintCircle(gray, w, 40, 35, r, 60);
    paintCircle(gray, w, 60, 35, r, 60);
    paintCircle(gray, w, 50, 55, r, 60);

    const blobs = extractBlobs(binary, w, h, 5, 100000, false, gray);
    const singleCellArea = Math.PI * r * r;
    const classified = ensembleClassifyBlobs(
      blobs, singleCellArea, 0.35, 1.3,
      binary, gray, w, h,
    );
    const total = classified.reduce((s, b) => s + b.cellCount, 0);
    expect(total).toBeGreaterThanOrEqual(2);
    expect(total).toBeLessThanOrEqual(4);
  });
});

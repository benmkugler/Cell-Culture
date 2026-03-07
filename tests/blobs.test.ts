import { describe, it, expect } from 'vitest';
import { extractBlobs, computeMedianCellArea, classifyBlobs } from '../src/cv/blobs';
import type { Blob } from '../src/cv/types';

/** Helper: paint a filled circle into a binary buffer. */
const paintCircle = (
  buf: Uint8Array, w: number, h: number,
  cx: number, cy: number, r: number,
) => {
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) {
        buf[y * w + x] = 1;
      }
    }
  }
};

describe('extractBlobs', () => {
  it('returns empty array for blank image', () => {
    const binary = new Uint8Array(100).fill(0);
    const blobs = extractBlobs(binary, 10, 10, 1, 10000, false);
    expect(blobs).toHaveLength(0);
  });

  it('detects a single circular blob', () => {
    const w = 50, h = 50;
    const binary = new Uint8Array(w * h).fill(0);
    paintCircle(binary, w, h, 25, 25, 8);

    const blobs = extractBlobs(binary, w, h, 5, 10000, false);
    expect(blobs.length).toBe(1);
    expect(blobs[0].area).toBeGreaterThan(150);
    expect(blobs[0].area).toBeLessThan(250);
    expect(blobs[0].cx).toBeCloseTo(25, 0);
    expect(blobs[0].cy).toBeCloseTo(25, 0);
    expect(blobs[0].circularity).toBeGreaterThan(0.65);
  });

  it('detects two separate blobs', () => {
    const w = 100, h = 50;
    const binary = new Uint8Array(w * h).fill(0);
    paintCircle(binary, w, h, 20, 25, 8);
    paintCircle(binary, w, h, 70, 25, 8);

    const blobs = extractBlobs(binary, w, h, 5, 10000, false);
    expect(blobs.length).toBe(2);
  });

  it('filters blobs smaller than minArea', () => {
    const w = 50, h = 50;
    const binary = new Uint8Array(w * h).fill(0);
    paintCircle(binary, w, h, 25, 25, 2);
    paintCircle(binary, w, h, 10, 10, 6);

    const blobs = extractBlobs(binary, w, h, 50, 10000, false);
    expect(blobs.length).toBe(1);
    expect(blobs[0].area).toBeGreaterThan(50);
  });

  it('filters blobs larger than maxArea', () => {
    const w = 50, h = 50;
    const binary = new Uint8Array(w * h).fill(0);
    paintCircle(binary, w, h, 25, 25, 15);

    const blobs = extractBlobs(binary, w, h, 5, 200, false);
    expect(blobs.length).toBe(0);
  });

  it('excludes blobs touching the border', () => {
    const w = 50, h = 50;
    const binary = new Uint8Array(w * h).fill(0);
    paintCircle(binary, w, h, 2, 25, 5);
    paintCircle(binary, w, h, 30, 25, 5);

    const blobsExclude = extractBlobs(binary, w, h, 5, 10000, true);
    expect(blobsExclude.length).toBe(1);
    expect(blobsExclude[0].cx).toBeCloseTo(30, 0);

    const blobsInclude = extractBlobs(binary, w, h, 5, 10000, false);
    expect(blobsInclude.length).toBe(2);
  });

  it('correctly computes bounding box', () => {
    const w = 40, h = 40;
    const binary = new Uint8Array(w * h).fill(0);
    paintCircle(binary, w, h, 20, 20, 5);

    const blobs = extractBlobs(binary, w, h, 5, 10000, false);
    expect(blobs.length).toBe(1);
    const bb = blobs[0].bbox;
    expect(bb.minX).toBeGreaterThanOrEqual(15);
    expect(bb.maxX).toBeLessThanOrEqual(25);
    expect(bb.minY).toBeGreaterThanOrEqual(15);
    expect(bb.maxY).toBeLessThanOrEqual(25);
  });

  it('computes solidity and eccentricity for circular blobs', () => {
    const w = 50, h = 50;
    const binary = new Uint8Array(w * h).fill(0);
    paintCircle(binary, w, h, 25, 25, 8);

    const blobs = extractBlobs(binary, w, h, 5, 10000, false);
    expect(blobs.length).toBe(1);
    expect(blobs[0].solidity).toBeGreaterThan(0.85);
    expect(blobs[0].eccentricity).toBeLessThan(0.3);
  });
});

describe('computeMedianCellArea', () => {
  const makeBlob = (area: number, circ: number): Blob => ({
    area, perimeter: 1, cx: 0, cy: 0,
    bbox: { minX: 0, maxX: 0, minY: 0, maxY: 0 },
    circularity: circ, cellCount: 1, cellLocations: [], pixels: [], touchesBorder: false,
    solidity: 0.9, eccentricity: 0.2, intensityStdDev: 20,
  });

  it('returns minArea for empty array', () => {
    expect(computeMedianCellArea([], 0.5, 20)).toBe(20);
  });

  it('returns IQM for sufficient circular blobs', () => {
    const blobs = [100, 110, 120, 130, 140, 150, 160, 170].map(a => makeBlob(a, 0.8));
    const median = computeMedianCellArea(blobs, 0.5, 10);
    expect(median).toBeCloseTo(135, 0);
  });

  it('falls back to simple median with < 3 circular blobs', () => {
    const blobs = [makeBlob(100, 0.8), makeBlob(200, 0.8)];
    const median = computeMedianCellArea(blobs, 0.5, 10);
    expect(median).toBe(200);
  });
});

describe('classifyBlobs', () => {
  const makeBlob = (
    area: number, circ: number, aspect: number = 1,
    solidity = 0.9, eccentricity = 0.2, intensityStdDev = 20,
  ): Blob => {
    const side = Math.sqrt(area);
    const w = side * Math.sqrt(aspect);
    const h = side / Math.sqrt(aspect);
    return {
      area, perimeter: 1, cx: 0, cy: 0,
      bbox: { minX: 0, maxX: w - 1, minY: 0, maxY: h - 1 },
      circularity: circ, cellCount: 1, cellLocations: [], pixels: [], touchesBorder: false,
      solidity, eccentricity, intensityStdDev,
    };
  };

  it('rejects blobs with low circularity', () => {
    const blobs = [makeBlob(100, 0.05)];
    const result = classifyBlobs(blobs, 100, 0.5, 1.8);
    expect(result).toHaveLength(0);
  });

  it('assigns cellCount=1 for normal-sized blobs', () => {
    const blobs = [makeBlob(100, 0.8)];
    const result = classifyBlobs(blobs, 100, 0.5, 1.8);
    expect(result).toHaveLength(1);
    expect(result[0].cellCount).toBe(1);
  });

  it('assigns cellCount>1 for cluster blobs', () => {
    const blobs = [makeBlob(300, 0.7)];
    const result = classifyBlobs(blobs, 100, 0.5, 1.8);
    expect(result).toHaveLength(1);
    expect(result[0].cellCount).toBe(3);
  });

  it('rejects elongated blobs (aspect ratio > 4)', () => {
    const blobs = [makeBlob(100, 0.8, 20)];
    const result = classifyBlobs(blobs, 100, 0.5, 1.8);
    expect(result).toHaveLength(0);
  });

  it('rejects blobs with low solidity (irregular debris)', () => {
    const blobs = [makeBlob(100, 0.8, 1, 0.3, 0.2, 20)];
    const result = classifyBlobs(blobs, 100, 0.5, 1.8, 0.5, 0.92, 80);
    expect(result).toHaveLength(0);
  });

  it('rejects blobs with high eccentricity (elongated scratches)', () => {
    const blobs = [makeBlob(100, 0.8, 1, 0.9, 0.95, 20)];
    const result = classifyBlobs(blobs, 100, 0.5, 1.8, 0.5, 0.92, 80);
    expect(result).toHaveLength(0);
  });

  it('rejects blobs with high intensity stddev (bubbles/ripples)', () => {
    const blobs = [makeBlob(100, 0.8, 1, 0.9, 0.2, 100)];
    const result = classifyBlobs(blobs, 100, 0.5, 1.8, 0.5, 0.92, 80);
    expect(result).toHaveLength(0);
  });

  it('accepts blobs passing all quality filters', () => {
    const blobs = [makeBlob(100, 0.8, 1, 0.9, 0.3, 30)];
    const result = classifyBlobs(blobs, 100, 0.5, 1.8, 0.5, 0.92, 80);
    expect(result).toHaveLength(1);
  });
});

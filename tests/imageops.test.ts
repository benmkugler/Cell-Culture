import { describe, it, expect } from 'vitest';
import {
  toGrayscale, otsuThreshold, gaussianBlur,
  binaryThreshold, removeGridLines, erode, dilate, morphOpen,
  detectGridLines, buildGridZone,
} from '../src/cv/imageops';

describe('toGrayscale', () => {
  it('converts a single white pixel', () => {
    const rgba = new Uint8ClampedArray([255, 255, 255, 255]);
    const gray = toGrayscale(rgba, 1);
    expect(gray[0]).toBe(255);
  });

  it('converts a single black pixel', () => {
    const rgba = new Uint8ClampedArray([0, 0, 0, 255]);
    const gray = toGrayscale(rgba, 1);
    expect(gray[0]).toBe(0);
  });

  it('uses BT.601 weights correctly', () => {
    // Pure red: 0.299*255 ≈ 76
    const rgba = new Uint8ClampedArray([255, 0, 0, 255]);
    const gray = toGrayscale(rgba, 1);
    expect(gray[0]).toBe(76);
  });

  it('converts multiple pixels', () => {
    const rgba = new Uint8ClampedArray([
      100, 100, 100, 255,
      200, 200, 200, 255,
    ]);
    const gray = toGrayscale(rgba, 2);
    expect(gray[0]).toBe(100);
    expect(gray[1]).toBe(200);
  });
});

describe('otsuThreshold', () => {
  it('returns 0 for uniform black image', () => {
    const gray = new Uint8Array(100).fill(0);
    expect(otsuThreshold(gray)).toBe(0);
  });

  it('finds threshold between two peaks', () => {
    // Bimodal: 50 pixels at intensity 50, 50 pixels at intensity 200
    const gray = new Uint8Array(100);
    for (let i = 0; i < 50; i++) gray[i] = 50;
    for (let i = 50; i < 100; i++) gray[i] = 200;
    const thresh = otsuThreshold(gray);
    // Threshold should be between 50 and 200
    expect(thresh).toBeGreaterThan(49);
    expect(thresh).toBeLessThan(201);
  });

  it('handles strongly bimodal distribution', () => {
    const gray = new Uint8Array(1000);
    for (let i = 0; i < 500; i++) gray[i] = 30;
    for (let i = 500; i < 1000; i++) gray[i] = 220;
    const thresh = otsuThreshold(gray);
    // Should split cleanly between the two modes
    expect(thresh).toBeGreaterThan(29);
    expect(thresh).toBeLessThan(221);
  });
});

describe('gaussianBlur', () => {
  it('does not change a uniform image', () => {
    const gray = new Uint8Array(25).fill(128);
    const blurred = gaussianBlur(gray, 5, 5, 1.0);
    for (let i = 0; i < 25; i++) {
      expect(blurred[i]).toBeCloseTo(128, 0);
    }
  });

  it('smooths a point impulse', () => {
    const gray = new Uint8Array(9 * 9).fill(0);
    gray[4 * 9 + 4] = 255; // center pixel
    const blurred = gaussianBlur(gray, 9, 9, 1.0);
    // Center should be reduced, neighbors should gain energy
    expect(blurred[4 * 9 + 4]).toBeLessThan(255);
    expect(blurred[4 * 9 + 3]).toBeGreaterThan(0);
    expect(blurred[3 * 9 + 4]).toBeGreaterThan(0);
  });

  it('returns copy when sigma=0', () => {
    const gray = new Uint8Array([10, 20, 30, 40]);
    const blurred = gaussianBlur(gray, 2, 2, 0);
    expect(Array.from(blurred)).toEqual([10, 20, 30, 40]);
    // Should be a copy, not same reference
    gray[0] = 99;
    expect(blurred[0]).toBe(10);
  });
});

describe('binaryThreshold', () => {
  it('thresholds correctly (non-inverted)', () => {
    const gray = new Uint8Array([50, 100, 150, 200]);
    const bin = binaryThreshold(gray, 120, false);
    expect(Array.from(bin)).toEqual([0, 0, 1, 1]);
  });

  it('thresholds correctly (inverted)', () => {
    const gray = new Uint8Array([50, 100, 150, 200]);
    const bin = binaryThreshold(gray, 120, true);
    expect(Array.from(bin)).toEqual([1, 1, 0, 0]);
  });
});

describe('removeGridLines', () => {
  it('removes a full horizontal line and its dilation band', () => {
    const w = 20, h = 20;
    const binary = new Uint8Array(w * h).fill(0);
    // Fill row 10 entirely (grid line)
    for (let x = 0; x < w; x++) binary[10 * w + x] = 1;
    // Add a blob far from the line at (1, 1)
    binary[1 * w + 1] = 1;

    const result = removeGridLines(binary, w, h, 0.5);
    // Row 10 and dilation band should be cleared
    for (let dy = -2; dy <= 2; dy++) {
      for (let x = 0; x < w; x++) expect(result[(10 + dy) * w + x]).toBe(0);
    }
    // Blob far from line should survive
    expect(result[1 * w + 1]).toBe(1);
  });

  it('removes a full vertical line and its dilation band', () => {
    const w = 20, h = 20;
    const binary = new Uint8Array(w * h).fill(0);
    // Fill column 10 entirely
    for (let y = 0; y < h; y++) binary[y * w + 10] = 1;
    binary[1 * w + 1] = 1;

    const result = removeGridLines(binary, w, h, 0.5);
    for (let dx = -2; dx <= 2; dx++) {
      for (let y = 0; y < h; y++) expect(result[y * w + (10 + dx)]).toBe(0);
    }
    expect(result[1 * w + 1]).toBe(1);
  });
});

describe('detectGridLines', () => {
  it('detects horizontal grid lines', () => {
    const w = 20, h = 20;
    const binary = new Uint8Array(w * h).fill(0);
    for (let x = 0; x < w; x++) binary[5 * w + x] = 1;
    const { rows, cols } = detectGridLines(binary, w, h, 0.5);
    expect(rows.has(5)).toBe(true);
    expect(cols.size).toBe(0);
  });

  it('detects vertical grid lines', () => {
    const w = 20, h = 20;
    const binary = new Uint8Array(w * h).fill(0);
    for (let y = 0; y < h; y++) binary[y * w + 8] = 1;
    const { rows, cols } = detectGridLines(binary, w, h, 0.5);
    expect(cols.has(8)).toBe(true);
    expect(rows.size).toBe(0);
  });
});

describe('buildGridZone', () => {
  it('marks pixels within radius of grid lines', () => {
    const w = 20, h = 20;
    const rows = new Set([10]);
    const cols = new Set([5]);
    const zone = buildGridZone(w, h, rows, cols, 2);
    // Row 10 ± 2
    for (let dy = -2; dy <= 2; dy++) {
      expect(zone[(10 + dy) * w + 0]).toBe(1);
    }
    // Row 7 is outside zone
    expect(zone[7 * w + 0]).toBe(0);
    // Col 5 ± 2
    expect(zone[0 * w + 5]).toBe(1);
    expect(zone[0 * w + 3]).toBe(1);
    expect(zone[0 * w + 1]).toBe(0);
  });
});

describe('morphological operations', () => {
  it('erode removes single-pixel protrusions', () => {
    const w = 5, h = 5;
    const binary = new Uint8Array(w * h).fill(0);
    // Single pixel at center
    binary[2 * w + 2] = 1;
    const result = erode(binary, w, h, 1);
    // Single pixel should vanish
    expect(result[2 * w + 2]).toBe(0);
  });

  it('dilate expands a single pixel', () => {
    const w = 5, h = 5;
    const binary = new Uint8Array(w * h).fill(0);
    binary[2 * w + 2] = 1;
    const result = dilate(binary, w, h, 1);
    // Center and 4-connected neighbors should be set
    expect(result[2 * w + 2]).toBe(1);
    expect(result[1 * w + 2]).toBe(1);
    expect(result[3 * w + 2]).toBe(1);
    expect(result[2 * w + 1]).toBe(1);
    expect(result[2 * w + 3]).toBe(1);
  });

  it('morphOpen removes noise but preserves larger structures', () => {
    const w = 10, h = 10;
    const binary = new Uint8Array(w * h).fill(0);
    // 4x4 block at (3,3)
    for (let dy = 0; dy < 4; dy++)
      for (let dx = 0; dx < 4; dx++)
        binary[(3 + dy) * w + (3 + dx)] = 1;
    // Noise pixel at (0,0)
    binary[0] = 1;

    const result = morphOpen(binary, w, h, 1);
    // Noise should be removed
    expect(result[0]).toBe(0);
    // Center of block should survive
    expect(result[4 * w + 4]).toBe(1);
    expect(result[5 * w + 5]).toBe(1);
  });
});

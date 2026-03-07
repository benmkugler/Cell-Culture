/**
 * Degraded-image tests: verify that the pipeline correctly counts cells
 * while rejecting artifacts caused by warping, debris, ripples, uneven
 * illumination, scratches, and salt-and-pepper noise.
 */
import { describe, it, expect } from 'vitest';
import { processGrayscaleBuffer } from '../src/cv/pipeline';
import {
  adaptiveIlluminationCorrection,
  medianFilter,
  blobIntensityStdDev,
} from '../src/cv/imageops';
import {
  computeSolidity,
  computeEccentricity,
} from '../src/cv/blobs';
import type { PipelineParams } from '../src/cv/types';

// ---------- Helpers ----------

/** Paint a filled circle. */
const paintCircle = (
  buf: Uint8Array, w: number, h: number,
  cx: number, cy: number, r: number, val: number,
) => {
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) buf[y * w + x] = val;
};

/** Paint a thin horizontal scratch (1px high, long). */
const paintScratch = (
  buf: Uint8Array, w: number,
  y0: number, x0: number, length: number, val: number,
) => {
  for (let x = x0; x < x0 + length && x < w; x++) buf[y0 * w + x] = val;
};

/** Add salt-and-pepper noise to a buffer. */
const addSaltAndPepper = (
  buf: Uint8Array, w: number, h: number, density: number, seed = 42,
) => {
  let s = seed;
  const next = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  for (let i = 0; i < w * h; i++) {
    const r = next();
    if (r < density / 2) buf[i] = 0;
    else if (r < density) buf[i] = 255;
  }
};

/** Add a smooth gradient (vignette-like uneven illumination). */
const addGradient = (
  buf: Uint8Array, w: number, h: number, deltaY: number,
) => {
  for (let y = 0; y < h; y++) {
    const offset = Math.round((y / h) * deltaY);
    for (let x = 0; x < w; x++) {
      buf[y * w + x] = Math.min(255, Math.max(0, buf[y * w + x] + offset));
    }
  }
};

/** Add sinusoidal ripple pattern. */
const addRipple = (
  buf: Uint8Array, w: number, h: number,
  amplitude: number, frequency: number,
) => {
  for (let y = 0; y < h; y++) {
    const ripple = Math.round(amplitude * Math.sin(2 * Math.PI * frequency * y / h));
    for (let x = 0; x < w; x++) {
      buf[y * w + x] = Math.min(255, Math.max(0, buf[y * w + x] + ripple));
    }
  }
};

/** Paint an irregular (non-convex) debris blob — jagged star shape. */
const paintDebris = (
  buf: Uint8Array, w: number,
  cx: number, cy: number, r: number, val: number,
) => {
  for (let angle = 0; angle < 360; angle += 5) {
    const rad = (angle * Math.PI) / 180;
    const reach = angle % 30 < 15 ? r : r * 0.3;
    for (let t = 0; t <= reach; t += 0.5) {
      const px = Math.round(cx + t * Math.cos(rad));
      const py = Math.round(cy + t * Math.sin(rad));
      if (px >= 0 && px < w && py >= 0) buf[py * w + px] = val;
    }
  }
};

const robustParams: Omit<PipelineParams, 'showMask'> = {
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
  adaptiveIllumination: true,
  illuminationBlockSize: 51,
  medianFilterRadius: 1,
  minSolidity: 0.5,
  maxEccentricity: 0.92,
  maxIntensityStdDev: 80,
  use8Connected: false,
  enableCLAHE: false,
  claheTileSize: 8,
  claheClipLimit: 2.0,
  minAgreement: 1,
};

// ---------- Unit tests: new primitives ----------

describe('adaptiveIlluminationCorrection', () => {
  it('reduces variance in a region with gradient + cell', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h);
    // Gradient background 100→220
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        gray[y * w + x] = Math.round(100 + (120 * y) / h);
    // Dark cell in the center
    paintCircle(gray, w, h, 50, 50, 8, 40);

    const corrected = adaptiveIlluminationCorrection(gray, w, h, 51);
    // Background std dev should decrease after correction
    const bgPixels: number[] = [];
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++)
        if ((x - 50) ** 2 + (y - 50) ** 2 > 15 * 15) bgPixels.push(corrected[y * w + x]);
    const mean = bgPixels.reduce((s, v) => s + v, 0) / bgPixels.length;
    const stddev = Math.sqrt(bgPixels.reduce((s, v) => s + (v - mean) ** 2, 0) / bgPixels.length);
    // Original bg stddev of uniform gradient 100-220 ≈ 35. Corrected should be tighter.
    expect(stddev).toBeLessThan(25);
  });

  it('preserves local contrast (cell remains visible)', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h).fill(180);
    paintCircle(gray, w, h, 50, 50, 8, 60);
    addGradient(gray, w, h, 80);

    const corrected = adaptiveIlluminationCorrection(gray, w, h, 31);
    const cellVal = corrected[50 * w + 50];
    const bgVal = corrected[10 * w + 10];
    expect(Math.abs(cellVal - bgVal)).toBeGreaterThan(20);
  });
});

describe('medianFilter', () => {
  it('removes salt-and-pepper noise', () => {
    const w = 50, h = 50;
    const gray = new Uint8Array(w * h).fill(128);
    addSaltAndPepper(gray, w, h, 0.1);

    const filtered = medianFilter(gray, w, h, 1);
    let restored = 0;
    for (let i = 0; i < filtered.length; i++) {
      if (Math.abs(filtered[i] - 128) < 10) restored++;
    }
    expect(restored / (w * h)).toBeGreaterThan(0.9);
  });

  it('returns copy when radius=0', () => {
    const gray = new Uint8Array([10, 20, 30, 40]);
    const out = medianFilter(gray, 2, 2, 0);
    expect(Array.from(out)).toEqual([10, 20, 30, 40]);
  });
});

describe('blobIntensityStdDev', () => {
  it('returns 0 for uniform-value pixels', () => {
    const gray = new Uint8Array(100).fill(128);
    expect(blobIntensityStdDev(gray, [0, 1, 2, 3, 4])).toBe(0);
  });

  it('computes correct stddev for varied pixels', () => {
    const gray = new Uint8Array(100);
    gray[0] = 100; gray[1] = 200;
    const stddev = blobIntensityStdDev(gray, [0, 1]);
    expect(stddev).toBeCloseTo(50, 0);
  });

  it('returns 0 for empty pixel set', () => {
    const gray = new Uint8Array(100);
    expect(blobIntensityStdDev(gray, [])).toBe(0);
  });
});

describe('computeSolidity', () => {
  it('returns ~1 for a filled circle', () => {
    const w = 50;
    const pixels: number[] = [];
    for (let y = 10; y < 40; y++)
      for (let x = 10; x < 40; x++)
        if ((x - 25) ** 2 + (y - 25) ** 2 <= 12 * 12) pixels.push(y * w + x);
    expect(computeSolidity(pixels, w)).toBeGreaterThan(0.9);
  });

  it('returns low value for star-shaped debris', () => {
    const w = 100;
    const pixels: number[] = [];
    const cx = 50, cy = 50, r = 15;
    for (let angle = 0; angle < 360; angle += 5) {
      const rad = (angle * Math.PI) / 180;
      const reach = angle % 30 < 15 ? r : r * 0.3;
      for (let t = 0; t <= reach; t += 1) {
        const px = Math.round(cx + t * Math.cos(rad));
        const py = Math.round(cy + t * Math.sin(rad));
        pixels.push(py * w + px);
      }
    }
    const unique = [...new Set(pixels)];
    expect(computeSolidity(unique, w)).toBeLessThan(0.7);
  });

  it('returns 1 for < 3 pixels', () => {
    expect(computeSolidity([0, 1], 10)).toBe(1);
  });
});

describe('computeEccentricity', () => {
  it('returns low value for a filled circle', () => {
    const w = 50;
    const pixels: number[] = [];
    for (let y = 10; y < 40; y++)
      for (let x = 10; x < 40; x++)
        if ((x - 25) ** 2 + (y - 25) ** 2 <= 12 * 12) pixels.push(y * w + x);
    expect(computeEccentricity(pixels, w)).toBeLessThan(0.3);
  });

  it('returns high value for a thin line', () => {
    const w = 100;
    const pixels: number[] = [];
    for (let x = 10; x < 90; x++) pixels.push(50 * w + x);
    expect(computeEccentricity(pixels, w)).toBeGreaterThan(0.9);
  });

  it('returns 0 for single pixel', () => {
    expect(computeEccentricity([55], 10)).toBe(0);
  });
});

// ---------- Integration: degraded-image scenarios ----------

describe('processGrayscaleBuffer — degraded images', () => {
  it('counts cells despite uneven illumination (gradient)', () => {
    const w = 200, h = 200;
    const gray = new Uint8Array(w * h).fill(210);
    paintCircle(gray, w, h, 60, 60, 10, 50);
    paintCircle(gray, w, h, 140, 140, 10, 50);
    addGradient(gray, w, h, 80);

    const result = processGrayscaleBuffer(gray, w, h, robustParams);
    expect(result.count).toBe(2);
  });

  it('counts cells with salt-and-pepper noise', () => {
    const w = 200, h = 200;
    const gray = new Uint8Array(w * h).fill(210);
    paintCircle(gray, w, h, 60, 60, 10, 40);
    paintCircle(gray, w, h, 140, 60, 10, 40);
    paintCircle(gray, w, h, 100, 140, 10, 40);
    addSaltAndPepper(gray, w, h, 0.05);

    const result = processGrayscaleBuffer(gray, w, h, robustParams);
    expect(result.count).toBe(3);
  });

  it('rejects elongated scratch artifacts', () => {
    const w = 200, h = 200;
    const gray = new Uint8Array(w * h).fill(210);
    paintCircle(gray, w, h, 100, 100, 10, 40);
    for (let dy = 0; dy < 2; dy++) {
      paintScratch(gray, w, 50 + dy, 20, 160, 40);
      paintScratch(gray, w, 150 + dy, 20, 160, 40);
    }

    const result = processGrayscaleBuffer(gray, w, h, robustParams);
    expect(result.count).toBe(1);
  });

  it('rejects irregular debris while counting cells', () => {
    const w = 200, h = 200;
    const gray = new Uint8Array(w * h).fill(210);
    // Two real circular cells
    paintCircle(gray, w, h, 50, 50, 10, 40);
    paintCircle(gray, w, h, 150, 50, 10, 40);
    // L-shaped debris (contiguous, low solidity) — dark on light bg
    for (let y = 120; y < 180; y++) gray[y * w + 100] = 40; // vertical arm
    for (let y = 120; y < 125; y++)
      for (let x = 100; x < 160; x++) gray[y * w + x] = 40; // horizontal arm

    const result = processGrayscaleBuffer(gray, w, h, {
      ...robustParams,
      adaptiveIllumination: false,
      medianFilterRadius: 0,
      excludeBorder: false,
      minSolidity: 0.7,
      maxEccentricity: 0.95,
    });
    // The 2 circular cells should survive; L-shape should be rejected
    // (either by eccentricity or solidity or aspect ratio)
    expect(result.count).toBe(2);
  });

  it('counts cells with sinusoidal ripple overlay', () => {
    const w = 200, h = 200;
    const gray = new Uint8Array(w * h).fill(210);
    paintCircle(gray, w, h, 60, 60, 10, 40);
    paintCircle(gray, w, h, 140, 140, 10, 40);
    addRipple(gray, w, h, 15, 6);

    const result = processGrayscaleBuffer(gray, w, h, robustParams);
    expect(result.count).toBe(2);
  });

  it('handles combined degradation: gradient + noise + ripple', () => {
    const w = 250, h = 250;
    const gray = new Uint8Array(w * h).fill(200);
    paintCircle(gray, w, h, 60, 60, 10, 40);
    paintCircle(gray, w, h, 190, 60, 10, 40);
    paintCircle(gray, w, h, 60, 190, 10, 40);
    paintCircle(gray, w, h, 190, 190, 10, 40);
    addGradient(gray, w, h, 60);
    addRipple(gray, w, h, 15, 6);
    addSaltAndPepper(gray, w, h, 0.02);

    const result = processGrayscaleBuffer(gray, w, h, robustParams);
    expect(result.count).toBe(4);
  });

  it('reports zero cells on a noisy blank image', () => {
    const w = 150, h = 150;
    const gray = new Uint8Array(w * h).fill(200);
    addSaltAndPepper(gray, w, h, 0.08);
    addRipple(gray, w, h, 20, 5);

    const result = processGrayscaleBuffer(gray, w, h, robustParams);
    expect(result.count).toBe(0);
  });

  it('counts cells under strong vignette', () => {
    const w = 200, h = 200;
    const gray = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const dist = Math.sqrt((x - 100) ** 2 + (y - 100) ** 2);
        gray[y * w + x] = Math.max(0, Math.min(255, Math.round(230 - dist * 0.8)));
      }
    }
    paintCircle(gray, w, h, 60, 100, 10, 30);
    paintCircle(gray, w, h, 140, 100, 10, 30);

    const result = processGrayscaleBuffer(gray, w, h, robustParams);
    expect(result.count).toBe(2);
  });

  it('does not double-count when adaptive illumination is enabled', () => {
    const w = 200, h = 200;
    const gray = new Uint8Array(w * h).fill(210);
    paintCircle(gray, w, h, 100, 100, 10, 40);

    const resultOn = processGrayscaleBuffer(gray, w, h, robustParams);
    const resultOff = processGrayscaleBuffer(gray, w, h, {
      ...robustParams,
      adaptiveIllumination: false,
      medianFilterRadius: 0,
    });
    expect(resultOn.count).toBe(1);
    expect(resultOff.count).toBe(1);
  });
});

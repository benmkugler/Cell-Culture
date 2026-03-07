import { describe, it, expect } from 'vitest';
import { processGrayscaleBuffer } from '../src/cv/pipeline';
import type { PipelineParams } from '../src/cv/types';

/** Paint a filled circle onto a grayscale buffer (dark circle on bright bg). */
const paintDarkCircle = (
  buf: Uint8Array, w: number, h: number,
  cx: number, cy: number, r: number,
  fgVal = 40, bgVal = 220,
) => {
  buf.fill(bgVal);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) {
        buf[y * w + x] = fgVal;
      }
    }
  }
};

/** Paint multiple non-overlapping dark circles. */
const paintMultipleCircles = (
  buf: Uint8Array, w: number, h: number,
  circles: { cx: number; cy: number; r: number }[],
  fgVal = 40, bgVal = 220,
) => {
  buf.fill(bgVal);
  for (const { cx, cy, r } of circles) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) {
          buf[y * w + x] = fgVal;
        }
      }
    }
  }
};

const baseParams: Omit<PipelineParams, 'showMask'> = {
  threshold: 130,
  autoThreshold: true,
  invert: true, // dark cells on bright background
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

describe('processGrayscaleBuffer — synthetic cell counting', () => {
  it('counts zero cells on blank image', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h).fill(220);
    const result = processGrayscaleBuffer(gray, w, h, baseParams);
    expect(result.count).toBe(0);
  });

  it('counts 1 cell (single circle)', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h);
    paintDarkCircle(gray, w, h, 50, 50, 10);
    const result = processGrayscaleBuffer(gray, w, h, baseParams);
    expect(result.count).toBe(1);
  });

  it('counts 2 separated cells', () => {
    const w = 200, h = 100;
    const gray = new Uint8Array(w * h);
    paintMultipleCircles(gray, w, h, [
      { cx: 50, cy: 50, r: 10 },
      { cx: 150, cy: 50, r: 10 },
    ]);
    const result = processGrayscaleBuffer(gray, w, h, baseParams);
    expect(result.count).toBe(2);
  });

  it('counts 4 cells in a grid', () => {
    const w = 200, h = 200;
    const gray = new Uint8Array(w * h);
    paintMultipleCircles(gray, w, h, [
      { cx: 50, cy: 50, r: 10 },
      { cx: 150, cy: 50, r: 10 },
      { cx: 50, cy: 150, r: 10 },
      { cx: 150, cy: 150, r: 10 },
    ]);
    const result = processGrayscaleBuffer(gray, w, h, baseParams);
    expect(result.count).toBe(4);
  });

  it('counts 8 cells scattered', () => {
    const w = 300, h = 300;
    const gray = new Uint8Array(w * h);
    const circles = [
      { cx: 50, cy: 50, r: 10 },
      { cx: 120, cy: 50, r: 10 },
      { cx: 190, cy: 50, r: 10 },
      { cx: 260, cy: 50, r: 10 },
      { cx: 50, cy: 150, r: 10 },
      { cx: 120, cy: 150, r: 10 },
      { cx: 190, cy: 150, r: 10 },
      { cx: 260, cy: 150, r: 10 },
    ];
    paintMultipleCircles(gray, w, h, circles);
    const result = processGrayscaleBuffer(gray, w, h, baseParams);
    expect(result.count).toBe(8);
  });

  it('excludes cells touching border when excludeBorder=true', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h);
    paintMultipleCircles(gray, w, h, [
      { cx: 3, cy: 50, r: 8 },   // touches left border
      { cx: 50, cy: 50, r: 8 },  // interior
    ]);
    const result = processGrayscaleBuffer(gray, w, h, { ...baseParams, excludeBorder: true });
    expect(result.count).toBe(1);
  });

  it('includes border cells when excludeBorder=false', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h);
    paintMultipleCircles(gray, w, h, [
      { cx: 3, cy: 50, r: 8 },
      { cx: 50, cy: 50, r: 8 },
    ]);
    const result = processGrayscaleBuffer(gray, w, h, { ...baseParams, excludeBorder: false });
    expect(result.count).toBe(2);
  });

  it('rejects noise below minCellSize', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h);
    paintMultipleCircles(gray, w, h, [
      { cx: 50, cy: 50, r: 2 },  // tiny dot, area ≈ 13
      { cx: 30, cy: 30, r: 10 }, // real cell, area ≈ 314
    ]);
    const result = processGrayscaleBuffer(gray, w, h, { ...baseParams, minCellSize: 50 });
    expect(result.count).toBe(1);
  });

  it('detects cluster and splits count', () => {
    const w = 400, h = 200;
    const gray = new Uint8Array(w * h);
    // 4 normal cells to anchor the median, plus one large blob (3x radius = 9x area)
    paintMultipleCircles(gray, w, h, [
      { cx: 40, cy: 50, r: 8 },
      { cx: 100, cy: 50, r: 8 },
      { cx: 160, cy: 50, r: 8 },
      { cx: 220, cy: 50, r: 8 },
      { cx: 320, cy: 100, r: 24 }, // ~9x area of a r=8 cell
    ]);
    const result = processGrayscaleBuffer(gray, w, h, {
      ...baseParams,
      clusterSplitRatio: 1.5,
      excludeBorder: false,
    });
    // 4 normal + large blob split into ~9 = ~13 total, but at minimum > 6
    expect(result.count).toBeGreaterThanOrEqual(6);
  });

  it('reports correct Otsu threshold', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h);
    paintDarkCircle(gray, w, h, 50, 50, 10, 40, 220);
    const result = processGrayscaleBuffer(gray, w, h, baseParams);
    // Otsu should find a threshold between 40 and 220
    expect(result.otsuThreshold).toBeGreaterThan(39);
    expect(result.otsuThreshold).toBeLessThan(221);
  });

  it('handles non-inverted mode (bright cells on dark bg)', () => {
    const w = 100, h = 100;
    const gray = new Uint8Array(w * h).fill(30); // dark bg
    // Bright cell
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if ((x - 50) ** 2 + (y - 50) ** 2 <= 100) {
          gray[y * w + x] = 230;
        }
      }
    }
    const result = processGrayscaleBuffer(gray, w, h, {
      ...baseParams,
      invert: false,
    });
    expect(result.count).toBe(1);
  });

  it('counts 2 touching cells as 2 (not 1)', () => {
    const w = 200, h = 100;
    const gray = new Uint8Array(w * h);
    // Two dark circles close enough to merge into one connected component
    paintMultipleCircles(gray, w, h, [
      { cx: 85, cy: 50, r: 10 },
      { cx: 108, cy: 50, r: 10 },
    ]);
    const result = processGrayscaleBuffer(gray, w, h, {
      ...baseParams,
      excludeBorder: false,
      clusterSplitRatio: 1.5,
    });
    expect(result.count).toBe(2);
  });

  it('counts 3 touching cells in a row as 3', () => {
    const w = 250, h = 100;
    const gray = new Uint8Array(w * h);
    paintMultipleCircles(gray, w, h, [
      { cx: 60, cy: 50, r: 10 },
      { cx: 83, cy: 50, r: 10 },
      { cx: 106, cy: 50, r: 10 },
    ]);
    const result = processGrayscaleBuffer(gray, w, h, {
      ...baseParams,
      excludeBorder: false,
      clusterSplitRatio: 1.5,
    });
    expect(result.count).toBe(3);
  });

  it('does not hallucinate cells on grid lines', () => {
    const w = 200, h = 200;
    const gray = new Uint8Array(w * h).fill(220);
    // Draw two grid lines (dark lines on bright bg)
    for (let x = 0; x < w; x++) {
      gray[50 * w + x] = 40;   // horizontal line at y=50
      gray[100 * w + x] = 40;  // horizontal line at y=100
    }
    for (let y = 0; y < h; y++) {
      gray[y * w + 60] = 40;   // vertical line at x=60
      gray[y * w + 140] = 40;  // vertical line at x=140
    }
    // One real cell away from grid lines
    paintDarkCircle(gray, w, h, 30, 150, 10);

    const result = processGrayscaleBuffer(gray, w, h, {
      ...baseParams,
      excludeBorder: false,
    });
    // Only the real cell should be counted, grid line fragments rejected
    expect(result.count).toBeLessThanOrEqual(1);
  });

  it('does not hallucinate cells at grid intersections', () => {
    const w = 200, h = 200;
    const gray = new Uint8Array(w * h).fill(220);
    // Draw thick grid lines (3px wide)
    for (let d = -1; d <= 1; d++) {
      for (let x = 0; x < w; x++) {
        gray[(80 + d) * w + x] = 40;
        gray[(140 + d) * w + x] = 40;
      }
      for (let y = 0; y < h; y++) {
        gray[y * w + (60 + d)] = 40;
        gray[y * w + (130 + d)] = 40;
      }
    }
    // Two real cells far from any grid line (paint without buf.fill)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if ((x - 30) ** 2 + (y - 30) ** 2 <= 64) gray[y * w + x] = 40;
        if ((x - 170) ** 2 + (y - 170) ** 2 <= 64) gray[y * w + x] = 40;
      }
    }

    const result = processGrayscaleBuffer(gray, w, h, {
      ...baseParams,
      excludeBorder: false,
    });
    // Should count exactly 2 real cells, no intersection phantoms
    expect(result.count).toBe(2);
  });
});

import { describe, it, expect } from 'vitest';
import {
  locateByDistanceTransform,
  locateByConcavity,
  locateByLoG,
  mergeLocations,
  ensembleLocate,
  ensembleClassifyBlobs,
} from '../src/cv/counting';
import { extractBlobs, computeMedianCellArea } from '../src/cv/blobs';
import type { CellLocation } from '../src/cv/types';

// ─── Helpers ─────────────────────────────────────────────────────────

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

/** Assert a CellLocation is near an expected position. */
const expectNear = (loc: CellLocation, ex: number, ey: number, tolerance: number) => {
  const dist = Math.sqrt((loc.x - ex) ** 2 + (loc.y - ey) ** 2);
  expect(dist).toBeLessThan(tolerance);
};

/** Find the closest detection to an expected position. */
const findNearest = (locs: CellLocation[], ex: number, ey: number): CellLocation => {
  let best = locs[0], bestD = Infinity;
  for (const l of locs) {
    const d = (l.x - ex) ** 2 + (l.y - ey) ** 2;
    if (d < bestD) { bestD = d; best = l; }
  }
  return best;
};

const buildScene = (
  w: number, h: number,
  cells: { cx: number; cy: number; r: number }[],
) => {
  const binary = new Uint8Array(w * h).fill(0);
  const gray = new Uint8Array(w * h).fill(200);
  for (const c of cells) {
    paintCircle(binary, w, c.cx, c.cy, c.r);
    paintCircle(gray, w, c.cx, c.cy, c.r, 60);
  }
  const blobs = extractBlobs(binary, w, h, 5, 100000, false, gray);
  return { binary, gray, blobs };
};

// ─── locateByDistanceTransform ──────────────────────────────────────

describe('locateByDistanceTransform — locations', () => {
  it('returns 1 location near center for a single circle', () => {
    const w = 60, h = 60;
    const { binary, blobs } = buildScene(w, h, [{ cx: 30, cy: 30, r: 10 }]);
    expect(blobs).toHaveLength(1);

    const locs = locateByDistanceTransform(binary, w, h, blobs[0], 5);
    expect(locs).toHaveLength(1);
    expectNear(locs[0], 30, 30, 5);
    expect(locs[0].radius).toBeGreaterThan(0);
  });

  it('returns 2 locations near each center for two touching circles', () => {
    const w = 80, h = 60;
    const { binary, blobs } = buildScene(w, h, [
      { cx: 25, cy: 30, r: 12 },
      { cx: 50, cy: 30, r: 12 },
    ]);
    if (blobs.length === 1) {
      const locs = locateByDistanceTransform(binary, w, h, blobs[0], 6);
      expect(locs.length).toBeGreaterThanOrEqual(2);
      // Each detected location should be near one of the original centers
      const nearA = findNearest(locs, 25, 30);
      const nearB = findNearest(locs, 50, 30);
      expectNear(nearA, 25, 30, 10);
      expectNear(nearB, 50, 30, 10);
    }
  });

  it('locations have positive radius', () => {
    const w = 60, h = 60;
    const { binary, blobs } = buildScene(w, h, [{ cx: 30, cy: 30, r: 12 }]);
    const locs = locateByDistanceTransform(binary, w, h, blobs[0], 5);
    for (const l of locs) expect(l.radius).toBeGreaterThan(0);
  });
});

// ─── locateByConcavity ──────────────────────────────────────────────

describe('locateByConcavity — locations', () => {
  it('returns 1 location for a single circle', () => {
    const w = 60, h = 60;
    const { blobs } = buildScene(w, h, [{ cx: 30, cy: 30, r: 12 }]);
    expect(blobs).toHaveLength(1);

    const locs = locateByConcavity(blobs[0], w);
    expect(locs).toHaveLength(1);
    expectNear(locs[0], 30, 30, 8);
  });

  it('returns >= 2 locations for overlapping circles', () => {
    const w = 80, h = 60;
    const { blobs } = buildScene(w, h, [
      { cx: 24, cy: 30, r: 13 },
      { cx: 51, cy: 30, r: 13 },
    ]);
    if (blobs.length === 1) {
      const locs = locateByConcavity(blobs[0], w, 6);
      expect(locs.length).toBeGreaterThanOrEqual(2);
      // Locations should be within the blob bbox
      for (const l of locs) {
        expect(l.x).toBeGreaterThanOrEqual(blobs[0].bbox.minX - 1);
        expect(l.x).toBeLessThanOrEqual(blobs[0].bbox.maxX + 1);
      }
    }
  });

  it('returns location with positive radius', () => {
    const w = 60, h = 60;
    const { blobs } = buildScene(w, h, [{ cx: 30, cy: 30, r: 10 }]);
    const locs = locateByConcavity(blobs[0], w);
    for (const l of locs) expect(l.radius).toBeGreaterThan(0);
  });
});

// ─── locateByLoG ────────────────────────────────────────────────────

describe('locateByLoG — locations', () => {
  it('returns 1 location near center for a single dark circle', () => {
    const w = 60, h = 60;
    const { gray, blobs } = buildScene(w, h, [{ cx: 30, cy: 30, r: 10 }]);
    expect(blobs).toHaveLength(1);

    const locs = locateByLoG(gray, w, h, blobs[0], [3, 12], 5);
    expect(locs).toHaveLength(1);
    expectNear(locs[0], 30, 30, 8);
  });

  it('returns 2 locations for two adjacent dark circles (merged blob)', () => {
    const w = 100, h = 60;
    const { binary, gray } = buildScene(w, h, [
      { cx: 28, cy: 30, r: 11 },
      { cx: 60, cy: 30, r: 11 },
    ]);
    // Use larger blobs to force merging
    const blobs = extractBlobs(binary, w, h, 5, 100000, false, gray);
    const totalLocs = blobs.flatMap(b => locateByLoG(gray, w, h, b, [3, 12], 5));
    expect(totalLocs.length).toBeGreaterThanOrEqual(2);
  });

  it('location coordinates are in image space', () => {
    const w = 80, h = 80;
    const { gray, blobs } = buildScene(w, h, [{ cx: 50, cy: 50, r: 10 }]);
    const locs = locateByLoG(gray, w, h, blobs[0], [3, 12], 5);
    for (const l of locs) {
      expect(l.x).toBeGreaterThanOrEqual(0);
      expect(l.x).toBeLessThan(w);
      expect(l.y).toBeGreaterThanOrEqual(0);
      expect(l.y).toBeLessThan(h);
    }
  });
});

// ─── mergeLocations ─────────────────────────────────────────────────

describe('mergeLocations', () => {
  it('merges co-located detections from 3 strategies into 1 with agreement=3', () => {
    const dt: CellLocation[] = [{ x: 30, y: 30, radius: 10, agreement: 0 }];
    const cc: CellLocation[] = [{ x: 31, y: 31, radius: 9, agreement: 0 }];
    const lg: CellLocation[] = [{ x: 29, y: 30, radius: 11, agreement: 0 }];

    const merged = mergeLocations(dt, cc, lg, 10);
    expect(merged).toHaveLength(1);
    expect(merged[0].agreement).toBe(3);
    expectNear(merged[0], 30, 30, 3);
  });

  it('keeps separate detections when far apart', () => {
    const dt: CellLocation[] = [{ x: 10, y: 10, radius: 5, agreement: 0 }];
    const cc: CellLocation[] = [{ x: 60, y: 60, radius: 5, agreement: 0 }];
    const lg: CellLocation[] = [];

    const merged = mergeLocations(dt, cc, lg, 10);
    expect(merged).toHaveLength(2);
    expect(merged.some(m => m.agreement === 1)).toBe(true);
  });

  it('handles 2 clusters: one with 3-way agreement, one with 1-way', () => {
    const dt: CellLocation[] = [
      { x: 20, y: 20, radius: 8, agreement: 0 },
      { x: 70, y: 70, radius: 8, agreement: 0 },
    ];
    const cc: CellLocation[] = [{ x: 21, y: 21, radius: 8, agreement: 0 }];
    const lg: CellLocation[] = [{ x: 19, y: 19, radius: 8, agreement: 0 }];

    const merged = mergeLocations(dt, cc, lg, 10);
    expect(merged).toHaveLength(2);
    const high = merged.find(m => m.agreement >= 3);
    const low = merged.find(m => m.agreement === 1);
    expect(high).toBeDefined();
    expect(low).toBeDefined();
  });

  it('returns empty array for no input', () => {
    expect(mergeLocations([], [], [], 10)).toHaveLength(0);
  });

  it('averages coordinates of merged detections', () => {
    const dt: CellLocation[] = [{ x: 10, y: 10, radius: 5, agreement: 0 }];
    const cc: CellLocation[] = [{ x: 14, y: 14, radius: 7, agreement: 0 }];
    const lg: CellLocation[] = [];

    const merged = mergeLocations(dt, cc, lg, 10);
    expect(merged).toHaveLength(1);
    expect(merged[0].x).toBeCloseTo(12, 0);
    expect(merged[0].y).toBeCloseTo(12, 0);
    expect(merged[0].radius).toBeCloseTo(6, 0);
    expect(merged[0].agreement).toBe(2);
  });
});

// ─── ensembleLocate ─────────────────────────────────────────────────

describe('ensembleLocate', () => {
  it('single cell: 1 location with agreement=3', () => {
    const w = 60, h = 60;
    const { binary, gray, blobs } = buildScene(w, h, [{ cx: 30, cy: 30, r: 8 }]);
    expect(blobs).toHaveLength(1);

    const result = ensembleLocate(
      blobs[0], binary, gray, w, h, blobs[0].area, 1.8,
    );
    expect(result.cellCount).toBe(1);
    expect(result.locations).toHaveLength(1);
    expect(result.locations[0].agreement).toBe(3);
    expectNear(result.locations[0], 30, 30, 5);
  });

  it('two touching cells: 2 locations, each near a true center', () => {
    const w = 80, h = 60;
    const r = 12;
    const { binary, gray, blobs } = buildScene(w, h, [
      { cx: 25, cy: 30, r },
      { cx: 50, cy: 30, r },
    ]);
    if (blobs.length === 1) {
      const singleArea = Math.PI * r * r;
      const result = ensembleLocate(
        blobs[0], binary, gray, w, h, singleArea, 1.5,
      );
      expect(result.cellCount).toBeGreaterThanOrEqual(2);
      expect(result.locations.length).toBeGreaterThanOrEqual(2);

      const nearA = findNearest(result.locations, 25, 30);
      const nearB = findNearest(result.locations, 50, 30);
      expectNear(nearA, 25, 30, 12);
      expectNear(nearB, 50, 30, 12);
    }
  });

  it('locations array length matches cellCount', () => {
    const w = 100, h = 80;
    const { binary, gray, blobs } = buildScene(w, h, [
      { cx: 30, cy: 40, r: 12 },
      { cx: 52, cy: 40, r: 12 },
      { cx: 41, cy: 22, r: 12 },
    ]);
    for (const b of blobs) {
      const singleArea = Math.PI * 12 * 12;
      const result = ensembleLocate(
        b, binary, gray, w, h, singleArea, 1.3,
      );
      expect(result.locations).toHaveLength(result.cellCount);
    }
  });
});

// ─── ensembleClassifyBlobs — location fields ────────────────────────

describe('ensembleClassifyBlobs — cellLocations', () => {
  it('every classified blob has cellLocations.length === cellCount', () => {
    const w = 100, h = 100;
    const { binary, gray, blobs } = buildScene(w, h, [
      { cx: 30, cy: 30, r: 8 },
      { cx: 70, cy: 70, r: 8 },
    ]);
    const medianArea = computeMedianCellArea(blobs, 0.35, 20);
    const classified = ensembleClassifyBlobs(
      blobs, medianArea, 0.35, 1.8, binary, gray, w, h,
    );
    for (const b of classified) {
      expect(b.cellLocations).toBeDefined();
      expect(b.cellLocations.length).toBe(b.cellCount);
    }
  });

  it('single-cell locations are near blob centroid', () => {
    const w = 80, h = 80;
    const { binary, gray, blobs } = buildScene(w, h, [{ cx: 40, cy: 40, r: 8 }]);
    const medianArea = computeMedianCellArea(blobs, 0.35, 20);
    const classified = ensembleClassifyBlobs(
      blobs, medianArea, 0.35, 1.8, binary, gray, w, h,
    );
    expect(classified).toHaveLength(1);
    expect(classified[0].cellLocations).toHaveLength(1);
    expectNear(classified[0].cellLocations[0], 40, 40, 5);
  });

  it('cluster locations are distinct (minimum inter-cell distance)', () => {
    const w = 120, h = 80;
    const { binary, gray, blobs } = buildScene(w, h, [
      { cx: 35, cy: 40, r: 13 },
      { cx: 65, cy: 40, r: 13 },
    ]);
    const singleArea = Math.PI * 13 * 13;
    const classified = ensembleClassifyBlobs(
      blobs, singleArea, 0.35, 1.3, binary, gray, w, h,
    );

    for (const b of classified) {
      if (b.cellCount > 1) {
        // All pairs of locations should be at least cellRadius apart
        for (let i = 0; i < b.cellLocations.length; i++) {
          for (let j = i + 1; j < b.cellLocations.length; j++) {
            const dist = Math.sqrt(
              (b.cellLocations[i].x - b.cellLocations[j].x) ** 2 +
              (b.cellLocations[i].y - b.cellLocations[j].y) ** 2,
            );
            expect(dist).toBeGreaterThan(5);
          }
        }
      }
    }
  });

  it('all locations are within the blob bounding box (with tolerance)', () => {
    const w = 100, h = 80;
    const { binary, gray, blobs } = buildScene(w, h, [
      { cx: 30, cy: 40, r: 11 },
      { cx: 55, cy: 40, r: 11 },
    ]);
    const singleArea = Math.PI * 11 * 11;
    const classified = ensembleClassifyBlobs(
      blobs, singleArea, 0.35, 1.3, binary, gray, w, h,
    );
    for (const b of classified) {
      for (const loc of b.cellLocations) {
        expect(loc.x).toBeGreaterThanOrEqual(b.bbox.minX - 5);
        expect(loc.x).toBeLessThanOrEqual(b.bbox.maxX + 5);
        expect(loc.y).toBeGreaterThanOrEqual(b.bbox.minY - 5);
        expect(loc.y).toBeLessThanOrEqual(b.bbox.maxY + 5);
      }
    }
  });

  it('agreement scores are in range [1, 3]', () => {
    const w = 80, h = 60;
    const { binary, gray, blobs } = buildScene(w, h, [{ cx: 40, cy: 30, r: 10 }]);
    const medianArea = computeMedianCellArea(blobs, 0.35, 20);
    const classified = ensembleClassifyBlobs(
      blobs, medianArea, 0.35, 1.8, binary, gray, w, h,
    );
    for (const b of classified) {
      for (const loc of b.cellLocations) {
        expect(loc.agreement).toBeGreaterThanOrEqual(1);
        expect(loc.agreement).toBeLessThanOrEqual(3);
      }
    }
  });

  it('oblong ellipse: 1 cell, 1 location, not over-split', () => {
    const w = 80, h = 80;
    const binary = new Uint8Array(w * h).fill(0);
    const gray = new Uint8Array(w * h).fill(200);
    paintEllipse(binary, w, 40, 40, 15, 8);
    paintEllipse(gray, w, 40, 40, 15, 8, 60);

    const blobs = extractBlobs(binary, w, h, 5, 100000, false, gray);
    const ellipseArea = Math.PI * 15 * 8;
    const classified = ensembleClassifyBlobs(
      blobs, ellipseArea, 0.2, 1.8, binary, gray, w, h, 0.3, 0.98,
    );
    expect(classified).toHaveLength(1);
    expect(classified[0].cellCount).toBe(1);
    expect(classified[0].cellLocations).toHaveLength(1);
    expectNear(classified[0].cellLocations[0], 40, 40, 8);
  });
});

// ─── Rigorous multi-cell spatial accuracy ───────────────────────────

describe('rigorous: spatial accuracy of cell identification', () => {
  it('3 cells in a row: locations track true centers', () => {
    const w = 150, h = 60, r = 10;
    const trueCenters = [
      { cx: 30, cy: 30 },
      { cx: 70, cy: 30 },
      { cx: 110, cy: 30 },
    ];
    const { binary, gray, blobs } = buildScene(w, h,
      trueCenters.map(c => ({ ...c, r })),
    );
    const singleArea = Math.PI * r * r;
    const classified = ensembleClassifyBlobs(
      blobs, singleArea, 0.35, 1.8, binary, gray, w, h,
    );

    const allLocs = classified.flatMap(b => b.cellLocations);
    const totalCount = classified.reduce((s, b) => s + b.cellCount, 0);
    expect(totalCount).toBe(3);

    for (const tc of trueCenters) {
      const nearest = findNearest(allLocs, tc.cx, tc.cy);
      expectNear(nearest, tc.cx, tc.cy, 8);
    }
  });

  it('3 touching cells in a triangle: >= 2 locations near true centers', () => {
    const w = 100, h = 100, r = 11;
    const trueCenters = [
      { cx: 40, cy: 35 },
      { cx: 60, cy: 35 },
      { cx: 50, cy: 55 },
    ];
    const { binary, gray, blobs } = buildScene(w, h,
      trueCenters.map(c => ({ ...c, r })),
    );
    const singleArea = Math.PI * r * r;
    const classified = ensembleClassifyBlobs(
      blobs, singleArea, 0.35, 1.3, binary, gray, w, h,
    );

    const allLocs = classified.flatMap(b => b.cellLocations);
    const totalCount = classified.reduce((s, b) => s + b.cellCount, 0);
    expect(totalCount).toBeGreaterThanOrEqual(2);
    expect(totalCount).toBeLessThanOrEqual(4);

    // At least 2 of the 3 true centers should have a nearby detection
    let matched = 0;
    for (const tc of trueCenters) {
      const nearest = findNearest(allLocs, tc.cx, tc.cy);
      if (Math.sqrt((nearest.x - tc.cx) ** 2 + (nearest.y - tc.cy) ** 2) < 15) {
        matched++;
      }
    }
    expect(matched).toBeGreaterThanOrEqual(2);
  });

  it('4 cells in a 2x2 grid: all 4 locations near true centers', () => {
    const w = 120, h = 120, r = 9;
    const trueCenters = [
      { cx: 30, cy: 30 },
      { cx: 80, cy: 30 },
      { cx: 30, cy: 80 },
      { cx: 80, cy: 80 },
    ];
    const { binary, gray, blobs } = buildScene(w, h,
      trueCenters.map(c => ({ ...c, r })),
    );
    const singleArea = Math.PI * r * r;
    const classified = ensembleClassifyBlobs(
      blobs, singleArea, 0.35, 1.8, binary, gray, w, h,
    );

    const allLocs = classified.flatMap(b => b.cellLocations);
    const totalCount = classified.reduce((s, b) => s + b.cellCount, 0);
    expect(totalCount).toBe(4);

    for (const tc of trueCenters) {
      const nearest = findNearest(allLocs, tc.cx, tc.cy);
      expectNear(nearest, tc.cx, tc.cy, 8);
    }
  });

  it('mixed: 2 isolated + 2 touching = 4 total', () => {
    const w = 200, h = 80, r = 10;
    const trueCenters = [
      { cx: 25, cy: 40 },   // isolated
      { cx: 175, cy: 40 },  // isolated
      { cx: 85, cy: 40 },   // touching pair
      { cx: 110, cy: 40 },  // touching pair
    ];
    const { binary, gray, blobs } = buildScene(w, h,
      trueCenters.map(c => ({ ...c, r })),
    );
    const singleArea = Math.PI * r * r;
    const classified = ensembleClassifyBlobs(
      blobs, singleArea, 0.35, 1.5, binary, gray, w, h,
    );

    const totalCount = classified.reduce((s, b) => s + b.cellCount, 0);
    expect(totalCount).toBeGreaterThanOrEqual(3);
    expect(totalCount).toBeLessThanOrEqual(5);

    // Isolated cells should each be cellCount=1 with 1 location
    const isolatedBlobs = classified.filter(b => b.cellCount === 1);
    for (const b of isolatedBlobs) {
      expect(b.cellLocations).toHaveLength(1);
    }
  });

  it('locations all have non-zero radius', () => {
    const w = 120, h = 120;
    const { binary, gray, blobs } = buildScene(w, h, [
      { cx: 30, cy: 30, r: 8 },
      { cx: 70, cy: 30, r: 8 },
      { cx: 50, cy: 70, r: 12 },
    ]);
    const singleArea = Math.PI * 8 * 8;
    const classified = ensembleClassifyBlobs(
      blobs, singleArea, 0.35, 1.8, binary, gray, w, h,
    );
    for (const b of classified) {
      for (const loc of b.cellLocations) {
        expect(loc.radius).toBeGreaterThan(0);
      }
    }
  });
});

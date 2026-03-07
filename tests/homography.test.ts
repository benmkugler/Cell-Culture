import { describe, it, expect } from 'vitest';
import { solveLinearSystem, computeHomography } from '../src/cv/homography';

describe('solveLinearSystem', () => {
  it('solves a 2x2 identity system', () => {
    const A = [[1, 0], [0, 1]];
    const B = [3, 7];
    const x = solveLinearSystem(A, B);
    expect(x[0]).toBeCloseTo(3);
    expect(x[1]).toBeCloseTo(7);
  });

  it('solves a 3x3 system', () => {
    // 2x + y - z = 8
    //  -3x - y + 2z = -11
    //  -2x + y + 2z = -3
    const A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]];
    const B = [8, -11, -3];
    const x = solveLinearSystem(A, B);
    expect(x[0]).toBeCloseTo(2);
    expect(x[1]).toBeCloseTo(3);
    expect(x[2]).toBeCloseTo(-1);
  });

  it('does not mutate input arrays', () => {
    const A = [[1, 2], [3, 4]];
    const B = [5, 6];
    const aCopy = JSON.parse(JSON.stringify(A));
    const bCopy = [...B];
    solveLinearSystem(A, B);
    expect(A).toEqual(aCopy);
    expect(B).toEqual(bCopy);
  });

  it('throws on singular matrix', () => {
    const A = [[1, 2], [2, 4]];
    const B = [3, 6];
    expect(() => solveLinearSystem(A, B)).toThrow(/singular/i);
  });
});

describe('computeHomography', () => {
  it('computes identity-like homography for coincident points', () => {
    const pts = [
      { x: 0, y: 0 }, { x: 100, y: 0 },
      { x: 100, y: 100 }, { x: 0, y: 100 },
    ];
    const H = computeHomography(pts, pts);
    // Should be close to identity (h0≈1, h4≈1, h8=1, rest ≈0)
    expect(H[0]).toBeCloseTo(1, 4);
    expect(H[4]).toBeCloseTo(1, 4);
    expect(H[8]).toBeCloseTo(1, 4);
    expect(H[1]).toBeCloseTo(0, 4);
    expect(H[3]).toBeCloseTo(0, 4);
  });

  it('throws if not exactly 4 points', () => {
    const pts = [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }];
    expect(() => computeHomography(pts, pts)).toThrow(/4 point/);
  });
});

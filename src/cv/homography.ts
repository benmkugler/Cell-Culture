import type { Point } from './types';

/**
 * Solve Ax = B via Gaussian elimination with partial pivoting.
 * Does NOT mutate inputs.
 */
export const solveLinearSystem = (Ain: number[][], Bin: number[]): number[] => {
  const n = Ain.length;
  // Deep-copy augmented matrix
  const A = Ain.map((row, i) => [...row, Bin[i]]);

  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxVal = Math.abs(A[col][col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(A[row][col]);
      if (v > maxVal) { maxVal = v; maxRow = row; }
    }
    if (maxVal < 1e-12) throw new Error(`Singular matrix at column ${col}`);
    [A[maxRow], A[col]] = [A[col], A[maxRow]];

    // Forward elimination
    for (let row = col + 1; row < n; row++) {
      const factor = A[row][col] / A[col][col];
      for (let j = col; j <= n; j++) {
        A[row][j] -= factor * A[col][j];
      }
    }
  }

  // Back substitution
  const x = new Array<number>(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = A[i][n];
    for (let j = i + 1; j < n; j++) sum -= A[i][j] * x[j];
    x[i] = sum / A[i][i];
  }
  return x;
};

/**
 * Compute 3×3 homography matrix (flattened, row-major) mapping src→dst.
 * Both arrays must have exactly 4 corresponding points.
 */
export const computeHomography = (src: Point[], dst: Point[]): Float64Array => {
  if (src.length !== 4 || dst.length !== 4) throw new Error('Exactly 4 point correspondences required');

  const A: number[][] = [];
  const B: number[] = [];
  for (let i = 0; i < 4; i++) {
    const { x: sx, y: sy } = src[i];
    const { x: dx, y: dy } = dst[i];
    A.push([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx]);
    A.push([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy]);
    B.push(dx, dy);
  }
  const h = solveLinearSystem(A, B);
  const H = new Float64Array(9);
  for (let i = 0; i < 8; i++) H[i] = h[i];
  H[8] = 1;
  return H;
};

/**
 * Warp source image data into the destination canvas using inverse mapping.
 * Uses bilinear interpolation for sub-pixel accuracy.
 */
export const warpPerspective = (
  srcData: Uint8ClampedArray,
  srcW: number,
  srcH: number,
  corners: Point[],
  outW: number,
  outH: number,
): ImageData => {
  const srcPixels = corners.map(p => ({ x: p.x * srcW, y: p.y * srcH }));
  const dstPixels: Point[] = [
    { x: 0, y: 0 }, { x: outW, y: 0 },
    { x: outW, y: outH }, { x: 0, y: outH },
  ];
  const H = computeHomography(dstPixels, srcPixels);

  const out = new ImageData(outW, outH);
  const od = out.data;

  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const w = H[6] * x + H[7] * y + H[8];
      const u = (H[0] * x + H[1] * y + H[2]) / w;
      const v = (H[3] * x + H[4] * y + H[5]) / w;

      // Bilinear interpolation
      const u0 = Math.floor(u), v0 = Math.floor(v);
      const u1 = u0 + 1, v1 = v0 + 1;
      const du = u - u0, dv = v - v0;

      const di = (y * outW + x) * 4;
      if (u0 >= 0 && u1 < srcW && v0 >= 0 && v1 < srcH) {
        const i00 = (v0 * srcW + u0) * 4;
        const i10 = (v0 * srcW + u1) * 4;
        const i01 = (v1 * srcW + u0) * 4;
        const i11 = (v1 * srcW + u1) * 4;

        for (let c = 0; c < 3; c++) {
          od[di + c] = Math.round(
            (1 - du) * (1 - dv) * srcData[i00 + c] +
            du * (1 - dv) * srcData[i10 + c] +
            (1 - du) * dv * srcData[i01 + c] +
            du * dv * srcData[i11 + c]
          );
        }
        od[di + 3] = 255;
      }
    }
  }
  return out;
};

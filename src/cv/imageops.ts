/**
 * Image processing primitives operating on flat grayscale arrays.
 * All functions are pure — they return new arrays.
 */

/** Convert RGBA ImageData to grayscale using BT.601 luminance weights. */
export const toGrayscale = (data: Uint8ClampedArray, len: number): Uint8Array => {
  const gray = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    const j = i * 4;
    gray[i] = Math.round(0.299 * data[j] + 0.587 * data[j + 1] + 0.114 * data[j + 2]);
  }
  return gray;
};

/**
 * Compute Otsu's threshold for a grayscale image.
 * Returns the optimal threshold that maximises inter-class variance.
 */
export const otsuThreshold = (gray: Uint8Array): number => {
  const hist = new Int32Array(256);
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++;

  const total = gray.length;
  let sumAll = 0;
  for (let t = 0; t < 256; t++) sumAll += t * hist[t];

  let sumBg = 0, wBg = 0;
  let maxVariance = 0, bestThresh = 0;

  for (let t = 0; t < 256; t++) {
    wBg += hist[t];
    if (wBg === 0) continue;
    const wFg = total - wBg;
    if (wFg === 0) break;

    sumBg += t * hist[t];
    const meanBg = sumBg / wBg;
    const meanFg = (sumAll - sumBg) / wFg;
    const variance = wBg * wFg * (meanBg - meanFg) ** 2;

    if (variance > maxVariance) {
      maxVariance = variance;
      bestThresh = t;
    }
  }
  return bestThresh;
};

/**
 * 1D Gaussian kernel (normalised).
 */
const makeGaussianKernel = (sigma: number): Float64Array => {
  const radius = Math.ceil(sigma * 3);
  const size = 2 * radius + 1;
  const kernel = new Float64Array(size);
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const x = i - radius;
    kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += kernel[i];
  }
  for (let i = 0; i < size; i++) kernel[i] /= sum;
  return kernel;
};

/**
 * Separable Gaussian blur on a grayscale image.
 */
export const gaussianBlur = (gray: Uint8Array, w: number, h: number, sigma: number): Uint8Array => {
  if (sigma <= 0) return new Uint8Array(gray);
  const kernel = makeGaussianKernel(sigma);
  const r = (kernel.length - 1) / 2;

  // Horizontal pass
  const tmp = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let val = 0;
      for (let k = -r; k <= r; k++) {
        const sx = Math.min(w - 1, Math.max(0, x + k));
        val += gray[y * w + sx] * kernel[k + r];
      }
      tmp[y * w + x] = val;
    }
  }

  // Vertical pass
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let val = 0;
      for (let k = -r; k <= r; k++) {
        const sy = Math.min(h - 1, Math.max(0, y + k));
        val += tmp[sy * w + x] * kernel[k + r];
      }
      out[y * w + x] = Math.round(Math.max(0, Math.min(255, val)));
    }
  }
  return out;
};

/** Binary threshold. Returns Uint8Array of 0/1 values. */
export const binaryThreshold = (gray: Uint8Array, thresh: number, invert: boolean): Uint8Array => {
  const out = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    out[i] = (invert ? gray[i] < thresh : gray[i] > thresh) ? 1 : 0;
  }
  return out;
};

/**
 * Detect grid row and column indices from projection profiles.
 * Returns sets of row (y) and column (x) indices classified as grid lines.
 */
export const detectGridLines = (
  binary: Uint8Array, w: number, h: number, fillRatio = 0.55,
): { rows: Set<number>; cols: Set<number> } => {
  const rowSum = new Int32Array(h);
  const colSum = new Int32Array(w);

  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (binary[y * w + x]) { rowSum[y]++; colSum[x]++; }

  const rowThresh = w * fillRatio;
  const colThresh = h * fillRatio;

  const rows = new Set<number>();
  const cols = new Set<number>();

  for (let y = 0; y < h; y++) if (rowSum[y] > rowThresh) rows.add(y);
  for (let x = 0; x < w; x++) if (colSum[x] > colThresh) cols.add(x);

  return { rows, cols };
};

/**
 * Build a boolean mask of all pixels within `radius` of any detected grid line.
 * Used downstream to filter phantom blobs that are grid-line fragments.
 */
export const buildGridZone = (
  w: number, h: number,
  gridRows: Set<number>, gridCols: Set<number>,
  radius: number,
): Uint8Array => {
  const zone = new Uint8Array(w * h);
  for (const y of gridRows) {
    for (let dy = -radius; dy <= radius; dy++) {
      const ny = y + dy;
      if (ny >= 0 && ny < h) {
        for (let x = 0; x < w; x++) zone[ny * w + x] = 1;
      }
    }
  }
  for (const x of gridCols) {
    for (let dx = -radius; dx <= radius; dx++) {
      const nx = x + dx;
      if (nx >= 0 && nx < w) {
        for (let y = 0; y < h; y++) zone[y * w + nx] = 1;
      }
    }
  }
  return zone;
};

/**
 * Remove grid lines using projection profiles with dilation.
 * A row/column is classified as a grid line if its fill ratio exceeds `fillRatio`.
 * Clears a band of ±`dilateRadius` around each detected line to handle thick lines.
 * Returns a new binary array with grid lines zeroed.
 */
export const removeGridLines = (
  binary: Uint8Array, w: number, h: number,
  fillRatio = 0.55, dilateRadius = 2,
): Uint8Array => {
  const out = new Uint8Array(binary);
  const { rows, cols } = detectGridLines(binary, w, h, fillRatio);

  // Zero grid rows + dilation band
  for (const y of rows) {
    for (let dy = -dilateRadius; dy <= dilateRadius; dy++) {
      const ny = y + dy;
      if (ny >= 0 && ny < h) {
        for (let x = 0; x < w; x++) out[ny * w + x] = 0;
      }
    }
  }
  // Zero grid columns + dilation band
  for (const x of cols) {
    for (let dx = -dilateRadius; dx <= dilateRadius; dx++) {
      const nx = x + dx;
      if (nx >= 0 && nx < w) {
        for (let y = 0; y < h; y++) out[y * w + nx] = 0;
      }
    }
  }
  return out;
};

/**
 * Morphological erosion with a square structuring element of given radius.
 * A pixel survives only if all pixels within the kernel are set.
 */
export const erode = (binary: Uint8Array, w: number, h: number, radius: number): Uint8Array => {
  const out = new Uint8Array(w * h);
  for (let y = radius; y < h - radius; y++) {
    for (let x = radius; x < w - radius; x++) {
      let all = true;
      outer:
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (!binary[(y + dy) * w + (x + dx)]) { all = false; break outer; }
        }
      }
      if (all) out[y * w + x] = 1;
    }
  }
  return out;
};

/**
 * Morphological dilation with a square structuring element of given radius.
 */
export const dilate = (binary: Uint8Array, w: number, h: number, radius: number): Uint8Array => {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!binary[y * w + x]) continue;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const ny = y + dy, nx = x + dx;
          if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
            out[ny * w + nx] = 1;
          }
        }
      }
    }
  }
  return out;
};

/** Morphological opening = erode then dilate. */
export const morphOpen = (binary: Uint8Array, w: number, h: number, radius: number): Uint8Array => {
  return dilate(erode(binary, w, h, radius), w, h, radius);
};

/** Morphological closing = dilate then erode. */
export const morphClose = (binary: Uint8Array, w: number, h: number, radius: number): Uint8Array => {
  return erode(dilate(binary, w, h, radius), w, h, radius);
};

/**
 * Adaptive illumination correction via local-mean subtraction.
 * Estimates the background illumination field using a large-kernel box blur,
 * then subtracts it to flatten uneven lighting, ripples, and vignetting.
 * Output is normalised to [0, 255].
 *
 * @param blockSize Side length of the local neighbourhood (must be odd, ≥3).
 */
export const adaptiveIlluminationCorrection = (
  gray: Uint8Array, w: number, h: number, blockSize: number,
): Uint8Array => {
  const bs = Math.max(3, blockSize | 1); // force odd
  const r = (bs - 1) / 2;

  // Integral image for O(1) box mean
  const integral = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    for (let x = 0; x < w; x++) {
      rowSum += gray[y * w + x];
      integral[(y + 1) * (w + 1) + (x + 1)] =
        rowSum + integral[y * (w + 1) + (x + 1)];
    }
  }

  const boxMean = (x: number, y: number): number => {
    const x0 = Math.max(0, x - r), y0 = Math.max(0, y - r);
    const x1 = Math.min(w - 1, x + r), y1 = Math.min(h - 1, y + r);
    const area = (x1 - x0 + 1) * (y1 - y0 + 1);
    const sum =
      integral[(y1 + 1) * (w + 1) + (x1 + 1)] -
      integral[y0 * (w + 1) + (x1 + 1)] -
      integral[(y1 + 1) * (w + 1) + x0] +
      integral[y0 * (w + 1) + x0];
    return sum / area;
  };

  // Subtract local mean, remap to [0,255]
  const diff = new Float64Array(w * h);
  let minVal = Infinity, maxVal = -Infinity;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const d = gray[y * w + x] - boxMean(x, y);
      diff[y * w + x] = d;
      if (d < minVal) minVal = d;
      if (d > maxVal) maxVal = d;
    }
  }

  const out = new Uint8Array(w * h);
  const range = maxVal - minVal || 1;
  for (let i = 0; i < w * h; i++) {
    out[i] = Math.round(((diff[i] - minVal) / range) * 255);
  }
  return out;
};

/**
 * 2D median filter for salt-and-pepper noise / small debris removal.
 * Uses a square window of `2*radius+1`.
 */
export const medianFilter = (
  gray: Uint8Array, w: number, h: number, radius: number,
): Uint8Array => {
  if (radius <= 0) return new Uint8Array(gray);
  const out = new Uint8Array(w * h);
  const windowSize = (2 * radius + 1) ** 2;
  const buf = new Uint8Array(windowSize);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const ny = Math.min(h - 1, Math.max(0, y + dy));
          const nx = Math.min(w - 1, Math.max(0, x + dx));
          buf[count++] = gray[ny * w + nx];
        }
      }
      // Partial sort to find median
      buf.subarray(0, count).sort();
      out[y * w + x] = buf[count >> 1];
    }
  }
  return out;
};

/**
 * Compute local intensity variance within each blob's pixels.
 * Returns the standard deviation of gray values within the pixel set.
 */
export const blobIntensityStdDev = (
  gray: Uint8Array, pixels: number[],
): number => {
  if (pixels.length === 0) return 0;
  let sum = 0, sumSq = 0;
  for (const p of pixels) {
    const v = gray[p];
    sum += v;
    sumSq += v * v;
  }
  const mean = sum / pixels.length;
  const variance = sumSq / pixels.length - mean * mean;
  return Math.sqrt(Math.max(0, variance));
};

/**
 * Contrast Limited Adaptive Histogram Equalization (CLAHE).
 * Divides the image into tiles, computes clipped histograms per tile,
 * equalises each, then bilinearly interpolates between tiles for seamless output.
 *
 * @param tileSize Number of tiles along each axis (e.g., 8 → 8×8 grid).
 * @param clipLimit Contrast clip limit (normalised to tile histogram). Higher = more contrast.
 */
export const clahe = (
  gray: Uint8Array, w: number, h: number,
  tileSize = 8, clipLimit = 2.0,
): Uint8Array => {
  const nTilesX = Math.max(1, tileSize);
  const nTilesY = Math.max(1, tileSize);
  const tileW = w / nTilesX;
  const tileH = h / nTilesY;

  // Compute clipped + redistributed CDF per tile
  const cdfs: Float64Array[] = [];

  for (let ty = 0; ty < nTilesY; ty++) {
    for (let tx = 0; tx < nTilesX; tx++) {
      const x0 = Math.round(tx * tileW);
      const y0 = Math.round(ty * tileH);
      const x1 = Math.round((tx + 1) * tileW);
      const y1 = Math.round((ty + 1) * tileH);

      const hist = new Float64Array(256);
      let count = 0;
      for (let y = y0; y < y1 && y < h; y++) {
        for (let x = x0; x < x1 && x < w; x++) {
          hist[gray[y * w + x]]++;
          count++;
        }
      }
      if (count === 0) {
        cdfs.push(new Float64Array(256));
        continue;
      }

      // Clip histogram
      const avgBin = count / 256;
      const limit = Math.max(1, clipLimit * avgBin);
      let excess = 0;
      for (let i = 0; i < 256; i++) {
        if (hist[i] > limit) {
          excess += hist[i] - limit;
          hist[i] = limit;
        }
      }
      // Redistribute excess uniformly
      const redistPerBin = excess / 256;
      for (let i = 0; i < 256; i++) hist[i] += redistPerBin;

      // CDF
      const cdf = new Float64Array(256);
      cdf[0] = hist[0];
      for (let i = 1; i < 256; i++) cdf[i] = cdf[i - 1] + hist[i];
      // Normalise to [0, 255]
      const cdfMin = cdf[0];
      const denom = cdf[255] - cdfMin || 1;
      for (let i = 0; i < 256; i++) {
        cdf[i] = ((cdf[i] - cdfMin) / denom) * 255;
      }
      cdfs.push(cdf);
    }
  }

  const getCdf = (tx: number, ty: number): Float64Array =>
    cdfs[ty * nTilesX + tx];

  // Bilinear interpolation across tiles
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // Map pixel to tile-center coordinates
      const fx = (x / tileW) - 0.5;
      const fy = (y / tileH) - 0.5;

      const tx0 = Math.max(0, Math.min(nTilesX - 1, Math.floor(fx)));
      const ty0 = Math.max(0, Math.min(nTilesY - 1, Math.floor(fy)));
      const tx1 = Math.min(nTilesX - 1, tx0 + 1);
      const ty1 = Math.min(nTilesY - 1, ty0 + 1);

      const ax = Math.max(0, Math.min(1, fx - tx0));
      const ay = Math.max(0, Math.min(1, fy - ty0));

      const v = gray[y * w + x];

      const f00 = getCdf(tx0, ty0)[v];
      const f10 = getCdf(tx1, ty0)[v];
      const f01 = getCdf(tx0, ty1)[v];
      const f11 = getCdf(tx1, ty1)[v];

      const val = (1 - ax) * (1 - ay) * f00 +
                  ax * (1 - ay) * f10 +
                  (1 - ax) * ay * f01 +
                  ax * ay * f11;

      out[y * w + x] = Math.round(Math.max(0, Math.min(255, val)));
    }
  }
  return out;
};

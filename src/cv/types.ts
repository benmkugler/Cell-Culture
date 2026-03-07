/** Normalized [0,1] coordinate. */
export type Point = { x: number; y: number };

/** Bounding box in pixel space. */
export type BBox = {
  minX: number; maxX: number;
  minY: number; maxY: number;
};

/** Estimated location of a single cell within a blob. */
export type CellLocation = {
  x: number;
  y: number;
  /** Estimated cell radius in pixels. */
  radius: number;
  /** Number of strategies (1–3) that agreed on this location. */
  agreement: number;
};

/** Connected-component blob descriptor. */
export type Blob = {
  area: number;
  perimeter: number;
  cx: number;
  cy: number;
  bbox: BBox;
  circularity: number;
  /** Number of cells this blob represents (≥1 for clusters). */
  cellCount: number;
  /** Estimated locations of individual cells within this blob. */
  cellLocations: CellLocation[];
  /** Indices into the flat pixel array. */
  pixels: number[];
  /** Whether the blob touches the image boundary. */
  touchesBorder: boolean;
  /** Solidity = area / convex_hull_area. 1.0 = perfectly convex. */
  solidity: number;
  /** Eccentricity from moment analysis. 0 = circle, →1 = elongated. */
  eccentricity: number;
  /** Standard deviation of grayscale intensity within the blob. */
  intensityStdDev: number;
};

/** Parameters for the CV pipeline. */
export type PipelineParams = {
  threshold: number;
  autoThreshold: boolean;
  invert: boolean;
  showMask: boolean;
  minCellSize: number;
  maxCellSize: number;
  circularityThresh: number;
  morphRadius: number;
  excludeBorder: boolean;
  gaussianSigma: number;
  clusterSplitRatio: number;
  /** Enable adaptive illumination correction for uneven lighting / ripples. */
  adaptiveIllumination: boolean;
  /** Block size for adaptive illumination (must be odd, ≥3). Larger = coarser correction. */
  illuminationBlockSize: number;
  /** Median filter radius for salt-and-pepper noise / small debris. 0 = disabled. */
  medianFilterRadius: number;
  /** Minimum solidity to accept a blob as a cell (rejects irregular debris). */
  minSolidity: number;
  /** Maximum eccentricity to accept (rejects elongated debris/scratches). */
  maxEccentricity: number;
  /** Maximum intensity std-dev within a blob (rejects bubbles/ripples). */
  maxIntensityStdDev: number;
  /** Use 8-connected flood fill instead of 4-connected. Merges diagonally-touching cells. */
  use8Connected: boolean;
  /** Enable CLAHE (Contrast Limited Adaptive Histogram Equalization). */
  enableCLAHE: boolean;
  /** CLAHE tile grid size (side length). */
  claheTileSize: number;
  /** CLAHE clip limit for contrast limiting. */
  claheClipLimit: number;
  /** Minimum ensemble agreement (1–3) to accept a cell location. */
  minAgreement: number;
};

/** Per-blob confidence record for downstream analysis. */
export type BlobStats = {
  blobIndex: number;
  area: number;
  circularity: number;
  solidity: number;
  eccentricity: number;
  cellCount: number;
  confidence: number;
  agreement: number;
};

/** Aggregate statistical output for the pipeline. */
export type PipelineStats = {
  /** Total cell count. */
  totalCount: number;
  /** Mean area of reference single cells (px²). */
  meanCellArea: number;
  /** Std-dev of single-cell areas (px²). */
  stdCellArea: number;
  /** Coefficient of variation (stddev / mean) for single-cell areas. */
  cvCellArea: number;
  /** 95% confidence interval for the total count [lower, upper]. */
  ci95: [number, number];
  /** Mean ensemble confidence across all blobs. */
  meanConfidence: number;
  /** Per-blob statistics. */
  perBlob: BlobStats[];
  /** Median cell diameter (px). */
  medianDiameter: number;
  /** [min, max] diameter range of detected single cells (px). */
  diameterRange: [number, number];
};

export type PipelineResult = {
  count: number;
  blobs: Blob[];
  dataUrl: string;
  otsuThreshold: number;
  /** Statistical analysis of the detection results. */
  stats: PipelineStats;
};

export type ProcessedImage = {
  id: string;
  originalSrc: string;
  count: number;
  processedSrc?: string;
};

export type FlaskData = {
  id: string;
  name: string;
  images: ProcessedImage[];
};

export const DEFAULT_PARAMS: PipelineParams = {
  threshold: 128,
  autoThreshold: true,
  invert: false,
  showMask: false,
  minCellSize: 15,
  maxCellSize: 8000,
  circularityThresh: 0.30,
  morphRadius: 1,
  excludeBorder: true,
  gaussianSigma: 1.0,
  clusterSplitRatio: 1.5,
  adaptiveIllumination: true,
  illuminationBlockSize: 61,
  medianFilterRadius: 1,
  minSolidity: 0.45,
  maxEccentricity: 0.95,
  maxIntensityStdDev: 90,
  use8Connected: true,
  enableCLAHE: false,
  claheTileSize: 8,
  claheClipLimit: 2.0,
  minAgreement: 1,
};

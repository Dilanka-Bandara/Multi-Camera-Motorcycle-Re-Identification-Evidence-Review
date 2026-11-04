// Shared contracts from backend/app/schemas.py. Coordinates are normalized.
export type Camera = 'A' | 'B';
export type XY = [number, number];
export interface BBox { x1: number; y1: number; x2: number; y2: number }
export interface Detection {
  frame_idx: number; t_s: number; bbox: BBox; conf: number; cls: number; track_id: number;
}
export interface CropInfo {
  frame_idx: number; t_s: number; bbox: BBox; quality: number; sharpness: number;
  truncated: boolean; occluded: boolean; path_full: string;
  path_rider: string | null; path_vehicle: string;
}
export interface FrameBox { frame_idx: number; bbox: BBox }
export interface Track {
  track_id: string; camera: Camera; raw_track_ids: number[];
  first_frame: number; last_frame: number; t_start_s: number; t_end_s: number;
  boxes: FrameBox[]; n_frames: number; mean_conf: number; median_box_h_px: number;
  entry_point: XY; exit_point: XY; motion_vector: XY; quality: number;
  quality_breakdown: Record<string, number>; crops: CropInfo[]; has_rider: boolean;
}
export interface TrackRepresentation {
  track_id: string; emb_full: number[][]; emb_full_mean: number[];
  emb_rider: number[][] | null; emb_vehicle: number[][]; color_hist: number[];
  lab_mean: [number, number, number];
}
export interface EvidenceItem {
  check: string; polarity: 'supports' | 'contradicts' | 'neutral' | 'missing';
  severity: 'low' | 'medium' | 'high' | 'critical'; value: number | string | null;
  threshold: number | string | null; message: string;
}
export interface Verification {
  status: 'accepted' | 'uncertain' | 'rejected'; confidence: number;
  features: Record<string, number>; evidence: EvidenceItem[]; summary: string;
  keypoint_viz_path: string | null;
}
export interface CandidateScores {
  visual: number; color: number; temporal: number; direction: number; combined: number;
}
export interface Candidate {
  pair_id: string; query_track_id: string; candidate_track_id: string; rank: number;
  scores: CandidateScores; delta_t_s: number; delta_t_adjusted_s: number;
  ambiguous: boolean; competing_pair_ids: string[]; assigned: boolean;
  verification: Verification | null;
}
export interface QueryResult {
  query_track_id: string; outcome: 'matched' | 'uncertain' | 'no_reliable_match';
  best_pair_id: string | null; candidates: Candidate[];
}
export interface RunStatus {
  run_id: string; state: 'queued' | 'running' | 'done' | 'failed'; stage: string;
  progress: number; error: string | null; timings_s: Record<string, number>;
  fps_processed: Record<string, number>;
}
export interface VideoMeta {
  path: string; fps: number; frame_count: number; width: number; height: number;
  duration_s: number; start_time_s: number;
}
export interface ClockOffset {
  mode: 'estimate' | 'fixed' | 'none'; estimated_s: number; n_pairs_used: number;
  reliable: boolean;
}
export interface RunResult {
  run_id: string; created_at: string; config_snapshot: Record<string, unknown>;
  videos: Partial<Record<Camera, VideoMeta>>; clock_offset: ClockOffset;
  tracks_A: Track[]; tracks_B: Track[]; queries: QueryResult[];
  metrics: Record<string, unknown> | null; timings: Record<string, number>;
}

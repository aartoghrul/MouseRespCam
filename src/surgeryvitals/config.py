from dataclasses import dataclass


@dataclass
class Params:
    # --- automatic localisation (whole-frame periodicity map) ---
    loc_width: int = 480            # frame is resized to this width for the whole-frame flow
    loc_block: int = 8              # block size (px at loc_width) -> ~32 px at 1080p
    loc_flow: str = 'farneback'     # 'farneback' (validated) or 'dis' (cheaper, misses weak breathing)
    loc_dis_preset: str = 'fast'    # ultrafast | fast | medium
    loc_stride: int = 2             # whole-frame flow on every n-th frame (15 Hz at 30 fps)
    loc_window_s: float = 20.0      # history used to score periodicity
    loc_min_s: float = 12.0         # minimum history before a region can be proposed
    loc_every_s: float = 5.0        # how often the map is refreshed while locked
    loc_search_every_s: float = 2.0 # how often while searching
    search_bpm: tuple = (30.0, 210.0)  # search band; floor excludes the surgeon's own breathing
    loc_min_score: float = 0.45     # mean periodicity of a region to accept it

    # --- anchor tracking (rigid scene around the ROI) ---
    reg_every: int = 15             # frames between registrations
    reg_scale: float = 0.5
    min_inliers: int = 25
    rekey_every_s: float = 5.0      # refresh the anchor reference while tracking is good
    lost_after_s: float = 3.0       # registration failing this long -> region lost

    # --- signal ---
    roi_target_px: int = 24         # min side of the down-scaled ROI fed to optical flow
    display_band: tuple = (0.2, 4.0)  # Hz, causal band-pass for the displayed waveform
    artifact_k: float = 3.0         # artifact if ROI flow magnitude > k * typical per-second peak (breath)
    artifact_floor: float = 0.3     # threshold limits (px/frame at ROI scale)
    artifact_ceiling: float = 3.0
    artifact_hold_s: float = 0.5

    # --- rate ---
    rate_window_s: float = 15.0
    rate_every_s: float = 1.0
    rate_bpm: tuple = (25.0, 200.0)  # floor excludes the surgeon's breathing (12-20 / min), which
                                     # reaches the region when their hand rests on the frame
    min_clean_s: float = 8.0        # rate is estimated on the longest artifact-free stretch of the window
    min_update_sqi: float = 0.3     # windows less periodic than this do not update the rate tracker
    hmm_sigma_bpm: float = 2.0      # expected drift per update
    relocate_after_s: float = 20.0  # poor signal this long -> accept a better region elsewhere

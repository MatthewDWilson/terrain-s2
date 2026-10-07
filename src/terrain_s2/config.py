"""Parameters. All distances in metres, areas in m², heights in m (T1: no cell units).

Defaults follow EDDIE_terrain_stage2_design.md §3–§4; all are initial values to
be tuned in Phase 2b.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class FeatureParams:
    relief_median_m: tuple[float, ...] = (5.0, 11.0, 21.0)   # high-pass median scales (channels)
    tophat_m: tuple[float, ...] = (11.0, 21.0, 41.0)          # morphological scales (ditches/channels/embankments)
    hessian_sigma_m: tuple[float, ...] = (2.0, 4.0)           # Gaussian derivative scales
    openness_radius_m: float = 20.0
    openness_directions: int = 8


@dataclass
class ChannelParams:
    # unsupervised channel map (§4.1 step 1); valley measures in 1/m, incision in m
    narrow_valley: float = 0.03           # Hessian valley measure, sigma 2 m
    narrow_incision: float = 0.15         # black top-hat 11 m
    wide_valley: float = 0.015            # Hessian valley measure, sigma 4 m
    wide_incision: float = 0.3            # black top-hat 21 m
    wide_min_local_incision: float = 0.1  # black top-hat 11 m, so broad hollows are not channels
    min_area_m2: float = 30.0
    min_length_m: float = 25.0            # centreline length per fragment
    slope_gate: float = 0.10              # on slopes steeper than this (m/m)...
    slope_min_upstream_m2: float = 1000.0  # ...a channel cell must also have this upstream area
    spur_m: float = 10.0                  # §4.1 step 4
    end_direction_m: float = 8.0          # outward direction of a channel end
    approach_m: float = 15.0              # approach direction/length of a channel end (Test A kind)
    # gap bridging (§4.1 step 5)
    raised_scale_m: float = 41.0          # white top-hat scale for "points at a raised feature"
    raised_min_m: float = 0.3
    lookahead_m: float = 30.0
    toe_distance_m: float = 10.0          # a channel this close to a raised feature...
    toe_parallel_deg: float = 30.0        # ...and within this angle of its axis runs along it
    toe_min_gradient: float = 0.02        # m/m of the smoothed top-hat: a raised feature is really there
    orientation_radius_m: float = 4.0
    bridge_max_m: float = 60.0
    geodesic_exclusion_m: float = 150.0   # network distances are traced up to this far...
    geodesic_ratio: float = 5.0           # ...and a target is the end's own channel if reachable along the
                                          # network within this multiple of the straight-line distance
    max_turn_deg: float = 60.0            # target must lie within this angle of the end's direction
    cost_floor: float = 0.05              # cost per metre on channel cells
    min_barrier_m: float = 0.3            # a bridged gap must rise at least this above both beds (= h_b)
    max_targets: int = 3                  # bridging targets kept per channel end (one per separate channel)


@dataclass
class HydroParams:
    breach_max_length_m: float = 100.0    # least-cost breach search limit (path length)
    breach_max_cost: float = -1.0         # m² of cut cross-section along path; <0 = unlimited
    breach_store_min_cut_m: float = 0.1   # keep path geometry only for breaches at least this deep
    breach_flat_step_cost: float = 1e-3   # tiny per-metre cost so flats prefer short paths


@dataclass
class CandidateParams:
    min_barrier_height_m: float = 0.3     # h_b
    max_barrier_length_m: float = 60.0    # L_b
    cut_threshold_m: float = 0.05         # a path cell is "in the barrier" if cut exceeds this
    bed_window_m: float = 5.0             # channel bed search either side of the barrier
    min_depression_depth_m: float = 0.3   # Test B
    min_depression_area_m2: float = 50.0  # Test B
    raised_tophat_scale_m: float = 41.0   # which top-hat scale defines the raised feature
    raised_min_height_m: float = 0.3
    elongation_window_m: float = 50.0     # radius; design says 30 m, but 30 m caps L/W at 60/W (see DESIGN_NOTES)
    min_elongation: float = 3.0
    channel_relief_scale_m: float = 11.0  # which median scale defines channel incision
    channel_min_incision_m: float = 0.15
    channel_probe_m: float = 5.0          # probe distance beyond the barrier ends
    merge_distance_m: float = 10.0        # candidates whose crests are closer are merged
    hard_gates: bool = True               # False: kind / vegetation / approach / L_b>=4 become attributes,
                                          # not exclusions (candidates for a learned ranking)
    min_cross_angle_deg: float = 45.0     # Test A: path vs raised-feature axis
    min_barrier_length_m: float = 4.0     # Test A: narrower barriers are fences, hedges, spoil banks
    barrier_frac: float = 0.25            # Test A: barrier = path cells above beds + frac * h_b
    veg_dsm_m: float = 1.0                # DSM - DEM above this counts as vegetation/structure
    veg_frac: float = 0.5                 # vegetation-affected if this fraction of the barrier is covered
    bank_parallel_deg: float = 30.0       # Test B kind "bank": raised axis within this of the channel
    transverse_deg: float = 45.0          # Test A: a channel at least this far off the barrier axis approaches it
    axis_local_m: float = 10.0            # barrier axis from the crest within this distance each side
    crest_snap_penalty: float = 0.02      # m per m of sideways offset when re-centring on the crest
    min_approach_m: float = 10.0          # Test A: the channel end must approach over at least this length
    continuation_min_deg: float = 30.0    # Test A continuation: both channels at least this far off the barrier axis
    continuation_max_dev_deg: float = 30.0  # ... and within this angle of each other (the same channel continues
                                          # through the barrier; Matt's rule, AOI2 culvert #2 at 42.6 deg, Oct 2026)
    obstruction_radius_m: float = 5.0     # Test B: cutline this close to the channel map...
    obstruction_frac: float = 0.95        # ...over this fraction of its length = channel_obstruction
    pond_max_cells: int = 2_000_000       # flood-fill limit when measuring the retained pond


@dataclass
class Params:
    features: FeatureParams = field(default_factory=FeatureParams)
    channels: ChannelParams = field(default_factory=ChannelParams)
    hydro: HydroParams = field(default_factory=HydroParams)
    candidates: CandidateParams = field(default_factory=CandidateParams)

    def to_dict(self) -> dict:
        return asdict(self)

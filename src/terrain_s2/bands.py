"""What each band of features.tif and hydro.tif is (written beside them as rasters.json)."""

FEATURES = {
    "relief_med{s}": ("m", "DEM minus its median over an {s} m window: negative in channels and ditches narrower "
                           "than ~{s} m, positive on banks and ridges"),
    "tophat_white{s}": ("m", "white top-hat at {s} m: height of raised features narrower than {s} m "
                             "(embankments, stopbanks, road formations, spoil banks)"),
    "tophat_black{s}": ("m", "black top-hat at {s} m: depth of incised features narrower than {s} m (drains, channels)"),
    "laplacian_s{s}": ("1/m", "Laplacian (Hessian trace) of the DEM smoothed with sigma {s} m: negative on crests, "
                             "positive in hollows"),
    "ridge_s{s}": ("1/m", "ridge line-likeness from Hessian eigenvalues at sigma {s} m (crests of embankments)"),
    "valley_s{s}": ("1/m", "valley line-likeness from Hessian eigenvalues at sigma {s} m (channel beds)"),
    "openness_pos": ("deg", "positive topographic openness, 20 m radius, 8 directions: low in incisions"),
    "openness_neg": ("deg", "negative topographic openness, 20 m radius, 8 directions: low on crests"),
}

HYDRO = {
    "hand_m": ("m", "height above nearest drainage (HAND) on the breached DEM; the floodplain is HAND <= --max-hand"),
    "floodplain": ("0/1", "floodplain mask (HAND <= --max-hand), before the 50 m buffer"),
    "log10_upstream_area_m2": ("log10 m2", "upstream area from D8 flow on the breached DEM, within this window only "
                                           "(truncated where flow enters across the window edge)"),
    "breach_depth_m": ("m", "DEM minus breached DEM: where least-cost breaching cut through a barrier"),
    "depression_depth_m": ("m", "filled DEM minus DEM: depressions (ponds behind barriers, true hollows)"),
    "channel_mask": ("0/1", "channel map (drain network mask, floodplain-limited)"),
    "channel_centreline": ("0/1", "channel centrelines as rasterised (with culvert links and river centrelines)"),
    "dsm_minus_dem_m": ("m", "DSM minus DEM: vegetation, buildings, bridge decks"),
    "road_distance_m": ("m", "distance to the nearest road centreline (from roads.gpkg)"),
}


def describe(names, table):
    """{band: {units, description}} for the bands present, matching the templated names above."""
    import re
    out = {}
    for n in names:
        for pat, (u, d) in table.items():
            rx = "^" + re.escape(pat).replace(re.escape("{s}"), r"(?P<s>[0-9.]+)") + "$"
            m = re.match(rx, n)
            if m:
                s = m.groupdict().get("s", "")
                out[n] = dict(units=u, description=d.replace("{s}", s))
                break
        else:
            out[n] = dict(units="", description="")
    return out


CONDITIONED = {
    "z_conditioned_m": ("m", "DEM conditioned with the mapped network: channel cells at their 3 x 3 bed less the burn "
                             "depth, culvert links and gap repairs at the bed interpolated between their ends"),
    "hand_m": ("m", "height above the mapped network (original DEM), along flow on the conditioned DEM; NaN where "
                    "flow leaves the window without meeting the network"),
    "log10_upstream_area_m2": ("log10 m2", "upstream area on the conditioned DEM, seeded at the window edge from REC2 "
                                           "where a river enters (see upstream_inflow_m2)"),
    "upstream_inflow_m2": ("m2", "upstream area added at the cell where a REC2 river enters the window (0 elsewhere)"),
    "residual_depression_m": ("m", "fill of the conditioned DEM minus the original DEM (>= 0): water the mapped "
                                   "network does not drain (the burn itself is not counted)"),
    "network": ("0/1", "the mapped network as rasterised for conditioning (drainage for HAND)"),
}

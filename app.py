import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import MiniMap
from streamlit_folium import st_folium
import branca.colormap as cm
from shapely.geometry import Point
from pyproj import Transformer
import matplotlib.colors as mcolors
import matplotlib.cm as mpl_cm

# ==========================================================
# Page configuration
# ==========================================================
st.set_page_config(
    layout="wide",
    page_title="Bakken Inventory Engine",
    page_icon="🛢️",
)

# ==========================================================
# Global constants
# ==========================================================
NULL_STYLE = {
    "fillColor": "#ffffff",
    "fillOpacity": 0,
    "color": "#888",
    "weight": 0.25,
}

DEFAULT_BUFFER_M = 500

# ==========================================================
# Helper utilities
# ==========================================================

def safe_range(series: pd.Series):
    vals = series.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        return (0.0, 1.0) if lo == 0.0 else (lo - abs(lo) * 0.1, lo + abs(lo) * 0.1)
    return lo, hi


def zscore(s: pd.Series) -> pd.Series:
    vals = s.replace([np.inf, -np.inf], np.nan)
    std = vals.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (vals - vals.mean()) / std


def midpoint_of_geom(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return geom.interpolate(0.5, normalized=True)
    elif geom.geom_type == "MultiLineString":
        longest = max(geom.geoms, key=lambda g: g.length)
        return longest.interpolate(0.5, normalized=True)
    elif geom.geom_type == "Point":
        return geom
    else:
        return geom.centroid


def get_ylgn_hex(value, vmin, vmax):
    if pd.isna(value) or vmin == vmax:
        return "#cccccc"
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl_cm.get_cmap("YlGn")
    rgba = cmap(norm(value))
    return mcolors.to_hex(rgba)


# ==========================================================
# Data loading (cached)
# ==========================================================

@st.cache_resource(show_spinner="Loading spatial data …")
def load_data():
    lines = gpd.read_file("lines.shp")
    points = gpd.read_file("points.shp")
    grid = gpd.read_file("ooipsectiongrid.shp")
    infills = gpd.read_file("2M_Infills_plyln.shp")
    merged = gpd.read_file("merged_inventory.shp")
    lease_lines = gpd.read_file("2M_LL_plyln.shp")
    units = gpd.read_file("Bakken Units.shp")
    land = gpd.read_file("Bakken Land.shp")

    # --- Excel: Sheet 1 = well-level, Sheet 2 = section-level ---
    well_df = pd.read_excel("wells.xlsx", sheet_name=0)
    section_df = pd.read_excel("wells.xlsx", sheet_name=1)

    all_gdfs = [lines, points, grid, units, infills, lease_lines, merged, land]
    for gdf in all_gdfs:
        if gdf.crs is None:
            gdf.set_crs(epsg=26913, inplace=True)
        gdf.to_crs(epsg=26913, inplace=True)

    grid["Section"] = grid["Section"].astype(str).str.strip()
    grid["geometry"] = grid.geometry.simplify(50, preserve_topology=True)

    # --- Well-level cleaning ---
    well_df["UWI"] = well_df["UWI"].astype(str).str.strip()
    well_numeric_cols = [
        c for c in well_df.columns if c != "UWI" and pd.api.types.is_numeric_dtype(well_df[c])
    ]
    well_non_numeric_cols = [
        c for c in well_df.columns if c != "UWI" and c not in well_numeric_cols
    ]
    for col in well_numeric_cols:
        well_df[col] = pd.to_numeric(well_df[col], errors="coerce")

    # --- Section-level cleaning ---
    section_df["Section"] = section_df["Section"].astype(str).str.strip()
    sec_numeric_cols = [
        c for c in section_df.columns if c != "Section" and pd.api.types.is_numeric_dtype(section_df[c])
    ]
    for col in sec_numeric_cols:
        section_df[col] = pd.to_numeric(section_df[col], errors="coerce")

    lines["UWI"] = lines["UWI"].astype(str).str.strip()
    points["UWI"] = points["UWI"].astype(str).str.strip()

    return (
        lines, points, grid, units, infills, lease_lines, merged, land,
        well_df, section_df,
        well_numeric_cols, well_non_numeric_cols, sec_numeric_cols,
    )


(
    lines_gdf, points_gdf, grid_gdf, units_gdf,
    infills_gdf, lease_lines_gdf, merged_gdf, land_gdf,
    well_df, section_df,
    WELL_NUMERIC_COLS, WELL_NON_NUMERIC_COLS, SEC_NUMERIC_COLS,
) = load_data()

ALL_METRIC_COLS = WELL_NUMERIC_COLS + SEC_NUMERIC_COLS

# ==========================================================
# Merge section-level data onto grid
# ==========================================================
section_enriched = grid_gdf.merge(section_df, on="Section", how="left")

# ==========================================================
# Build proximal well pool
# ==========================================================
lines_with_uwi = lines_gdf[["UWI", "geometry"]].copy()
points_with_uwi = points_gdf[["UWI", "geometry"]].copy()
points_only = points_with_uwi[~points_with_uwi["UWI"].isin(lines_with_uwi["UWI"])]
existing_wells = pd.concat([lines_with_uwi, points_only], ignore_index=True)
existing_wells = gpd.GeoDataFrame(existing_wells, geometry="geometry", crs=lines_gdf.crs)

proximal_wells = existing_wells.merge(well_df, on="UWI", how="inner")
proximal_wells = gpd.GeoDataFrame(proximal_wells, geometry="geometry", crs=existing_wells.crs)
proximal_wells["_midpoint"] = proximal_wells.geometry.apply(midpoint_of_geom)

# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.title("Map Settings")

# ---- Buffer distance ----
st.sidebar.markdown("---")
st.sidebar.subheader("📏 Buffer Distance")
buffer_distance = st.sidebar.slider(
    "Buffer radius (m)", 100, 2000, DEFAULT_BUFFER_M, step=50, key="buf_dist",
)

# ---- Section gradient ----
st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Section Grid Gradient")
gradient_options = ["None"] + SEC_NUMERIC_COLS
section_gradient = st.sidebar.selectbox("Colour sections by", gradient_options, key="p_gradient")

# ---- Layer toggles ----
show_infills = st.sidebar.checkbox("Show Infills", value=True)
show_lease_lines = st.sidebar.checkbox("Show Lease Lines", value=True)
show_merged = st.sidebar.checkbox("Show Merged", value=True)

# ==========================================================
# Build prospects
# ==========================================================
prospect_frames = []
if show_infills:
    inf_copy = infills_gdf.copy()
    inf_copy["_prospect_type"] = "Infill"
    prospect_frames.append(inf_copy)
if show_lease_lines:
    ll_copy = lease_lines_gdf.copy()
    ll_copy["_prospect_type"] = "Lease Line"
    prospect_frames.append(ll_copy)
if show_merged:
    merged_copy = merged_gdf.copy()
    merged_copy["_prospect_type"] = "Merged"
    prospect_frames.append(merged_copy)

prospects = pd.concat(prospect_frames, ignore_index=True)
prospects = gpd.GeoDataFrame(prospects, geometry="geometry", crs=infills_gdf.crs)

# ==========================================================
# Prospect analysis — IDW² (with fixed caching)
# ==========================================================

def hash_with_buffer(val, buffer_m):
    """Custom hash function incorporating buffer distance"""
    return hash((buffer_m, id(val)))

@st.cache_data(
    show_spinner="Analysing prospects (IDW²) …",
    hash_funcs={
        gpd.GeoDataFrame: lambda gdf: hash_with_buffer(gdf, buffer_distance),
        list: lambda lst: hash(tuple(lst)),
    }
)
def analyze_prospects_idw(_prospects, _proximal_wells, _section_enriched, _buffer_m, _well_metrics, _sec_metrics):
    # [Keep the original function body unchanged]
    ...
    # [Existing function body remains unchanged]
    pros = _prospects.copy()
    prox = _proximal_wells.copy()
    sections = _section_enriched.copy()

    results = []
    for idx, prospect in pros.iterrows():
        geom = prospect.geometry
        record = {"_idx": idx, "_prospect_type": prospect["_prospect_type"]}

        prospect_mid = midpoint_of_geom(geom)
        if prospect_mid is None:
            for col in _well_metrics + _sec_metrics:
                record[col] = np.nan
            record["Proximal_Count"] = 0
            record["_proximal_uwis"] = ""
            record["_section_label"] = "Unknown"
            results.append(record)
            continue

        # Section label from endpoint
        if geom.geom_type == "MultiLineString":
            endpoint = Point(list(geom.geoms[-1].coords)[-1])
        elif geom.geom_type == "LineString":
            endpoint = Point(list(geom.coords)[-1])
        else:
            endpoint = prospect_mid

        ep_gdf = gpd.GeoDataFrame([{"geometry": endpoint}], crs=pros.crs)
        sec_hit = gpd.sjoin(
            ep_gdf, sections[["Section", "geometry"]], how="left", predicate="within",
        )
        if not sec_hit.empty and pd.notna(sec_hit.iloc[0].get("Section")):
            record["_section_label"] = str(sec_hit.iloc[0]["Section"])
        else:
            record["_section_label"] = "Unknown"

        # Buffer & proximal wells
        buffer_geom = geom.buffer(_buffer_m, cap_style=2)
        midpoint_mask = prox["_midpoint"].apply(
            lambda mp: buffer_geom.contains(mp) if mp is not None else False,
        )
        hits = prox[midpoint_mask].copy()

        record["Proximal_Count"] = len(hits)
        record["_proximal_uwis"] = ",".join(hits["UWI"].tolist()) if len(hits) > 0 else ""

        # IDW² for well-level metrics
        if len(hits) > 0:
            pmx, pmy = prospect_mid.x, prospect_mid.y
            hit_dists = np.sqrt(
                (hits["_midpoint"].apply(lambda m: m.x) - pmx) ** 2
                + (hits["_midpoint"].apply(lambda m: m.y) - pmy) ** 2,
            ).replace(0, 1.0)
            weights = (1.0 / (hit_dists ** 2)).replace([np.inf, -np.inf], np.nan)
            valid_w = weights.dropna()

            if valid_w.sum() > 0:
                for col in _well_metrics:
                    if col not in hits.columns:
                        record[col] = np.nan
                        continue
                    col_vals = hits.loc[valid_w.index, col]
                    mask = col_vals.notna() & valid_w.notna()
                    if mask.sum() > 0:
                        w = valid_w[mask]
                        record[col] = (col_vals[mask] * w).sum() / w.sum()
                    else:
                        record[col] = np.nan
            else:
                for col in _well_metrics:
                    record[col] = hits[col].mean() if col in hits.columns else np.nan
        else:
            for col in _well_metrics:
                record[col] = np.nan

        # Section-level metrics via spatial overlap
        buffer_series = gpd.GeoSeries([buffer_geom], crs=pros.crs)
        buffer_clip_gdf = gpd.GeoDataFrame(geometry=buffer_series)
        sec_cols_for_overlay = ["Section", "geometry"] + [
            c for c in _sec_metrics if c in sections.columns
        ]
        overlaps = gpd.overlay(
            sections[sec_cols_for_overlay], buffer_clip_gdf, how="intersection",
        )
        if not overlaps.empty:
            for col in _sec_metrics:
                if col in overlaps.columns:
                    valid = overlaps[col].dropna()
                    record[col] = valid.mean() if not valid.empty else np.nan
                else:
                    record[col] = np.nan
        else:
            for col in _sec_metrics:
                record[col] = np.nan

        results.append(record)

    results_df = pd.DataFrame(results)

    # De-duplicate section labels
    label_counts = results_df["_section_label"].value_counts()
    dup_labels = label_counts[label_counts > 1].index
    for label in dup_labels:
        mask = results_df["_section_label"] == label
        indices = results_df[mask].index
        for i, row_idx in enumerate(indices, 1):
            results_df.loc[row_idx, "_section_label"] = f"{label}-{i}"

    results_df = results_df.set_index("_idx")
    return results_df


prospect_metrics = analyze_prospects_idw(
    prospects, proximal_wells, section_enriched, buffer_distance,
    WELL_NUMERIC_COLS, SEC_NUMERIC_COLS,
)

prospects = prospects.join(
    prospect_metrics.drop(columns=["_prospect_type"], errors="ignore"),
)
prospects["Label"] = prospects["_section_label"]

for col in ALL_METRIC_COLS:
    if col in prospects.columns:
        prospects[col] = prospects[col].replace([np.inf, -np.inf], np.nan)

# ==========================================================
# Sidebar — Dynamic filters
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Prospect Filters")

p = prospects.copy()
has_proximal = p["Proximal_Count"] > 0
filter_mask = has_proximal.copy()

filter_ranges = {}
for col in ALL_METRIC_COLS:
    if col not in p.columns:
        continue
    lo, hi = safe_range(p[col])
    if lo == hi:
        continue
    f_lo, f_hi = st.sidebar.slider(
        col, lo, hi, (lo, hi),
        key=f"filter_{col}",
    )
    filter_ranges[col] = (f_lo, f_hi)
    filter_mask = filter_mask & (
        ((p[col] >= f_lo) & (p[col] <= f_hi)) | p[col].isna()
    )

p["_passes_filter"] = filter_mask
p["_no_proximal"] = ~has_proximal

n_total = len(p)
n_passing = int(filter_mask.sum())
n_no_proximal = int((~has_proximal).sum())

st.sidebar.markdown(
    f"**{n_passing}** / {n_total} prospects pass filters "
    f"({n_passing / max(n_total, 1) * 100:.0f}%)"
)
if n_no_proximal > 0:
    st.sidebar.warning(f"⚠️ {n_no_proximal} prospects have no nearby proximal wells")

# ==========================================================
# Sidebar — Ranking metric
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Ranking Metric")

available_for_ranking = [c for c in ALL_METRIC_COLS if c in p.columns]
ranking_options = available_for_ranking + ["High-Grade Score"]
selected_metric = st.sidebar.selectbox("Rank prospects by", ranking_options, key="p_metric")

# ==========================================================
# Sidebar — High-Grade Score configuration
# ==========================================================
if selected_metric == "High-Grade Score":
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ High-Grade Score")

    st.sidebar.markdown("**Select metrics to include:**")
    hg_selected = {}
    hg_lower_better = {}
    hg_weights = {}

    for col in available_for_ranking:
        col_on = st.sidebar.checkbox(col, value=False, key=f"hg_chk_{col}")
        if col_on:
            hg_selected[col] = True
            c1, c2 = st.sidebar.columns([1, 1])
            hg_lower_better[col] = c1.checkbox(
                "Lower=better", value=False, key=f"hg_lb_{col}",
            )
            hg_weights[col] = c2.number_input(
                "Weight %", 0, 100, 0, key=f"hg_w_{col}",
            )

    total_weight = sum(hg_weights.values())
    if hg_selected:
        if total_weight == 100:
            st.sidebar.success(f"Total: {total_weight}%")
        elif total_weight > 100:
            st.sidebar.error(f"Total: {total_weight}% (Over by {total_weight - 100}%)")
        else:
            st.sidebar.warning(f"Total: {total_weight}% ({100 - total_weight}% remaining)")
        st.sidebar.progress(min(total_weight / 100, 1.0))
    else:
        st.sidebar.info("Check at least one metric.")
        total_weight = 0
else:
    total_weight = 0
    hg_selected = {}
    hg_lower_better = {}
    hg_weights = {}

# ==========================================================
# Compute High-Grade Score
# ==========================================================
if selected_metric == "High-Grade Score" and total_weight == 100 and hg_selected:
    passing = p[p["_passes_filter"]].copy()
    hgs = pd.Series(0.0, index=passing.index)
    for col in hg_selected:
        z = zscore(passing[col])
        if hg_lower_better.get(col, False):
            z = -z
        hgs += (hg_weights[col] / 100.0) * z
    p["HighGradeScore"] = np.nan
    p.loc[passing.index, "HighGradeScore"] = hgs
else:
    p["HighGradeScore"] = np.nan

# ==========================================================
# Determine ranking direction
# ==========================================================
if selected_metric == "High-Grade Score":
    metric_col = "HighGradeScore"
    ascending = False
else:
    metric_col = selected_metric
    # For non-HG ranking, ask user
    ascending = st.sidebar.checkbox(
        f"Lower {selected_metric} = better?", value=False, key="rank_asc",
    )

# ==========================================================
# Colour map for buffers
# ==========================================================
passing_metric_vals = p[p["_passes_filter"]][metric_col].dropna()
if not passing_metric_vals.empty:
    if ascending:
        gmap_vmin = float(-passing_metric_vals.max())
        gmap_vmax = float(-passing_metric_vals.min())
    else:
        gmap_vmin = float(passing_metric_vals.min())
        gmap_vmax = float(passing_metric_vals.max())
else:
    gmap_vmin, gmap_vmax = 0.0, 1.0

label_color_map = {}
for idx_val in p[p["_passes_filter"]].index:
    row = p.loc[idx_val]
    val = row.get(metric_col, np.nan)
    if pd.notna(val):
        gmap_val = -val if ascending else val
        label_color_map[row["Label"]] = get_ylgn_hex(gmap_val, gmap_vmin, gmap_vmax)
    else:
        label_color_map[row["Label"]] = "#cccccc"

p["_buffer_color"] = p["Label"].map(label_color_map).fillna("#cccccc")

# ==========================================================
# Display data preparation
# ==========================================================
section_display = section_enriched.copy().to_crs(4326)
units_display = units_gdf.copy().to_crs(4326)
land_display = land_gdf.copy().to_crs(4326)

existing_display_cols = ["UWI", "geometry"] + [
    c for c in well_df.columns if c != "UWI" and c in proximal_wells.columns
]
existing_display = proximal_wells[existing_display_cols].copy().to_crs(4326)

transformer_to_4326 = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)
label_to_latlon = {}
for idx_val, row in p.iterrows():
    mid = midpoint_of_geom(row.geometry)
    if mid is not None:
        lon, lat = transformer_to_4326.transform(mid.x, mid.y)
        label_to_latlon[row["Label"]] = (lat, lon)

# Buffer GeoDataFrame
buffer_records = []
for idx, row in p.iterrows():
    rec = {
        "Label": row["Label"],
        "_passes_filter": row["_passes_filter"],
        "_no_proximal": row["_no_proximal"],
        "_buffer_color": row["_buffer_color"],
        "Proximal_Count": row.get("Proximal_Count", 0),
        "geometry": row.geometry.buffer(buffer_distance, cap_style=2),
    }
    for col in ALL_METRIC_COLS:
        if col in row.index:
            rec[col] = row[col]
    if metric_col not in rec:
        rec[metric_col] = row.get(metric_col, np.nan)
    buffer_records.append(rec)

buffer_gdf = gpd.GeoDataFrame(buffer_records, crs=p.crs).to_crs(4326)

# Prospect lines for display
p_lines = p.copy()
keep_cols = ["Label", "_prospect_type", "Proximal_Count", "geometry"] + [
    c for c in ALL_METRIC_COLS if c in p_lines.columns
]
if "HighGradeScore" in p_lines.columns:
    keep_cols.append("HighGradeScore")
keep_cols = list(dict.fromkeys(keep_cols))  # deduplicate preserving order
p_lines = p_lines[[c for c in keep_cols if c in p_lines.columns]].copy()
for c in p_lines.columns:
    if c == "geometry":
        continue
    if p_lines[c].dtype == object:
        try:
            p_lines[c] = p_lines[c].astype(str)
        except Exception:
            p_lines = p_lines.drop(columns=[c])
p_lines_display = p_lines.to_crs(4326)

# ================================================================
# EXECUTIVE SUMMARY
# ================================================================
st.title("🛢️ Bakken Inventory Engine")

if n_passing > 0:
    best_pool = p[p["_passes_filter"]].dropna(subset=[metric_col])
    if not best_pool.empty:
        best_row = best_pool.sort_values(metric_col, ascending=ascending).iloc[0]
        best_name = best_row["Label"]
        best_val = best_row[metric_col]
        avg_proximal = p[p["_passes_filter"]]["Proximal_Count"].mean()
        st.success(
            f"**{n_passing}** of {n_total} prospects pass filters. "
            f"Top prospect by **{selected_metric}**: **{best_name}** "
            f"({metric_col} = {best_val:,.2f}). "
            f"Avg proximal wells/prospect: **{avg_proximal:.1f}**."
        )
    else:
        st.info(f"**{n_passing}** prospects pass filters but none have valid {selected_metric} data.")
else:
    st.warning("No prospects pass the current filters. Try relaxing your criteria.")

# ==============================================================
# Ranking dataframe
# ==============================================================
rank_df = None
if selected_metric == "High-Grade Score" and total_weight != 100:
    pass
else:
    rank_df_raw = p[p["_passes_filter"]].copy()

    display_cols = ["Label", "_prospect_type", "Proximal_Count"] + [
        c for c in ALL_METRIC_COLS if c in rank_df_raw.columns
    ]
    if "HighGradeScore" in rank_df_raw.columns and selected_metric == "High-Grade Score":
        display_cols.append("HighGradeScore")
    if metric_col not in display_cols:
        display_cols.append(metric_col)
    display_cols = list(dict.fromkeys(display_cols))
    display_cols = [c for c in display_cols if c in rank_df_raw.columns]

    rank_df_raw = rank_df_raw[display_cols].copy()
    rank_df_raw = rank_df_raw.dropna(subset=[metric_col])

    if not rank_df_raw.empty:
        rank_df_raw["Percentile"] = (
            rank_df_raw[metric_col].rank(pct=True, ascending=(not ascending)) * 100
        )
        rank_df_raw = rank_df_raw.sort_values(metric_col, ascending=ascending).reset_index(drop=True)
        rank_df_raw.index = rank_df_raw.index + 1
        rank_df_raw.index.name = "Rank"
        rank_df = rank_df_raw.rename(columns={
            "_prospect_type": "Type", "Proximal_Count": "Proximal",
        })

# ==============================================================
# Layout: Map + Ranking
# ==============================================================
col_map, col_rank = st.columns([7, 4])

# ----------------------------------------------------------
# RIGHT — Ranking & Detail
# ----------------------------------------------------------
with col_rank:
    st.header("📊 Prospect Ranking")

    if selected_metric == "High-Grade Score" and total_weight != 100:
        st.warning("Adjust weights to total 100 % to see rankings.")
    elif rank_df is None or rank_df.empty:
        st.warning(f"No valid data for **{selected_metric}**.")
    else:
        st.caption(
            f"Ranked by **{selected_metric}** · {len(rank_df)} prospects · "
            f"Buffer: {buffer_distance}m · IDW²"
        )

        # Auto-format: integers for large numbers, 3 decimals for small
        fmt = {}
        for c in rank_df.columns:
            if c in ("Label", "Type"):
                continue
            if c == "Percentile":
                fmt[c] = "{:.0f}%"
            elif c == "Proximal":
                fmt[c] = "{:.0f}"
            elif rank_df[c].dtype in [np.float64, np.float32, float]:
                col_max = rank_df[c].abs().max()
                if pd.notna(col_max) and col_max > 100:
                    fmt[c] = "{:,.0f}"
                else:
                    fmt[c] = "{:.3f}"

        gmap_vals = rank_df[metric_col] if not ascending else -rank_df[metric_col]

        styled = rank_df.style.background_gradient(
            subset=[metric_col], cmap="YlGn", gmap=gmap_vals,
        ).background_gradient(
            subset=["Percentile"], cmap="RdYlGn",
        ).format(fmt)

        st.dataframe(styled, use_container_width=True, height=500)

        csv = rank_df.to_csv().encode("utf-8")
        st.download_button(
            "⬇️ Download Rankings (CSV)", data=csv,
            file_name="bakken_prospect_rankings.csv", mime="text/csv",
        )

        # Detail panel
        st.markdown("---")
        st.subheader("🔬 Prospect Detail")

        label_list = rank_df["Label"].tolist()
        detail_label = st.selectbox("Select a prospect", label_list, index=0, key="detail_select")

        if detail_label and detail_label in rank_df["Label"].values:
            dr = rank_df[rank_df["Label"] == detail_label].iloc[0]

            # Dynamic metric cards — 4 per row
            detail_metrics = [
                c for c in rank_df.columns
                if c not in ("Label", "Type", "Percentile")
                and rank_df[c].dtype in [np.float64, np.float32, np.int64, float, int]
            ]
            rows_needed = (len(detail_metrics) + 3) // 4
            metric_idx = 0
            for _ in range(rows_needed):
                cols = st.columns(4)
                for ci in range(4):
                    if metric_idx < len(detail_metrics):
                        mc = detail_metrics[metric_idx]
                        val = dr.get(mc, np.nan)
                        if pd.notna(val):
                            if abs(val) > 100:
                                display_val = f"{val:,.0f}"
                            else:
                                display_val = f"{val:.3f}"
                        else:
                            display_val = "—"
                        cols[ci].metric(mc, display_val)
                        metric_idx += 1

    # No-proximal table
    no_proximal_prospects = p[p["_no_proximal"]].copy()
    if not no_proximal_prospects.empty:
        st.markdown("---")
        st.subheader("⚠️ No Proximal Wells Found")
        st.caption(
            f"These {len(no_proximal_prospects)} prospects have no proximal wells within "
            f"the {buffer_distance}m buffer. Consider increasing buffer distance."
        )
        st.dataframe(
            no_proximal_prospects[["Label", "_prospect_type"]]
            .rename(columns={"_prospect_type": "Type"})
            .reset_index(drop=True),
            use_container_width=True,
        )

# ----------------------------------------------------------
# LEFT — Map
# ----------------------------------------------------------
with col_map:
    bounds = p.total_bounds
    centre_x = (bounds[0] + bounds[2]) / 2
    centre_y = (bounds[1] + bounds[3]) / 2
    transformer = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)
    centre_lon, centre_lat = transformer.transform(centre_x, centre_y)

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=11, tiles="CartoDB positron")
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)

    # --- Layer 0: Bakken Land ---
    land_fg = folium.FeatureGroup(name="Bakken Land", show=True)
    folium.GeoJson(
        land_display.to_json(),
        style_function=lambda _: {
            "fillColor": "#fff9c4", "color": "#fff9c4",
            "weight": 0.5, "fillOpacity": 0.2,
        },
    ).add_to(land_fg)
    land_fg.add_to(m)

    # --- Layer 1: Section grid ---
    if section_gradient != "None":
        grad_col = section_gradient
        grad_vals = section_display[grad_col].dropna() if grad_col in section_display.columns else pd.Series(dtype=float)

        if not grad_vals.empty:
            colormap = cm.LinearColormap(
                colors=["#f7fcf5", "#74c476", "#00441b"],
                vmin=float(grad_vals.min()),
                vmax=float(grad_vals.max()),
            ).to_step(n=7)
            colormap.caption = section_gradient
            m.add_child(colormap)

            def section_style(feature):
                val = feature["properties"].get(grad_col)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return NULL_STYLE
                return {
                    "fillColor": colormap(val), "fillOpacity": 0.45,
                    "color": "white", "weight": 0.3,
                }
        else:
            section_style = lambda _: NULL_STYLE
    else:
        section_style = lambda _: NULL_STYLE

    # Section tooltip: Section + all Sheet 2 columns
    sec_tooltip_fields = ["Section"] + [
        c for c in SEC_NUMERIC_COLS if c in section_display.columns
    ]

    section_fg = folium.FeatureGroup(name="Section Grid", show=(section_gradient != "None"))
    folium.GeoJson(
        section_display.to_json(), name="Sections",
        style_function=section_style,
        highlight_function=lambda _: {"weight": 2, "color": "black", "fillOpacity": 0.5},
        tooltip=folium.GeoJsonTooltip(
            fields=sec_tooltip_fields,
            aliases=[f"{f}:" for f in sec_tooltip_fields],
            localize=True, sticky=True,
            style="font-size:11px;padding:4px 8px;background:rgba(255,255,255,0.9);"
                  "border:1px solid #333;border-radius:3px;",
        ),
    ).add_to(section_fg)
    section_fg.add_to(m)

    # --- Layer 2: Units ---
    units_fg = folium.FeatureGroup(name="Units", show=True)
    folium.GeoJson(
        units_display.to_json(),
        style_function=lambda _: {
            "color": "black", "weight": 2, "fillOpacity": 0, "interactive": False,
        },
    ).add_to(units_fg)
    units_fg.add_to(m)

    # --- Layer 3: Buffers ---
    buffer_fg = folium.FeatureGroup(name="Prospect Buffers")

    passing_buf = buffer_gdf[buffer_gdf["_passes_filter"]].copy()
    for _, brow in passing_buf.iterrows():
        fill_color = label_color_map.get(brow["Label"], "#cccccc")

        tip_parts = [f"<b>{brow['Label']}</b>"]
        tip_parts.append(f"Proximal Wells: {brow.get('Proximal_Count', '—')}")
        for col in ALL_METRIC_COLS:
            if col in brow.index and pd.notna(brow[col]):
                val = brow[col]
                tip_parts.append(
                    f"{col}: {val:,.0f}" if abs(val) > 100 else f"{col}: {val:.3f}"
                )

        folium.GeoJson(
            brow.geometry.__geo_interface__,
            style_function=lambda _, fc=fill_color: {
                "fillColor": fc, "fillOpacity": 0.4,
                "color": fc, "weight": 1, "opacity": 0.7,
            },
            tooltip=folium.Tooltip(
                "<br>".join(tip_parts), sticky=True,
                style="font-size:11px;padding:3px 6px;background:rgba(255,255,255,0.92);"
                      "border:1px solid #333;border-radius:3px;",
            ),
        ).add_to(buffer_fg)

    filtered_buf = buffer_gdf[~buffer_gdf["_passes_filter"] & ~buffer_gdf["_no_proximal"]]
    for _, brow in filtered_buf.iterrows():
        folium.GeoJson(
            brow.geometry.__geo_interface__,
            style_function=lambda _: {
                "fillColor": "#d3d3d3", "fillOpacity": 0.15,
                "color": "#aaa", "weight": 0.5, "opacity": 0.3,
            },
        ).add_to(buffer_fg)

    no_proximal_buf = buffer_gdf[buffer_gdf["_no_proximal"]]
    for _, brow in no_proximal_buf.iterrows():
        folium.GeoJson(
            brow.geometry.__geo_interface__,
            style_function=lambda _: {
                "fillColor": "#ffe0b2", "fillOpacity": 0.1,
                "color": "orange", "weight": 1, "dashArray": "5 5", "opacity": 0.4,
            },
        ).add_to(buffer_fg)

    buffer_fg.add_to(m)

    # --- Layer 4: Existing wells ---
    well_fg = folium.FeatureGroup(name="Existing Wells")
    line_wells = existing_display[existing_display.geometry.type != "Point"]
    point_wells = existing_display[existing_display.geometry.type == "Point"]

    # Tooltip fields: UWI + all Sheet 1 columns present
    well_tooltip_fields = ["UWI"] + [
        c for c in well_df.columns
        if c != "UWI" and c in existing_display.columns and existing_display[c].notna().any()
    ]

    if not line_wells.empty:
        folium.GeoJson(
            line_wells.to_json(),
            style_function=lambda _: {"color": "black", "weight": 1, "opacity": 0.7},
            highlight_function=lambda _: {"weight": 2.5, "color": "#555"},
            tooltip=folium.GeoJsonTooltip(
                fields=well_tooltip_fields,
                aliases=[f"{f}:" for f in well_tooltip_fields],
                localize=True, sticky=True,
                style="font-size:11px;padding:3px 6px;background:rgba(255,255,255,0.92);"
                      "border:1px solid #333;border-radius:3px;",
            ),
        ).add_to(well_fg)

    for _, row in point_wells.iterrows():
        tip_parts = [f"<b>UWI:</b> {row.get('UWI', '—')}"]
        for col in well_df.columns:
            if col == "UWI":
                continue
            if col in row.index and pd.notna(row[col]):
                val = row[col]
                if isinstance(val, (int, float)):
                    tip_parts.append(
                        f"<b>{col}:</b> {val:,.0f}" if abs(val) > 100 else f"<b>{col}:</b> {val:.3f}"
                    )
                else:
                    tip_parts.append(f"<b>{col}:</b> {val}")

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=1, color="black", fill=True, fill_color="black",
            fill_opacity=0.7, weight=1,
            tooltip=folium.Tooltip(
                "<br>".join(tip_parts), sticky=True,
                style="font-size:11px;padding:3px 6px;background:rgba(255,255,255,0.92);"
                      "border:1px solid #333;border-radius:3px;",
            ),
        ).add_to(well_fg)
    well_fg.add_to(m)

    # --- Layer 5: Prospect lines ---
    prospect_fg = folium.FeatureGroup(name="Prospect Wells", show=True)

    pt_fields = [c for c in p_lines_display.columns if c != "geometry"]
    folium.GeoJson(
        p_lines_display.to_json(),
        style_function=lambda _: {"color": "red", "weight": 3, "opacity": 0.9},
        highlight_function=lambda _: {"weight": 5, "color": "#ff4444"},
        tooltip=folium.GeoJsonTooltip(
            fields=pt_fields,
            aliases=[f"{f}:" for f in pt_fields],
            localize=True, sticky=True,
            style="font-size:12px",
        ),
    ).add_to(prospect_fg)
    prospect_fg.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, use_container_width=True, height=900, returned_objects=[])
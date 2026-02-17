import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import MiniMap
from streamlit_folium import st_folium
import branca.colormap as cm
from shapely.geometry import Point
import plotly.graph_objects as go
from pyproj import Transformer
import matplotlib.colors as mcolors
import matplotlib.cm as mpl_cm

# ==========================================================
# Page configuration
# ==========================================================
st.set_page_config(
    layout="wide",
    page_title="Bakken Prospect Analysis",
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

WELL_LEVEL_METRICS = ["EUR", "IP90", "1YCuml", "Wcut"]
SECTION_LEVEL_METRICS = ["OOIP", "RFTD", "URF"]

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
        return (0.0, 1.0) if lo == 0.0 else (lo, lo + abs(lo) * 0.1)
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
    """Return a hex colour from the YlGn matplotlib colormap, matching
    the pandas Styler `background_gradient(cmap='YlGn')` behaviour."""
    if pd.isna(value) or vmin == vmax:
        return "#cccccc"
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl_cm.get_cmap("YlGn")
    rgba = cmap(norm(value))
    return mcolors.to_hex(rgba)


# ==========================================================
# Session state for map zoom target
# ==========================================================
if "zoom_to_label" not in st.session_state:
    st.session_state["zoom_to_label"] = None


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

    prod_all = pd.read_excel("wells.xlsx")

    all_gdfs = [lines, points, grid, units, infills, lease_lines, merged, land]
    for gdf in all_gdfs:
        if gdf.crs is None:
            gdf.set_crs(epsg=26913, inplace=True)
        gdf.to_crs(epsg=26913, inplace=True)

    grid["Section"] = grid["Section"].astype(str).str.strip()
    grid["OOIP"] = pd.to_numeric(grid["OOIP"], errors="coerce")

    PROD_NUMERIC = ["Cuml", "EUR", "IP90", "1YCuml", "Wcut"]
    df = prod_all
    df["UWI"] = df["UWI"].astype(str).str.strip()
    df["Section"] = df["Section"].astype(str).str.strip()

    for col in PROD_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    lines["UWI"] = lines["UWI"].astype(str).str.strip()
    points["UWI"] = points["UWI"].astype(str).str.strip()

    grid["geometry"] = grid.geometry.simplify(50, preserve_topology=True)

    return lines, points, grid, units, infills, lease_lines, merged, prod_all, land


(
    lines_gdf, points_gdf, grid_gdf, units_gdf,
    infills_gdf, lease_lines_gdf, merged_gdf, prod_all_df, land_gdf,
) = load_data()

# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.title("Map Settings")

# ==========================================================
# Build the proximal well pool
# ==========================================================
prod_pool = prod_all_df.copy()
lines_with_uwi = lines_gdf[["UWI", "geometry"]].copy()
points_with_uwi = points_gdf[["UWI", "geometry"]].copy()
points_only = points_with_uwi[~points_with_uwi["UWI"].isin(lines_with_uwi["UWI"])]
existing_wells = pd.concat([lines_with_uwi, points_only], ignore_index=True)
existing_wells = gpd.GeoDataFrame(existing_wells, geometry="geometry", crs=lines_gdf.crs)

proximal_wells = existing_wells.merge(prod_pool, on="UWI", how="inner")
proximal_wells = gpd.GeoDataFrame(proximal_wells, geometry="geometry", crs=existing_wells.crs)
proximal_wells["_midpoint"] = proximal_wells.geometry.apply(midpoint_of_geom)

# ==========================================================
# Section enrichment
# ==========================================================

@st.cache_data(show_spinner="Computing section metrics …")
def compute_section_metrics(_proximal_wells, _grid_df):
    aw = _proximal_wells.copy()
    g = _grid_df[["Section", "OOIP", "geometry"]].copy()

    endpoints = []
    for idx, row in aw.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            endpoints.append(None)
        elif geom.geom_type == "MultiLineString":
            endpoints.append(Point(list(geom.geoms[-1].coords)[-1]))
        elif geom.geom_type == "LineString":
            endpoints.append(Point(list(geom.coords)[-1]))
        elif geom.geom_type == "Point":
            endpoints.append(geom)
        else:
            endpoints.append(geom.centroid)

    aw["_endpoint"] = endpoints
    valid = aw[aw["_endpoint"].notna()].copy()
    endpoint_gdf = gpd.GeoDataFrame(
        valid.drop(columns=["geometry"]),
        geometry="_endpoint",
        crs=_proximal_wells.crs,
    )

    joined = gpd.sjoin(endpoint_gdf, g, how="left", predicate="within")

    sec_col = None
    for c in joined.columns:
        if "Section" in c and c.endswith("right"):
            sec_col = c
            break
    if sec_col is None:
        for c in joined.columns:
            if c == "Section" or (c.startswith("Section") and c != "Section_left"):
                sec_col = c
                break
    if sec_col and sec_col != "Assigned_Section":
        joined = joined.rename(columns={sec_col: "Assigned_Section"})

    keep_cols = [c for c in ["UWI", "Assigned_Section", "EUR", "IP90", "1YCuml",
                             "Wcut", "Cuml"] if c in joined.columns]
    ws = joined[keep_cols].drop_duplicates(subset="UWI", keep="first")

    section_agg = (
        ws.groupby("Assigned_Section")
        .agg(
            Well_Count=("UWI", "count"),
            Section_Cuml=("Cuml", "sum"),
            Section_EUR=("EUR", "sum"),
            Avg_EUR=("EUR", "mean"),
            Avg_IP90=("IP90", "mean"),
            Avg_1YCuml=("1YCuml", "mean"),
            Avg_Wcut=("Wcut", "mean"),
        )
        .reset_index()
        .rename(columns={"Assigned_Section": "Section"})
    )

    g = g.merge(section_agg, on="Section", how="left")
    ooip_safe = g["OOIP"].replace(0, np.nan)
    g["RFTD"] = g["Section_Cuml"] / ooip_safe
    g["URF"] = g["Section_EUR"] / ooip_safe
    for col in ["RFTD", "URF"]:
        g[col] = g[col].replace([np.inf, -np.inf], np.nan)

    return g


section_enriched = compute_section_metrics(proximal_wells, grid_gdf)

# ==========================================================
# PROSPECT ANALYSIS
# ==========================================================

# ---- Buffer distance slider ----
st.sidebar.markdown("---")
st.sidebar.subheader("📏 Buffer Distance")
buffer_distance = st.sidebar.slider(
    "Buffer radius (m)", 100, 2000, DEFAULT_BUFFER_M, step=50, key="buf_dist",
)

# ---- Gradient selector ----
st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Section Grid Gradient")
GRADIENT_OPTIONS = ["None", "OOIP", "Avg EUR", "Avg IP90", "Avg 1YCuml",
                    "Avg Wcut", "Section Cuml", "RFTD", "URF"]
GRADIENT_COL_MAP = {
    "OOIP": "OOIP", "Avg EUR": "Avg_EUR", "Avg IP90": "Avg_IP90",
    "Avg 1YCuml": "Avg_1YCuml", "Avg Wcut": "Avg_Wcut",
    "Section Cuml": "Section_Cuml", "RFTD": "RFTD", "URF": "URF",
}
section_gradient = st.sidebar.selectbox("Colour sections by", GRADIENT_OPTIONS, key="p_gradient")

show_infills = st.sidebar.checkbox("Show Infills", value=True)
show_lease_lines = st.sidebar.checkbox("Show Lease Lines", value=True)
show_merged = st.sidebar.checkbox("Show Merged", value=True)

# ---- Build prospects ----
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

# ---- Prospect analysis with IDW ----
@st.cache_data(show_spinner="Analysing prospects (IDW²) …")
def analyze_prospects_idw(_prospects, _proximal_wells, _section_enriched, _buffer_m):
    pros = _prospects.copy()
    prox = _proximal_wells.copy()
    sections = _section_enriched.copy()

    prox_mid_x = prox["_midpoint"].apply(lambda mp: mp.x if mp is not None else np.nan)
    prox_mid_y = prox["_midpoint"].apply(lambda mp: mp.y if mp is not None else np.nan)

    results = []
    for idx, prospect in pros.iterrows():
        geom = prospect.geometry
        record = {"_idx": idx, "_prospect_type": prospect["_prospect_type"]}

        prospect_mid = midpoint_of_geom(geom)
        if prospect_mid is None:
            for col in WELL_LEVEL_METRICS + SECTION_LEVEL_METRICS:
                record[col] = np.nan
            record["Proximal_Count"] = 0
            record["_proximal_uwis"] = ""
            record["_section_label"] = "Unknown"
            record["EUR_median"] = np.nan
            record["EUR_p10"] = np.nan
            record["EUR_p90"] = np.nan
            results.append(record)
            continue

        if geom.geom_type == "MultiLineString":
            endpoint = Point(list(geom.geoms[-1].coords)[-1])
        elif geom.geom_type == "LineString":
            endpoint = Point(list(geom.coords)[-1])
        else:
            endpoint = prospect_mid

        ep_gdf = gpd.GeoDataFrame([{"geometry": endpoint}], crs=pros.crs)
        sec_hit = gpd.sjoin(ep_gdf, sections[["Section", "geometry"]], how="left", predicate="within")
        if not sec_hit.empty and pd.notna(sec_hit.iloc[0].get("Section")):
            record["_section_label"] = str(sec_hit.iloc[0]["Section"])
        else:
            record["_section_label"] = "Unknown"

        buffer_geom = geom.buffer(_buffer_m, cap_style=2)

        midpoint_mask = prox["_midpoint"].apply(
            lambda mp: buffer_geom.contains(mp) if mp is not None else False
        )

        hits = prox[midpoint_mask].copy()

        if not hits.empty:
            pmx, pmy = prospect_mid.x, prospect_mid.y
            hit_mids_x = hits["_midpoint"].apply(lambda m: m.x)
            hit_mids_y = hits["_midpoint"].apply(lambda m: m.y)
            hit_dists = np.sqrt((hit_mids_x - pmx)**2 + (hit_mids_y - pmy)**2)
        else:
            hit_dists = pd.Series(dtype=float)

        record["Proximal_Count"] = len(hits)
        record["_proximal_uwis"] = ",".join(hits["UWI"].tolist()) if len(hits) > 0 else ""

        if len(hits) > 0:
            hit_mids = hit_dists.replace(0, 1.0)
            weights = 1.0 / (hit_mids ** 2)
            weights = weights.replace([np.inf, -np.inf], np.nan)
            valid_w = weights.dropna()

            if valid_w.sum() > 0:
                for col in WELL_LEVEL_METRICS:
                    col_vals = hits.loc[valid_w.index, col]
                    mask = col_vals.notna() & valid_w.notna()
                    if mask.sum() > 0:
                        w = valid_w[mask]
                        record[col] = (col_vals[mask] * w).sum() / w.sum()
                    else:
                        record[col] = np.nan
            else:
                for col in WELL_LEVEL_METRICS:
                    record[col] = hits[col].mean()

            record["EUR_median"] = hits["EUR"].median()
            record["EUR_p10"] = hits["EUR"].quantile(0.9)
            record["EUR_p90"] = hits["EUR"].quantile(0.1)
        else:
            for col in WELL_LEVEL_METRICS:
                record[col] = np.nan
            record["EUR_median"] = np.nan
            record["EUR_p10"] = np.nan
            record["EUR_p90"] = np.nan

        buffer_series = gpd.GeoSeries([buffer_geom], crs=pros.crs)
        buffer_clip_gdf = gpd.GeoDataFrame(geometry=buffer_series)

        overlaps = gpd.overlay(
            sections[["Section", "OOIP", "RFTD", "URF", "geometry"]],
            buffer_clip_gdf,
            how="intersection",
        )

        if not overlaps.empty:
            for col in SECTION_LEVEL_METRICS:
                valid = overlaps[col].dropna()
                record[col] = valid.mean() if not valid.empty else np.nan
        else:
            for col in SECTION_LEVEL_METRICS:
                record[col] = np.nan

        results.append(record)

    results_df = pd.DataFrame(results)

    label_counts = results_df["_section_label"].value_counts()
    dup_labels = label_counts[label_counts > 1].index
    for label in dup_labels:
        mask = results_df["_section_label"] == label
        indices = results_df[mask].index
        for i, row_idx in enumerate(indices, 1):
            results_df.loc[row_idx, "_section_label"] = f"{label}-{i}"

    results_df = results_df.set_index("_idx")
    return results_df

prospect_metrics = analyze_prospects_idw(prospects, proximal_wells, section_enriched, buffer_distance)

prospects = prospects.join(
    prospect_metrics.drop(columns=["_prospect_type"], errors="ignore")
)
prospects["Label"] = prospects["_section_label"]

for col in WELL_LEVEL_METRICS + SECTION_LEVEL_METRICS:
    if col in prospects.columns:
        prospects[col] = prospects[col].replace([np.inf, -np.inf], np.nan)

# ---- Filters ----
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Prospect Filters")

p = prospects.copy()

_wcut_lo, _wcut_hi = safe_range(p["Wcut"])
f_wcut = st.sidebar.slider("Max Water Cut (%)", _wcut_lo, _wcut_hi, _wcut_hi, step=1.0, key="p_wcut")

_ooip_lo, _ooip_hi = safe_range(p["OOIP"])
f_ooip = st.sidebar.slider("Min OOIP (bbl)", _ooip_lo, _ooip_hi, _ooip_lo, step=1.0, format="%.0f", key="p_ooip")

_eur_lo, _eur_hi = safe_range(p["EUR"])
f_eur = st.sidebar.slider("Min EUR (bbl)", _eur_lo, _eur_hi, _eur_lo, step=1.0, format="%.0f", key="p_eur")

_ip90_lo, _ip90_hi = safe_range(p["IP90"])
f_ip90 = st.sidebar.slider("Min IP90 (bbl/d)", _ip90_lo, _ip90_hi, _ip90_lo, step=1.0, format="%.0f", key="p_ip90")

_1y_lo, _1y_hi = safe_range(p["1YCuml"])
f_1y = st.sidebar.slider("Min 1Y Cuml (bbl)", _1y_lo, _1y_hi, _1y_lo, step=1.0, format="%.0f", key="p_1y")

_urf_lo, _urf_hi = safe_range(p["URF"])
f_urf = st.sidebar.slider("Max URF", _urf_lo, _urf_hi, _urf_hi, step=0.01, format="%.2f", key="p_urf")

_rftd_lo, _rftd_hi = safe_range(p["RFTD"])
f_rftd = st.sidebar.slider("Max RFTD", _rftd_lo, _rftd_hi, _rftd_hi, step=0.01, format="%.2f", key="p_rftd")

has_proximal = p["Proximal_Count"] > 0
filter_mask = (
    has_proximal
    & ((p["Wcut"] <= f_wcut) | p["Wcut"].isna())
    & ((p["OOIP"] >= f_ooip) | p["OOIP"].isna())
    & ((p["EUR"] >= f_eur) | p["EUR"].isna())
    & ((p["IP90"] >= f_ip90) | p["IP90"].isna())
    & ((p["1YCuml"] >= f_1y) | p["1YCuml"].isna())
    & ((p["URF"] <= f_urf) | p["URF"].isna())
    & ((p["RFTD"] <= f_rftd) | p["RFTD"].isna())
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

# ---- Ranking metric ----
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Ranking Metric")

selected_metric = st.sidebar.selectbox(
    "Rank prospects by",
    ["EUR", "IP90", "1YCuml", "Wcut", "OOIP", "URF", "RFTD", "High-Grade Score"],
    key="p_metric",
)

# ---- HG Score weights ----
if selected_metric == "High-Grade Score":
    st.sidebar.markdown("---")
    st.sidebar.subheader("High-Grade Score Weights")
    st.sidebar.caption("Must total 100 %")

    c1, c2 = st.sidebar.columns(2)
    w_eur = c1.number_input("EUR", 0, 100, 20, key="pw_eur")
    w_ip90 = c2.number_input("IP90", 0, 100, 15, key="pw_ip90")
    w_1ycuml = c1.number_input("1Y Cuml", 0, 100, 15, key="pw_1y")
    w_wcut = c2.number_input("Wcut", 0, 100, 10, key="pw_wcut")
    w_ooip = c1.number_input("OOIP", 0, 100, 20, key="pw_ooip")
    w_urf = c2.number_input("URF", 0, 100, 10, key="pw_urf")
    w_rftd = c1.number_input("RFTD", 0, 100, 10, key="pw_rftd")

    total_weight = w_eur + w_ip90 + w_1ycuml + w_wcut + w_ooip + w_urf + w_rftd
    if total_weight == 100:
        st.sidebar.success(f"Total: {total_weight}%")
    elif total_weight > 100:
        st.sidebar.error(f"Total: {total_weight}% (Over by {total_weight - 100}%)")
    else:
        st.sidebar.warning(f"Total: {total_weight}% ({100 - total_weight}% remaining)")
    st.sidebar.progress(min(total_weight / 100, 1.0))
else:
    total_weight = 0

# ---- Compute HG Score ----
if selected_metric == "High-Grade Score" and total_weight == 100:
    passing = p[p["_passes_filter"]].copy()

    z_eur = zscore(passing["EUR"])
    z_ip90 = zscore(passing["IP90"])
    z_1y = zscore(passing["1YCuml"])
    z_ooip = zscore(passing["OOIP"])
    z_wcut = -zscore(passing["Wcut"])
    z_urf = -zscore(passing["URF"])
    z_rftd = -zscore(passing["RFTD"])

    hgs = (
        (w_eur / 100) * z_eur
        + (w_ip90 / 100) * z_ip90
        + (w_1ycuml / 100) * z_1y
        + (w_wcut / 100) * z_wcut
        + (w_ooip / 100) * z_ooip
        + (w_urf / 100) * z_urf
        + (w_rftd / 100) * z_rftd
    )

    p["HighGradeScore"] = np.nan
    p.loc[passing.index, "HighGradeScore"] = hgs
else:
    p["HighGradeScore"] = np.nan

metric_col = "HighGradeScore" if selected_metric == "High-Grade Score" else selected_metric
ascending = selected_metric in ["Wcut", "URF", "RFTD"]

# ---- Rank stability ----
@st.cache_data(show_spinner="Computing rank stability …")
def compute_rank_stability(_p_df, _proximal_wells, _buffer_m, _metric_col, _ascending):
    p_local = _p_df.copy()
    passing = p_local[p_local["_passes_filter"]].copy()
    if passing.empty or _metric_col not in passing.columns:
        return pd.Series(dtype=float)

    baseline = passing.dropna(subset=[_metric_col]).sort_values(
        _metric_col, ascending=_ascending
    ).reset_index()
    baseline["_base_rank"] = range(1, len(baseline) + 1)
    base_map = dict(zip(baseline["index"], baseline["_base_rank"]))

    prox = _proximal_wells.copy()
    prox_mid_x = prox["_midpoint"].apply(lambda mp: mp.x if mp is not None else np.nan)
    prox_mid_y = prox["_midpoint"].apply(lambda mp: mp.y if mp is not None else np.nan)
    stability = {}

    for pidx in passing.index:
        geom = p_local.loc[pidx, "geometry"]
        prospect_mid = midpoint_of_geom(geom)
        uwi_str = p_local.loc[pidx].get("_proximal_uwis", "")
        if not uwi_str or pd.isna(uwi_str):
            stability[pidx] = 0
            continue

        uwi_list = [u.strip() for u in str(uwi_str).split(",") if u.strip()]
        if len(uwi_list) <= 1:
            stability[pidx] = np.nan
            continue

        prox_sub = prox[prox["UWI"].isin(uwi_list)]
        if prox_sub.empty or prospect_mid is None:
            stability[pidx] = 0
            continue

        if _metric_col in WELL_LEVEL_METRICS:
            if _ascending:
                best_idx = prox_sub[_metric_col].idxmin()
            else:
                best_idx = prox_sub[_metric_col].idxmax()
            if pd.isna(best_idx):
                stability[pidx] = 0
                continue
            best_uwi = prox_sub.loc[best_idx, "UWI"]
        else:
            pmx = prospect_mid.x
            pmy = prospect_mid.y
            dists = np.sqrt(
                (prox_mid_x.loc[prox_sub.index] - pmx) ** 2
                + (prox_mid_y.loc[prox_sub.index] - pmy) ** 2
            )
            best_uwi = prox_sub.loc[dists.idxmin(), "UWI"]

        remaining = prox_sub[prox_sub["UWI"] != best_uwi]
        if remaining.empty:
            stability[pidx] = np.nan
            continue

        pmx = prospect_mid.x
        pmy = prospect_mid.y
        hit_dists = np.sqrt(
            (prox_mid_x.loc[remaining.index] - pmx) ** 2
            + (prox_mid_y.loc[remaining.index] - pmy) ** 2
        ).replace(0, 1.0)
        weights = (1.0 / (hit_dists ** 2)).replace([np.inf, -np.inf], np.nan).dropna()

        new_val = np.nan
        if _metric_col in WELL_LEVEL_METRICS and weights.sum() > 0:
            col_vals = remaining.loc[weights.index, _metric_col]
            mask = col_vals.notna()
            if mask.sum() > 0:
                w = weights[mask]
                new_val = (col_vals[mask] * w).sum() / w.sum()

        if pd.isna(new_val):
            stability[pidx] = 0
            continue

        comparison = passing[[_metric_col]].copy()
        comparison.loc[pidx, _metric_col] = new_val
        comparison = comparison.dropna(subset=[_metric_col])
        new_ranks = comparison.sort_values(_metric_col, ascending=_ascending).reset_index()
        new_ranks["_new_rank"] = range(1, len(new_ranks) + 1)
        new_map = dict(zip(new_ranks["index"], new_ranks["_new_rank"]))

        old_r = base_map.get(pidx, np.nan)
        new_r = new_map.get(pidx, np.nan)
        if pd.notna(old_r) and pd.notna(new_r):
            stability[pidx] = int(new_r - old_r)
        else:
            stability[pidx] = 0

    return pd.Series(stability)

if n_passing > 0 and metric_col in p.columns:
    rank_stability = compute_rank_stability(p, proximal_wells, buffer_distance, metric_col, ascending)
    p["RankStability"] = rank_stability
else:
    p["RankStability"] = np.nan

# ---- Compute colour map values for buffers (using same logic as table) ----
passing_metric_vals = p[p["_passes_filter"]][metric_col].dropna()
if not passing_metric_vals.empty:
    # For the table, gmap_vals = metric_col values if not ascending, else negated
    # The YlGn colormap maps low→yellow, high→green
    # So for "higher is better" metrics, raw values work (high value = green)
    # For "lower is better" (ascending), we negate so lower original = higher negated = green
    if ascending:
        gmap_vmin = float(-passing_metric_vals.max())
        gmap_vmax = float(-passing_metric_vals.min())
    else:
        gmap_vmin = float(passing_metric_vals.min())
        gmap_vmax = float(passing_metric_vals.max())
else:
    gmap_vmin, gmap_vmax = 0.0, 1.0

# Build a label→colour mapping for passing prospects
label_color_map = {}
for idx_val in p[p["_passes_filter"]].index:
    row = p.loc[idx_val]
    label = row["Label"]
    val = row.get(metric_col, np.nan)
    if pd.notna(val):
        gmap_val = -val if ascending else val
        label_color_map[label] = get_ylgn_hex(gmap_val, gmap_vmin, gmap_vmax)
    else:
        label_color_map[label] = "#cccccc"

# Also store the metric value per label for buffer colouring
p["_buffer_color"] = p["Label"].map(label_color_map).fillna("#cccccc")

# ---- Prepare display data ----
section_display = section_enriched.copy().to_crs(4326)
units_display = units_gdf.copy().to_crs(4326)
land_display = land_gdf.copy().to_crs(4326)

existing_display_cols = ["UWI", "geometry"] + [
    c for c in ["EUR", "IP90", "1YCuml", "Wcut", "Section"] if c in proximal_wells.columns
]
existing_display = proximal_wells[existing_display_cols].copy().to_crs(4326)

# Build a lookup: Label → midpoint in EPSG:4326 for zoom
transformer_to_4326 = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)
label_to_latlon = {}
for idx_val, row in p.iterrows():
    mid = midpoint_of_geom(row.geometry)
    if mid is not None:
        lon, lat = transformer_to_4326.transform(mid.x, mid.y)
        label_to_latlon[row["Label"]] = (lat, lon)

# Build buffer GeoDataFrame in projected CRS, then reproject
buffer_records = []
for idx, row in p.iterrows():
    buffer_records.append({
        "Label": row["Label"],
        "_passes_filter": row["_passes_filter"],
        "_no_proximal": row["_no_proximal"],
        "_buffer_color": row["_buffer_color"],
        "Proximal_Count": row.get("Proximal_Count", 0),
        "EUR": row.get("EUR", np.nan),
        "IP90": row.get("IP90", np.nan),
        "1YCuml": row.get("1YCuml", np.nan),
        "Wcut": row.get("Wcut", np.nan),
        "OOIP": row.get("OOIP", np.nan),
        "RFTD": row.get("RFTD", np.nan),
        "URF": row.get("URF", np.nan),
        metric_col: row.get(metric_col, np.nan),
        "geometry": row.geometry.buffer(buffer_distance),
    })
buffer_gdf = gpd.GeoDataFrame(buffer_records, crs=p.crs).to_crs(4326)

# Prospect lines for display
p_lines = p.copy()
drop_cols = [c for c in p_lines.columns if c.startswith("_") and c not in ["_prospect_type"]]
for c in p_lines.columns:
    if c == "geometry":
        continue
    if p_lines[c].dtype == object:
        try:
            p_lines[c] = p_lines[c].astype(str)
        except Exception:
            drop_cols.append(c)
p_lines = p_lines.drop(columns=[c for c in drop_cols if c in p_lines.columns], errors="ignore")
p_lines_display = p_lines.to_crs(4326)

# ================================================================
# EXECUTIVE SUMMARY
# ================================================================
st.title("🛢️ Bakken Prospect Analyzer")
st.caption("Identifying the best locations to drill next based on proximal well performance and reservoir quality.")

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
# We render the RANKING TABLE FIRST so that clicking a row can
# set the zoom target BEFORE we draw the map.
# ==============================================================

# Prepare ranking dataframe once (used by both table and map)
rank_df = None
if selected_metric == "High-Grade Score" and total_weight != 100:
    pass  # will show warning later
else:
    rank_df_raw = p[p["_passes_filter"]].copy()

    display_cols = [
        "Label", "_prospect_type", "Proximal_Count",
        "EUR", "IP90", "1YCuml", "Wcut",
        "OOIP", "RFTD", "URF",
        "EUR_median", "EUR_p10", "EUR_p90",
        "RankStability",
    ]
    if selected_metric == "High-Grade Score":
        display_cols.append("HighGradeScore")
    if metric_col not in display_cols:
        display_cols.append(metric_col)

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

        rename_map = {
            "_prospect_type": "Type",
            "Proximal_Count": "Proximal",
            "EUR_median": "EUR Med",
            "EUR_p10": "EUR P10",
            "EUR_p90": "EUR P90",
            "RankStability": "Rank Δ",
        }
        rank_df = rank_df_raw.rename(columns=rename_map)


col_map, col_rank = st.columns([7, 4])

# ----------------------------------------------------------
# RIGHT COLUMN — Ranking & Detail (rendered first to capture zoom target)
# ----------------------------------------------------------
with col_rank:
    st.header("📊 Prospect Ranking")

    if selected_metric == "High-Grade Score" and total_weight != 100:
        st.warning("Adjust weights to total 100 % to see rankings.")
    elif rank_df is None or rank_df.empty:
        st.warning(f"No valid data for **{selected_metric}**.")
    else:
        st.caption(f"Ranked by **{selected_metric}** · {len(rank_df)} prospects · "
                   f"Buffer: {buffer_distance}m · IDW²")

        summary_cols = st.columns(3)
        is_score = "Score" in selected_metric
        with summary_cols[0]:
            v = rank_df[metric_col].iloc[0]
            st.metric("Best", f"{v:,.2f}" if is_score else f"{v:,.0f}")
        with summary_cols[1]:
            med = rank_df[metric_col].median()
            st.metric("Median", f"{med:,.2f}" if is_score else f"{med:,.0f}")
        with summary_cols[2]:
            v = rank_df[metric_col].iloc[-1]
            st.metric("Worst", f"{v:,.2f}" if is_score else f"{v:,.0f}")

        fmt = {
            "EUR": "{:,.0f}", "IP90": "{:,.0f}", "1YCuml": "{:,.0f}",
            "Wcut": "{:.1f}", "OOIP": "{:,.0f}",
            "RFTD": "{:.3f}", "URF": "{:.3f}", "Percentile": "{:.0f}%",
            "EUR Med": "{:,.0f}", "EUR P10": "{:,.0f}", "EUR P90": "{:,.0f}",
            "Proximal": "{:.0f}", "Rank Δ": "{:+.0f}",
        }
        if "HighGradeScore" in rank_df.columns:
            fmt["HighGradeScore"] = "{:.3f}"

        gmap_vals = rank_df[metric_col] if not ascending else -rank_df[metric_col]

        styled = rank_df.style.background_gradient(
            subset=[metric_col], cmap="YlGn", gmap=gmap_vals,
        ).background_gradient(
            subset=["Percentile"], cmap="RdYlGn",
        ).format(fmt)

        st.dataframe(styled, use_container_width=True, height=500)

        # ----------------------------------------------------------
        # Clickable table: use selectbox to pick a prospect → zoom
        # ----------------------------------------------------------
        st.markdown("##### 🔎 Click to Zoom on Map")
        label_list = rank_df["Label"].tolist()

        # Determine default index from session state
        default_idx = 0
        if st.session_state.get("zoom_to_label") in label_list:
            default_idx = label_list.index(st.session_state["zoom_to_label"])

        selected_label = st.selectbox(
            "Select a prospect to zoom to on map",
            label_list,
            index=default_idx,
            key="zoom_select",
        )
        st.session_state["zoom_to_label"] = selected_label

        csv = rank_df.to_csv().encode("utf-8")
        st.download_button("⬇️ Download Rankings (CSV)", data=csv,
                           file_name="bakken_prospect_rankings.csv", mime="text/csv")

        # ==================================================
        # PROSPECT DETAIL PANEL
        # ==================================================
        st.markdown("---")
        st.subheader("🔬 Prospect Detail")

        # Use the same selected label for detail
        detail_label = selected_label

        if detail_label and detail_label in rank_df["Label"].values:
            dr = rank_df[rank_df["Label"] == detail_label].iloc[0]

            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("EUR", f"{dr['EUR']:,.0f}" if pd.notna(dr.get("EUR")) else "—")
            dc2.metric("IP90", f"{dr['IP90']:,.0f}" if pd.notna(dr.get("IP90")) else "—")
            dc3.metric("1Y Cuml", f"{dr['1YCuml']:,.0f}" if pd.notna(dr.get("1YCuml")) else "—")
            dc4.metric("Wcut", f"{dr['Wcut']:.1f}%" if pd.notna(dr.get("Wcut")) else "—")

            dc5, dc6, dc7, dc8 = st.columns(4)
            dc5.metric("OOIP", f"{dr['OOIP']:,.0f}" if pd.notna(dr.get("OOIP")) else "—")
            dc6.metric("URF", f"{dr['URF']:.3f}" if pd.notna(dr.get("URF")) else "—")
            dc7.metric("RFTD", f"{dr['RFTD']:.3f}" if pd.notna(dr.get("RFTD")) else "—")
            dc8.metric("Proximal", f"{dr['Proximal']:.0f}" if pd.notna(dr.get("Proximal")) else "—")

            if pd.notna(dr.get("Rank Δ")):
                delta_val = int(dr["Rank Δ"])
                st.metric("Rank Stability",
                          f"{delta_val:+d} ranks" if delta_val != 0 else "Stable")

            if pd.notna(dr.get("EUR P10")) and pd.notna(dr.get("EUR P90")):
                st.caption(
                    f"EUR range: P90 = {dr['EUR P90']:,.0f} · "
                    f"Median = {dr.get('EUR Med', 0):,.0f} · "
                    f"P10 = {dr['EUR P10']:,.0f}"
                )

            # ---- Spider Chart ----
            st.markdown("##### Prospect Profile (Radar)")
            radar_cats = ["EUR", "IP90", "1YCuml", "Wcut", "OOIP", "URF", "RFTD"]
            invert_set = {"Wcut", "URF", "RFTD"}

            radar_vals = []
            for c in radar_cats:
                col_data = rank_df[c].dropna()
                if col_data.empty or pd.isna(dr.get(c)):
                    radar_vals.append(50)
                    continue
                pct = (col_data < dr[c]).sum() / len(col_data) * 100
                if c in invert_set:
                    pct = 100 - pct
                radar_vals.append(pct)

            radar_vals.append(radar_vals[0])
            radar_labels = radar_cats + [radar_cats[0]]

            fig_radar = go.Figure(data=go.Scatterpolar(
                r=radar_vals, theta=radar_labels,
                fill="toself",
                fillcolor="rgba(33,150,243,0.25)",
                line_color="#2196F3",
                name=detail_label,
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                margin=dict(l=40, r=40, t=30, b=30),
                height=320,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # ---- No-proximal prospects ----
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
# LEFT COLUMN — Map (rendered after table so zoom target is set)
# ----------------------------------------------------------
with col_map:
    # Determine map centre and zoom
    zoom_label = st.session_state.get("zoom_to_label")
    if zoom_label and zoom_label in label_to_latlon:
        centre_lat, centre_lon = label_to_latlon[zoom_label]
        zoom_level = 14  # zoomed in on the selected prospect
    else:
        bounds = p.total_bounds
        centre_x = (bounds[0] + bounds[2]) / 2
        centre_y = (bounds[1] + bounds[3]) / 2
        transformer = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)
        centre_lon, centre_lat = transformer.transform(centre_x, centre_y)
        zoom_level = 11

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=zoom_level, tiles="CartoDB positron")
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)

    # Layer 0: Bakken Land
    land_fg = folium.FeatureGroup(name="Bakken Land", show=True)
    folium.GeoJson(
        land_display.to_json(),
        style_function=lambda _: {
            "fillColor": "#fff9c4",
            "color": "#fff9c4",
            "weight": 0.5,
            "fillOpacity": 0.2,
        },
    ).add_to(land_fg)
    land_fg.add_to(m)

    # Layer 1: Section grid
    if section_gradient != "None":
        grad_col = GRADIENT_COL_MAP[section_gradient]
        grad_vals = section_display[grad_col].dropna()
        lower_is_better = section_gradient in ["Avg Wcut", "RFTD", "URF"]

        if not grad_vals.empty:
            colors = (["#006837", "#78c679", "#ffffcc"] if lower_is_better
                      else ["#f7fcf5", "#74c476", "#00441b"])
            colormap = cm.LinearColormap(
                colors=colors,
                vmin=float(grad_vals.min()),
                vmax=float(grad_vals.max()),
            ).to_step(n=7)
            colormap.caption = section_gradient
            m.add_child(colormap)

            def section_style(feature):
                val = feature["properties"].get(grad_col)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return NULL_STYLE
                return {"fillColor": colormap(val), "fillOpacity": 0.45,
                        "color": "white", "weight": 0.3}
        else:
            section_style = lambda _: NULL_STYLE
    else:
        section_style = lambda _: NULL_STYLE

    sec_fields = ["Section", "OOIP", "Well_Count", "Avg_EUR", "Avg_IP90",
                  "Avg_1YCuml", "Avg_Wcut", "RFTD", "URF"]
    sec_aliases = ["Section:", "OOIP:", "Wells:", "Avg EUR:", "Avg IP90:",
                  "Avg 1Y Cuml:", "Avg Wcut:", "RFTD:", "URF:"]

    section_fg = folium.FeatureGroup(name="Section Grid", show=(section_gradient != "None"))
    folium.GeoJson(
        section_display.to_json(), name="Sections",
        style_function=section_style,
        highlight_function=lambda _: {"weight": 2, "color": "black", "fillOpacity": 0.5},
        tooltip=folium.GeoJsonTooltip(
            fields=sec_fields, aliases=sec_aliases,
            localize=True, sticky=True,
            style="font-size:11px;padding:4px 8px;background:rgba(255,255,255,0.9);border:1px solid #333;border-radius:3px;",
        ),
    ).add_to(section_fg)
    section_fg.add_to(m)

    # Layer 2: Units
    units_fg = folium.FeatureGroup(name="Units", show=True)
    folium.GeoJson(
        units_display.to_json(),
        style_function=lambda _: {"color": "black", "weight": 2, "fillOpacity": 0, "interactive": False},
    ).add_to(units_fg)
    units_fg.add_to(m)

    # Layer 3: Buffers — colour matches table YlGn gradient exactly
    buffer_fg = folium.FeatureGroup(name="Prospect Buffers")

    # Passing buffers
    passing_buf = buffer_gdf[buffer_gdf["_passes_filter"]].copy()

    for bidx, brow in passing_buf.iterrows():
        fill_color = label_color_map.get(brow["Label"], "#cccccc")

        # Highlight the selected/zoomed prospect with a thicker border
        is_selected = (brow["Label"] == zoom_label)
        border_weight = 3 if is_selected else 1
        border_color = "#ff0000" if is_selected else fill_color
        fill_opacity = 0.55 if is_selected else 0.4

        tip_parts = [
            f"<b>{brow['Label']}</b>",
            f"Proximal Wells: {brow.get('Proximal_Count', '—')}",
        ]
        for col, label, fmt_str in [
            ("EUR", "EUR", ",.0f"),
            ("IP90", "IP90", ",.0f"),
            ("1YCuml", "1Y Cuml", ",.0f"),
            ("Wcut", "Wcut", ".1f"),
            ("OOIP", "OOIP", ",.0f"),
            ("RFTD", "RFTD", ".3f"),
            ("URF", "URF", ".3f"),
        ]:
            if pd.notna(brow.get(col)):
                tip_parts.append(f"{label}: {brow[col]:{fmt_str}}")

        tip_text = "<br>".join(tip_parts)

        folium.GeoJson(
            brow.geometry.__geo_interface__,
            style_function=lambda _, fc=fill_color, bw=border_weight, bc=border_color, fo=fill_opacity: {
                "fillColor": fc,
                "fillOpacity": fo,
                "color": bc,
                "weight": bw,
                "opacity": 0.7,
            },
            tooltip=folium.Tooltip(tip_text, sticky=True,
                style="font-size:11px;padding:3px 6px;background:rgba(255,255,255,0.92);border:1px solid #333;border-radius:3px;"),
        ).add_to(buffer_fg)

    # Filtered-out buffers
    filtered_buf = buffer_gdf[
        ~buffer_gdf["_passes_filter"] & ~buffer_gdf["_no_proximal"]
    ]
    for bidx, brow in filtered_buf.iterrows():
        folium.GeoJson(
            brow.geometry.__geo_interface__,
            style_function=lambda _: {
                "fillColor": "#d3d3d3",
                "fillOpacity": 0.15,
                "color": "#aaa",
                "weight": 0.5,
                "opacity": 0.3,
            }
        ).add_to(buffer_fg)

    # No-proximal buffers
    no_proximal_buf = buffer_gdf[buffer_gdf["_no_proximal"]]
    for bidx, brow in no_proximal_buf.iterrows():
        folium.GeoJson(
            brow.geometry.__geo_interface__,
            style_function=lambda _: {
                "fillColor": "#ffe0b2",
                "fillOpacity": 0.1,
                "color": "orange",
                "weight": 1,
                "dashArray": "5 5",
                "opacity": 0.4,
            }
        ).add_to(buffer_fg)

    buffer_fg.add_to(m)

    # Layer 4: Existing wells
    well_fg = folium.FeatureGroup(name="Existing Wells")
    line_wells = existing_display[existing_display.geometry.type != "Point"]
    point_wells = existing_display[existing_display.geometry.type == "Point"]

    if not line_wells.empty:
        wl_fields = ["UWI"]
        wl_aliases = ["UWI:"]
        for wf in ["EUR", "IP90", "1YCuml", "Wcut", "Section"]:
            if wf in line_wells.columns:
                wl_fields.append(wf)
                wl_aliases.append(f"{wf}:")
        folium.GeoJson(
            line_wells.to_json(),
            style_function=lambda _: {"color": "black", "weight": 1, "opacity": 0.7},
            highlight_function=lambda _: {"weight": 2.5, "color": "#555"},
            tooltip=folium.GeoJsonTooltip(
                fields=wl_fields, aliases=wl_aliases,
                localize=True, sticky=True,
                style="font-size:11px;padding:3px 6px;background:rgba(255,255,255,0.92);border:1px solid #333;border-radius:3px;",
            ),
        ).add_to(well_fg)

    for _, row in point_wells.iterrows():
        tip_parts = [f"<b>UWI:</b> {row.get('UWI', '—')}"]
        for col, label, fmt_str in [("EUR", "EUR", ",.0f"), ("IP90", "IP90", ",.0f"),
                                ("1YCuml", "1Y Cuml", ",.0f"), ("Wcut", "Wcut", ".1f")]:
            if col in row.index and pd.notna(row[col]):
                tip_parts.append(f"<b>{label}:</b> {row[col]:{fmt_str}}")
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=1, color="black", fill=True, fill_color="black",
            fill_opacity=0.7, weight=1,
            tooltip=folium.Tooltip("<br>".join(tip_parts), sticky=True,
                style="font-size:11px;padding:3px 6px;background:rgba(255,255,255,0.92);border:1px solid #333;border-radius:3px;"),
        ).add_to(well_fg)
    well_fg.add_to(m)

    # Layer 5: Prospect lines
    prospect_fg = folium.FeatureGroup(name="Prospect Wells", show=True)

    pt_fields_wanted = [
        ("Label", "Prospect:"), ("_prospect_type", "Type:"),
        ("Proximal_Count", "Proximal Wells:"),
        ("EUR", "EUR:"), ("IP90", "IP90:"), ("1YCuml", "1Y Cuml:"),
        ("Wcut", "Wcut:"), ("OOIP", "OOIP:"), ("RFTD", "RFTD:"), ("URF", "URF:"),
    ]
    pt_fields = [f for f, _ in pt_fields_wanted if f in p_lines_display.columns]
    pt_aliases = [a for f, a in pt_fields_wanted if f in p_lines_display.columns]

    folium.GeoJson(
        p_lines_display[pt_fields + ["geometry"]].to_json(),
        style_function=lambda _: {"color": "red", "weight": 3, "opacity": 0.9},
        highlight_function=lambda _: {"weight": 5, "color": "#ff4444"},
        tooltip=folium.GeoJsonTooltip(
            fields=pt_fields, aliases=pt_aliases,
            localize=True, sticky=True,
            style="font-size:12px",
        ),
    ).add_to(prospect_fg)
    prospect_fg.add_to(m)

    # Add a marker for the selected/zoomed prospect
    if zoom_label and zoom_label in label_to_latlon:
        sel_lat, sel_lon = label_to_latlon[zoom_label]
        sel_color = label_color_map.get(zoom_label, "#2196F3")
        folium.Marker(
            location=[sel_lat, sel_lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:bold;color:#222;'
                     f'background:{sel_color};border:2px solid #333;border-radius:4px;'
                     f'padding:2px 6px;white-space:nowrap;transform:translate(-50%,-100%);">'
                     f'📍 {zoom_label}</div>',
                icon_size=(0, 0),
                icon_anchor=(0, 0),
            ),
        ).add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m, use_container_width=True, height=900, returned_objects=[])

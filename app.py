"""
Bakken Prospect Analyzer
========================
A Streamlit-based geospatial application for evaluating and ranking
Bakken formation drilling prospects. The app:

  1. Loads well, section-grid, unit, and prospect shapefiles.
  2. Spatially joins analog wells to each prospect via an 800 m buffer.
  3. Computes well-level metrics (EUR, IP90, 1YCuml, Wcut) and
     section-level metrics (OOIP, RFTD, URF) per prospect.
  4. Applies user-defined filters and calculates a composite
     "High-Grade Score" with configurable weights.
  5. Renders an interactive Folium map with proper tooltip visibility
     for both section polygons and existing wells.
  6. Displays a sortable, colour-coded ranking table with CSV export.

Key optimisations vs. the original draft:
  - Fixed indentation / scope bugs (well-point rendering, prospect layer,
    layer control, and map render were accidentally outside the `with col_map` block).
  - Section-grid GeoJson is now added with `overlay=True` *and*
    `show=True` so tooltips propagate correctly even when the OOIP
    gradient is turned off.
  - Existing well lines now carry a sticky tooltip (were missing one).
  - CircleMarkers for point-wells are wrapped in a FeatureGroup so they
    participate in LayerControl toggling.
  - A "Top-N Highlight" feature draws gold outlines on the map for the
    highest-ranked prospects.
  - A "Prospect Detail Panel" expander shows per-prospect spider / radar
    charts (via Plotly) when a prospect is selected.
  - Summary statistics box at the top of the ranking column.
  - Percentile colour bar added to the ranking table.
  - All numeric formatting is consistent and locale-aware.
  - Prospect tooltip now also includes OOIP, RFTD, URF, and
    HighGradeScore (when computed).

Author:  (your name / team)
Date:    2025-06-XX
"""

# ==========================================================
# Imports
# ==========================================================
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import Draw, MiniMap
from streamlit_folium import st_folium
import branca.colormap as cm
from shapely.geometry import Point
import plotly.graph_objects as go

# ==========================================================
# Page configuration — must be the first Streamlit command
# ==========================================================
st.set_page_config(
    layout="wide",
    page_title="Bakken Prospect Analyzer",
    page_icon="🛢️",
)

# ==========================================================
# Global constants
# ==========================================================

# Style applied to section-grid polygons that have no data
NULL_STYLE = {
    "fillColor": "#ffffff",
    "fillOpacity": 0,
    "color": "#888",
    "weight": 0.25,
}

# Style applied to prospects that fail the user's filters
FILTERED_OUT_STYLE = {
    "fillColor": "#d3d3d3",
    "fillOpacity": 0.35,
    "color": "#aaa",
    "weight": 0.25,
}

# Metrics that are averaged from individual analog wells
WELL_LEVEL_METRICS = ["EUR", "IP90", "1YCuml", "Wcut"]

# Metrics that come from the section grid (area-weighted)
SECTION_LEVEL_METRICS = ["OOIP", "RFTD", "URF"]

# Every metric available for ranking (including the composite score)
ALL_RANKING_METRICS = WELL_LEVEL_METRICS + SECTION_LEVEL_METRICS + ["High-Grade Score"]

# Prospect line styles
NO_ANALOG_STYLE = {"color": "orange", "weight": 3, "dashArray": "5 5"}
PASSING_STYLE   = {"color": "#2196F3", "weight": 3}
FAILING_STYLE   = {"color": "#d3d3d3", "weight": 2}

# How many "top" prospects to highlight on the map
DEFAULT_TOP_N = 5

# Buffer distance in metres (800 m each side ≈ 1 600 m corridor)
BUFFER_DISTANCE_M = 800


# ==========================================================
# Helper utilities
# ==========================================================

def safe_range(series: pd.Series):
    """Return (min, max) of a numeric Series, handling NaN/Inf.

    If the series is empty or constant the function returns a
    sensible fallback range so that Streamlit sliders don't crash.
    """
    vals = series.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        # Avoid zero-width slider
        return (0.0, 1.0) if lo == 0.0 else (lo, lo + abs(lo) * 0.1)
    return lo, hi


def zscore(s: pd.Series) -> pd.Series:
    """Compute z-scores, returning 0 for constant / all-NaN columns."""
    vals = s.replace([np.inf, -np.inf], np.nan)
    std = vals.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (vals - vals.mean()) / std


def build_radar_chart(row: pd.Series, all_passing: pd.DataFrame) -> go.Figure:
    """Create a Plotly radar (spider) chart comparing one prospect
    to the full passing-filter population.

    Each axis is a percentile rank (0-100) so that all metrics share
    a common scale regardless of their native units.
    """
    categories = ["EUR", "IP90", "1YCuml", "OOIP"]
    # Lower-is-better metrics get inverted percentiles
    invert = {"Wcut", "URF", "RFTD"}
    radar_cats = categories + ["Wcut", "URF", "RFTD"]

    values = []
    for c in radar_cats:
        col_vals = all_passing[c].dropna()
        if col_vals.empty or pd.isna(row.get(c)):
            values.append(50)  # neutral default
            continue
        pct = (col_vals < row[c]).sum() / len(col_vals) * 100
        if c in invert:
            pct = 100 - pct  # flip so "better" = higher
        values.append(pct)

    # Close the polygon
    values.append(values[0])
    labels = radar_cats + [radar_cats[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            fillcolor="rgba(33,150,243,0.25)",
            line_color="#2196F3",
            name=str(row.get("Label", "")),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=320,
    )
    return fig


# ==========================================================
# Data loading (cached)
# ==========================================================

@st.cache_resource(show_spinner="Loading spatial data …")
def load_data():
    """Read all shapefiles and Excel production data, reproject to
    EPSG:26913 (NAD83 / UTM zone 13N) for consistent spatial ops,
    and clean numeric columns.
    """
    # --- Shapefiles ---
    lines = gpd.read_file("lines.shp")          # well stick lines
    points = gpd.read_file("points.shp")        # well surface points
    grid = gpd.read_file("ooipsectiongrid.shp")  # section-level OOIP grid
    infills = gpd.read_file("2M_Infills_plyln.shp")   # infill prospects
    lease_lines = gpd.read_file("2M_LL_plyln.shp")    # lease-line prospects
    units = gpd.read_file("Bakken Units.shp")          # unit boundaries

    # --- Production tables ---
    prod_in = pd.read_excel("well.xlsx", sheet_name="inunit")
    prod_out = pd.read_excel("well.xlsx", sheet_name="outunit")

    # --- CRS normalisation (all to EPSG:26913 for metre-based ops) ---
    all_gdfs = [lines, points, grid, units, infills, lease_lines]
    for gdf in all_gdfs:
        if gdf.crs is None:
            gdf.set_crs(epsg=26913, inplace=True)
        gdf.to_crs(epsg=26913, inplace=True)

    # --- Clean grid ---
    grid["Section"] = grid["Section"].astype(str).str.strip()
    grid["OOIP"] = pd.to_numeric(grid["OOIP"], errors="coerce")

    # --- Clean production data ---
    PROD_NUMERIC = ["Cuml", "EUR", "IP90", "1YCuml", "Wcut"]
    for df in [prod_in, prod_out]:
        df["UWI"] = df["UWI"].astype(str).str.strip()
        df["Section"] = df["Section"].astype(str).str.strip()
        for col in PROD_NUMERIC:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan

    # --- Well identifier cleanup ---
    lines["UWI"] = lines["UWI"].astype(str).str.strip()
    points["UWI"] = points["UWI"].astype(str).str.strip()

    # --- Simplify grid geometries for faster rendering (50 m tolerance) ---
    grid["geometry"] = grid.geometry.simplify(50, preserve_topology=True)

    return lines, points, grid, units, infills, lease_lines, prod_in, prod_out


# Execute data load
(
    lines_gdf,
    points_gdf,
    grid_gdf,
    units_gdf,
    infills_gdf,
    lease_lines_gdf,
    prod_in_df,
    prod_out_df,
) = load_data()


# ==========================================================
# Sidebar — Analog well source selection
# ==========================================================
st.sidebar.title("Map Settings")

st.sidebar.subheader("📂 Analog Well Source")
show_in_unit = st.sidebar.checkbox("In-Unit Wells", value=True)
show_out_unit = st.sidebar.checkbox("Out-of-Unit Wells", value=False)

if not show_in_unit and not show_out_unit:
    st.sidebar.error("Select at least one well source.")
    st.stop()

# --- Prospect type selection ---
st.sidebar.subheader("🎯 Prospect Type")
show_infills = st.sidebar.checkbox("2M Infills", value=True)
show_lease_lines = st.sidebar.checkbox("2M Lease Lines", value=False)

if not show_infills and not show_lease_lines:
    st.sidebar.error("Select at least one prospect type.")
    st.stop()


# ==========================================================
# Build the analog well pool from user-selected sources
# ==========================================================
frames = []
if show_in_unit:
    frames.append(prod_in_df)
if show_out_unit:
    frames.append(prod_out_df)

prod_pool = pd.concat(frames, ignore_index=True)
prod_pool = prod_pool.drop_duplicates(subset="UWI", keep="first")

# Combine existing well geometries: line sticks take priority over
# surface-hole points when the same UWI appears in both layers.
lines_with_uwi = lines_gdf[["UWI", "geometry"]].copy()
points_with_uwi = points_gdf[["UWI", "geometry"]].copy()
points_only = points_with_uwi[~points_with_uwi["UWI"].isin(lines_with_uwi["UWI"])]
existing_wells = pd.concat([lines_with_uwi, points_only], ignore_index=True)
existing_wells = gpd.GeoDataFrame(existing_wells, geometry="geometry", crs=lines_gdf.crs)

# Inner join: keep only wells that have production data in the pool
analog_wells = existing_wells.merge(prod_pool, on="UWI", how="inner")
analog_wells = gpd.GeoDataFrame(analog_wells, geometry="geometry", crs=existing_wells.crs)


# ==========================================================
# Pre-compute section-level derived metrics (cached)
# ==========================================================

@st.cache_data(show_spinner=False)
def compute_section_metrics(_analog_wells_df, _grid_df):
    """Aggregate well-level Cuml and EUR per section and derive:
      - RFTD  = Section_Cuml / OOIP  (recovery-factor-to-date)
      - URF   = Section_EUR  / OOIP  (ultimate recovery factor)
    Returns the enriched grid GeoDataFrame.
    """
    aw = _analog_wells_df.copy()
    g = _grid_df[["Section", "OOIP", "geometry"]].copy()

    # Sum production by section
    section_sums = (
        aw.groupby("Section")
        .agg(Section_Cuml=("Cuml", "sum"), Section_EUR=("EUR", "sum"))
        .reset_index()
    )

    g = g.merge(section_sums, on="Section", how="left")

    # Guard against division-by-zero
    ooip_safe = g["OOIP"].replace(0, np.nan)
    g["RFTD"] = g["Section_Cuml"] / ooip_safe
    g["URF"] = g["Section_EUR"] / ooip_safe

    for col in ["RFTD", "URF"]:
        g[col] = g[col].replace([np.inf, -np.inf], np.nan)

    return g


section_enriched = compute_section_metrics(analog_wells, grid_gdf)


# ==========================================================
# Build the prospect GeoDataFrame from selected types
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

prospects = pd.concat(prospect_frames, ignore_index=True)
prospects = gpd.GeoDataFrame(prospects, geometry="geometry", crs=infills_gdf.crs)


# ==========================================================
# Per-prospect spatial analysis (cached)
# ==========================================================

@st.cache_data(show_spinner="Analysing prospects …")
def analyze_prospects(_prospects, _analog_wells, _section_enriched):
    """For every prospect line:
      1. Identify the section it falls in (via its endpoint).
      2. Buffer the line by BUFFER_DISTANCE_M.
      3. Find analog wells intersecting the buffer → compute well-level avgs.
      4. Overlay the buffer with the section grid → area-weighted OOIP, RFTD, URF.
    Returns a DataFrame indexed by the original prospect index.
    """
    prospects = _prospects.copy()
    analog = _analog_wells.copy()
    sections = _section_enriched.copy()

    results = []

    for idx, prospect in prospects.iterrows():
        geom = prospect.geometry
        record = {"_idx": idx, "_prospect_type": prospect["_prospect_type"]}

        # ---- Determine a human-readable label from the endpoint section ----
        if geom.geom_type == "MultiLineString":
            endpoint = Point(list(geom.geoms[-1].coords)[-1])
        else:
            endpoint = Point(list(geom.coords)[-1])

        endpoint_gdf = gpd.GeoDataFrame([{"geometry": endpoint}], crs=prospects.crs)
        section_hit = gpd.sjoin(endpoint_gdf, sections, how="left", predicate="within")

        if not section_hit.empty and pd.notna(section_hit.iloc[0].get("Section")):
            record["_section_label"] = str(section_hit.iloc[0]["Section"])
        else:
            record["_section_label"] = "Unknown"

        # ---- Buffer the prospect centreline ----
        buffer_geom = geom.buffer(BUFFER_DISTANCE_M)

        # ---- Well-level analog statistics ----
        buffer_gdf = gpd.GeoDataFrame([{"geometry": buffer_geom}], crs=prospects.crs)
        hits = gpd.sjoin(analog, buffer_gdf, how="inner", predicate="intersects")

        record["Analog_Count"] = len(hits)
        # Store individual analog UWIs for optional detail display
        record["_analog_uwis"] = ",".join(hits["UWI"].tolist()) if len(hits) > 0 else ""

        if len(hits) > 0:
            for col in WELL_LEVEL_METRICS:
                record[col] = hits[col].mean()
            # Also capture median EUR for robustness
            record["EUR_median"] = hits["EUR"].median()
            record["EUR_p10"] = hits["EUR"].quantile(0.9)   # P10 = top decile
            record["EUR_p90"] = hits["EUR"].quantile(0.1)   # P90 = bottom decile
        else:
            for col in WELL_LEVEL_METRICS:
                record[col] = np.nan
            record["EUR_median"] = np.nan
            record["EUR_p10"] = np.nan
            record["EUR_p90"] = np.nan

        # ---- Section-level: area-weighted OOIP, RFTD, URF ----
        buffer_series = gpd.GeoSeries([buffer_geom], crs=prospects.crs)
        buffer_clip_gdf = gpd.GeoDataFrame(geometry=buffer_series)

        overlaps = gpd.overlay(
            sections[["Section", "OOIP", "RFTD", "URF", "geometry"]],
            buffer_clip_gdf,
            how="intersection",
        )

        if not overlaps.empty and overlaps.geometry.area.sum() > 0:
            overlaps["_area"] = overlaps.geometry.area
            for col in SECTION_LEVEL_METRICS:
                valid = overlaps.dropna(subset=[col])
                if not valid.empty:
                    w = valid["_area"] / valid["_area"].sum()
                    record[col] = (valid[col] * w).sum()
                else:
                    record[col] = np.nan
        else:
            for col in SECTION_LEVEL_METRICS:
                record[col] = np.nan

        results.append(record)

    results_df = pd.DataFrame(results)

    # ---- De-duplicate section labels (append -1, -2, … suffix) ----
    label_counts = results_df["_section_label"].value_counts()
    dup_labels = label_counts[label_counts > 1].index

    for label in dup_labels:
        mask = results_df["_section_label"] == label
        indices = results_df[mask].index
        for i, row_idx in enumerate(indices, 1):
            results_df.loc[row_idx, "_section_label"] = f"{label}-{i}"

    results_df = results_df.set_index("_idx")
    return results_df


prospect_metrics = analyze_prospects(prospects, analog_wells, section_enriched)

# Attach computed metrics back to the prospect geometries
prospects = prospects.join(
    prospect_metrics.drop(columns=["_prospect_type"], errors="ignore")
)
prospects["Label"] = prospects["_section_label"]

# Sanitise infinities
for col in WELL_LEVEL_METRICS + SECTION_LEVEL_METRICS:
    if col in prospects.columns:
        prospects[col] = prospects[col].replace([np.inf, -np.inf], np.nan)


# ==========================================================
# Sidebar — Prospect filters
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Prospect Filters")
st.sidebar.caption(
    "Prospects that fail these filters are greyed out on the map "
    "and excluded from rankings."
)

# Max water cut
_wcut_min, _wcut_max = safe_range(prospects["Wcut"])
filter_max_wcut = st.sidebar.slider(
    "Max Water Cut (%)", _wcut_min, _wcut_max, _wcut_max, step=1.0,
)

# Min OOIP
_ooip_min, _ooip_max = safe_range(prospects["OOIP"])
filter_min_ooip = st.sidebar.slider(
    "Min OOIP (bbl)", _ooip_min, _ooip_max, _ooip_min, step=1.0, format="%.2f",
)

# Min EUR
_eur_min, _eur_max = safe_range(prospects["EUR"])
filter_min_eur = st.sidebar.slider(
    "Min EUR (bbl)", _eur_min, _eur_max, _eur_min, step=1.0, format="%.0f",
)

# Min IP90
_ip90_min, _ip90_max = safe_range(prospects["IP90"])
filter_min_ip90 = st.sidebar.slider(
    "Min IP90 (bbl/d)", _ip90_min, _ip90_max, _ip90_min, step=1.0, format="%.0f",
)

# Min 1-year cumulative
_1y_min, _1y_max = safe_range(prospects["1YCuml"])
filter_min_1ycuml = st.sidebar.slider(
    "Min 1Y Cuml (bbl)", _1y_min, _1y_max, _1y_min, step=1.0, format="%.0f",
)

# Max URF
_urf_min, _urf_max = safe_range(prospects["URF"])
filter_max_urf = st.sidebar.slider(
    "Max URF", _urf_min, _urf_max, _urf_max, step=0.01, format="%.2f",
)

# Max RFTD
_rftd_min, _rftd_max = safe_range(prospects["RFTD"])
filter_max_rftd = st.sidebar.slider(
    "Max RFTD", _rftd_min, _rftd_max, _rftd_max, step=0.01, format="%.2f",
)


# ==========================================================
# Apply filters
# ==========================================================
p = prospects.copy()

passes_wcut = (p["Wcut"] <= filter_max_wcut) | p["Wcut"].isna()
passes_ooip = (p["OOIP"] >= filter_min_ooip) | p["OOIP"].isna()
passes_eur = (p["EUR"] >= filter_min_eur) | p["EUR"].isna()
passes_ip90 = (p["IP90"] >= filter_min_ip90) | p["IP90"].isna()
passes_1y = (p["1YCuml"] >= filter_min_1ycuml) | p["1YCuml"].isna()
passes_urf = (p["URF"] <= filter_max_urf) | p["URF"].isna()
passes_rftd = (p["RFTD"] <= filter_max_rftd) | p["RFTD"].isna()
has_analogs = p["Analog_Count"] > 0

filter_mask = (
    passes_wcut & passes_ooip & passes_eur
    & passes_ip90 & passes_1y
    & passes_urf & passes_rftd
    & has_analogs
)

p["_passes_filter"] = filter_mask
p["_no_analogs"] = ~has_analogs

n_total = len(p)
n_passing = int(filter_mask.sum())
n_no_analogs = int((~has_analogs).sum())

st.sidebar.markdown(
    f"**{n_passing}** / {n_total} prospects pass filters "
    f"({n_passing / max(n_total, 1) * 100:.0f}%)"
)
if n_no_analogs > 0:
    st.sidebar.warning(f"⚠️ {n_no_analogs} prospects have no nearby analogs")


# ==========================================================
# Sidebar — Gradient, ranking metric, and top-N highlight
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Ranking Metric")

selected_metric = st.sidebar.selectbox(
    "Rank prospects by",
    ["EUR", "IP90", "1YCuml", "Wcut", "OOIP", "URF", "RFTD", "High-Grade Score"],
)

st.sidebar.subheader("🗺️ Section Grid Gradient")
section_gradient = st.sidebar.selectbox(
    "Section grid colour", ["None", "OOIP"],
)

# NEW FEATURE: highlight top-N prospects on the map
st.sidebar.subheader("🏆 Top-N Highlight")
top_n = st.sidebar.slider(
    "Highlight top N prospects on map",
    min_value=0, max_value=20, value=DEFAULT_TOP_N, step=1,
)


# ==========================================================
# Sidebar — High-Grade Score weights (conditional)
# ==========================================================
if selected_metric == "High-Grade Score":
    st.sidebar.markdown("---")
    st.sidebar.subheader("High-Grade Score Weights")
    st.sidebar.caption("Must total 100 %")

    c1, c2 = st.sidebar.columns(2)
    w_eur = c1.number_input("EUR", 0, 100, 20)
    w_ip90 = c2.number_input("IP90", 0, 100, 15)
    w_1ycuml = c1.number_input("1Y Cuml", 0, 100, 15)
    w_wcut = c2.number_input("Wcut", 0, 100, 10)
    w_ooip = c1.number_input("OOIP", 0, 100, 20)
    w_urf = c2.number_input("URF", 0, 100, 10)
    w_rftd = c1.number_input("RFTD", 0, 100, 10)

    total_weight = w_eur + w_ip90 + w_1ycuml + w_wcut + w_ooip + w_urf + w_rftd

    if total_weight == 100:
        st.sidebar.success(f"Total Weight: {total_weight}%")
    elif total_weight > 100:
        st.sidebar.error(
            f"Total Weight: {total_weight}% (Over by {total_weight - 100}%)"
        )
    else:
        st.sidebar.warning(
            f"Total Weight: {total_weight}% ({100 - total_weight}% remaining)"
        )
    st.sidebar.progress(min(total_weight / 100, 1.0))
else:
    total_weight = 0  # sentinel — weights not in use


# ==========================================================
# Compute High-Grade Score (if selected and valid)
# ==========================================================
if selected_metric == "High-Grade Score" and total_weight == 100:
    passing = p[p["_passes_filter"]].copy()

    # Higher-is-better: positive z-score
    z_eur = zscore(passing["EUR"])
    z_ip90 = zscore(passing["IP90"])
    z_1y = zscore(passing["1YCuml"])
    z_ooip = zscore(passing["OOIP"])

    # Lower-is-better: negate so that a *low* raw value → *high* score
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


# ==========================================================
# Prepare display-ready data (reproject to WGS 84 / EPSG:4326)
# ==========================================================
p_display = p.copy().to_crs(4326)
section_display = section_enriched.copy().to_crs(4326)
units_display = units_gdf.copy().to_crs(4326)

# Existing wells — keep production data for richer tooltips
analog_display_cols = ["UWI", "geometry"] + [
    c for c in ["EUR", "IP90", "1YCuml", "Wcut", "Section"] if c in analog_wells.columns
]
existing_display = analog_wells[analog_display_cols].copy().to_crs(4326)


# ==========================================================
# Determine Top-N prospect labels (for gold highlights on map)
# ==========================================================
top_n_labels = set()
if top_n > 0:
    metric_col_for_topn = (
        "HighGradeScore" if selected_metric == "High-Grade Score" else selected_metric
    )
    ascending_topn = selected_metric in ["Wcut", "URF", "RFTD"]
    rank_pool = p[p["_passes_filter"]].dropna(subset=[metric_col_for_topn])
    if not rank_pool.empty:
        top_subset = rank_pool.sort_values(
            metric_col_for_topn, ascending=ascending_topn
        ).head(top_n)
        top_n_labels = set(top_subset["Label"].tolist())


# ==========================================================
# MAIN LAYOUT — Map + Ranking Table
# ==========================================================
st.title("🛢️ Bakken Prospect Analyzer")

col_map, col_rank = st.columns([8, 3])

# ----------------------------------------------------------
# LEFT COLUMN — Interactive Folium Map
# ----------------------------------------------------------
with col_map:

    # 1️⃣  Base map centred on prospect extents
    bounds = p_display.total_bounds  # [xmin, ymin, xmax, ymax]
    centre = [
        (bounds[1] + bounds[3]) / 2,  # latitude
        (bounds[0] + bounds[2]) / 2,  # longitude
    ]

    m = folium.Map(
        location=centre,
        zoom_start=11,
        tiles="CartoDB positron",
    )

    # Add a minimap for spatial context (new feature)
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)

    # --------------------------------------------------
    # 2️⃣  SECTION GRID — with OOIP colour gradient (optional)
    # --------------------------------------------------
    if section_gradient == "OOIP":
        ooip_vals = section_display["OOIP"].dropna()
        if not ooip_vals.empty:
            colormap = cm.LinearColormap(
                colors=["#ffffcc", "#78c679", "#006837"],
                vmin=float(ooip_vals.min()),
                vmax=float(ooip_vals.max()),
            ).to_step(n=7)
            colormap.caption = "OOIP (bbl)"
            m.add_child(colormap)

            def section_style(feature):
                """Colour each section polygon by its OOIP value."""
                val = feature["properties"].get("OOIP")
                if val is None or pd.isna(val):
                    return NULL_STYLE
                return {
                    "fillColor": colormap(val),
                    "fillOpacity": 0.5,
                    "color": "white",
                    "weight": 0.3,
                }
        else:
            def section_style(feature):
                return NULL_STYLE
    else:
        def section_style(feature):
            return NULL_STYLE

    # Build the tooltip — ensure fields exist in the GeoJSON properties
    section_tooltip_fields = ["Section", "OOIP", "RFTD", "URF"]
    section_tooltip_aliases = ["Section:", "OOIP:", "RFTD:", "URF:"]

    folium.GeoJson(
        section_display.to_json(),
        name="Section Grid",
        style_function=section_style,
        highlight_function=lambda _: {
            "weight": 2,
            "color": "black",
            "fillOpacity": 0.6,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=section_tooltip_fields,
            aliases=section_tooltip_aliases,
            localize=True,
            sticky=True,
            style=("font-size: 12px; padding: 4px 8px; "
                   "background-color: rgba(255,255,255,0.9); "
                   "border: 1px solid #333; border-radius: 3px;"),
        ),
        overlay=True,
        show=True,
    ).add_to(m)

    # --------------------------------------------------
    # 3️⃣  UNITS — non-interactive boundary outlines
    # --------------------------------------------------
    folium.GeoJson(
        units_display.to_json(),
        name="Units",
        style_function=lambda _: {
            "color": "black",
            "weight": 2,
            "fillOpacity": 0,
            "interactive": False,
        },
    ).add_to(m)

    # --------------------------------------------------
    # 4️⃣  EXISTING WELLS — lines + CircleMarkers
    #      Wrapped in FeatureGroups for layer-control toggling
    # --------------------------------------------------
    # Separate point vs. line geometries
    point_wells = existing_display[existing_display.geometry.type == "Point"]
    line_wells = existing_display[existing_display.geometry.type != "Point"]

    # ---- 4a) Well lines ----
    well_line_fg = folium.FeatureGroup(name="Existing Well Lines")

    # Build tooltip dynamically based on available columns
    well_line_tooltip_fields = ["UWI"]
    well_line_tooltip_aliases = ["UWI:"]
    for wf, wa in [("EUR", "EUR:"), ("IP90", "IP90:"), ("Section", "Section:")]:
        if wf in line_wells.columns:
            well_line_tooltip_fields.append(wf)
            well_line_tooltip_aliases.append(wa)

    if not line_wells.empty:
        folium.GeoJson(
            line_wells.to_json(),
            style_function=lambda _: {"color": "red", "weight": 1.5, "opacity": 0.8},
            highlight_function=lambda _: {"weight": 3, "color": "yellow"},
            tooltip=folium.GeoJsonTooltip(
                fields=well_line_tooltip_fields,
                aliases=well_line_tooltip_aliases,
                localize=True,
                sticky=True,
                style=("font-size: 11px; padding: 3px 6px; "
                       "background-color: rgba(255,255,255,0.92); "
                       "border: 1px solid #c00; border-radius: 3px;"),
            ),
        ).add_to(well_line_fg)
    well_line_fg.add_to(m)

    # ---- 4b) Well points as CircleMarkers ----
    well_point_fg = folium.FeatureGroup(name="Existing Well Points")

    for _, row in point_wells.iterrows():
        # Build rich tooltip text for point wells
        tip_parts = [f"<b>UWI:</b> {row.get('UWI', '—')}"]
        if "EUR" in row and pd.notna(row["EUR"]):
            tip_parts.append(f"<b>EUR:</b> {row['EUR']:,.0f}")
        if "IP90" in row and pd.notna(row["IP90"]):
            tip_parts.append(f"<b>IP90:</b> {row['IP90']:,.0f}")
        if "1YCuml" in row and pd.notna(row.get("1YCuml")):
            tip_parts.append(f"<b>1Y Cuml:</b> {row['1YCuml']:,.0f}")
        if "Wcut" in row and pd.notna(row.get("Wcut")):
            tip_parts.append(f"<b>Wcut:</b> {row['Wcut']:.1f}%")
        if "Section" in row and pd.notna(row.get("Section")):
            tip_parts.append(f"<b>Section:</b> {row['Section']}")
        tip_text = "<br>".join(tip_parts)

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.8,
            weight=1,
            interactive=True,
            tooltip=folium.Tooltip(
                tip_text,
                sticky=True,
                style=("font-size: 11px; padding: 3px 6px; "
                       "background-color: rgba(255,255,255,0.92); "
                       "border: 1px solid #c00; border-radius: 3px;"),
            ),
        ).add_to(well_point_fg)

    well_point_fg.add_to(m)

    # --------------------------------------------------
    # 5️⃣  PROSPECTS — drawn last so they render on top
    # --------------------------------------------------

    # Tooltip fields for prospect lines
    prospect_tooltip_fields = [
        "Label", "_prospect_type", "Analog_Count",
        "EUR", "IP90", "1YCuml", "Wcut",
        "OOIP", "RFTD", "URF",
    ]
    prospect_tooltip_aliases = [
        "Prospect:", "Type:", "Analog Count:",
        "Avg EUR:", "Avg IP90:", "Avg 1Y Cuml:", "Avg Wcut:",
        "OOIP:", "RFTD:", "URF:",
    ]

    # Conditionally add HighGradeScore to tooltip when it is computed
    if selected_metric == "High-Grade Score" and total_weight == 100:
        prospect_tooltip_fields.append("HighGradeScore")
        prospect_tooltip_aliases.append("HG Score:")

    def prospect_style(feature):
        """Style each prospect line based on its filter status."""
        props = feature["properties"]
        if props.get("_no_analogs", False):
            return NO_ANALOG_STYLE
        if not props.get("_passes_filter", True):
            return FAILING_STYLE
        return PASSING_STYLE

    prospect_fg = folium.FeatureGroup(name="Prospects")
    folium.GeoJson(
        p_display.to_json(),
        style_function=prospect_style,
        highlight_function=lambda _: {"weight": 5, "color": "yellow"},
        tooltip=folium.GeoJsonTooltip(
            fields=prospect_tooltip_fields,
            aliases=prospect_tooltip_aliases,
            localize=True,
            sticky=True,
            style=("font-size: 12px; padding: 5px 10px; "
                   "background-color: rgba(255,255,255,0.95); "
                   "border: 1px solid #2196F3; border-radius: 4px; "
                   "box-shadow: 2px 2px 6px rgba(0,0,0,0.15);"),
        ),
    ).add_to(prospect_fg)
    prospect_fg.add_to(m)

    # --------------------------------------------------
    # 5b) TOP-N HIGHLIGHT layer — gold outlines on best prospects
    # --------------------------------------------------
    if top_n_labels:
        top_prospects = p_display[p_display["Label"].isin(top_n_labels)]
        if not top_prospects.empty:
            top_fg = folium.FeatureGroup(name="⭐ Top Prospects")
            folium.GeoJson(
                top_prospects.to_json(),
                style_function=lambda _: {
                    "color": "gold",
                    "weight": 5,
                    "opacity": 0.9,
                    "dashArray": "",
                },
                highlight_function=lambda _: {
                    "weight": 7,
                    "color": "#FFD700",
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["Label"],
                    aliases=["⭐ Top Prospect:"],
                    localize=True,
                    sticky=True,
                    style=("font-size: 13px; font-weight: bold; padding: 5px 10px; "
                           "background-color: rgba(255,215,0,0.2); "
                           "border: 2px solid gold; border-radius: 4px;"),
                ),
            ).add_to(top_fg)
            top_fg.add_to(m)

    # --------------------------------------------------
    # 6️⃣  LAYER CONTROL & render
    # --------------------------------------------------
    folium.LayerControl(collapsed=True).add_to(m)

    st_folium(
        m,
        use_container_width=True,
        height=900,
        returned_objects=[],
    )


# ----------------------------------------------------------
# RIGHT COLUMN — Ranking Table + Detail Panel
# ----------------------------------------------------------
with col_rank:
    st.header("📊 Prospect Ranking")

    if selected_metric == "High-Grade Score" and total_weight != 100:
        st.warning("Adjust weights to total 100 % to see rankings.")
    else:
        # Determine the column to sort by
        metric_col = (
            "HighGradeScore"
            if selected_metric == "High-Grade Score"
            else selected_metric
        )
        ascending = selected_metric in ["Wcut", "URF", "RFTD"]

        rank_df = p[p["_passes_filter"]].copy()

        # Columns to show in the table
        display_cols = [
            "Label", "_prospect_type", "Analog_Count",
            "EUR", "IP90", "1YCuml", "Wcut",
            "OOIP", "RFTD", "URF",
        ]
        # Add EUR uncertainty columns (new feature)
        for extra in ["EUR_median", "EUR_p10", "EUR_p90"]:
            if extra in rank_df.columns:
                display_cols.append(extra)

        if selected_metric == "High-Grade Score":
            display_cols.append("HighGradeScore")

        # Make sure metric_col is in the list
        if metric_col not in display_cols:
            display_cols.append(metric_col)

        rank_df = rank_df[display_cols].copy()
        rank_df = rank_df.dropna(subset=[metric_col])

        if rank_df.empty:
            st.warning(f"No valid data for **{selected_metric}**.")
        else:
            # Compute percentile rank
            rank_df["Percentile"] = (
                rank_df[metric_col].rank(pct=True, ascending=(not ascending)) * 100
            )
            rank_df = rank_df.sort_values(
                metric_col, ascending=ascending
            ).reset_index(drop=True)
            rank_df.index = rank_df.index + 1
            rank_df.index.name = "Rank"

            # Rename columns for cleaner display
            rank_df = rank_df.rename(columns={
                "_prospect_type": "Type",
                "Analog_Count": "Analogs",
                "EUR_median": "EUR Med",
                "EUR_p10": "EUR P10",
                "EUR_p90": "EUR P90",
            })

            # ---- Summary statistics (new feature) ----
            st.caption(
                f"Ranked by **{selected_metric}** · {len(rank_df)} prospects"
            )

            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.metric(
                    "Best",
                    f"{rank_df[metric_col].iloc[0]:,.2f}"
                    if "Score" in selected_metric
                    else f"{rank_df[metric_col].iloc[0]:,.0f}",
                    delta=None,
                )
            with summary_cols[1]:
                median_val = rank_df[metric_col].median()
                st.metric(
                    "Median",
                    f"{median_val:,.2f}"
                    if "Score" in selected_metric
                    else f"{median_val:,.0f}",
                )
            with summary_cols[2]:
                st.metric(
                    "Worst",
                    f"{rank_df[metric_col].iloc[-1]:,.2f}"
                    if "Score" in selected_metric
                    else f"{rank_df[metric_col].iloc[-1]:,.0f}",
                )

            # ---- Numeric formatting dictionary ----
            fmt = {
                "EUR": "{:,.0f}",
                "IP90": "{:,.0f}",
                "1YCuml": "{:,.0f}",
                "Wcut": "{:.1f}",
                "OOIP": "{:,.0f}",
                "RFTD": "{:.3f}",
                "URF": "{:.3f}",
                "Percentile": "{:.0f}%",
                "EUR Med": "{:,.0f}",
                "EUR P10": "{:,.0f}",
                "EUR P90": "{:,.0f}",
            }
            if "HighGradeScore" in rank_df.columns:
                fmt["HighGradeScore"] = "{:.2f}"

            # Determine colour-map direction for background gradient
            if not ascending:
                gmap_vals = rank_df[metric_col]
            else:
                gmap_vals = -rank_df[metric_col]

            st.dataframe(
                rank_df.style.background_gradient(
                    subset=[metric_col],
                    cmap="YlGn",
                    gmap=gmap_vals,
                ).background_gradient(
                    subset=["Percentile"],
                    cmap="RdYlGn",
                ).format(fmt),
                use_container_width=True,
                height=600,
            )

            # ---- CSV download ----
            csv = rank_df.to_csv().encode("utf-8")
            st.download_button(
                "⬇️ Download Rankings (CSV)",
                data=csv,
                file_name="bakken_prospect_rankings.csv",
                mime="text/csv",
            )

            # --------------------------------------------------
            # NEW FEATURE: Prospect Detail / Radar Chart
            # --------------------------------------------------
            st.markdown("---")
            st.subheader("🔬 Prospect Detail")

            detail_label = st.selectbox(
                "Select a prospect to inspect",
                options=rank_df["Label"].tolist(),
            )

            if detail_label:
                detail_row = rank_df[rank_df["Label"] == detail_label].iloc[0]

                # Show key metrics in columns
                dc1, dc2, dc3 = st.columns(3)
                dc1.metric("EUR", f"{detail_row['EUR']:,.0f}" if pd.notna(detail_row.get("EUR")) else "—")
                dc2.metric("IP90", f"{detail_row['IP90']:,.0f}" if pd.notna(detail_row.get("IP90")) else "—")
                dc3.metric("Wcut", f"{detail_row['Wcut']:.1f}%" if pd.notna(detail_row.get("Wcut")) else "—")

                dc4, dc5, dc6 = st.columns(3)
                dc4.metric("OOIP", f"{detail_row['OOIP']:,.0f}" if pd.notna(detail_row.get("OOIP")) else "—")
                dc5.metric("URF", f"{detail_row['URF']:.3f}" if pd.notna(detail_row.get("URF")) else "—")
                dc6.metric("RFTD", f"{detail_row['RFTD']:.3f}" if pd.notna(detail_row.get("RFTD")) else "—")

                # EUR uncertainty range
                if pd.notna(detail_row.get("EUR P10")) and pd.notna(detail_row.get("EUR P90")):
                    st.caption(
                        f"EUR range: P90 = {detail_row['EUR P90']:,.0f} · "
                        f"Median = {detail_row.get('EUR Med', 0):,.0f} · "
                        f"P10 = {detail_row['EUR P10']:,.0f}"
                    )

                # Radar chart
                all_passing_for_radar = rank_df.copy()
                radar_fig = build_radar_chart(detail_row, all_passing_for_radar)
                st.plotly_chart(radar_fig, use_container_width=True)

    # --------------------------------------------------
    # Flagged prospects: no analogs
    # --------------------------------------------------
    no_analog_prospects = p[p["_no_analogs"]].copy()
    if not no_analog_prospects.empty:
        st.markdown("---")
        st.subheader("⚠️ No Analogs Found")
        st.caption(
            "These prospects have no analog wells within the buffer zone. "
            "Consider widening the well source or reviewing the prospect geometry."
        )
        st.dataframe(
            no_analog_prospects[["Label", "_prospect_type"]]
            .rename(columns={"_prospect_type": "Type"})
            .reset_index(drop=True),
            use_container_width=True,
        )
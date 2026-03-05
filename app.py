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
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ==========================================================
# Page configuration
# ==========================================================
st.set_page_config(layout="wide", page_title="Bakken Inventory Optimizer", page_icon="🛢️")

# ==========================================================
# Constants
# ==========================================================
NULL_STYLE = {"fillColor": "#ffffff", "fillOpacity": 0, "color": "#888", "weight": 0.25}
DEFAULT_BUFFER_M = 800
COLOR_MAP_CLASS = {
    "Above Trend": "#2ca02c",
    "On Trend": "#1f77b4",
    "Below Trend": "#d62728",
}

# Hard-coded column names
WELL_COLS = ["Norm EUR", "Norm 1Y Cuml", "Norm IP90"]
SECTION_OOIP_COL = "SectionOOIP"

# ==========================================================
# Helpers
# ==========================================================


def safe_range(series):
    vals = series.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        return (0.0, 1.0) if lo == 0.0 else (lo - abs(lo) * 0.1, lo + abs(lo) * 0.1)
    return lo, hi


def zscore_full(series, full_series):
    vals = full_series.replace([np.inf, -np.inf], np.nan)
    mu, sigma = vals.mean(), vals.std()
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(0.0, index=series.index)
    return (series.replace([np.inf, -np.inf], np.nan) - mu) / sigma


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
    return geom.centroid


def endpoint_of_geom(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return Point(list(geom.coords)[-1])
    elif geom.geom_type == "MultiLineString":
        return Point(list(geom.geoms[-1].coords)[-1])
    elif geom.geom_type == "Point":
        return geom
    return None


def get_ylgn_hex(value, vmin, vmax):
    if pd.isna(value) or vmin == vmax:
        return "#cccccc"
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = mpl_cm.get_cmap("YlGn")(norm(value))
    return mcolors.to_hex(rgba)


def fit_trend(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    model = LinearRegression().fit(x[mask].values.reshape(-1, 1), y[mask].values)
    return model


def classify_label(z, threshold):
    if z > threshold:
        return "Above Trend"
    elif z < -threshold:
        return "Below Trend"
    return "On Trend"


def prospect_coords_latlon(geom, transformer):
    """Return (endpoint_lat, endpoint_lon) for a prospect geometry in 4326."""
    ep = endpoint_of_geom(geom)
    if ep is None:
        return np.nan, np.nan
    lon, lat = transformer.transform(ep.x, ep.y)
    return round(lat, 6), round(lon, 6)


# ==========================================================
# Load data
# ==========================================================

@st.cache_resource(show_spinner="Loading spatial data …")
def load_data():
    lines = gpd.read_file("lines.shp")
    points = gpd.read_file("points.shp")
    grid = gpd.read_file("ooipsectiongrid.shp")
    infills = gpd.read_file("Infills_plyln.shp")
    merged = gpd.read_file("merged_inventory.shp")
    lease_lines = gpd.read_file("LL_plyln.shp")
    units = gpd.read_file("Bakken Units.shp")
    land = gpd.read_file("Bakken Land.shp")

    # Sheet 0: must have UWI, Section, Norm EUR, Norm 1Y Cuml, Norm IP90, WF
    well_df = pd.read_excel("wells.xlsx", sheet_name=0)
    # Sheet 1: must have Section, SectionOOIP
    section_df = pd.read_excel("wells.xlsx", sheet_name=1)

    for gdf in [lines, points, grid, units, infills, lease_lines, merged, land]:
        if gdf.crs is None:
            gdf.set_crs(epsg=26913, inplace=True)
        gdf.to_crs(epsg=26913, inplace=True)

    grid["Section"] = grid["Section"].astype(str).str.strip()
    grid["geometry"] = grid.geometry.simplify(50, preserve_topology=True)

    # Clean well data
    well_df["UWI"] = well_df["UWI"].astype(str).str.strip()
    well_df["Section"] = well_df["Section"].astype(str).str.strip()
    for col in WELL_COLS:
        well_df[col] = pd.to_numeric(well_df[col], errors="coerce")

    # Ensure WF column exists and is numeric
    if "WF" in well_df.columns:
        well_df["WF"] = pd.to_numeric(well_df["WF"], errors="coerce")
    else:
        well_df["WF"] = np.nan

    # Clean section data
    section_df["Section"] = section_df["Section"].astype(str).str.strip()
    section_df[SECTION_OOIP_COL] = pd.to_numeric(section_df[SECTION_OOIP_COL], errors="coerce")

    sec_numeric_cols = [
        c for c in section_df.columns
        if c != "Section" and pd.api.types.is_numeric_dtype(section_df[c])
    ]

    lines["UWI"] = lines["UWI"].astype(str).str.strip()
    points["UWI"] = points["UWI"].astype(str).str.strip()

    # Merge SectionOOIP onto well_df via Section
    well_df = well_df.merge(
        section_df[["Section", SECTION_OOIP_COL]], on="Section", how="left"
    )

    return (lines, points, grid, units, infills, lease_lines, merged, land,
            well_df, section_df, sec_numeric_cols)


(lines_gdf, points_gdf, grid_gdf, units_gdf, infills_gdf, lease_lines_gdf,
 merged_gdf, land_gdf, well_df, section_df, SEC_NUMERIC_COLS) = load_data()

# All metric columns available on prospects (IDW cols + WF + section OOIP)
ALL_METRIC_COLS = WELL_COLS + ["WF", SECTION_OOIP_COL]

# ==========================================================
# Derived spatial data
# ==========================================================
section_enriched = grid_gdf.merge(section_df, on="Section", how="left")

# Combine lines and points into one existing-wells layer
lines_with_uwi = lines_gdf[["UWI", "geometry"]].copy()
points_only = points_gdf[~points_gdf["UWI"].isin(lines_with_uwi["UWI"])][["UWI", "geometry"]].copy()
existing_wells = gpd.GeoDataFrame(
    pd.concat([lines_with_uwi, points_only], ignore_index=True),
    geometry="geometry", crs=lines_gdf.crs,
)

# Merge well data onto spatial wells
proximal_wells = gpd.GeoDataFrame(
    existing_wells.merge(well_df, on="UWI", how="inner"),
    geometry="geometry", crs=existing_wells.crs,
)
proximal_wells["_midpoint"] = proximal_wells.geometry.apply(midpoint_of_geom)

# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.title("Map Settings")

st.sidebar.markdown("---")
st.sidebar.subheader("📏 Buffer Distance")
buffer_distance = st.sidebar.slider("Buffer radius (m)", 100, 2000, DEFAULT_BUFFER_M, step=50)

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Section Grid Gradient")
section_gradient = st.sidebar.selectbox("Colour sections by", ["None"] + SEC_NUMERIC_COLS)

show_infills = st.sidebar.checkbox("Show Infills", value=True)
show_lease_lines = st.sidebar.checkbox("Show Lease Lines", value=True)
show_merged = st.sidebar.checkbox("Show Merged", value=True)

# ==========================================================
# Build prospect set
# ==========================================================
prospect_frames = []
if show_infills:
    f = infills_gdf.copy(); f["_prospect_type"] = "Infill"; prospect_frames.append(f)
if show_lease_lines:
    f = lease_lines_gdf.copy(); f["_prospect_type"] = "Lease Line"; prospect_frames.append(f)
if show_merged:
    f = merged_gdf.copy(); f["_prospect_type"] = "Merged"; prospect_frames.append(f)

if not prospect_frames:
    st.error("Enable at least one prospect layer.")
    st.stop()

prospects = gpd.GeoDataFrame(
    pd.concat(prospect_frames, ignore_index=True),
    geometry="geometry", crs=infills_gdf.crs,
)

# ==========================================================
# Analyse prospects — IDW² for well-level cols, geometry-intersect for WF,
#                      mean for SectionOOIP
# ==========================================================

def analyze_prospects(pros, prox, sections, buffer_m):
    pros = pros.copy()
    pros["_midpoint"] = pros.geometry.apply(midpoint_of_geom)
    pros["_buffer"] = pros.geometry.buffer(buffer_m, cap_style=2)

    # Section label from endpoint
    endpoints = gpd.GeoDataFrame(
        {"_pidx": pros.index, "geometry": pros.geometry.apply(endpoint_of_geom)},
        crs=pros.crs,
    )
    endpoints = endpoints[endpoints.geometry.notna()]
    ep_join = gpd.sjoin(endpoints, sections[["Section", "geometry"]], how="left", predicate="within")
    ep_join = ep_join.drop_duplicates(subset="_pidx", keep="first")
    label_map = ep_join.set_index("_pidx")["Section"].fillna("Unknown").astype(str)
    pros["_section_label"] = pros.index.map(label_map).fillna("Unknown")

    # De-duplicate labels
    counts = pros["_section_label"].value_counts()
    for lab in counts[counts > 1].index:
        idxs = pros[pros["_section_label"] == lab].index
        for i, ix in enumerate(idxs, 1):
            pros.at[ix, "_section_label"] = f"{lab}-{i}"

    buffer_gdf = gpd.GeoDataFrame(
        {"_pidx": pros.index, "geometry": pros["_buffer"]}, crs=pros.crs,
    )

    # --- IDW² for well-level metrics (midpoint-based) ---
    midpt_gdf = prox[prox["_midpoint"].notna()].copy()
    midpt_gdf = midpt_gdf.set_geometry(
        gpd.GeoSeries(midpt_gdf["_midpoint"], crs=prox.crs)
    )
    well_hits = gpd.sjoin(midpt_gdf, buffer_gdf, how="inner", predicate="within")

    px_pts = well_hits["index_right"].map(lambda i: pros.at[i, "_midpoint"])
    well_hits["_dist"] = np.sqrt(
        (well_hits["_midpoint"].apply(lambda pt: pt.x) - px_pts.apply(lambda pt: pt.x if pt else np.nan)) ** 2 +
        (well_hits["_midpoint"].apply(lambda pt: pt.y) - px_pts.apply(lambda pt: pt.y if pt else np.nan)) ** 2
    ).replace(0, 1.0)
    well_hits["_w"] = 1.0 / (well_hits["_dist"] ** 2)

    idw_results = {}
    for col in WELL_COLS:
        valid = well_hits[well_hits[col].notna() & well_hits["_w"].notna()].copy()
        if valid.empty:
            idw_results[col] = pd.Series(np.nan, index=pros.index)
            continue
        valid["_wv"] = valid[col] * valid["_w"]
        g = valid.groupby("index_right").agg(_wv_sum=("_wv", "sum"), _w_sum=("_w", "sum"))
        idw_results[col] = (g["_wv_sum"] / g["_w_sum"]).reindex(pros.index)

    proximal_count = well_hits.groupby("index_right").size().reindex(pros.index, fill_value=0)
    proximal_uwis = (
        well_hits.groupby("index_right")["UWI"]
        .apply(lambda x: ", ".join(x.astype(str)))
        .reindex(pros.index, fill_value="")
    )

    # --- WF: geometry-intersect based (any part of well in buffer) ---
    # Use the original well geometries (lines/points), not midpoints
    wf_wells = prox[["UWI", "WF", "geometry"]].copy()
    wf_wells = wf_wells[wf_wells["WF"].notna()].copy()

    if not wf_wells.empty:
        # Spatial join using actual well geometry intersecting the buffer
        wf_hits = gpd.sjoin(wf_wells, buffer_gdf, how="inner", predicate="intersects")
        # For each prospect buffer, compute IDW² of WF using midpoints of
        # the intersecting wells for distance calculation
        wf_hits["_well_midpoint"] = wf_hits.geometry.apply(midpoint_of_geom)
        wf_px_pts = wf_hits["index_right"].map(lambda i: pros.at[i, "_midpoint"])
        wf_hits["_dist"] = np.sqrt(
            (wf_hits["_well_midpoint"].apply(lambda pt: pt.x if pt else np.nan) -
             wf_px_pts.apply(lambda pt: pt.x if pt else np.nan)) ** 2 +
            (wf_hits["_well_midpoint"].apply(lambda pt: pt.y if pt else np.nan) -
             wf_px_pts.apply(lambda pt: pt.y if pt else np.nan)) ** 2
        ).replace(0, 1.0)
        wf_hits["_w"] = 1.0 / (wf_hits["_dist"] ** 2)
        wf_valid = wf_hits[wf_hits["WF"].notna() & wf_hits["_w"].notna()].copy()
        if not wf_valid.empty:
            wf_valid["_wv"] = wf_valid["WF"] * wf_valid["_w"]
            wf_g = wf_valid.groupby("index_right").agg(
                _wv_sum=("_wv", "sum"), _w_sum=("_w", "sum")
            )
            wf_idw = (wf_g["_wv_sum"] / wf_g["_w_sum"]).reindex(pros.index)
        else:
            wf_idw = pd.Series(np.nan, index=pros.index)
    else:
        wf_idw = pd.Series(np.nan, index=pros.index)

    # --- Mean SectionOOIP from intersecting grid cells ---
    sec_join = gpd.sjoin(
        sections[["geometry", SECTION_OOIP_COL]], buffer_gdf,
        how="inner", predicate="intersects",
    )
    ooip_mean = sec_join.groupby("index_right")[SECTION_OOIP_COL].mean().reindex(pros.index)

    # Assemble
    out = pd.DataFrame(index=pros.index)
    out["_prospect_type"] = pros["_prospect_type"].values
    out["_section_label"] = pros["_section_label"].values
    out["Proximal_Count"] = proximal_count.values
    out["_proximal_uwis"] = proximal_uwis.values
    for col in WELL_COLS:
        out[col] = idw_results[col].values
    out["WF"] = wf_idw.values
    out[SECTION_OOIP_COL] = ooip_mean.values

    return out


prospect_metrics = analyze_prospects(
    prospects, proximal_wells, section_enriched, buffer_distance,
)

for c in prospect_metrics.columns:
    prospects[c] = prospect_metrics[c].values
prospects["Label"] = prospects["_section_label"]

for col in ALL_METRIC_COLS:
    if col in prospects.columns:
        prospects[col] = prospects[col].replace([np.inf, -np.inf], np.nan)

# ==========================================================
# Compute prospect coordinates (lat/lon of midpoint)
# ==========================================================
_transformer_coord = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)
_coords = prospects.geometry.apply(lambda g: prospect_coords_latlon(g, _transformer_coord))
prospects["Latitude"] = _coords.apply(lambda x: x[0])
prospects["Longitude"] = _coords.apply(lambda x: x[1])

# ==========================================================
# Classification
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Classification Settings")

st.sidebar.markdown("**Classification weights (must sum to 100):**")
cw_eur = st.sidebar.number_input("EUR weight %", 0, 100, 50, key="cw_eur")
cw_1y = st.sidebar.number_input("1Y weight %", 0, 100, 25, key="cw_1y")
cw_ip90 = st.sidebar.number_input("IP90 weight %", 0, 100, 25, key="cw_ip90")
cw_sum = cw_eur + cw_1y + cw_ip90

classification_ready = False

if cw_sum != 100:
    st.sidebar.error(f"Weights sum to {cw_sum}%, must be 100%")
else:
    st.sidebar.success(f"Weights: EUR {cw_eur}%, 1Y {cw_1y}%, IP90 {cw_ip90}%")
    cls_threshold = st.sidebar.slider("Composite Z threshold (σ)", 0.1, 2.0, 0.5, 0.05, key="cls_thresh")

    # Build field data: each UWI's metric / SectionOOIP
    field = well_df.dropna(subset=[SECTION_OOIP_COL, "Norm EUR", "Norm 1Y Cuml", "Norm IP90"]).copy()
    field = field[field[SECTION_OOIP_COL] > 0].copy()

    if len(field) >= 2:
        field["EUR_ratio"] = field["Norm EUR"] / field[SECTION_OOIP_COL]
        field["Y1_ratio"] = field["Norm 1Y Cuml"] / field[SECTION_OOIP_COL]
        field["IP90_ratio"] = field["Norm IP90"] / field[SECTION_OOIP_COL]

        eur_model = fit_trend(field[SECTION_OOIP_COL], field["Norm EUR"])
        ip90_model = fit_trend(field[SECTION_OOIP_COL], field["Norm IP90"])
        y1_model = fit_trend(field[SECTION_OOIP_COL], field["Norm 1Y Cuml"])

        if all(m is not None for m in [eur_model, ip90_model, y1_model]):
            field["EUR_resid"] = field["Norm EUR"] - eur_model.predict(field[SECTION_OOIP_COL].values.reshape(-1, 1))
            field["IP90_resid"] = field["Norm IP90"] - ip90_model.predict(field[SECTION_OOIP_COL].values.reshape(-1, 1))
            field["Y1_resid"] = field["Norm 1Y Cuml"] - y1_model.predict(field[SECTION_OOIP_COL].values.reshape(-1, 1))
            eur_std = field["EUR_resid"].std()
            ip90_std = field["IP90_resid"].std()
            y1_std = field["Y1_resid"].std()

            # Classify prospects
            pros_cls = prospects.dropna(
                subset=[SECTION_OOIP_COL, "Norm EUR", "Norm 1Y Cuml", "Norm IP90"]
            ).copy()
            pros_cls = pros_cls[pros_cls[SECTION_OOIP_COL] > 0].copy()

            if not pros_cls.empty:
                ooip_vals = pros_cls[SECTION_OOIP_COL].values.reshape(-1, 1)
                pros_cls["EUR_pred"] = eur_model.predict(ooip_vals)
                pros_cls["IP90_pred"] = ip90_model.predict(ooip_vals)
                pros_cls["Y1_pred"] = y1_model.predict(ooip_vals)

                pros_cls["Z_EUR"] = (pros_cls["Norm EUR"] - pros_cls["EUR_pred"]) / eur_std if eur_std > 0 else 0
                pros_cls["Z_IP90"] = (pros_cls["Norm IP90"] - pros_cls["IP90_pred"]) / ip90_std if ip90_std > 0 else 0
                pros_cls["Z_1Y"] = (pros_cls["Norm 1Y Cuml"] - pros_cls["Y1_pred"]) / y1_std if y1_std > 0 else 0

                pros_cls["Composite_Z"] = (
                    (cw_eur / 100) * pros_cls["Z_EUR"] +
                    (cw_1y / 100) * pros_cls["Z_1Y"] +
                    (cw_ip90 / 100) * pros_cls["Z_IP90"]
                )
                pros_cls["Classification"] = pros_cls["Composite_Z"].apply(
                    lambda z: classify_label(z, cls_threshold)
                )

                for col in ["Classification", "Composite_Z", "Z_EUR", "Z_IP90", "Z_1Y"]:
                    prospects[col] = np.nan
                    prospects.loc[pros_cls.index, col] = pros_cls[col]

                classification_ready = True
            else:
                st.sidebar.warning("No prospects have valid data for classification.")
        else:
            st.sidebar.warning("Could not fit trend lines (insufficient data).")
    else:
        st.sidebar.warning(f"Only {len(field)} UWIs with complete data — need ≥ 2.")

# ==========================================================
# Sidebar — Filters
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Prospect Filters")

p = prospects.copy()
has_proximal = p["Proximal_Count"] > 0
filter_mask = has_proximal.copy()

for col in ALL_METRIC_COLS:
    if col not in p.columns:
        continue
    lo, hi = safe_range(p[col])
    if lo == hi:
        continue
    f_lo, f_hi = st.sidebar.slider(col, lo, hi, (lo, hi), key=f"filter_{col}")
    filter_mask = filter_mask & (((p[col] >= f_lo) & (p[col] <= f_hi)) | p[col].isna())

p["_passes_filter"] = filter_mask
p["_no_proximal"] = ~has_proximal

n_total = len(p)
n_passing = int(filter_mask.sum())
n_no_proximal = int((~has_proximal).sum())

st.sidebar.markdown(
    f"**{n_passing}** / {n_total} prospects pass filters "
    f"({n_passing / max(n_total, 1) * 100:.0f}%)"
)
if n_no_proximal:
    st.sidebar.warning(f"⚠️ {n_no_proximal} prospects have no nearby proximal wells")

# ==========================================================
# Sidebar — Ranking
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Ranking Metric")

available_for_ranking = [c for c in ALL_METRIC_COLS if c in p.columns]
ranking_options = available_for_ranking + ["High-Grade Score"]
selected_metric = st.sidebar.selectbox("Rank prospects by", ranking_options)

# High-Grade Score config
if selected_metric == "High-Grade Score":
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ High-Grade Score")
    st.sidebar.markdown("**Select metrics to include:**")
    hg_selected, hg_lower_better, hg_weights = {}, {}, {}
    for col in available_for_ranking:
        if st.sidebar.checkbox(col, value=False, key=f"hg_chk_{col}"):
            hg_selected[col] = True
            c1, c2 = st.sidebar.columns(2)
            hg_lower_better[col] = c1.checkbox("Lower=better", value=False, key=f"hg_lb_{col}")
            hg_weights[col] = c2.number_input("Weight %", 0, 100, 0, key=f"hg_w_{col}")
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
    hg_selected, hg_lower_better, hg_weights, total_weight = {}, {}, {}, 0

# Compute High-Grade Score
if selected_metric == "High-Grade Score" and total_weight == 100 and hg_selected:
    passing = p[p["_passes_filter"]].copy()
    hgs = pd.Series(0.0, index=passing.index)
    for col in hg_selected:
        z = zscore_full(passing[col], p[col])
        if hg_lower_better.get(col, False):
            z = -z
        hgs += (hg_weights[col] / 100.0) * z
    p["HighGradeScore"] = np.nan
    p.loc[passing.index, "HighGradeScore"] = hgs
else:
    p["HighGradeScore"] = np.nan

if selected_metric == "High-Grade Score":
    metric_col = "HighGradeScore"
    ascending = False
else:
    metric_col = selected_metric
    ascending = st.sidebar.checkbox(f"Lower {selected_metric} = better?", value=False, key="rank_asc")

# ==========================================================
# Buffer colours for map
# ==========================================================
passing_vals = p[p["_passes_filter"]][metric_col].dropna()
if not passing_vals.empty:
    if ascending:
        gmap_vmin, gmap_vmax = float(-passing_vals.max()), float(-passing_vals.min())
    else:
        gmap_vmin, gmap_vmax = float(passing_vals.min()), float(passing_vals.max())
else:
    gmap_vmin, gmap_vmax = 0.0, 1.0

label_color_map = {}
for idx_val in p[p["_passes_filter"]].index:
    row = p.loc[idx_val]
    val = row.get(metric_col, np.nan)
    if pd.notna(val):
        label_color_map[row["Label"]] = get_ylgn_hex(-val if ascending else val, gmap_vmin, gmap_vmax)
    else:
        label_color_map[row["Label"]] = "#cccccc"

p["_buffer_color"] = p["Label"].map(label_color_map).fillna("#cccccc")

# ==========================================================
# Display-ready data (4326)
# ==========================================================
transformer_to_4326 = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)

section_display = section_enriched.to_crs(4326)
units_display = units_gdf.to_crs(4326)
land_display = land_gdf.to_crs(4326)

# Existing wells display — all available columns
existing_display = proximal_wells.copy().to_crs(4326)

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

# Prospect lines for display — everything available
skip_cols = {"geometry", "_buffer", "_midpoint", "_passes_filter", "_no_proximal",
             "_buffer_color", "_section_label", "_proximal_uwis"}
keep_cols = [c for c in p.columns if c not in skip_cols]
p_lines = p[keep_cols + ["geometry"]].copy()
for c in p_lines.columns:
    if c != "geometry" and p_lines[c].dtype == object:
        p_lines[c] = p_lines[c].astype(str)
p_lines_display = p_lines.to_crs(4326)

# ==========================================================
# Build ranking table
# ==========================================================
rank_df = None
if not (selected_metric == "High-Grade Score" and total_weight != 100):
    display_cols = ["Label", "_prospect_type", "Latitude", "Longitude", "Proximal_Count"]
    display_cols += [c for c in ALL_METRIC_COLS if c in p.columns]
    if selected_metric == "High-Grade Score" and "HighGradeScore" in p.columns:
        display_cols.append("HighGradeScore")
    if metric_col not in display_cols:
        display_cols.append(metric_col)
    if "Classification" in p.columns:
        display_cols.append("Classification")
    display_cols = list(dict.fromkeys(display_cols))
    display_cols = [c for c in display_cols if c in p.columns]

    rdf = p[p["_passes_filter"]][display_cols].dropna(subset=[metric_col]).copy()
    if not rdf.empty:
        rdf["Percentile"] = rdf[metric_col].rank(pct=True, ascending=(not ascending)) * 100
        rdf = rdf.sort_values(metric_col, ascending=ascending).reset_index(drop=True)
        rdf.index = rdf.index + 1
        rdf.index.name = "Rank"
        rank_df = rdf.rename(columns={"_prospect_type": "Type", "Proximal_Count": "Proximal"})

# ==========================================================
# Executive summary
# ==========================================================
st.title("Bakken Inventory Optimizer")

if n_passing > 0:
    best_pool = p[p["_passes_filter"]].dropna(subset=[metric_col])
    if not best_pool.empty:
        best_row = best_pool.sort_values(metric_col, ascending=ascending).iloc[0]
        avg_prox = p[p["_passes_filter"]]["Proximal_Count"].mean()
        st.success(
            f"**{n_passing}** of {n_total} prospects pass filters. "
            f"Top prospect by **{selected_metric}**: **{best_row['Label']}** "
            f"({metric_col} = {best_row[metric_col]:,.2f}). "
            f"Avg proximal wells/prospect: **{avg_prox:.1f}**."
        )
    else:
        st.info(f"**{n_passing}** prospects pass filters but none have valid {selected_metric} data.")
else:
    st.warning("No prospects pass the current filters. Try relaxing your criteria.")

# ==========================================================
# MAP
# ==========================================================
bounds = p.total_bounds
cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
clon, clat = transformer_to_4326.transform(cx, cy)

m = folium.Map(location=[clat, clon], zoom_start=11, tiles="CartoDB positron")
MiniMap(toggle_display=True, position="bottomleft").add_to(m)

# Land
folium.FeatureGroup(name="Bakken Land", show=True).add_child(
    folium.GeoJson(land_display.to_json(), style_function=lambda _: {
        "fillColor": "#fff9c4", "color": "#fff9c4", "weight": 0.5, "fillOpacity": 0.2,
    })
).add_to(m)

# Section grid
if section_gradient != "None" and section_gradient in section_display.columns:
    grad_vals = section_display[section_gradient].dropna()
    if not grad_vals.empty:
        colormap = cm.LinearColormap(
            ["#f7fcf5", "#74c476", "#00441b"],
            vmin=float(grad_vals.min()), vmax=float(grad_vals.max()),
        ).to_step(n=7)
        colormap.caption = section_gradient
        m.add_child(colormap)
        sec_style = lambda feat, _col=section_gradient, _cm=colormap: (
            {"fillColor": _cm(feat["properties"].get(_col)), "fillOpacity": 0.45,
             "color": "white", "weight": 0.3}
            if feat["properties"].get(_col) is not None
            and not (isinstance(feat["properties"].get(_col), float)
                     and np.isnan(feat["properties"].get(_col)))
            else NULL_STYLE
        )
    else:
        sec_style = lambda _: NULL_STYLE
else:
    sec_style = lambda _: NULL_STYLE

# Section tooltip
sec_tip_fields = [c for c in section_display.columns if c != "geometry"]
section_fg = folium.FeatureGroup(name="Section Grid", show=(section_gradient != "None"))
folium.GeoJson(
    section_display.to_json(), style_function=sec_style,
    highlight_function=lambda _: {"weight": 2, "color": "black", "fillOpacity": 0.5},
    tooltip=folium.GeoJsonTooltip(
        fields=sec_tip_fields, aliases=[f"{f}:" for f in sec_tip_fields],
        localize=True, sticky=True,
        style="font-size:11px;padding:4px 8px;background:rgba(255,255,255,0.9);"
              "border:1px solid #333;border-radius:3px;",
    ),
).add_to(section_fg)
section_fg.add_to(m)

# Units
folium.FeatureGroup(name="Units", show=True).add_child(
    folium.GeoJson(units_display.to_json(), style_function=lambda _: {
        "color": "black", "weight": 2, "fillOpacity": 0, "interactive": False,
    })
).add_to(m)

# Buffers
buffer_fg = folium.FeatureGroup(name="Prospect Buffers")
for _, brow in buffer_gdf[buffer_gdf["_passes_filter"]].iterrows():
    fc = label_color_map.get(brow["Label"], "#cccccc")
    tip_parts = [f"<b>{brow['Label']}</b>", f"Proximal Wells: {brow.get('Proximal_Count', '—')}"]
    for col in ALL_METRIC_COLS:
        if col in brow.index and pd.notna(brow[col]):
            v = brow[col]
            tip_parts.append(f"{col}: {v:,.0f}" if abs(v) > 100 else f"{col}: {v:.3f}")
    if metric_col in brow.index and pd.notna(brow[metric_col]) and metric_col not in ALL_METRIC_COLS:
        v = brow[metric_col]
        tip_parts.append(f"{metric_col}: {v:.3f}")
    folium.GeoJson(
        brow.geometry.__geo_interface__,
        style_function=lambda _, _fc=fc: {
            "fillColor": _fc, "fillOpacity": 0.4, "color": _fc, "weight": 1, "opacity": 0.7
        },
        tooltip=folium.Tooltip(
            "<br>".join(tip_parts), sticky=True,
            style="font-size:11px;padding:3px 6px;background:rgba(255,255,255,0.92);"
                  "border:1px solid #333;border-radius:3px;"
        ),
    ).add_to(buffer_fg)

for _, brow in buffer_gdf[~buffer_gdf["_passes_filter"] & ~buffer_gdf["_no_proximal"]].iterrows():
    folium.GeoJson(
        brow.geometry.__geo_interface__,
        style_function=lambda _: {
            "fillColor": "#d3d3d3", "fillOpacity": 0.15, "color": "#aaa", "weight": 0.5, "opacity": 0.3
        },
    ).add_to(buffer_fg)

for _, brow in buffer_gdf[buffer_gdf["_no_proximal"]].iterrows():
    folium.GeoJson(
        brow.geometry.__geo_interface__,
        style_function=lambda _: {
            "fillColor": "#ffe0b2", "fillOpacity": 0.1, "color": "orange",
            "weight": 1, "dashArray": "5 5", "opacity": 0.4
        },
    ).add_to(buffer_fg)
buffer_fg.add_to(m)

# Existing wells
well_fg = folium.FeatureGroup(name="Existing Wells")
line_wells = existing_display[existing_display.geometry.type != "Point"]
point_wells = existing_display[existing_display.geometry.type == "Point"]

well_tip_fields = [c for c in existing_display.columns if c not in ("geometry", "_midpoint")]

if not line_wells.empty:
    folium.GeoJson(
        line_wells[[c for c in well_tip_fields] + ["geometry"]].to_json(),
        style_function=lambda _: {"color": "transparent", "weight": 15, "opacity": 0},
        highlight_function=lambda _: {"weight": 15, "color": "#555", "opacity": 0.3},
        tooltip=folium.GeoJsonTooltip(
            fields=well_tip_fields, aliases=[f"{f}:" for f in well_tip_fields],
            localize=True, sticky=True,
            style="font-size:11px;padding:3px 6px;background:rgba(255,255,255,0.92);"
                  "border:1px solid #333;border-radius:3px;",
        ),
    ).add_to(well_fg)

    line_wells_clean = line_wells.drop(columns=["_midpoint"], errors="ignore").copy()
    for c in line_wells_clean.columns:
        if c != "geometry" and line_wells_clean[c].dtype == object:
            line_wells_clean[c] = line_wells_clean[c].astype(str)

    folium.GeoJson(
        line_wells_clean.to_json(),
        style_function=lambda _: {"color": "black", "weight": 0.5, "opacity": 0.8},
    ).add_to(well_fg)

    for _, row in line_wells.iterrows():
        ep = endpoint_of_geom(row.geometry)
        if ep is not None:
            folium.CircleMarker(
                location=[ep.y, ep.x], radius=2,
                color="black", fill=True, fill_color="black", fill_opacity=0.9, weight=0.5,
            ).add_to(well_fg)

for _, row in point_wells.iterrows():
    tip_parts = []
    for col in well_tip_fields:
        if col not in row.index or pd.isna(row[col]):
            continue
        v = row[col]
        if isinstance(v, (int, float)):
            tip_parts.append(f"<b>{col}:</b> {v:,.0f}" if abs(v) > 100 else f"<b>{col}:</b> {v:.3f}")
        else:
            tip_parts.append(f"<b>{col}:</b> {v}")
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=2,
        color="black", fill=True, fill_color="black", fill_opacity=0.9, weight=1,
        tooltip=folium.Tooltip(
            "<br>".join(tip_parts), sticky=True,
            style="font-size:11px;padding:3px 6px;background:rgba(255,255,255,0.92);"
                  "border:1px solid #333;border-radius:3px;"
        ),
    ).add_to(well_fg)
well_fg.add_to(m)

# Prospect lines — colour by classification if available
prospect_fg = folium.FeatureGroup(name="Prospect Wells", show=True)

has_classification = classification_ready and "Classification" in p.columns

if has_classification:
    for _, row in p_lines_display.iterrows():
        cls_val = row.get("Classification", None)
        line_color = COLOR_MAP_CLASS.get(cls_val, "red") if pd.notna(cls_val) else "red"

        tip_parts = [
            f"<b>{c}:</b> {row[c]}"
            for c in p_lines_display.columns if c != "geometry"
        ]
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, _lc=line_color: {"color": _lc, "weight": 3, "opacity": 0.9},
            highlight_function=lambda _: {"weight": 5, "color": "#ff4444"},
            tooltip=folium.Tooltip("<br>".join(tip_parts), sticky=True, style="font-size:12px"),
        ).add_to(prospect_fg)

        ep = endpoint_of_geom(row.geometry)
        if ep is not None:
            folium.CircleMarker(
                location=[ep.y, ep.x], radius=3,
                color=line_color, fill=True, fill_color=line_color,
                fill_opacity=0.9, weight=1,
            ).add_to(prospect_fg)
else:
    pt_fields = [c for c in p_lines_display.columns if c != "geometry"]
    folium.GeoJson(
        p_lines_display.to_json(),
        style_function=lambda _: {"color": "red", "weight": 3, "opacity": 0.9},
        highlight_function=lambda _: {"weight": 5, "color": "#ff4444"},
        tooltip=folium.GeoJsonTooltip(
            fields=pt_fields, aliases=[f"{f}:" for f in pt_fields],
            localize=True, sticky=True, style="font-size:12px"
        ),
    ).add_to(prospect_fg)

    for _, row in p_lines_display.iterrows():
        ep = endpoint_of_geom(row.geometry)
        if ep is not None:
            folium.CircleMarker(
                location=[ep.y, ep.x], radius=3,
                color="red", fill=True, fill_color="red",
                fill_opacity=0.9, weight=1,
            ).add_to(prospect_fg)

prospect_fg.add_to(m)

folium.LayerControl(collapsed=True).add_to(m)
st_folium(m, use_container_width=True, height=900, returned_objects=[])

# ==========================================================
# CLASSIFICATION RESULTS
# ==========================================================
if classification_ready and "Classification" in p.columns:
    st.markdown("---")
    st.header("📐 Classification Results")

    pros_chart = p[p["_passes_filter"] & p["Classification"].notna()].copy()

    if not pros_chart.empty:
        col1, col2 = st.columns(2)

        x_range = np.linspace(field[SECTION_OOIP_COL].min(), field[SECTION_OOIP_COL].max(), 100)

        with col1:
            fig_eur = px.scatter(
                pros_chart, x=SECTION_OOIP_COL, y="Norm EUR",
                color="Classification", color_discrete_map=COLOR_MAP_CLASS,
                hover_data=["Label"], title="Norm EUR vs SectionOOIP",
            )
            fig_eur.add_trace(go.Scatter(
                x=field[SECTION_OOIP_COL], y=field["Norm EUR"],
                mode="markers", name="Field UWIs",
                marker=dict(color="lightgrey", size=4, opacity=0.5),
            ))
            fig_eur.add_trace(go.Scatter(
                x=x_range, y=eur_model.predict(x_range.reshape(-1, 1)),
                mode="lines", name="Trend", line=dict(color="black", dash="dash"),
            ))
            st.plotly_chart(fig_eur, use_container_width=True)

            fig_ip90 = px.scatter(
                pros_chart, x=SECTION_OOIP_COL, y="Norm IP90",
                color="Classification", color_discrete_map=COLOR_MAP_CLASS,
                hover_data=["Label"], title="Norm IP90 vs SectionOOIP",
            )
            fig_ip90.add_trace(go.Scatter(
                x=field[SECTION_OOIP_COL], y=field["Norm IP90"],
                mode="markers", name="Field UWIs",
                marker=dict(color="lightgrey", size=4, opacity=0.5),
            ))
            fig_ip90.add_trace(go.Scatter(
                x=x_range, y=ip90_model.predict(x_range.reshape(-1, 1)),
                mode="lines", name="Trend", line=dict(color="black", dash="dash"),
            ))
            st.plotly_chart(fig_ip90, use_container_width=True)

        with col2:
            fig_1y = px.scatter(
                pros_chart, x=SECTION_OOIP_COL, y="Norm 1Y Cuml",
                color="Classification", color_discrete_map=COLOR_MAP_CLASS,
                hover_data=["Label"], title="Norm 1Y Cuml vs SectionOOIP",
            )
            fig_1y.add_trace(go.Scatter(
                x=field[SECTION_OOIP_COL], y=field["Norm 1Y Cuml"],
                mode="markers", name="Field UWIs",
                marker=dict(color="lightgrey", size=4, opacity=0.5),
            ))
            fig_1y.add_trace(go.Scatter(
                x=x_range, y=y1_model.predict(x_range.reshape(-1, 1)),
                mode="lines", name="Trend", line=dict(color="black", dash="dash"),
            ))
            st.plotly_chart(fig_1y, use_container_width=True)

            fig_comp = px.scatter(
                pros_chart, x=SECTION_OOIP_COL, y="Composite_Z",
                color="Classification", color_discrete_map=COLOR_MAP_CLASS,
                hover_data=["Label"], title="Composite Z-Score",
            )
            fig_comp.add_hline(y=cls_threshold, line_dash="dot", line_color="green",
                               annotation_text="Above Trend")
            fig_comp.add_hline(y=-cls_threshold, line_dash="dot", line_color="red",
                               annotation_text="Below Trend")
            fig_comp.add_hline(y=0, line_dash="dash", line_color="black")
            st.plotly_chart(fig_comp, use_container_width=True)

        # Summary
        st.subheader("Summary")
        summary = pros_chart["Classification"].value_counts().reset_index()
        summary.columns = ["Classification", "Count"]
        st.dataframe(summary, use_container_width=True)

        # Detailed table — now with coordinates
        cls_display = pros_chart[[
            "Label", "Latitude", "Longitude",
            SECTION_OOIP_COL, "Norm EUR", "Norm 1Y Cuml", "Norm IP90", "WF",
            "Z_EUR", "Z_IP90", "Z_1Y", "Composite_Z", "Classification"
        ]].copy()
        cls_display = cls_display.sort_values("Composite_Z", ascending=False).reset_index(drop=True)
        st.dataframe(cls_display, use_container_width=True)
        st.download_button(
            "📥 Download Classified Prospects",
            data=cls_display.to_csv(index=False),
            file_name="classified_prospects.csv", mime="text/csv",
        )
    else:
        st.warning("No prospects with complete data for classification charts.")

# ==========================================================
# RANKING
# ==========================================================
st.markdown("---")
st.header("📊 Prospect Ranking")

if selected_metric == "High-Grade Score" and total_weight != 100:
    st.warning("Adjust weights to total 100% to see rankings.")
elif rank_df is None or rank_df.empty:
    st.warning(f"No valid data for **{selected_metric}**.")
else:
    st.caption(
        f"Ranked by **{selected_metric}** · {len(rank_df)} prospects · "
        f"Buffer: {buffer_distance}m · IDW² interpolation"
    )
    fmt = {}
    for c in rank_df.columns:
        if c in ("Label", "Type", "Classification"):
            continue
        if c == "Percentile":
            fmt[c] = "{:.0f}%"
        elif c == "Proximal":
            fmt[c] = "{:.0f}"
        elif c in ("Latitude", "Longitude"):
            fmt[c] = "{:.6f}"
        elif rank_df[c].dtype in [np.float64, np.float32, float]:
            fmt[c] = (
                "{:,.0f}"
                if pd.notna(rank_df[c].abs().max()) and rank_df[c].abs().max() > 100
                else "{:.3f}"
            )

    styled = rank_df.style.background_gradient(
        subset=[metric_col], cmap="YlGn",
        gmap=rank_df[metric_col] if not ascending else -rank_df[metric_col],
    ).background_gradient(subset=["Percentile"], cmap="RdYlGn").format(fmt)

    st.dataframe(styled, use_container_width=True, height=500)
    st.download_button(
        "⬇️ Download Rankings (CSV)",
        data=rank_df.to_csv().encode("utf-8"),
        file_name="bakken_prospect_rankings.csv", mime="text/csv",
    )

# No-proximal table
no_prox = p[p["_no_proximal"]]
if not no_prox.empty:
    st.markdown("---")
    st.subheader("⚠️ No Proximal Wells Found")
    st.caption(f"{len(no_prox)} prospects have no proximal wells within {buffer_distance}m.")
    st.dataframe(
        no_prox[["Label", "_prospect_type", "Latitude", "Longitude"]].rename(
            columns={"_prospect_type": "Type"}
        ).reset_index(drop=True),
        use_container_width=True,
    )
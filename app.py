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
from sklearn.linear_model import LinearRegression, RANSACRegressor

# ==========================================================
# Page configuration & Constants
# ==========================================================
st.set_page_config(layout="wide", page_title="Bakken Inventory Optimizer", page_icon="🛢️")

NULL_STYLE = {"fillColor": "#ffffff", "fillOpacity": 0, "color": "#888", "weight": 0.25}
DEFAULT_BUFFER_M = 900

# ---- 4-quadrant colour scheme ----
COLOR_MAP_CLASS = {
    "High Prod / High Resource": "#2ca02c",
    "Low Prod / High Resource":  "#ff7f0e",
    "High Prod / Low Resource":  "#1f77b4",
    "Low Prod / Low Resource":   "#d62728",
}

WELL_COLS = ["Norm EUR", "Norm 1Y Cuml", "Norm IP90"]
SECTION_OOIP_COL = "SectionOOIP"
SECTION_ROIP_COL = "SectionROIP"                                                    # <-- NEW
ALL_METRIC_COLS = WELL_COLS + ["WF", SECTION_OOIP_COL, SECTION_ROIP_COL]            # <-- UPDATED
TOOLTIP_STYLE = (
    "font-size:11px;padding:3px 6px;background:rgba(255,255,255,0.92);"
    "border:1px solid #333;border-radius:3px;"
)

# ==========================================================
# Helpers
# ==========================================================
def safe_range(series):
    vals = series.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        return (0.0, 1.0) if lo == 0 else (lo - abs(lo) * 0.1, lo + abs(lo) * 0.1)
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
    if geom.geom_type == "MultiLineString":
        return max(geom.geoms, key=lambda g: g.length).interpolate(0.5, normalized=True)
    if geom.geom_type == "Point":
        return geom
    return geom.centroid


def endpoint_of_geom(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return Point(list(geom.coords)[-1])
    if geom.geom_type == "MultiLineString":
        return Point(list(geom.geoms[-1].coords)[-1])
    if geom.geom_type == "Point":
        return geom
    return None


def get_ylgn_hex(value, vmin, vmax):
    if pd.isna(value) or vmin == vmax:
        return "#cccccc"
    rgba = mpl_cm.get_cmap("YlGn")(mcolors.Normalize(vmin, vmax)(value))
    return mcolors.to_hex(rgba)


def fit_trend(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    X, Y = x[mask].values.reshape(-1, 1), y[mask].values
    try:
        model = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=max(3, int(0.5 * len(X))),
            residual_threshold=None, random_state=42,
        )
        model.fit(X, Y)
        return model
    except Exception:
        return None


def classify_quadrant(prod_z, resource_z, prod_thresh, resource_thresh):
    high_prod = prod_z >= prod_thresh
    high_resource = resource_z >= resource_thresh
    if high_prod and high_resource:
        return "High Prod / High Resource"
    if (not high_prod) and high_resource:
        return "Low Prod / High Resource"
    if high_prod and (not high_resource):
        return "High Prod / Low Resource"
    return "Low Prod / Low Resource"


def prospect_coords_latlon(geom, transformer):
    ep = endpoint_of_geom(geom)
    if ep is None:
        return np.nan, np.nan
    lon, lat = transformer.transform(ep.x, ep.y)
    return round(lat, 6), round(lon, 6)


def make_prospect_label(gdf):
    if "UWI" in gdf.columns:
        return gdf["UWI"].astype(str).str.strip().replace("", np.nan).replace("nan", np.nan)
    return pd.Series("", index=gdf.index)


def fmt_val(col, v):
    if pd.isna(v):
        return "—"
    return f"{v:,.0f}" if abs(v) > 100 else f"{v:.3f}"


# ==========================================================
# Load data
# ==========================================================
@st.cache_resource(show_spinner="Loading spatial data …")
def load_data():
    lines = gpd.read_file("lines.shp")
    points = gpd.read_file("points.shp")
    grid = gpd.read_file("ooipsectiongrid.shp")
    infills = gpd.read_file("inf.shp")
    merged = gpd.read_file("merged_inventory.shp")
    lease_lines = gpd.read_file("ll.shp")
    units = gpd.read_file("Bakken Units.shp")
    land = gpd.read_file("Bakken Land.shp")

    well_df = pd.read_excel("wells.xlsx", sheet_name=0)
    section_df = pd.read_excel("wells.xlsx", sheet_name=1)

    for gdf in [lines, points, grid, units, infills, lease_lines, merged, land]:
        if gdf.crs is None:
            gdf.set_crs(epsg=26913, inplace=True)
        gdf.to_crs(epsg=26913, inplace=True)

    grid["Section"] = grid["Section"].astype(str).str.strip()
    grid["geometry"] = grid.geometry.simplify(50, preserve_topology=True)

    well_df["UWI"] = well_df["UWI"].astype(str).str.strip()
    well_df["Section"] = well_df["Section"].astype(str).str.strip()
    for col in WELL_COLS:
        well_df[col] = pd.to_numeric(well_df[col], errors="coerce")
    well_df["WF"] = pd.to_numeric(well_df.get("WF", np.nan), errors="coerce")

    section_df["Section"] = section_df["Section"].astype(str).str.strip()
    section_df[SECTION_OOIP_COL] = pd.to_numeric(section_df[SECTION_OOIP_COL], errors="coerce")
    section_df[SECTION_ROIP_COL] = pd.to_numeric(section_df[SECTION_ROIP_COL], errors="coerce")  # <-- NEW

    sec_numeric_cols = [
        c for c in section_df.columns
        if c != "Section" and pd.api.types.is_numeric_dtype(section_df[c])
    ]

    lines["UWI"] = lines["UWI"].astype(str).str.strip()
    points["UWI"] = points["UWI"].astype(str).str.strip()

    # Merge both OOIP and ROIP onto well_df
    well_df_out = well_df.merge(
        section_df[["Section", SECTION_OOIP_COL, SECTION_ROIP_COL]],  # <-- UPDATED
        on="Section", how="left",
    )

    return (lines, points, grid, units, infills, lease_lines, merged, land,
            well_df_out, section_df, sec_numeric_cols)


(lines_gdf, points_gdf, grid_gdf, units_gdf, infills_gdf, lease_lines_gdf,
 merged_gdf, land_gdf, well_df, section_df, SEC_NUMERIC_COLS) = load_data()

# ==========================================================
# Derived spatial data
# ==========================================================
section_enriched = grid_gdf.merge(section_df, on="Section", how="left")

lines_with_uwi = lines_gdf[["UWI", "geometry"]].copy()
points_only = points_gdf[~points_gdf["UWI"].isin(lines_with_uwi["UWI"])][["UWI", "geometry"]].copy()
existing_wells = gpd.GeoDataFrame(
    pd.concat([lines_with_uwi, points_only], ignore_index=True),
    geometry="geometry", crs=lines_gdf.crs,
)

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

show_layers = {
    "Infill": st.sidebar.checkbox("Show Infills", value=True),
    "Lease Line": st.sidebar.checkbox("Show Lease Lines", value=True),
    "Merged": st.sidebar.checkbox("Show Merged", value=False),
}

LAYER_GDFS = {"Infill": infills_gdf, "Lease Line": lease_lines_gdf, "Merged": merged_gdf}

# ==========================================================
# Build prospect set
# ==========================================================
prospect_frames = []
for name, enabled in show_layers.items():
    if enabled:
        f = LAYER_GDFS[name].copy()
        f["_prospect_type"] = name
        prospect_frames.append(f)

if not prospect_frames:
    st.error("Enable at least one prospect layer.")
    st.stop()

prospects = gpd.GeoDataFrame(
    pd.concat(prospect_frames, ignore_index=True),
    geometry="geometry", crs=infills_gdf.crs,
)

prospects["Label"] = make_prospect_label(prospects)
unnamed_mask = prospects["Label"].isna() | (prospects["Label"] == "")

# For unnamed prospects, find the section that the last coordinate (endpoint) falls in
if unnamed_mask.any():
    # Build a GeoDataFrame of endpoints for unnamed prospects
    unnamed_endpoints = []
    for idx in prospects[unnamed_mask].index:
        ep = endpoint_of_geom(prospects.at[idx, "geometry"])
        unnamed_endpoints.append({"_pidx": idx, "geometry": ep})

    ep_gdf = gpd.GeoDataFrame(unnamed_endpoints, crs=prospects.crs)
    ep_gdf = ep_gdf[ep_gdf["geometry"].notna()]  # drop any that couldn't resolve

    if not ep_gdf.empty:
        # Spatial join: find which section grid cell each endpoint falls within
        ep_in_section = gpd.sjoin(
            ep_gdf,
            grid_gdf[["Section", "geometry"]],
            how="left",
            predicate="within",
        )

        # If "within" misses due to floating-point issues, fall back to "intersects"
        still_missing = ep_in_section["Section"].isna()
        if still_missing.any():
            fallback = gpd.sjoin(
                ep_gdf.loc[still_missing.index],
                grid_gdf[["Section", "geometry"]],
                how="left",
                predicate="intersects",
            )
            # Take the first match per endpoint
            fallback = fallback[~fallback["Section"].isna()].groupby("_pidx").first()
            for pidx, row in fallback.iterrows():
                ep_in_section.loc[
                    ep_in_section["_pidx"] == pidx, "Section"
                ] = row["Section"]

        # Assign section names as labels
        for _, row in ep_in_section.dropna(subset=["Section"]).iterrows():
            prospects.at[row["_pidx"], "Label"] = str(row["Section"]).strip()

# Final fallback: anything still unnamed gets a generic label
unnamed_mask = prospects["Label"].isna() | (prospects["Label"] == "")
prospects.loc[unnamed_mask, "Label"] = ""

# ==========================================================
# Analyse prospects
# ==========================================================
def idw_for_column(hits, col, pros_index):
    valid = hits[hits[col].notna() & hits["_w"].notna()].copy()
    if valid.empty:
        return pd.Series(np.nan, index=pros_index)
    valid["_wv"] = valid[col] * valid["_w"]
    g = valid.groupby("index_right").agg(_wv_sum=("_wv", "sum"), _w_sum=("_w", "sum"))
    return (g["_wv_sum"] / g["_w_sum"]).reindex(pros_index)


def analyze_prospects(pros, prox, sections, buffer_m):
    pros = pros.copy()
    pros["_midpoint"] = pros.geometry.apply(midpoint_of_geom)
    pros["_buffer"] = pros.geometry.buffer(buffer_m, cap_style=2)
    buffer_gdf = gpd.GeoDataFrame(
        {"_pidx": pros.index, "geometry": pros["_buffer"]}, crs=pros.crs,
    )

    # IDW² for well-level metrics (midpoint-based)
    midpt_gdf = prox[prox["_midpoint"].notna()].copy()
    midpt_gdf = midpt_gdf.set_geometry(gpd.GeoSeries(midpt_gdf["_midpoint"], crs=prox.crs))
    well_hits = gpd.sjoin(midpt_gdf, buffer_gdf, how="inner", predicate="within")

    px_pts = well_hits["index_right"].map(lambda i: pros.at[i, "_midpoint"])
    well_hits["_dist"] = np.sqrt(
        (well_hits["_midpoint"].apply(lambda pt: pt.x) - px_pts.apply(lambda pt: pt.x if pt else np.nan)) ** 2 +
        (well_hits["_midpoint"].apply(lambda pt: pt.y) - px_pts.apply(lambda pt: pt.y if pt else np.nan)) ** 2
    ).replace(0, 1.0)
    well_hits["_w"] = 1.0 / (well_hits["_dist"] ** 2)

    idw_results = {col: idw_for_column(well_hits, col, pros.index) for col in WELL_COLS}

    proximal_count = well_hits.groupby("index_right").size().reindex(pros.index, fill_value=0)
    proximal_uwis = (
        well_hits.groupby("index_right")["UWI"]
        .apply(lambda x: ", ".join(x.astype(str)))
        .reindex(pros.index, fill_value="")
    )

    # WF: geometry-intersect based
    wf_wells = prox[["UWI", "WF", "geometry"]].copy()
    wf_wells = wf_wells[wf_wells["WF"].notna()].copy()

    if not wf_wells.empty:
        wf_hits = gpd.sjoin(wf_wells, buffer_gdf, how="inner", predicate="intersects")
        wf_hits["_well_midpoint"] = wf_hits.geometry.apply(midpoint_of_geom)
        wf_px_pts = wf_hits["index_right"].map(lambda i: pros.at[i, "_midpoint"])
        wf_hits["_dist"] = np.sqrt(
            (wf_hits["_well_midpoint"].apply(lambda pt: pt.x if pt else np.nan) -
             wf_px_pts.apply(lambda pt: pt.x if pt else np.nan)) ** 2 +
            (wf_hits["_well_midpoint"].apply(lambda pt: pt.y if pt else np.nan) -
             wf_px_pts.apply(lambda pt: pt.y if pt else np.nan)) ** 2
        ).replace(0, 1.0)
        wf_hits["_w"] = 1.0 / (wf_hits["_dist"] ** 2)
        wf_idw = idw_for_column(wf_hits, "WF", pros.index)
    else:
        wf_idw = pd.Series(np.nan, index=pros.index)

    # Mean SectionOOIP AND Mean SectionROIP from intersecting grid cells
    sec_join = gpd.sjoin(
        sections[["geometry", SECTION_OOIP_COL, SECTION_ROIP_COL]],  # <-- UPDATED: include ROIP
        buffer_gdf,
        how="inner", predicate="intersects",
    )
    ooip_mean = sec_join.groupby("index_right")[SECTION_OOIP_COL].mean().reindex(pros.index)
    roip_mean = sec_join.groupby("index_right")[SECTION_ROIP_COL].mean().reindex(pros.index)  # <-- NEW

    out = pd.DataFrame(index=pros.index)
    out["_prospect_type"] = pros["_prospect_type"].values
    out["Proximal_Count"] = proximal_count.values
    out["_proximal_uwis"] = proximal_uwis.values
    for col in WELL_COLS:
        out[col] = idw_results[col].values
    out["WF"] = wf_idw.values
    out[SECTION_OOIP_COL] = ooip_mean.values
    out[SECTION_ROIP_COL] = roip_mean.values  # <-- NEW
    return out

prospect_metrics = analyze_prospects(prospects, proximal_wells, section_enriched, buffer_distance)

for c in prospect_metrics.columns:
    prospects[c] = prospect_metrics[c].values

for col in ALL_METRIC_COLS:
    if col in prospects.columns:
        prospects[col] = prospects[col].replace([np.inf, -np.inf], np.nan)

# Coordinates
_tf = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)
_coords = prospects.geometry.apply(lambda g: prospect_coords_latlon(g, _tf))
prospects["Latitude"] = _coords.apply(lambda x: x[0])
prospects["Longitude"] = _coords.apply(lambda x: x[1])

# ==========================================================
# Classification
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Classification Settings")

st.sidebar.markdown("**Productivity weights (must sum to 100):**")
cw_eur = st.sidebar.number_input("EUR weight %", 0, 100, 34, key="cw_eur")
cw_1y = st.sidebar.number_input("1Y weight %", 0, 100, 33, key="cw_1y")
cw_ip90 = st.sidebar.number_input("IP90 weight %", 0, 100, 33, key="cw_ip90")
cw_sum = cw_eur + cw_1y + cw_ip90

classification_ready = False
eur_model = ip90_model = y1_model = None
prod_threshold = resource_threshold = None
field = pd.DataFrame()

if cw_sum != 100:
    st.sidebar.error(f"Weights sum to {cw_sum}%, must be 100%")
else:
    st.sidebar.success(f"Weights: EUR {cw_eur}%, 1Y {cw_1y}%, IP90 {cw_ip90}%")

    st.sidebar.markdown("**Quadrant thresholds:**")
    prod_threshold = st.sidebar.slider(
        "Productivity Z threshold (σ)", -1.0, 2.0, 0.0, 0.05,
        help="Prospects above this composite-Z are 'High Prod'",
        key="prod_thresh",
    )
    resource_threshold = st.sidebar.slider(
        "Resource Z threshold (σ)", -1.0, 2.0, 0.0, 0.05,
        help="Prospects above this ROIP-Z are 'High Resource'",
        key="res_thresh",
    )

    # --- UPDATED: require ROIP instead of just OOIP for field wells ---
    field = well_df.dropna(subset=[SECTION_ROIP_COL, "Norm EUR", "Norm 1Y Cuml", "Norm IP90"]).copy()
    field = field[field[SECTION_ROIP_COL] > 0].copy()

    if len(field) >= 2:
        for ratio_name, src in [("EUR_ratio", "Norm EUR"), ("Y1_ratio", "Norm 1Y Cuml"), ("IP90_ratio", "Norm IP90")]:
            field[ratio_name] = field[src] / field[SECTION_ROIP_COL]

        # --- UPDATED: trend lines fit against SectionROIP ---
        eur_model = fit_trend(field[SECTION_ROIP_COL], field["Norm EUR"])
        ip90_model = fit_trend(field[SECTION_ROIP_COL], field["Norm IP90"])
        y1_model = fit_trend(field[SECTION_ROIP_COL], field["Norm 1Y Cuml"])

        if all(m is not None for m in [eur_model, ip90_model, y1_model]):
            resid_std = {}
            for tag, model, src in [("EUR", eur_model, "Norm EUR"),
                                     ("IP90", ip90_model, "Norm IP90"),
                                     ("Y1", y1_model, "Norm 1Y Cuml")]:
                field[f"{tag}_resid"] = (
                    field[src] - model.predict(field[SECTION_ROIP_COL].values.reshape(-1, 1))
                )
                resid_std[tag] = field[f"{tag}_resid"].std()

            # --- UPDATED: Resource Z based on ROIP ---
            field_roip_mean = field[SECTION_ROIP_COL].mean()
            field_roip_std = field[SECTION_ROIP_COL].std()

            pros_cls = prospects.dropna(subset=[SECTION_ROIP_COL] + WELL_COLS).copy()
            pros_cls = pros_cls[pros_cls[SECTION_ROIP_COL] > 0].copy()

            if not pros_cls.empty:
                roip_vals = pros_cls[SECTION_ROIP_COL].values.reshape(-1, 1)
                pros_cls["EUR_pred"] = eur_model.predict(roip_vals)
                pros_cls["IP90_pred"] = ip90_model.predict(roip_vals)
                pros_cls["Y1_pred"] = y1_model.predict(roip_vals)

                for tag, src, std in [("Z_EUR", "Norm EUR", resid_std["EUR"]),
                                       ("Z_IP90", "Norm IP90", resid_std["IP90"]),
                                       ("Z_1Y", "Norm 1Y Cuml", resid_std["Y1"])]:
                    pred_map = {"Z_EUR": "EUR_pred", "Z_IP90": "IP90_pred", "Z_1Y": "Y1_pred"}
                    pros_cls[tag] = (pros_cls[src] - pros_cls[pred_map[tag]]) / std if std > 0 else 0

                pros_cls["Productivity_Z"] = (
                    (cw_eur / 100) * pros_cls["Z_EUR"] +
                    (cw_1y / 100) * pros_cls["Z_1Y"] +
                    (cw_ip90 / 100) * pros_cls["Z_IP90"]
                )

                if field_roip_std > 0:
                    pros_cls["Resource_Z"] = (
                        (pros_cls[SECTION_ROIP_COL] - field_roip_mean) / field_roip_std
                    )
                else:
                    pros_cls["Resource_Z"] = 0.0

                pros_cls["Classification"] = pros_cls.apply(
                    lambda r: classify_quadrant(
                        r["Productivity_Z"], r["Resource_Z"],
                        prod_threshold, resource_threshold,
                    ),
                    axis=1,
                )

                for col in ["Classification", "Productivity_Z", "Resource_Z",
                             "Z_EUR", "Z_IP90", "Z_1Y"]:
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
# Filters
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
    filter_mask &= ((p[col] >= f_lo) & (p[col] <= f_hi)) | p[col].isna()

p["_passes_filter"] = filter_mask
p["_no_proximal"] = ~has_proximal

n_total, n_passing = len(p), int(filter_mask.sum())
n_no_proximal = int((~has_proximal).sum())

st.sidebar.markdown(
    f"**{n_passing}** / {n_total} prospects pass filters "
    f"({n_passing / max(n_total, 1) * 100:.0f}%)"
)
if n_no_proximal:
    st.sidebar.warning(f"⚠️ {n_no_proximal} prospects have no nearby proximal wells")

# ==========================================================
# Ranking
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Ranking Metric")

available_for_ranking = [c for c in ALL_METRIC_COLS if c in p.columns]
selected_metric = st.sidebar.selectbox("Rank prospects by", available_for_ranking + ["High-Grade Score"])

hg_selected, hg_lower_better, hg_weights, total_weight = {}, {}, {}, 0

if selected_metric == "High-Grade Score":
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ High-Grade Score")
    st.sidebar.markdown("**Select metrics to include:**")
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
# Buffer colours
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
    label_color_map[row["Label"]] = (
        get_ylgn_hex(-val if ascending else val, gmap_vmin, gmap_vmax)
        if pd.notna(val) else "#cccccc"
    )

p["_buffer_color"] = p["Label"].map(label_color_map).fillna("#cccccc")

# ==========================================================
# Display-ready data (4326)
# ==========================================================
transformer_to_4326 = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)
section_display = section_enriched.to_crs(4326)
units_display = units_gdf.to_crs(4326)
land_display = land_gdf.to_crs(4326)
existing_display = proximal_wells.copy().to_crs(4326)


def build_tooltip_label(row, include_metrics=True):
    parts = []
    if row.get("Label", ""):
        parts.append(f"<b>{row['Label']}</b>")
    if include_metrics:
        parts.append(f"Proximal Wells: {row.get('Proximal_Count', '—')}")
        for col in ALL_METRIC_COLS:
            if col in row.index and pd.notna(row[col]):
                parts.append(f"{col}: {fmt_val(col, row[col])}")
    return "<br>".join(parts) if parts else "Prospect"


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
skip_cols = {"geometry", "_buffer", "_midpoint", "_passes_filter", "_no_proximal",
             "_buffer_color", "_proximal_uwis"}
keep_cols = [c for c in p.columns if c not in skip_cols]
p_lines = p[keep_cols + ["geometry"]].copy()
for c in p_lines.columns:
    if c != "geometry" and p_lines[c].dtype == object:
        p_lines[c] = p_lines[c].astype(str)
p_lines_display = p_lines.to_crs(4326)

# ==========================================================
# Ranking table
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
    display_cols = list(dict.fromkeys(c for c in display_cols if c in p.columns))

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
        best_label = best_row["Label"] if best_row["Label"] else "Unnamed"
        st.success(
            f"**{n_passing}** of {n_total} prospects pass filters. "
            f"Top prospect by **{selected_metric}**: **{best_label}** "
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

sec_tip_fields = [c for c in section_display.columns if c != "geometry"]
section_fg = folium.FeatureGroup(name="Section Grid", show=(section_gradient != "None"))
folium.GeoJson(
    section_display.to_json(), style_function=sec_style,
    highlight_function=lambda _: {"weight": 2, "color": "black", "fillOpacity": 0.5},
    tooltip=folium.GeoJsonTooltip(
        fields=sec_tip_fields, aliases=[f"{f}:" for f in sec_tip_fields],
        localize=True, sticky=True,
        style=f"font-size:11px;padding:4px 8px;background:rgba(255,255,255,0.9);"
              f"border:1px solid #333;border-radius:3px;",
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
    tip_html = build_tooltip_label(brow)
    if metric_col in brow.index and pd.notna(brow[metric_col]) and metric_col not in ALL_METRIC_COLS:
        tip_html += f"<br>{metric_col}: {fmt_val(metric_col, brow[metric_col])}"
    folium.GeoJson(
        brow.geometry.__geo_interface__,
        style_function=lambda _, _fc=fc: {
            "fillColor": _fc, "fillOpacity": 0.4, "color": _fc, "weight": 1, "opacity": 0.7
        },
        tooltip=folium.Tooltip(tip_html, sticky=True, style=TOOLTIP_STYLE),
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
        line_wells[well_tip_fields + ["geometry"]].to_json(),
        style_function=lambda _: {"color": "transparent", "weight": 15, "opacity": 0},
        highlight_function=lambda _: {"weight": 15, "color": "#555", "opacity": 0.3},
        tooltip=folium.GeoJsonTooltip(
            fields=well_tip_fields, aliases=[f"{f}:" for f in well_tip_fields],
            localize=True, sticky=True, style=TOOLTIP_STYLE,
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
                location=[ep.y, ep.x], radius=1,
                color="black", fill=True, fill_color="black", fill_opacity=0.8, weight=1,
            ).add_to(well_fg)

for _, row in point_wells.iterrows():
    tip_parts = []
    for col in well_tip_fields:
        if col not in row.index or pd.isna(row[col]):
            continue
        v = row[col]
        if isinstance(v, (int, float)):
            tip_parts.append(f"<b>{col}:</b> {fmt_val(col, v)}")
        else:
            tip_parts.append(f"<b>{col}:</b> {v}")
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=2,
        color="black", fill=True, fill_color="black", fill_opacity=0.9, weight=1,
        tooltip=folium.Tooltip("<br>".join(tip_parts), sticky=True, style=TOOLTIP_STYLE),
    ).add_to(well_fg)
well_fg.add_to(m)

# Prospect lines
prospect_fg = folium.FeatureGroup(name="Prospect Wells", show=True)
has_classification = classification_ready and "Classification" in p.columns

for _, row in p_lines_display.iterrows():
    cls_val = row.get("Classification", None) if has_classification else None
    line_color = COLOR_MAP_CLASS.get(cls_val, "red") if pd.notna(cls_val) else "red"

    tip_parts = []
    for c in p_lines_display.columns:
        if c == "geometry":
            continue
        val = row[c]
        if pd.notna(val) and str(val).strip():
            tip_parts.append(f"<b>{c}:</b> {val}")

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

prospect_fg.add_to(m)

folium.LayerControl(collapsed=True).add_to(m)
st_folium(m, use_container_width=True, height=900, returned_objects=[])

# ==========================================================
# CLASSIFICATION RESULTS
# ==========================================================
if classification_ready and "Classification" in p.columns:
    st.markdown("---")
    st.header("📐 Classification Results — 4-Quadrant View")

    pros_chart = p[p["_passes_filter"] & p["Classification"].notna()].copy()

    if not pros_chart.empty:
        col1, col2 = st.columns(2)

        # --- UPDATED: x-axis range uses SectionROIP ---
        x_range = np.linspace(
            field[SECTION_ROIP_COL].min(),
            field[SECTION_ROIP_COL].max(),
            100,
        )

        chart_configs = [
            ("Norm EUR", eur_model, col1),
            ("Norm 1Y Cuml", y1_model, col2),
            ("Norm IP90", ip90_model, col1),
        ]

        for y_col, model, target_col in chart_configs:
            with target_col:
                fig = px.scatter(
                    pros_chart, x=SECTION_ROIP_COL, y=y_col,  # <-- UPDATED x-axis
                    color="Classification", color_discrete_map=COLOR_MAP_CLASS,
                    hover_data=["Label"],
                    title=f"{y_col} vs {SECTION_ROIP_COL}",
                )
                fig.add_trace(go.Scatter(
                    x=field[SECTION_ROIP_COL], y=field[y_col],  # <-- UPDATED
                    mode="markers", name="Field UWIs",
                    marker=dict(color="lightgrey", size=4, opacity=0.5),
                ))
                fig.add_trace(go.Scatter(
                    x=x_range, y=model.predict(x_range.reshape(-1, 1)),
                    mode="lines", name="Trend", line=dict(color="black", dash="dash"),
                ))
                st.plotly_chart(fig, use_container_width=True)

        # ---- Productivity vs Resource quadrant chart ----
        with col2:
            fig_quad = px.scatter(
                pros_chart, x="Resource_Z", y="Productivity_Z",
                color="Classification", color_discrete_map=COLOR_MAP_CLASS,
                hover_data=["Label"],
                title="Productivity Z vs Resource Z (4-Quadrant)",
                labels={
                    "Resource_Z": f"Resource Z ({SECTION_ROIP_COL})",  # <-- UPDATED label
                    "Productivity_Z": "Productivity Z (Composite)",
                },
            )

            rx_min = min(pros_chart["Resource_Z"].min(), -2) - 0.5
            rx_max = max(pros_chart["Resource_Z"].max(), 2) + 0.5
            ry_min = min(pros_chart["Productivity_Z"].min(), -2) - 0.5
            ry_max = max(pros_chart["Productivity_Z"].max(), 2) + 0.5

            quad_rects = [
                dict(x0=resource_threshold, x1=rx_max, y0=prod_threshold, y1=ry_max,
                     fillcolor=COLOR_MAP_CLASS["High Prod / High Resource"], opacity=0.07),
                dict(x0=resource_threshold, x1=rx_max, y0=ry_min, y1=prod_threshold,
                     fillcolor=COLOR_MAP_CLASS["Low Prod / High Resource"], opacity=0.07),
                dict(x0=rx_min, x1=resource_threshold, y0=prod_threshold, y1=ry_max,
                     fillcolor=COLOR_MAP_CLASS["High Prod / Low Resource"], opacity=0.07),
                dict(x0=rx_min, x1=resource_threshold, y0=ry_min, y1=prod_threshold,
                     fillcolor=COLOR_MAP_CLASS["Low Prod / Low Resource"], opacity=0.07),
            ]
            for rect in quad_rects:
                fig_quad.add_shape(
                    type="rect", xref="x", yref="y", layer="below",
                    line=dict(width=0), **rect,
                )

            fig_quad.add_hline(y=prod_threshold, line_dash="dot", line_color="grey",
                               annotation_text=f"Prod σ = {prod_threshold}")
            fig_quad.add_vline(x=resource_threshold, line_dash="dot", line_color="grey",
                               annotation_text=f"Resource σ = {resource_threshold}")
            fig_quad.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.5)
            fig_quad.add_vline(x=0, line_dash="dash", line_color="black", line_width=0.5)

            st.plotly_chart(fig_quad, use_container_width=True)

        # ---- Summary ----
        st.subheader("Summary")
        summary = pros_chart["Classification"].value_counts().reset_index()
        summary.columns = ["Classification", "Count"]
        quad_order = list(COLOR_MAP_CLASS.keys())
        summary["Classification"] = pd.Categorical(summary["Classification"], categories=quad_order, ordered=True)
        summary = summary.sort_values("Classification").reset_index(drop=True)
        st.dataframe(summary, use_container_width=True)

        # --- UPDATED: include ROIP in classification export ---
        cls_display = pros_chart[[
            "Label", "Latitude", "Longitude",
            SECTION_OOIP_COL, SECTION_ROIP_COL,
            "Norm EUR", "Norm 1Y Cuml", "Norm IP90", "WF",
            "Z_EUR", "Z_IP90", "Z_1Y", "Productivity_Z", "Resource_Z", "Classification"
        ]].sort_values("Productivity_Z", ascending=False).reset_index(drop=True)
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
                "{:,.0f}" if pd.notna(rank_df[c].abs().max()) and rank_df[c].abs().max() > 100
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
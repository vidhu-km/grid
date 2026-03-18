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
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, RANSACRegressor
import string
import hashlib

# ==========================================================
# Page configuration & Constants
# ==========================================================
st.set_page_config(layout="wide", page_title="Inventory Classifier", page_icon="🛢️")

NULL_STYLE = {"fillColor": "#ffffff", "fillOpacity": 0, "color": "#888", "weight": 0.25}
DEFAULT_BUFFER_M = 900

COLOR_MAP_CLASS = {
    "High Prod / High Resource": "#2ca02c",
    "Low Prod / High Resource":  "#ff7f0e",
    "High Prod / Low Resource":  "#1f77b4",
    "Low Prod / Low Resource":   "#d62728",
}

WELL_COLS = ["Norm EUR", "Norm 1Y Cuml", "Norm IP90"]
SECTION_OOIP_COL = "SectionOOIP"
SECTION_ROIP_COL = "SectionROIP"
ALL_METRIC_COLS = WELL_COLS + ["WF", SECTION_OOIP_COL, SECTION_ROIP_COL]
TOOLTIP_STYLE = (
    "font-size:11px;padding:3px 6px;background:rgba(255,255,255,0.92);"
    "border:1px solid #333;border-radius:3px;"
)

BUFFER_STYLE_PASS = {
    "fillOpacity": 0, "color": "#000000", "weight": 1.2,
    "opacity": 0.6, "dashArray": "6 4",
}
BUFFER_STYLE_FAIL = {
    "fillOpacity": 0, "color": "#000000", "weight": 0.8,
    "opacity": 0.25, "dashArray": "6 4",
}
BUFFER_STYLE_NO_PROX = {
    "fillOpacity": 0, "color": "#000000", "weight": 0.8,
    "opacity": 0.3, "dashArray": "4 6",
}


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


def midpoint_of_geom(geom):
    if geom is None or geom.is_empty:
        return None
    gt = geom.geom_type
    if gt == "LineString":
        return geom.interpolate(0.5, normalized=True)
    if gt == "MultiLineString":
        return max(geom.geoms, key=lambda g: g.length).interpolate(0.5, normalized=True)
    if gt == "Point":
        return geom
    return geom.centroid


def endpoint_of_geom(geom):
    if geom is None or geom.is_empty:
        return None
    gt = geom.geom_type
    if gt == "LineString":
        return Point(geom.coords[-1])
    if gt == "MultiLineString":
        return Point(geom.geoms[-1].coords[-1])
    if gt == "Point":
        return geom
    return None


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


def fmt_val(col, v):
    if pd.isna(v):
        return "—"
    return f"{v:,.0f}" if abs(v) > 100 else f"{v:.3f}"


def _suffix_generator():
    n = 1
    while True:
        for combo in _alpha_combos(n):
            yield combo
        n += 1


def _alpha_combos(length):
    if length == 1:
        return list(string.ascii_uppercase)
    base = _alpha_combos(length - 1)
    return [b + c for b in base for c in string.ascii_uppercase]


# ==========================================================
# Load data (cached once)
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
    section_df[SECTION_ROIP_COL] = pd.to_numeric(section_df[SECTION_ROIP_COL], errors="coerce")

    sec_numeric_cols = [
        c for c in section_df.columns
        if c != "Section" and pd.api.types.is_numeric_dtype(section_df[c])
    ]

    lines["UWI"] = lines["UWI"].astype(str).str.strip()
    points["UWI"] = points["UWI"].astype(str).str.strip()

    well_df_out = well_df.merge(
        section_df[["Section", SECTION_OOIP_COL, SECTION_ROIP_COL]],
        on="Section", how="left",
    )

    # Pre-build existing wells and proximal wells
    lines_with_uwi = lines[["UWI", "geometry"]]
    points_only = points[~points["UWI"].isin(lines_with_uwi["UWI"])][["UWI", "geometry"]]
    existing_wells = gpd.GeoDataFrame(
        pd.concat([lines_with_uwi, points_only], ignore_index=True),
        geometry="geometry", crs=lines.crs,
    )
    proximal_wells = gpd.GeoDataFrame(
        existing_wells.merge(well_df_out, on="UWI", how="inner"),
        geometry="geometry", crs=existing_wells.crs,
    )
    proximal_wells["_midpoint"] = proximal_wells.geometry.apply(midpoint_of_geom)

    # Pre-build section enriched
    section_enriched = grid.merge(section_df, on="Section", how="left")

    # Pre-build 4326 display layers
    section_display = section_enriched.to_crs(4326)
    units_display = units.to_crs(4326)
    land_display = land.to_crs(4326)
    existing_display = proximal_wells.copy().to_crs(4326)

    return (grid, infills, lease_lines, merged,
            well_df_out, section_df, sec_numeric_cols,
            proximal_wells, section_enriched,
            section_display, units_display, land_display, existing_display)


(grid_gdf, infills_gdf, lease_lines_gdf, merged_gdf,
 well_df, section_df, SEC_NUMERIC_COLS,
 proximal_wells, section_enriched,
 section_display, units_display, land_display, existing_display) = load_data()

LAYER_GDFS = {"Infill": infills_gdf, "Lease Line": lease_lines_gdf, "Merged": merged_gdf}
_TF_4326 = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)

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
    "Infill": st.sidebar.checkbox("Show Unit Infills", value=True),
    "Lease Line": st.sidebar.checkbox("Show Unit Lease Lines", value=True),
    "Merged": st.sidebar.checkbox("Show Mosaic Merged Inventory", value=True),
}

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Classification Settings")

st.sidebar.markdown("**Productivity weights (must sum to 100):**")
cw_eur = st.sidebar.number_input("EUR weight %", 0, 100, 34, key="cw_eur")
cw_1y = st.sidebar.number_input("1Y weight %", 0, 100, 33, key="cw_1y")
cw_ip90 = st.sidebar.number_input("IP90 weight %", 0, 100, 33, key="cw_ip90")
cw_sum = cw_eur + cw_1y + cw_ip90

prod_threshold = resource_threshold = None
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


# ==========================================================
# Build & label prospects (cached on layer selection)
# ==========================================================
@st.cache_data(show_spinner=False)
def build_prospects(_layer_gdfs, enabled_layers, _grid_gdf):
    """Build prospect GeoDataFrame with labels. Cached on which layers are enabled."""
    frames = []
    for name in enabled_layers:
        f = _layer_gdfs[name].copy()
        f["_prospect_type"] = name
        frames.append(f)

    if not frames:
        return None

    prospects = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry", crs=f.crs,
    )

    # Labels from UWI
    if "UWI" in prospects.columns:
        prospects["Label"] = prospects["UWI"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    else:
        prospects["Label"] = np.nan
    prospects["_label_is_section"] = False

    unnamed_mask = prospects["Label"].isna()
    if unnamed_mask.any():
        unnamed_endpoints = []
        for idx in prospects[unnamed_mask].index:
            ep = endpoint_of_geom(prospects.at[idx, "geometry"])
            unnamed_endpoints.append({"_pidx": idx, "geometry": ep})

        ep_gdf = gpd.GeoDataFrame(unnamed_endpoints, crs=prospects.crs)
        ep_gdf = ep_gdf[ep_gdf["geometry"].notna()]

        if not ep_gdf.empty:
            ep_in_section = gpd.sjoin(
                ep_gdf, _grid_gdf[["Section", "geometry"]],
                how="left", predicate="within",
            )
            still_missing = ep_in_section["Section"].isna()
            if still_missing.any():
                missing_pidxs = ep_in_section.loc[still_missing, "_pidx"].values
                fallback_src = ep_gdf[ep_gdf["_pidx"].isin(missing_pidxs)]
                if not fallback_src.empty:
                    fallback = gpd.sjoin(
                        fallback_src, _grid_gdf[["Section", "geometry"]],
                        how="left", predicate="intersects",
                    )
                    fallback_first = fallback.dropna(subset=["Section"]).groupby("_pidx").first()
                    for pidx, row in fallback_first.iterrows():
                        ep_in_section.loc[ep_in_section["_pidx"] == pidx, "Section"] = row["Section"]

            for _, row in ep_in_section.dropna(subset=["Section"]).iterrows():
                prospects.at[row["_pidx"], "Label"] = str(row["Section"]).strip()
                prospects.at[row["_pidx"], "_label_is_section"] = True

        # Disambiguate
        section_labeled = prospects[prospects["_label_is_section"]]
        dupes = section_labeled.groupby("Label").filter(lambda g: len(g) > 1)
        if not dupes.empty:
            for section_name, group in dupes.groupby("Label"):
                if len(group) == 1:
                    continue
                suffix_gen = _suffix_generator()
                for pidx in group.index:
                    prospects.at[pidx, "Label"] = f"{section_name}-{next(suffix_gen)}"

    prospects["Label"] = prospects["Label"].fillna("")

    # Coordinates
    coords = prospects.geometry.apply(lambda g: _prospect_coords(g))
    prospects["BH Latitude"] = coords.apply(lambda x: x[0])
    prospects["BH Longitude"] = coords.apply(lambda x: x[1])

    return prospects


def _prospect_coords(geom):
    ep = endpoint_of_geom(geom)
    if ep is None:
        return np.nan, np.nan
    lon, lat = _TF_4326.transform(ep.x, ep.y)
    return round(lat, 6), round(lon, 6)


enabled = [name for name, on in show_layers.items() if on]
if not enabled:
    st.error("Enable at least one prospect layer.")
    st.stop()

prospects = build_prospects(LAYER_GDFS, tuple(enabled), grid_gdf)
if prospects is None:
    st.error("Enable at least one prospect layer.")
    st.stop()


# ==========================================================
# Analyse prospects (cached on buffer distance + layer selection)
# ==========================================================
@st.cache_data(show_spinner="Analysing prospects …")
def analyze_prospects(_pros, _prox, _sections, buffer_m):
    pros = _pros.copy()
    pros["_midpoint"] = pros.geometry.apply(midpoint_of_geom)
    pros["_buffer"] = pros.geometry.buffer(buffer_m, cap_style=2)
    buffer_gdf = gpd.GeoDataFrame(
        {"_pidx": pros.index, "geometry": pros["_buffer"]}, crs=pros.crs,
    )

    midpt_gdf = _prox[_prox["_midpoint"].notna()].copy()
    midpt_gdf = midpt_gdf.set_geometry(gpd.GeoSeries(midpt_gdf["_midpoint"], crs=_prox.crs))
    well_hits = gpd.sjoin(midpt_gdf, buffer_gdf, how="inner", predicate="within")

    # Vectorized distance computation
    px_midpoints = well_hits["index_right"].map(lambda i: pros.at[i, "_midpoint"])
    wh_x = well_hits["_midpoint"].apply(lambda pt: pt.x)
    wh_y = well_hits["_midpoint"].apply(lambda pt: pt.y)
    px_x = px_midpoints.apply(lambda pt: pt.x if pt else np.nan)
    px_y = px_midpoints.apply(lambda pt: pt.y if pt else np.nan)
    well_hits["_dist"] = np.sqrt((wh_x - px_x) ** 2 + (wh_y - px_y) ** 2).replace(0, 1.0)
    well_hits["_w"] = 1.0 / (well_hits["_dist"] ** 2)

    def idw_col(hits, col):
        valid = hits[hits[col].notna() & hits["_w"].notna()]
        if valid.empty:
            return pd.Series(np.nan, index=pros.index)
        wv = valid[col] * valid["_w"]
        g = pd.DataFrame({"_wv": wv, "_w": valid["_w"], "ir": valid["index_right"]})
        g = g.groupby("ir").agg(_wv_sum=("_wv", "sum"), _w_sum=("_w", "sum"))
        return (g["_wv_sum"] / g["_w_sum"]).reindex(pros.index)

    idw_results = {col: idw_col(well_hits, col) for col in WELL_COLS}

    proximal_count = well_hits.groupby("index_right").size().reindex(pros.index, fill_value=0)
    proximal_uwis = (
        well_hits.groupby("index_right")["UWI"]
        .apply(lambda x: ", ".join(x.astype(str)))
        .reindex(pros.index, fill_value="")
    )

    # WF IDW
    wf_wells = _prox[_prox["WF"].notna()][["UWI", "WF", "geometry"]].copy()
    if not wf_wells.empty:
        wf_hits = gpd.sjoin(wf_wells, buffer_gdf, how="inner", predicate="intersects")
        wf_hits["_well_midpoint"] = wf_hits.geometry.apply(midpoint_of_geom)
        wf_px = wf_hits["index_right"].map(lambda i: pros.at[i, "_midpoint"])
        wf_hits["_dist"] = np.sqrt(
            (wf_hits["_well_midpoint"].apply(lambda pt: pt.x if pt else np.nan) -
             wf_px.apply(lambda pt: pt.x if pt else np.nan)) ** 2 +
            (wf_hits["_well_midpoint"].apply(lambda pt: pt.y if pt else np.nan) -
             wf_px.apply(lambda pt: pt.y if pt else np.nan)) ** 2
        ).replace(0, 1.0)
        wf_hits["_w"] = 1.0 / (wf_hits["_dist"] ** 2)
        wf_idw = idw_col(wf_hits, "WF")
    else:
        wf_idw = pd.Series(np.nan, index=pros.index)

    sec_join = gpd.sjoin(
        _sections[["geometry", SECTION_OOIP_COL, SECTION_ROIP_COL]],
        buffer_gdf, how="inner", predicate="intersects",
    )
    ooip_mean = sec_join.groupby("index_right")[SECTION_OOIP_COL].mean().reindex(pros.index)
    roip_mean = sec_join.groupby("index_right")[SECTION_ROIP_COL].mean().reindex(pros.index)

    out = pd.DataFrame(index=pros.index)
    out["_prospect_type"] = pros["_prospect_type"].values
    out["Proximal_Count"] = proximal_count.values
    out["_proximal_uwis"] = proximal_uwis.values
    for col in WELL_COLS:
        out[col] = idw_results[col].values
    out["WF"] = wf_idw.values
    out[SECTION_OOIP_COL] = ooip_mean.values
    out[SECTION_ROIP_COL] = roip_mean.values
    return out


prospect_metrics = analyze_prospects(prospects, proximal_wells, section_enriched, buffer_distance)

for c in prospect_metrics.columns:
    prospects[c] = prospect_metrics[c].values

for col in ALL_METRIC_COLS:
    if col in prospects.columns:
        prospects[col] = prospects[col].replace([np.inf, -np.inf], np.nan)


# ==========================================================
# Classification (cached on weights + thresholds + buffer)
# ==========================================================
@st.cache_data(show_spinner=False)
def run_classification(_prospects_df, _well_df, cw_eur, cw_1y, cw_ip90, prod_thresh, res_thresh):
    """Returns (classification_ready, pros_cls DataFrame, field DataFrame, models dict)."""
    field = _well_df.dropna(subset=[SECTION_ROIP_COL, "Norm EUR", "Norm 1Y Cuml", "Norm IP90"]).copy()
    field = field[field[SECTION_ROIP_COL] > 0].copy()

    if len(field) < 2:
        return False, None, field, None, None

    eur_model = fit_trend(field[SECTION_ROIP_COL], field["Norm EUR"])
    ip90_model = fit_trend(field[SECTION_ROIP_COL], field["Norm IP90"])
    y1_model = fit_trend(field[SECTION_ROIP_COL], field["Norm 1Y Cuml"])

    if not all(m is not None for m in [eur_model, ip90_model, y1_model]):
        return False, None, field, None, None

    # Field residuals
    resid_std = {}
    for tag, model, src in [("EUR", eur_model, "Norm EUR"),
                             ("IP90", ip90_model, "Norm IP90"),
                             ("Y1", y1_model, "Norm 1Y Cuml")]:
        field[f"{tag}_resid"] = field[src] - model.predict(field[SECTION_ROIP_COL].values.reshape(-1, 1))
        resid_std[tag] = field[f"{tag}_resid"].std()

    field_roip_mean = field[SECTION_ROIP_COL].mean()
    field_roip_std = field[SECTION_ROIP_COL].std()

    pros_cls = _prospects_df.dropna(subset=[SECTION_ROIP_COL] + WELL_COLS).copy()
    pros_cls = pros_cls[pros_cls[SECTION_ROIP_COL] > 0].copy()

    if pros_cls.empty:
        return False, None, field, None, None

    roip_vals = pros_cls[SECTION_ROIP_COL].values.reshape(-1, 1)
    pros_cls["EUR_pred"] = eur_model.predict(roip_vals)
    pros_cls["IP90_pred"] = ip90_model.predict(roip_vals)
    pros_cls["Y1_pred"] = y1_model.predict(roip_vals)

    for tag, src, pred_col, std in [
        ("Z_EUR", "Norm EUR", "EUR_pred", resid_std["EUR"]),
        ("Z_IP90", "Norm IP90", "IP90_pred", resid_std["IP90"]),
        ("Z_1Y", "Norm 1Y Cuml", "Y1_pred", resid_std["Y1"]),
    ]:
        pros_cls[tag] = (pros_cls[src] - pros_cls[pred_col]) / std if std > 0 else 0.0

    pros_cls["Productivity_Z"] = (
        (cw_eur / 100) * pros_cls["Z_EUR"] +
        (cw_1y / 100) * pros_cls["Z_1Y"] +
        (cw_ip90 / 100) * pros_cls["Z_IP90"]
    )

    pros_cls["Resource_Z"] = (
        (pros_cls[SECTION_ROIP_COL] - field_roip_mean) / field_roip_std
        if field_roip_std > 0 else 0.0
    )

    pros_cls["Classification"] = [
        classify_quadrant(pz, rz, prod_thresh, res_thresh)
        for pz, rz in zip(pros_cls["Productivity_Z"], pros_cls["Resource_Z"])
    ]

    models = {"eur": eur_model, "ip90": ip90_model, "y1": y1_model}
    return True, pros_cls, field, models, None


classification_ready = False
pros_cls_result = None
field = pd.DataFrame()
models = None

if cw_sum == 100 and prod_threshold is not None:
    # Build a hashable representation of prospect numeric data for caching
    classification_ready, pros_cls_result, field, models, _ = run_classification(
        prospects, well_df, cw_eur, cw_1y, cw_ip90, prod_threshold, resource_threshold
    )

    if classification_ready and pros_cls_result is not None:
        for col in ["Classification", "Productivity_Z", "Resource_Z", "Z_EUR", "Z_IP90", "Z_1Y"]:
            prospects[col] = np.nan
            prospects.loc[pros_cls_result.index, col] = pros_cls_result[col]
    elif len(field) < 2:
        st.sidebar.warning(f"Only {len(field)} UWIs with complete data — need ≥ 2.")
    elif models is None:
        st.sidebar.warning("Could not fit trend lines (insufficient data).")
    else:
        st.sidebar.warning("No prospects with valid data for classification.")


# ==========================================================
# Filters
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Prospect Filters")

p = prospects
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

p = p.copy()
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
# Build map (main bottleneck — optimized)
# ==========================================================
def build_map(p_df, section_disp, units_disp, land_disp, existing_disp,
              section_grad, buffer_dist):
    bounds = p_df.total_bounds
    cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
    clon, clat = _TF_4326.transform(cx, cy)

    m = folium.Map(location=[clat, clon], zoom_start=11, tiles="CartoDB positron",
                   prefer_canvas=True)
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)

    # ── Layer 1: Land ──
    land_fg = folium.FeatureGroup(name="Bakken Land", show=True)
    folium.GeoJson(
        land_disp.__geo_interface__,
        style_function=lambda _: {
            "fillColor": "#fff9c4", "color": "#fff9c4",
            "weight": 0.5, "fillOpacity": 0.2,
        },
    ).add_to(land_fg)
    land_fg.add_to(m)

    # ── Layer 2: Units ──
    units_fg = folium.FeatureGroup(name="Units", show=True)
    folium.GeoJson(
        units_disp.__geo_interface__,
        style_function=lambda _: {
            "color": "black", "weight": 2, "fillOpacity": 0, "interactive": False,
        },
    ).add_to(units_fg)
    units_fg.add_to(m)

    # ── Layer 3: Section Grid ──
    sec_style = lambda _: NULL_STYLE
    if section_grad != "None" and section_grad in section_disp.columns:
        grad_vals = section_disp[section_grad].dropna()
        if not grad_vals.empty:
            colormap = cm.LinearColormap(
                ["#f7fcf5", "#74c476", "#00441b"],
                vmin=float(grad_vals.min()), vmax=float(grad_vals.max()),
            ).to_step(n=7)
            colormap.caption = section_grad
            m.add_child(colormap)

            def sec_style(feat, _col=section_grad, _cm=colormap):
                v = feat["properties"].get(_col)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    return {"fillColor": _cm(v), "fillOpacity": 0.45,
                            "color": "white", "weight": 0.3}
                return NULL_STYLE

    sec_tip_fields = [c for c in section_disp.columns if c != "geometry"]
    section_fg = folium.FeatureGroup(name="Section Grid", show=(section_grad != "None"))
    folium.GeoJson(
        section_disp.__geo_interface__, style_function=sec_style,
        highlight_function=lambda _: {"weight": 2, "color": "black", "fillOpacity": 0.5},
        tooltip=folium.GeoJsonTooltip(
            fields=sec_tip_fields, aliases=[f"{f}:" for f in sec_tip_fields],
            localize=True, sticky=True,
            style="font-size:11px;padding:4px 8px;background:rgba(255,255,255,0.9);"
                  "border:1px solid #333;border-radius:3px;",
        ),
    ).add_to(section_fg)
    section_fg.add_to(m)

    # ── Layer 4: Existing Wells ──
    well_fg = folium.FeatureGroup(name="Existing Wells")
    well_tip_fields = [c for c in existing_disp.columns if c not in ("geometry", "_midpoint")]

    line_wells = existing_disp[existing_disp.geometry.type != "Point"]
    point_wells = existing_disp[existing_disp.geometry.type == "Point"]

    if not line_wells.empty:
        # Invisible wide hover target
        folium.GeoJson(
            line_wells[well_tip_fields + ["geometry"]].__geo_interface__,
            style_function=lambda _: {"color": "transparent", "weight": 15, "opacity": 0},
            highlight_function=lambda _: {"weight": 15, "color": "#555", "opacity": 0.3},
            tooltip=folium.GeoJsonTooltip(
                fields=well_tip_fields, aliases=[f"{f}:" for f in well_tip_fields],
                localize=True, sticky=True, style=TOOLTIP_STYLE,
            ),
        ).add_to(well_fg)

        # Visible thin line
        line_clean = line_wells.drop(columns=["_midpoint"], errors="ignore").copy()
        for c in line_clean.columns:
            if c != "geometry" and line_clean[c].dtype == object:
                line_clean[c] = line_clean[c].astype(str)
        folium.GeoJson(
            line_clean.__geo_interface__,
            style_function=lambda _: {"color": "black", "weight": 0.5, "opacity": 0.8},
        ).add_to(well_fg)

        # Endpoints
        for _, row in line_wells.iterrows():
            ep = endpoint_of_geom(row.geometry)
            if ep is not None:
                folium.CircleMarker(
                    location=[ep.y, ep.x], radius=1,
                    color="black", fill=True, fill_color="black",
                    fill_opacity=0.8, weight=1,
                ).add_to(well_fg)

    for _, row in point_wells.iterrows():
        tip_parts = []
        for col in well_tip_fields:
            if col not in row.index or pd.isna(row[col]):
                continue
            v = row[col]
            tip_parts.append(
                f"<b>{col}:</b> {fmt_val(col, v)}" if isinstance(v, (int, float))
                else f"<b>{col}:</b> {v}"
            )
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x], radius=2,
            color="black", fill=True, fill_color="black", fill_opacity=0.9, weight=1,
            tooltip=folium.Tooltip("<br>".join(tip_parts), sticky=True, style=TOOLTIP_STYLE),
        ).add_to(well_fg)
    well_fg.add_to(m)

    # ── Layer 5: Buffers — batch by style group ──
    buffer_fg = folium.FeatureGroup(name="Prospect Buffers")
    buf_geoms = p_df.geometry.buffer(buffer_dist, cap_style=2)
    buf_gdf = gpd.GeoDataFrame({
        "_passes_filter": p_df["_passes_filter"].values,
        "_no_proximal": p_df["_no_proximal"].values,
        "geometry": buf_geoms,
    }, crs=p_df.crs).to_crs(4326)

    mask_pass = buf_gdf["_passes_filter"]
    mask_no_prox = buf_gdf["_no_proximal"]
    mask_fail = ~mask_pass & ~mask_no_prox

    for mask, style in [(mask_pass, BUFFER_STYLE_PASS),
                         (mask_fail, BUFFER_STYLE_FAIL),
                         (mask_no_prox, BUFFER_STYLE_NO_PROX)]:
        subset = buf_gdf[mask]
        if not subset.empty:
            folium.GeoJson(
                subset.__geo_interface__,
                style_function=lambda _, s=style: s,
            ).add_to(buffer_fg)
    buffer_fg.add_to(m)

    # ── Layer 6: Prospect Wells ──
    prospect_fg = folium.FeatureGroup(name="Prospect Wells", show=True)
    has_cls = "Classification" in p_df.columns

    skip_cols = {"geometry", "_label_is_section", "_prospect_type", "_buffer_color",
                 "_passes_filter", "_no_proximal", "_proximal_uwis", "_midpoint", "_buffer"}
    keep = [c for c in p_df.columns if c not in skip_cols]

    p_display = p_df[keep + ["geometry", "_label_is_section"]].copy()
    for c in p_display.columns:
        if c != "geometry" and p_display[c].dtype == object:
            p_display[c] = p_display[c].astype(str)
    p_display = p_display.to_crs(4326)

    for _, row in p_display.iterrows():
        cls_val = row.get("Classification", None) if has_cls else None
        line_color = (
            COLOR_MAP_CLASS.get(cls_val, "red")
            if pd.notna(cls_val) and str(cls_val) != "nan"
            else "red"
        )
        is_section = str(row.get("_label_is_section", "False")) == "True"

        tip_parts = []
        for c in keep:
            val = row[c]
            if pd.isna(val) or str(val).strip() in ("", "nan"):
                continue
            if c == "Label":
                tip_parts.append(f"<b>{'Section' if is_section else 'UWI'}:</b> {val}")
            elif c == "_label_is_section":
                continue
            else:
                tip_parts.append(f"<b>{c}:</b> {val}")

        tip_html = "<br>".join(tip_parts) if tip_parts else "Prospect"

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, lc=line_color: {"color": lc, "weight": 3, "opacity": 0.9},
            highlight_function=lambda _: {"weight": 5, "color": "#ff4444"},
            tooltip=folium.Tooltip(tip_html, sticky=True, style="font-size:12px"),
        ).add_to(prospect_fg)

        ep = endpoint_of_geom(row.geometry)
        if ep is not None:
            folium.CircleMarker(
                location=[ep.y, ep.x], radius=3,
                color=line_color, fill=True, fill_color=line_color,
                fill_opacity=0.9, weight=1,
                tooltip=folium.Tooltip(tip_html, sticky=True, style="font-size:12px"),
            ).add_to(prospect_fg)

    prospect_fg.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    return m


# ==========================================================
# Render
# ==========================================================
st.title("🛢️ Inventory Classifier")
st.info(
    f"**{n_passing}** of {n_total} prospects pass filters "
    f"({n_passing / max(n_total, 1) * 100:.0f}%). "
    f"Buffer: {buffer_distance}m."
)

m = build_map(
    p, section_display, units_display, land_display, existing_display,
    section_gradient, buffer_distance,
)
st_folium(m, use_container_width=True, height=900, returned_objects=[])


# ==========================================================
# Classification Results
# ==========================================================
if classification_ready and "Classification" in p.columns:
    st.markdown("---")
    st.header("📐 Classification Results — 4-Quadrant View")

    pros_chart = p[p["_passes_filter"] & p["Classification"].notna()].copy()

    if not pros_chart.empty:
        col1, col2 = st.columns(2)

        x_range = np.linspace(
            field[SECTION_ROIP_COL].min(),
            field[SECTION_ROIP_COL].max(),
            100,
        )

        chart_configs = [
            ("Norm EUR", models["eur"], col1),
            ("Norm 1Y Cuml", models["y1"], col2),
            ("Norm IP90", models["ip90"], col1),
        ]

        for y_col, model, target_col in chart_configs:
            with target_col:
                fig = px.scatter(
                    pros_chart, x=SECTION_ROIP_COL, y=y_col,
                    color="Classification", color_discrete_map=COLOR_MAP_CLASS,
                    hover_data=["Label"],
                    title=f"{y_col} vs {SECTION_ROIP_COL}",
                )
                fig.add_trace(go.Scatter(
                    x=field[SECTION_ROIP_COL], y=field[y_col],
                    mode="markers", name="Field UWIs",
                    marker=dict(color="lightgrey", size=4, opacity=0.5),
                ))
                fig.add_trace(go.Scatter(
                    x=x_range, y=model.predict(x_range.reshape(-1, 1)),
                    mode="lines", name="Trend", line=dict(color="black", dash="dash"),
                ))
                st.plotly_chart(fig, use_container_width=True)

        # Quadrant chart
        with col2:
            fig_quad = px.scatter(
                pros_chart, x="Resource_Z", y="Productivity_Z",
                color="Classification", color_discrete_map=COLOR_MAP_CLASS,
                hover_data=["Label"],
                title="Productivity Z vs Resource Z (4-Quadrant)",
                labels={
                    "Resource_Z": f"Resource Z ({SECTION_ROIP_COL})",
                    "Productivity_Z": "Productivity Z (Composite)",
                },
            )

            rx_min = min(pros_chart["Resource_Z"].min(), -2) - 0.5
            rx_max = max(pros_chart["Resource_Z"].max(), 2) + 0.5
            ry_min = min(pros_chart["Productivity_Z"].min(), -2) - 0.5
            ry_max = max(pros_chart["Productivity_Z"].max(), 2) + 0.5

            for rect in [
                dict(x0=resource_threshold, x1=rx_max, y0=prod_threshold, y1=ry_max,
                     fillcolor=COLOR_MAP_CLASS["High Prod / High Resource"], opacity=0.07),
                dict(x0=resource_threshold, x1=rx_max, y0=ry_min, y1=prod_threshold,
                     fillcolor=COLOR_MAP_CLASS["Low Prod / High Resource"], opacity=0.07),
                dict(x0=rx_min, x1=resource_threshold, y0=prod_threshold, y1=ry_max,
                     fillcolor=COLOR_MAP_CLASS["High Prod / Low Resource"], opacity=0.07),
                dict(x0=rx_min, x1=resource_threshold, y0=ry_min, y1=prod_threshold,
                     fillcolor=COLOR_MAP_CLASS["Low Prod / Low Resource"], opacity=0.07),
            ]:
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

        # Summary table
        st.subheader("Classification/Ranking Table")
        summary = pros_chart["Classification"].value_counts().reset_index()
        summary.columns = ["Classification", "Count"]
        summary["Classification"] = pd.Categorical(
            summary["Classification"],
            categories=list(COLOR_MAP_CLASS.keys()), ordered=True,
        )
        summary = summary.sort_values("Classification").reset_index(drop=True)
        st.dataframe(summary, use_container_width=True)

        cls_display = pros_chart[[
            "Label", "BH Latitude", "BH Longitude",
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

# No-proximal table
no_prox = p[p["_no_proximal"]]
if not no_prox.empty:
    st.markdown("---")
    st.subheader("⚠️ No Proximal Wells Found")
    st.caption(f"{len(no_prox)} prospects have no proximal wells within {buffer_distance}m.")
    st.dataframe(
        no_prox[["Label", "_prospect_type", "BH Latitude", "BH Longitude"]].rename(
            columns={"_prospect_type": "Type"}
        ).reset_index(drop=True),
        use_container_width=True,
    )
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import MiniMap
from streamlit_folium import st_folium
import branca.colormap as cm
from shapely.geometry import Point, LineString
from pyproj import Transformer
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, RANSACRegressor
import string

# ==========================================================
# Page config & Constants
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
SUM_COLS = ["WF", "FOOZ"]
SECTION_OOIP_COL = "SectionOOIP"
SECTION_ROIP_COL = "SectionROIP"
ALL_METRIC_COLS = WELL_COLS + SUM_COLS + [SECTION_OOIP_COL, SECTION_ROIP_COL]
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


def midpoint_of_geom(geom):
    if geom is None or geom.is_empty:
        return None
    t = geom.geom_type
    if t == "LineString":
        return geom.interpolate(0.5, normalized=True)
    if t == "MultiLineString":
        return max(geom.geoms, key=lambda g: g.length).interpolate(0.5, normalized=True)
    if t == "Point":
        return geom
    return geom.centroid


def startpoint_of_geom(geom):
    if geom is None or geom.is_empty:
        return None
    t = geom.geom_type
    if t == "LineString":
        return Point(geom.coords[0])
    if t == "MultiLineString":
        return Point(geom.geoms[0].coords[0])
    if t == "Point":
        return geom
    return None


def endpoint_of_geom(geom):
    if geom is None or geom.is_empty:
        return None
    t = geom.geom_type
    if t == "LineString":
        return Point(geom.coords[-1])
    if t == "MultiLineString":
        return Point(geom.geoms[-1].coords[-1])
    if t == "Point":
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
            residual_threshold=None,
            random_state=42,
        )
        model.fit(X, Y)
        return model
    except Exception:
        return None


def classify_quadrant(prod_z, resource_z, pt, rt):
    hp, hr = prod_z >= pt, resource_z >= rt
    if hp and hr:
        return "High Prod / High Resource"
    if not hp and hr:
        return "Low Prod / High Resource"
    if hp and not hr:
        return "High Prod / Low Resource"
    return "Low Prod / Low Resource"


def fmt_val(col, v):
    if pd.isna(v):
        return "—"
    if col in SUM_COLS:
        return f"{int(v)}"
    return f"{v:,.0f}" if abs(v) > 100 else f"{v:.3f}"


def _coords_key(coords):
    return tuple(tuple(round(c, 6) for c in pt) for pt in coords)


# ---- Paste parser (EPSG:4326 lon,lat) ----
def parse_coord_line(line: str):
    line = line.strip()
    if not line:
        return None
    line = line.replace(";", " ").replace(",", " ")
    parts = [p for p in line.split() if p]
    if len(parts) < 2:
        return None
    try:
        lon = float(parts[0])
        lat = float(parts[1])
    except (ValueError, IndexError):
        return None
    return lon, lat


def parse_wells_from_text(text: str):
    wells = []
    current = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                wells.append(current)
                current = []
            continue
        parsed = parse_coord_line(line)
        if parsed is None:
            continue
        current.append(parsed)

    if current:
        wells.append(current)

    wells = [w for w in wells if len(w) >= 2]
    return wells


def midpoint_to_wgs84(g):
    if g is None or g.is_empty:
        return (np.nan, np.nan)
    return g.x, g.y


# ==========================================================
# Load data (cached)
# ==========================================================
@st.cache_resource(show_spinner="Loading spatial data …")
def load_data():
    # ---- FIX: validate all files exist before attempting to read ----
    import os
    required_files = {
        "lines.shp": "Wellbore lateral lines",
        "points.shp": "Well surface locations",
        "ooipsectiongrid.shp": "Section grid with OOIP",
        "inv.shp": "Inventory polygons",
        "Bakken Units.shp": "Unit boundaries",
        "Bakken Land.shp": "Land grid",
        "wells.xlsx": "Well & section metrics",
    }
    missing = []
    for fname, desc in required_files.items():
        if not os.path.exists(fname):
            # also check for .shp sidecar files
            if fname.endswith(".shp"):
                stem = fname[:-4]
                has_sidecar = any(
                    os.path.exists(stem + ext)
                    for ext in [".dbf", ".shx", ".prj"]
                )
                if not has_sidecar:
                    missing.append(f"{fname}  ({desc})")
            else:
                missing.append(f"{fname}  ({desc})")

    if missing:
        st.error(
            "**Missing required data files.** Place these in the working directory:\n\n"
            + "\n".join(f"• `{f}`" for f in missing)
        )
        st.stop()

    lines = gpd.read_file("lines.shp")
    points = gpd.read_file("points.shp")
    grid = gpd.read_file("ooipsectiongrid.shp")
    inv = gpd.read_file("inv.shp")
    units = gpd.read_file("Bakken Units.shp")
    land = gpd.read_file("Bakken Land.shp")

    # wells.xlsx – sheet 0 = well metrics, sheet 1 = section metrics
    xls = pd.ExcelFile("wells.xlsx")
    well_df = pd.read_excel(xls, sheet_name=0)
    section_df = pd.read_excel(xls, sheet_name=1)

    # CRS handling  (assume source CRS = 26913 if missing)
    for gdf in [lines, points, grid, units, inv, land]:
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
    well_df["FOOZ"] = pd.to_numeric(well_df.get("FOOZ", np.nan), errors="coerce")

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

    grid_enriched = grid.merge(section_df, on="Section", how="left")

    section_4326 = grid_enriched.to_crs(4326)
    units_4326 = units.to_crs(4326)
    land_4326 = land.to_crs(4326)

    land_json = land_4326.to_json()
    units_json = units_4326.to_json()

    # Proximal matching wells (existing lines + points)
    lines_with_uwi = lines[["UWI", "geometry"]].copy()
    points_only = points[~points["UWI"].isin(lines_with_uwi["UWI"])][["UWI", "geometry"]].copy()
    existing_wells = gpd.GeoDataFrame(
        pd.concat([lines_with_uwi, points_only], ignore_index=True),
        geometry="geometry", crs=lines.crs,
    )

    proximal_wells = gpd.GeoDataFrame(
        existing_wells.merge(well_df_out, on="UWI", how="inner"),
        geometry="geometry", crs=existing_wells.crs,
    )
    proximal_wells["_midpoint"] = proximal_wells.geometry.apply(midpoint_of_geom)

    return (
        lines, points, grid, inv,
        land, well_df_out, section_df, sec_numeric_cols,
        section_4326, units_4326, land_json, units_json,
        proximal_wells,
        units,
    )


(
    _lines_gdf, _points_gdf, grid_gdf, inv_gdf,
    land_gdf, well_df, section_df, SEC_NUMERIC_COLS,
    section_enriched_4326, units_4326, land_json, units_json, proximal_wells, units_gdf
) = load_data()

# ==========================================================
# Session state for custom wells
# ==========================================================
if "drawn_wells" not in st.session_state:
    st.session_state.drawn_wells = []
if "drawn_coords_set" not in st.session_state:
    st.session_state.drawn_coords_set = set()

# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.title("Settings")

buffer_distance = st.sidebar.slider("Buffer Distance (m)", 100, 2000, DEFAULT_BUFFER_M, step=50)

st.sidebar.markdown("---")
section_gradient = st.sidebar.selectbox("Section Grid Colour", ["None"] + SEC_NUMERIC_COLS)

st.sidebar.markdown("---")
st.sidebar.subheader("✏️ Custom Wells")
st.sidebar.caption("Paste lon/lat coordinates from the URL below.")
st.sidebar.markdown(
    '<a href="https://vidhu-km.github.io/invdraw/" target="_blank">'
    '<button style="width:100%;padding:8px;background-color:#4a90d9;color:white;'
    'border:none;border-radius:5px;font-size:14px;cursor:pointer;">'
    '🔗 Select Sections Here</button></a>',
    unsafe_allow_html=True,
)

st.sidebar.subheader("📌 Paste Coordinates (lon,lat)")
st.sidebar.caption("EPSG:4326. Format `lon,lat` per line. Blank line separates wells.")

coords_text = st.sidebar.text_area(
    "Pasted coords",
    height=150,
    placeholder=(
        "-103.2345,48.0123\n"
        "-103.2330,48.0130\n"
        "\n"
        "-103.2200,48.0200\n"
        "-103.2150,48.0220"
    ),
    key="coords_text",
)

if st.sidebar.button("➕ Add to Custom Wells", type="primary"):
    wells = parse_wells_from_text(coords_text)
    if not wells:
        st.sidebar.warning("No valid wells found. Need at least 2 points per well.")
    else:
        tf = Transformer.from_crs("EPSG:4326", "EPSG:26913", always_xy=True)
        added = 0
        for wcoords in wells:
            coord_list = [(float(lon), float(lat)) for lon, lat in wcoords]
            key = _coords_key(coord_list)
            if key in st.session_state.drawn_coords_set:
                continue
            st.session_state.drawn_coords_set.add(key)
            st.session_state.drawn_wells.append({"coords": coord_list, "label": None})
            added += 1
        if added:
            st.toast(f"{added} custom well(s) added.", icon="✅")
        else:
            st.toast("All wells already exist.", icon="ℹ️")
        st.rerun()

if st.session_state.drawn_wells:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Manage Wells")
    wells_to_delete = []
    for i, cw in enumerate(st.session_state.drawn_wells):
        col_lbl, col_del = st.sidebar.columns([3, 1])
        with col_lbl:
            new_label = st.text_input(
                f"Well {i+1}",
                value=cw.get("label") or f"Custom-{i+1}",
                key=f"cw_label_{i}",
                label_visibility="collapsed",
            )
            st.session_state.drawn_wells[i]["label"] = new_label
        with col_del:
            if st.button("🗑️", key=f"cw_del_{i}"):
                wells_to_delete.append(i)

    if wells_to_delete:
        for idx in sorted(wells_to_delete, reverse=True):
            cw = st.session_state.drawn_wells[idx]
            st.session_state.drawn_coords_set.discard(_coords_key(cw["coords"]))
            st.session_state.drawn_wells.pop(idx)
        st.rerun()

    if st.sidebar.button("🗑️ Clear All"):
        st.session_state.drawn_wells.clear()
        st.session_state.drawn_coords_set.clear()
        st.rerun()
else:
    st.sidebar.info("No custom wells added yet.")

# ==========================================================
# Build prospects from custom wells only
# ==========================================================
tf_to_proj = Transformer.from_crs("EPSG:4326", "EPSG:26913", always_xy=True)

prospects_rows = []
if st.session_state.drawn_wells:
    for i, cw in enumerate(st.session_state.drawn_wells):
        coords_proj = [tf_to_proj.transform(lon, lat) for lon, lat in cw["coords"]]
        if len(coords_proj) >= 2:
            line = LineString(coords_proj)
            lbl = cw.get("label") or f"Custom-{i+1}"
            prospects_rows.append(
                {
                    "geometry": line,
                    "_prospect_type": "Custom",
                    "_is_custom": True,
                    "Label": lbl,
                }
            )

if not prospects_rows:
    st.title("🛢️ Inventory Classifier")
    st.caption("Add at least one custom well (2+ points) to generate prospects.")
    st.stop()

prospects = gpd.GeoDataFrame(prospects_rows, geometry="geometry", crs="EPSG:26913")
prospects["_is_custom"] = True

# ==========================================================
# Label prospects by endpoint section (fallback)
# ==========================================================
prospects["_label_is_section"] = False
if "Label" not in prospects.columns:
    prospects["Label"] = ""

ep_series = prospects.geometry.apply(endpoint_of_geom)
valid_ep = ep_series.notna()

if valid_ep.any():
    ep_gdf = gpd.GeoDataFrame(
        {"_pidx": ep_series[valid_ep].index, "geometry": ep_series[valid_ep].values},
        crs=prospects.crs,
    )

    joined = gpd.sjoin(ep_gdf, grid_gdf[["Section", "geometry"]], how="left", predicate="within")
    valid_rows = joined.dropna(subset=["Section"])

    for _, r in valid_rows.iterrows():
        prospects.at[r["_pidx"], "Label"] = str(r["Section"]).strip()
        prospects.at[r["_pidx"], "_label_is_section"] = True

prospects["Label"] = prospects["Label"].fillna("")

# ==========================================================
# Analyse prospects (proximal wells + section means)
# ==========================================================
def idw_for_column(hits, col, pros_index):
    valid = hits.loc[hits[col].notna() & hits["_w"].notna()]
    if valid.empty:
        return pd.Series(np.nan, index=pros_index)
    wv = valid[col] * valid["_w"]
    g = (
        pd.DataFrame({"_wv": wv, "_w": valid["_w"], "ir": valid["index_right"]})
        .groupby("ir")
        .sum()
    )
    return (g["_wv"] / g["_w"]).reindex(pros_index)


def analyze_prospects(pros, prox, sections, buffer_m):
    pros = pros.copy()
    pros["_midpoint"] = pros.geometry.apply(midpoint_of_geom)
    pros["_buffer"] = pros.geometry.buffer(buffer_m, cap_style=2)

    buffer_gdf = gpd.GeoDataFrame(
        {"_pidx": pros.index, "geometry": pros["_buffer"]}, crs=pros.crs
    )

    midpt_gdf = prox[prox["_midpoint"].notna()].copy()
    midpt_gdf = midpt_gdf.set_geometry(gpd.GeoSeries(midpt_gdf["_midpoint"], crs=prox.crs))

    # wells within buffer
    well_hits = gpd.sjoin(midpt_gdf, buffer_gdf, how="inner", predicate="within")

    # inverse-distance-squared weight
    px_mp = well_hits["index_right"].map(pros["_midpoint"])
    hit_x = well_hits["_midpoint"].apply(lambda pt: pt.x)
    hit_y = well_hits["_midpoint"].apply(lambda pt: pt.y)
    px_x = px_mp.apply(lambda pt: pt.x if pt is not None else np.nan)
    px_y = px_mp.apply(lambda pt: pt.y if pt is not None else np.nan)

    well_hits["_dist"] = np.sqrt((hit_x - px_x) ** 2 + (hit_y - px_y) ** 2).replace(0, 1.0)
    well_hits["_w"] = 1.0 / (well_hits["_dist"] ** 2)

    idw_results = {col: idw_for_column(well_hits, col, pros.index) for col in WELL_COLS}

    proximal_count = well_hits.groupby("index_right").size().reindex(pros.index, fill_value=0)
    proximal_uwis = (
        well_hits.groupby("index_right")["UWI"]
        .apply(lambda x: ", ".join(x.astype(str)))
        .reindex(pros.index, fill_value="")
    )

    # sum columns for wells intersecting buffer
    sum_results = {}
    for sc in SUM_COLS:
        if sc in prox.columns:
            sc_wells = prox.loc[prox[sc].notna(), ["UWI", sc, "geometry"]].copy()
            if not sc_wells.empty:
                sc_hits = gpd.sjoin(sc_wells, buffer_gdf, how="inner", predicate="intersects")
                sum_results[sc] = sc_hits.groupby("index_right")[sc].sum().reindex(pros.index, fill_value=0)
            else:
                sum_results[sc] = pd.Series(0, index=pros.index)
        else:
            sum_results[sc] = pd.Series(0, index=pros.index)

    # section means by intersection with buffer
    sec_join = gpd.sjoin(
        sections[["geometry", SECTION_OOIP_COL, SECTION_ROIP_COL]],
        buffer_gdf,
        how="inner",
        predicate="intersects",
    )
    ooip_mean = sec_join.groupby("index_right")[SECTION_OOIP_COL].mean().reindex(pros.index)
    roip_mean = sec_join.groupby("index_right")[SECTION_ROIP_COL].mean().reindex(pros.index)

    out = pd.DataFrame(index=pros.index)
    out["_prospect_type"] = pros["_prospect_type"].values
    out["Proximal_Count"] = proximal_count.values
    out["_proximal_uwis"] = proximal_uwis.values
    for col in WELL_COLS:
        out[col] = idw_results[col].values
    for sc in SUM_COLS:
        out[sc] = sum_results[sc].values
    out[SECTION_OOIP_COL] = ooip_mean.values
    out[SECTION_ROIP_COL] = roip_mean.values
    return out


prospect_metrics = analyze_prospects(prospects, proximal_wells, grid_gdf, buffer_distance)
for c in prospect_metrics.columns:
    prospects[c] = prospect_metrics[c].values

for col in ALL_METRIC_COLS:
    if col in prospects.columns:
        prospects[col] = prospects[col].replace([np.inf, -np.inf], np.nan)

# ==========================================================
# Coordinates for display
# ==========================================================
_tf_wgs = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)

_endpoints = prospects.geometry.apply(endpoint_of_geom)
_ep_x = _endpoints.apply(lambda pt: pt.x if pt else np.nan)
_ep_y = _endpoints.apply(lambda pt: pt.y if pt else np.nan)
valid_ep_mask = _ep_x.notna()

_lon_bh = np.full(len(prospects), np.nan)
_lat_bh = np.full(len(prospects), np.nan)
if valid_ep_mask.any():
    _lon_bh[valid_ep_mask], _lat_bh[valid_ep_mask] = _tf_wgs.transform(
        _ep_x[valid_ep_mask].values, _ep_y[valid_ep_mask].values
    )
prospects["BH Latitude"] = np.round(_lat_bh, 6)
prospects["BH Longitude"] = np.round(_lon_bh, 6)

_startpoints = prospects.geometry.apply(startpoint_of_geom)
_sp_x = _startpoints.apply(lambda pt: pt.x if pt else np.nan)
_sp_y = _startpoints.apply(lambda pt: pt.y if pt else np.nan)
valid_sp_mask = _sp_x.notna()

_lon_heel = np.full(len(prospects), np.nan)
_lat_heel = np.full(len(prospects), np.nan)
if valid_sp_mask.any():
    _lon_heel[valid_sp_mask], _lat_heel[valid_sp_mask] = _tf_wgs.transform(
        _sp_x[valid_sp_mask].values, _sp_y[valid_sp_mask].values
    )
prospects["Heel Latitude"] = np.round(_lat_heel, 6)
prospects["Heel Longitude"] = np.round(_lon_heel, 6)

# ==========================================================
# Classification controls & model fitting
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Classification")

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
    prod_threshold = st.sidebar.slider(
        "Productivity Z threshold (σ)", -1.0, 2.0, 0.0, 0.05, key="prod_thresh"
    )
    resource_threshold = st.sidebar.slider(
        "Resource Z threshold (σ)", -1.0, 2.0, 0.0, 0.05, key="res_thresh"
    )

    field = well_df.dropna(subset=[SECTION_ROIP_COL] + WELL_COLS).copy()
    field = field[field[SECTION_ROIP_COL] > 0]

    if len(field) >= 2:
        eur_model = fit_trend(field[SECTION_ROIP_COL], field["Norm EUR"])
        ip90_model = fit_trend(field[SECTION_ROIP_COL], field["Norm IP90"])
        y1_model = fit_trend(field[SECTION_ROIP_COL], field["Norm 1Y Cuml"])

        if all(m is not None for m in [eur_model, ip90_model, y1_model]):
            resid_std = {}
            for tag, model, src in [
                ("EUR", eur_model, "Norm EUR"),
                ("IP90", ip90_model, "Norm IP90"),
                ("Y1", y1_model, "Norm 1Y Cuml"),
            ]:
                preds = model.predict(field[SECTION_ROIP_COL].values.reshape(-1, 1))
                field[f"{tag}_resid"] = field[src] - preds
                resid_std[tag] = field[f"{tag}_resid"].std()

            field_roip_mean = field[SECTION_ROIP_COL].mean()
            field_roip_std = field[SECTION_ROIP_COL].std()

            pros_cls = prospects.dropna(subset=[SECTION_ROIP_COL] + WELL_COLS).copy()
            pros_cls = pros_cls[pros_cls[SECTION_ROIP_COL] > 0]

            if not pros_cls.empty:
                roip_vals = pros_cls[SECTION_ROIP_COL].values.reshape(-1, 1)
                pros_cls["EUR_pred"] = eur_model.predict(roip_vals)
                pros_cls["IP90_pred"] = ip90_model.predict(roip_vals)
                pros_cls["Y1_pred"] = y1_model.predict(roip_vals)

                for tag, src, pred_col, std in [
                    ("Z_EUR", "Norm EUR", "EUR_pred", resid_std["EUR"]),
                    ("Z_IP90", "Norm IP90", "IP90_pred", resid_std["IP90"]),
                    ("Z_1Y", "Norm 1Y Cuml", "Y1_pred", resid_std["Y1"]),
                ]:
                    pros_cls[tag] = (
                        (pros_cls[src] - pros_cls[pred_col]) / std if std and std > 0 else 0.0
                    )

                pros_cls["Productivity_Z"] = (
                    (cw_eur / 100) * pros_cls["Z_EUR"]
                    + (cw_1y / 100) * pros_cls["Z_1Y"]
                    + (cw_ip90 / 100) * pros_cls["Z_IP90"]
                )
                pros_cls["Resource_Z"] = (
                    (pros_cls[SECTION_ROIP_COL] - field_roip_mean) / field_roip_std
                    if field_roip_std and field_roip_std > 0
                    else 0.0
                )
                pros_cls["Classification"] = pros_cls.apply(
                    lambda r: classify_quadrant(
                        r["Productivity_Z"], r["Resource_Z"],
                        prod_threshold, resource_threshold,
                    ),
                    axis=1,
                )

                # write back to prospects
                for col in [
                    "Classification", "Productivity_Z", "Resource_Z",
                    "Z_EUR", "Z_IP90", "Z_1Y",
                ]:
                    if col not in prospects.columns:
                        prospects[col] = np.nan
                    prospects.loc[pros_cls.index, col] = pros_cls[col].values

                classification_ready = True
    else:
        st.sidebar.warning("Need ≥ 2 wells with valid ROIP & metrics to classify.")

# ==========================================================
# Filters
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

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

n_total = len(p)
n_passing = int(filter_mask.sum())
n_no_proximal = int((~has_proximal).sum())
n_custom = int(p["_is_custom"].sum())

# ==========================================================
# Map prep
# ==========================================================
transformer_to_4326 = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)

p["_line_color"] = "red"
if classification_ready and "Classification" in p.columns:
    p["_line_color"] = p["Classification"].map(COLOR_MAP_CLASS).fillna("red")

# ---- FIX: broken boolean logic for custom wells without classification ----
if "Classification" in p.columns:
    custom_no_cls = p["_is_custom"] & p["Classification"].isna()
else:
    custom_no_cls = p["_is_custom"].copy()
p.loc[custom_no_cls, "_line_color"] = "#ff00ff"


def _build_tooltip_html(row):
    parts = []
    is_custom = bool(row.get("_is_custom", False))
    if is_custom:
        parts.append("✏️ <b>CUSTOM WELL</b>")

    label = row.get("Label", "")
    if label:
        if is_custom:
            parts.append(f"<b>Label:</b> {label}")
        else:
            tag = "Section" if row.get("_label_is_section", False) else "UWI"
            parts.append(f"<b>{tag}:</b> {label}")

    pc = row.get("Proximal_Count", "—")
    parts.append(f"Proximal Wells: {pc}")

    for col in ALL_METRIC_COLS:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"{col}: {fmt_val(col, row[col])}")

    cls = row.get("Classification", None)
    if pd.notna(cls):
        parts.append(f"<b>Class:</b> {cls}")
    pz = row.get("Productivity_Z", None)
    if pd.notna(pz):
        parts.append(f"Prod Z: {pz:.2f}")
    rz = row.get("Resource_Z", None)
    if pd.notna(rz):
        parts.append(f"Resource Z: {rz:.2f}")

    return "<br>".join(parts) if parts else "Prospect"


p["_tooltip"] = p.apply(_build_tooltip_html, axis=1)

buffer_geoms = p.geometry.buffer(buffer_distance, cap_style=2)

p_lines_4326 = gpd.GeoDataFrame(
    {
        "_tooltip": p["_tooltip"].values,
        "_line_color": p["_line_color"].values,
        "_passes_filter": p["_passes_filter"].values,
        "_is_custom": p["_is_custom"].values,
        "geometry": p.geometry,
    },
    crs=p.crs,
).to_crs(4326)

buffer_gdf = gpd.GeoDataFrame(
    {
        "_passes_filter": p["_passes_filter"].values,
        "_no_proximal": p["_no_proximal"].values,
        "_is_custom": p["_is_custom"].values,
        "geometry": buffer_geoms,
    },
    crs=p.crs,
).to_crs(4326)

inv_4326 = inv_gdf.to_crs(4326)

# ==========================================================
# Title
# ==========================================================
st.title("🛢️ Inventory Classifier")
st.caption(f"**{n_passing}** / {n_total} prospects pass filters · Buffer: {buffer_distance}m")

# ==========================================================
# Map
# ==========================================================
bounds = p.total_bounds
cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
clon, clat = transformer_to_4326.transform(cx, cy)

@st.fragment
def render_map():
    m = folium.Map(
        location=[clat, clon],
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
    )

    MiniMap(toggle_display=True).add_to(m)

    # ---------------------------
    # SAFE SIMPLIFICATION (critical fix)
    # ---------------------------
    def safe_geojson(gdf, simplify_tol=30):
        try:
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.simplify(simplify_tol, preserve_topology=True)
            return gdf.to_json()
        except Exception:
            return None

    # ---------------------------
    # Land
    # ---------------------------
    land_fg = folium.FeatureGroup(name="Bakken Land", show=True)

    land_data = safe_geojson(land_gdf.to_crs(4326), 100)
    if land_data:
        folium.GeoJson(
            land_data,
            style_function=lambda _: {
                "fillColor": "#fff9c4",
                "color": "#fff9c4",
                "weight": 0.5,
                "fillOpacity": 0.2,
            },
        ).add_to(land_fg)

    land_fg.add_to(m)

    # ---------------------------
    # Units
    # ---------------------------
    units_fg = folium.FeatureGroup(name="Units", show=True)

    units_data = safe_geojson(units_gdf.to_crs(4326), 50)
    if units_data:
        folium.GeoJson(
            units_data,
            style_function=lambda _: {
                "color": "black",
                "weight": 2,
                "fillOpacity": 0,
            },
        ).add_to(units_fg)

    units_fg.add_to(m)

    # ---------------------------
    # Section grid (SAFE)
    # ---------------------------
    section_fg = folium.FeatureGroup(name="Section Grid", show=False)

    sec_data = safe_geojson(section_enriched_4326, 80)
    if sec_data:
        folium.GeoJson(
            sec_data,
            style_function=lambda _: {
                "fillOpacity": 0,
                "color": "#999",
                "weight": 0.3,
            },
        ).add_to(section_fg)

    section_fg.add_to(m)

    # ---------------------------
    # Inventory
    # ---------------------------
    inv_fg = folium.FeatureGroup(name="Inventory", show=True)

    inv_data = safe_geojson(inv_4326, 80)
    if inv_data:
        folium.GeoJson(
            inv_data,
            style_function=lambda _: {
                "color": "#999",
                "weight": 1,
                "fillOpacity": 0.05,
            },
        ).add_to(inv_fg)

    inv_fg.add_to(m)

    # ---------------------------
    # Buffers
    # ---------------------------
    buf_fg = folium.FeatureGroup(name="Buffers")

    for _, row in buffer_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        folium.GeoJson(
            geom.__geo_interface__,
            style_function=lambda _: {
                "color": "#000",
                "weight": 1,
                "opacity": 0.4,
                "dashArray": "5,5",
            },
        ).add_to(buf_fg)

    buf_fg.add_to(m)

    # ---------------------------
    # Prospect lines (FIXED lambda)
    # ---------------------------
    prospect_fg = folium.FeatureGroup(name="Prospects", show=True)

    for _, row in p_lines_4326.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        color = row["_line_color"]
        weight = 5 if row["_is_custom"] else 3
        tooltip = row["_tooltip"]

        folium.GeoJson(
            geom.__geo_interface__,
            style_function=lambda _, c=color, w=weight: {
                "color": c,
                "weight": w,
                "opacity": 0.9,
            },
            tooltip=folium.Tooltip(tooltip),
        ).add_to(prospect_fg)

    prospect_fg.add_to(m)

    folium.LayerControl().add_to(m)

    # ---------------------------
    # CRITICAL FIX
    # ---------------------------
    st_folium(
        m,
        use_container_width=True,
        height=800,
    )
    
render_map()

# ==========================================================
# Results table
# ==========================================================
p_display = p[p["_passes_filter"]].copy()

st.markdown("---")
st.header("Prospects")

if p_display.empty:
    st.info("No prospects pass filters.")
else:
    table_cols = [
        "Label", "_is_custom",
        "Heel Latitude", "Heel Longitude",
        "BH Latitude", "BH Longitude",
        SECTION_OOIP_COL, SECTION_ROIP_COL,
        "Norm EUR", "Norm 1Y Cuml", "Norm IP90",
        "WF", "FOOZ",
        "Productivity_Z", "Resource_Z", "Classification",
    ]
    table_cols = [c for c in table_cols if c in p_display.columns]
    df = (
        p_display[table_cols]
        .sort_values("Proximal_Count" if "Proximal_Count" in p_display.columns else table_cols[0],
                      ascending=False)
        .reset_index(drop=True)
    )
    df.rename(columns={"_is_custom": "Custom Well"}, inplace=True)

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "📥 Download CSV",
        data=df.to_csv(index=False),
        file_name="prospects_classified.csv",
        mime="text/csv",
    )

# ==========================================================
# No-proximal expander
# ==========================================================
no_prox = p[p["_no_proximal"]].copy()
if not no_prox.empty:
    with st.expander(f"⚠️ {len(no_prox)} prospects with no proximal wells within {buffer_distance}m"):
        show_cols = [
            "Label", "_prospect_type", "_is_custom",
            "Heel Latitude", "Heel Longitude",
            "BH Latitude", "BH Longitude",
        ]
        show_cols = [c for c in show_cols if c in no_prox.columns]
        st.dataframe(
            no_prox[show_cols]
            .rename(columns={"_prospect_type": "Type", "_is_custom": "Custom Well"})
            .reset_index(drop=True),
            use_container_width=True,
        )
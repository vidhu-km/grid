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


def _suffix_generator():
    n = 1
    while True:
        if n == 1:
            yield from string.ascii_uppercase
        else:
            for combo in _alpha_combos(n):
                yield combo
        n += 1


def _alpha_combos(length):
    if length == 1:
        return list(string.ascii_uppercase)
    return [b + c for b in _alpha_combos(length - 1) for c in string.ascii_uppercase]


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
    lon = float(parts[0])
    lat = float(parts[1])
    return lon, lat


def parse_wells_from_text(text: str):
    """
    Multiple wells separated by blank lines.
    Each well: >=2 lines of "lon,lat".
    """
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

# ==========================================================
# Load data (cached)
# ==========================================================
@st.cache_resource(show_spinner="Loading spatial data …")
def load_data():
    lines = gpd.read_file("lines.shp")
    points = gpd.read_file("points.shp")
    grid = gpd.read_file("ooipsectiongrid.shp")
    inv = gpd.read_file("inv.shp")  # Single combined prospect shapefile
    units = gpd.read_file("Bakken Units.shp")
    land = gpd.read_file("Bakken Land.shp")
    well_df = pd.read_excel("wells.xlsx", sheet_name=0)
    section_df = pd.read_excel("wells.xlsx", sheet_name=1)

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

    return (lines, points, grid, units, inv, land,
            well_df_out, section_df, sec_numeric_cols,
            grid_enriched, section_4326, units_4326, land_4326,
            land_json, units_json, proximal_wells)


(
    lines_gdf, points_gdf, grid_gdf, units_gdf, inv_gdf,
    land_gdf, well_df, section_df, SEC_NUMERIC_COLS,
    section_enriched, section_4326, units_4326, land_4326,
    land_json, units_json, proximal_wells
) = load_data()

# ==========================================================
# Session state
# ==========================================================
if "drawn_wells" not in st.session_state:
    st.session_state.drawn_wells = []
if "drawn_coords_set" not in st.session_state:
    st.session_state.drawn_coords_set = set()
if "last_drawing_hash" not in st.session_state:
    st.session_state.last_drawing_hash = None

# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.title("Settings")

buffer_distance = st.sidebar.slider("Buffer Distance (m)", 100, 2000, DEFAULT_BUFFER_M, step=50)

st.sidebar.markdown("---")
section_gradient = st.sidebar.selectbox("Section Grid Colour", ["None"] + SEC_NUMERIC_COLS)

show_inventory = st.sidebar.checkbox("Prospect Inventory", value=True)

# Custom well management (now via paste box)
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

# ---- Paste box ----
st.sidebar.subheader("📌 Paste Coordinates (lon,lat)")
st.sidebar.caption(
    "Use EPSG:4326 (WGS84). Format: `lon,lat` per line. "
    "Blank line separates wells."
)

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
        for wcoords in wells:
            coord_list = [(float(lon), float(lat)) for lon, lat in wcoords]
            key = _coords_key(coord_list)
            if key in st.session_state.drawn_coords_set:
                continue
            st.session_state.drawn_coords_set.add(key)
            st.session_state.drawn_wells.append({"coords": coord_list, "label": None})
        st.toast("Custom well(s) added.", icon="✅")
        st.rerun()

# Edit/delete existing custom wells
if st.session_state.drawn_wells:
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
            key = _coords_key(cw["coords"])
            st.session_state.drawn_coords_set.discard(key)
            st.session_state.drawn_wells.pop(idx)
        st.rerun()

    if st.sidebar.button("🗑️ Clear All"):
        st.session_state.drawn_wells.clear()
        st.session_state.drawn_coords_set.clear()
        st.rerun()
else:
    st.sidebar.info("No custom wells added yet.")

# ==========================================================
# Build prospect set
# ==========================================================
prospect_frames = []
if show_inventory:
    f = inv_gdf.copy()
    f["_prospect_type"] = "Inventory"
    f["_is_custom"] = False
    prospect_frames.append(f)

if st.session_state.drawn_wells:
    tf_to_proj = Transformer.from_crs("EPSG:4326", "EPSG:26913", always_xy=True)
    custom_rows = []
    for i, cw in enumerate(st.session_state.drawn_wells):
        coords_proj = [tf_to_proj.transform(lon, lat) for lon, lat in cw["coords"]]
        if len(coords_proj) >= 2:
            line = LineString(coords_proj)
            lbl = cw.get("label") or f"Custom-{i+1}"
            custom_rows.append({
                "geometry": line,
                "_prospect_type": "Custom",
                "_is_custom": True,
                "Label": lbl,
            })
    if custom_rows:
        custom_gdf = gpd.GeoDataFrame(custom_rows, crs="EPSG:26913")
        prospect_frames.append(custom_gdf)

if not prospect_frames:
    st.error("Enable the prospect inventory or add custom wells.")
    st.stop()

prospects = gpd.GeoDataFrame(
    pd.concat(prospect_frames, ignore_index=True),
    geometry="geometry", crs=inv_gdf.crs,
)

if "_is_custom" not in prospects.columns:
    prospects["_is_custom"] = False
prospects["_is_custom"] = prospects["_is_custom"].fillna(False).astype(bool)

# ==========================================================
# Label prospects
# ==========================================================
custom_mask_label = prospects["_is_custom"]

if "Label" not in prospects.columns:
    prospects["Label"] = None

prospects["Label"] = prospects["Label"].astype("object")

if "UWI" in prospects.columns:
    non_custom_mask = ~custom_mask_label
    uwi_vals = (
        prospects.loc[non_custom_mask, "UWI"]
        .astype(str).str.strip()
        .replace({"": np.nan, "nan": np.nan})
    )
    prospects.loc[non_custom_mask, "Label"] = prospects.loc[non_custom_mask, "Label"].where(
        prospects.loc[non_custom_mask, "Label"].notna() &
        (prospects.loc[non_custom_mask, "Label"].astype(str).str.strip() != ""),
        uwi_vals
    )

unnamed_mask = prospects["Label"].isna() & ~custom_mask_label
if unnamed_mask.any():
    ep_data = prospects.loc[unnamed_mask, "geometry"].apply(endpoint_of_geom)
    valid_ep = ep_data.dropna()

    if not valid_ep.empty:
        ep_gdf = gpd.GeoDataFrame(
            {"_pidx": valid_ep.index, "geometry": valid_ep.values},
            crs=prospects.crs,
        )

        joined = gpd.sjoin(ep_gdf, grid_gdf[["Section", "geometry"]], how="left", predicate="within")

        still_missing = joined["Section"].isna()
        if still_missing.any():
            missing_ep = ep_gdf.loc[ep_gdf["_pidx"].isin(joined.loc[still_missing, "_pidx"])]
            if not missing_ep.empty:
                fallback = gpd.sjoin(missing_ep, grid_gdf[["Section", "geometry"]], how="left", predicate="intersects")
                fb_first = fallback.dropna(subset=["Section"]).groupby("_pidx").first()
                for pidx, row in fb_first.iterrows():
                    joined.loc[joined["_pidx"] == pidx, "Section"] = row["Section"]

        valid_rows = joined.dropna(subset=["Section"])
        for _, row in valid_rows.iterrows():
            prospects.at[row["_pidx"], "Label"] = str(row["Section"]).strip()
            prospects.at[row["_pidx"], "_label_is_section"] = True

    section_labeled = prospects[prospects.get("_label_is_section", pd.Series(False, index=prospects.index)).fillna(False).astype(bool)]
    if not section_labeled.empty:
        dupe_labels = section_labeled["Label"].value_counts()
        dupe_labels = dupe_labels[dupe_labels > 1].index

        for section_name in dupe_labels:
            idxs = section_labeled[section_labeled["Label"] == section_name].index
            suffix_gen = _suffix_generator()
            for pidx in idxs:
                prospects.at[pidx, "Label"] = f"{section_name}-{next(suffix_gen)}"

prospects["Label"] = prospects["Label"].fillna("")

# ==========================================================
# Analyse prospects
# ==========================================================
def idw_for_column(hits, col, pros_index):
    valid = hits.loc[hits[col].notna() & hits["_w"].notna()]
    if valid.empty:
        return pd.Series(np.nan, index=pros_index)
    wv = valid[col] * valid["_w"]
    g = pd.DataFrame({"_wv": wv, "_w": valid["_w"], "ir": valid["index_right"]}).groupby("ir").sum()
    return (g["_wv"] / g["_w"]).reindex(pros_index)


def analyze_prospects(pros, prox, sections, buffer_m):
    pros = pros.copy()
    pros["_midpoint"] = pros.geometry.apply(midpoint_of_geom)
    pros["_buffer"] = pros.geometry.buffer(buffer_m, cap_style=2)

    buffer_gdf = gpd.GeoDataFrame(
        {"_pidx": pros.index, "geometry": pros["_buffer"]}, crs=pros.crs,
    )

    midpt_gdf = prox[prox["_midpoint"].notna()].copy()
    midpt_gdf = midpt_gdf.set_geometry(gpd.GeoSeries(midpt_gdf["_midpoint"], crs=prox.crs))
    well_hits = gpd.sjoin(midpt_gdf, buffer_gdf, how="inner", predicate="within")

    px_mp = well_hits["index_right"].map(pros["_midpoint"])
    hit_x = well_hits["_midpoint"].apply(lambda pt: pt.x)
    hit_y = well_hits["_midpoint"].apply(lambda pt: pt.y)
    px_x = px_mp.apply(lambda pt: pt.x if pt else np.nan)
    px_y = px_mp.apply(lambda pt: pt.y if pt else np.nan)
    well_hits["_dist"] = np.sqrt((hit_x - px_x) ** 2 + (hit_y - px_y) ** 2).replace(0, 1.0)
    well_hits["_w"] = 1.0 / (well_hits["_dist"] ** 2)

    idw_results = {col: idw_for_column(well_hits, col, pros.index) for col in WELL_COLS}

    proximal_count = well_hits.groupby("index_right").size().reindex(pros.index, fill_value=0)
    proximal_uwis = (
        well_hits.groupby("index_right")["UWI"]
        .apply(lambda x: ", ".join(x.astype(str)))
        .reindex(pros.index, fill_value="")
    )

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

    sec_join = gpd.sjoin(
        sections[["geometry", SECTION_OOIP_COL, SECTION_ROIP_COL]],
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
    for sc in SUM_COLS:
        out[sc] = sum_results[sc].values
    out[SECTION_OOIP_COL] = ooip_mean.values
    out[SECTION_ROIP_COL] = roip_mean.values
    return out


prospect_metrics = analyze_prospects(prospects, proximal_wells, section_enriched, buffer_distance)
for c in prospect_metrics.columns:
    prospects[c] = prospect_metrics[c].values

for col in ALL_METRIC_COLS:
    if col in prospects.columns:
        prospects[col] = prospects[col].replace([np.inf, -np.inf], np.nan)

# Coordinates
_tf = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)

_endpoints = prospects.geometry.apply(endpoint_of_geom)
_ep_x = _endpoints.apply(lambda pt: pt.x if pt else np.nan)
_ep_y = _endpoints.apply(lambda pt: pt.y if pt else np.nan)
valid_ep = _ep_x.notna()
_lon_bh, _lat_bh = np.full(len(prospects), np.nan), np.full(len(prospects), np.nan)
if valid_ep.any():
    _lon_bh[valid_ep], _lat_bh[valid_ep] = _tf.transform(
        _ep_x[valid_ep].values, _ep_y[valid_ep].values
    )
prospects["BH Latitude"] = np.round(_lat_bh, 6)
prospects["BH Longitude"] = np.round(_lon_bh, 6)

_startpoints = prospects.geometry.apply(startpoint_of_geom)
_sp_x = _startpoints.apply(lambda pt: pt.x if pt else np.nan)
_sp_y = _startpoints.apply(lambda pt: pt.y if pt else np.nan)
valid_sp = _sp_x.notna()
_lon_heel, _lat_heel = np.full(len(prospects), np.nan), np.full(len(prospects), np.nan)
if valid_sp.any():
    _lon_heel[valid_sp], _lat_heel[valid_sp] = _tf.transform(
        _sp_x[valid_sp].values, _sp_y[valid_sp].values
    )
prospects["Heel Latitude"] = np.round(_lat_heel, 6)
prospects["Heel Longitude"] = np.round(_lon_heel, 6)

# ==========================================================
# Classification
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
    prod_threshold = st.sidebar.slider("Productivity Z threshold (σ)", -1.0, 2.0, 0.0, 0.05, key="prod_thresh")
    resource_threshold = st.sidebar.slider("Resource Z threshold (σ)", -1.0, 2.0, 0.0, 0.05, key="res_thresh")

    field = well_df.dropna(subset=[SECTION_ROIP_COL, "Norm EUR", "Norm 1Y Cuml", "Norm IP90"]).copy()
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
                field[f"{tag}_resid"] = field[src] - model.predict(field[SECTION_ROIP_COL].values.reshape(-1, 1))
                resid_std[tag] = field[f"{tag}_resid"].std()

            field_roip_mean = field[SECTION_ROIP_COL].mean()
            field_roip_std = field[SECTION_ROIP_COL].std()

            pros_cls = prospects.dropna(subset=[SECTION_ROIP_COL] + WELL_COLS)
            pros_cls = pros_cls[pros_cls[SECTION_ROIP_COL] > 0].copy()

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
                    pros_cls[tag] = (pros_cls[src] - pros_cls[pred_col]) / std if std > 0 else 0

                pros_cls["Productivity_Z"] = (
                    (cw_eur / 100) * pros_cls["Z_EUR"] +
                    (cw_1y / 100) * pros_cls["Z_1Y"] +
                    (cw_ip90 / 100) * pros_cls["Z_IP90"]
                )
                pros_cls["Resource_Z"] = (
                    (pros_cls[SECTION_ROIP_COL] - field_roip_mean) / field_roip_std
                    if field_roip_std > 0 else 0.0
                )
                pros_cls["Classification"] = pros_cls.apply(
                    lambda r: classify_quadrant(
                        r["Productivity_Z"], r["Resource_Z"], prod_threshold, resource_threshold
                    ),
                    axis=1,
                )

                for col in ["Classification", "Productivity_Z", "Resource_Z", "Z_EUR", "Z_IP90", "Z_1Y"]:
                    prospects.loc[pros_cls.index, col] = pros_cls[col]

                classification_ready = True

# ==========================================================
# Filters
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

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

p["_passes_filter"] = filter_mask.fillna(False)
p["_no_proximal"] = ~has_proximal

n_total, n_passing = len(p), int(filter_mask.sum())
n_no_proximal = int((~has_proximal).sum())
n_custom = int(p["_is_custom"].sum())

# ==========================================================
# Prepare display data
# ==========================================================
transformer_to_4326 = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)
existing_display = proximal_wells.to_crs(4326)

# Flag wells with WF > 0 for blue colouring
existing_display["_wf_positive"] = (
    existing_display["WF"].notna() & (existing_display["WF"] > 0)
)


def _build_tooltip_html(row):
    parts = []
    is_custom = row.get("_is_custom", False)
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

has_classification = classification_ready and "Classification" in p.columns

if has_classification:
    p["_line_color"] = p["Classification"].map(COLOR_MAP_CLASS).fillna("red")
else:
    p["_line_color"] = "red"

custom_no_cls = p["_is_custom"] & (
    (~p["Classification"].notna()) if "Classification" in p.columns else True
)
p.loc[custom_no_cls, "_line_color"] = "#ff00ff"

buffer_geoms = p.geometry.buffer(buffer_distance, cap_style=2)
buffer_gdf = gpd.GeoDataFrame(
    {
        "_passes_filter": p["_passes_filter"].values,
        "_no_proximal": p["_no_proximal"].values,
        "_is_custom": p["_is_custom"].values,
        "geometry": buffer_geoms,
    },
    crs=p.crs
).to_crs(4326)

p_lines_4326 = gpd.GeoDataFrame(
    {
        "_tooltip": p["_tooltip"].values,
        "_line_color": p["_line_color"].values,
        "_passes_filter": p["_passes_filter"].values,
        "_is_custom": p["_is_custom"].values,
        "geometry": p.geometry,
    },
    crs=p.crs
).to_crs(4326)

# ==========================================================
# Title
# ==========================================================
st.title("🛢️ Inventory Classifier")
st.caption(
    f"**{n_passing}** / {n_total} prospects pass filters · "
    f"Buffer: {buffer_distance}m"
    + (f" · ✏️ {n_custom} custom well(s)" if n_custom else "")
)

# ==========================================================
# MAP
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
        prefer_canvas=True,
    )
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)

    # ── Bakken Land ──
    land_fg = folium.FeatureGroup(name="Bakken Land", show=True)
    folium.GeoJson(
        land_json,
        style_function=lambda _: {"fillColor": "#fff9c4", "color": "#fff9c4", "weight": 0.5, "fillOpacity": 0.2},
    ).add_to(land_fg)
    land_fg.add_to(m)

    # ── Units ──
    units_fg = folium.FeatureGroup(name="Units", show=True)
    folium.GeoJson(
        units_json,
        style_function=lambda _: {"color": "black", "weight": 2, "fillOpacity": 0, "interactive": False},
    ).add_to(units_fg)
    units_fg.add_to(m)

    # ── Section Grid ──
    if section_gradient != "None" and section_gradient in section_4326.columns:
        grad_vals = section_4326[section_gradient].dropna()
        if not grad_vals.empty:
            colormap = cm.LinearColormap(
                ["#f7fcf5", "#74c476", "#00441b"],
                vmin=float(grad_vals.min()),
                vmax=float(grad_vals.max()),
            ).to_step(n=7)
            colormap.caption = section_gradient
            m.add_child(colormap)

            sec_style = lambda feat, _col=section_gradient, _cm=colormap: (
                {
                    "fillColor": _cm(feat["properties"].get(_col)),
                    "fillOpacity": 0.45,
                    "color": "white",
                    "weight": 0.3,
                }
                if feat["properties"].get(_col) is not None
                and not (
                    isinstance(feat["properties"].get(_col), float)
                    and np.isnan(feat["properties"].get(_col))
                )
                else NULL_STYLE
            )
        else:
            sec_style = lambda _: NULL_STYLE
    else:
        sec_style = lambda _: NULL_STYLE

    sec_tip_fields = [c for c in section_4326.columns if c != "geometry"]
    section_fg = folium.FeatureGroup(name="Section Grid", show=(section_gradient != "None"))
    folium.GeoJson(
        section_4326.to_json(),
        style_function=sec_style,
        highlight_function=lambda _: {"weight": 2, "color": "black", "fillOpacity": 0.5},
        tooltip=folium.GeoJsonTooltip(
            fields=sec_tip_fields,
            aliases=[f"{f}:" for f in sec_tip_fields],
            localize=True,
            sticky=True,
            style=TOOLTIP_STYLE,
        ),
    ).add_to(section_fg)
    section_fg.add_to(m)

    # ── Existing Wells (with WF>0 coloured blue) ──
    well_fg = folium.FeatureGroup(name="Existing Wells")
    line_wells = existing_display[existing_display.geometry.type != "Point"]
    point_wells = existing_display[existing_display.geometry.type == "Point"]
    well_tip_fields = [c for c in existing_display.columns if c not in ("geometry", "_midpoint", "_wf_positive")]

    if not line_wells.empty:
        # Invisible wide hitbox for tooltips
        folium.GeoJson(
            line_wells[well_tip_fields + ["geometry"]].to_json(),
            style_function=lambda _: {"color": "transparent", "weight": 15, "opacity": 0},
            highlight_function=lambda _: {"weight": 15, "color": "#555", "opacity": 0.3},
            tooltip=folium.GeoJsonTooltip(
                fields=well_tip_fields,
                aliases=[f"{f}:" for f in well_tip_fields],
                localize=True,
                sticky=True,
                style=TOOLTIP_STYLE,
            ),
        ).add_to(well_fg)

        # Draw each line well individually to apply WF>0 colouring
        line_clean = line_wells.drop(columns=["_midpoint"], errors="ignore")
        for c in line_clean.columns:
            if c != "geometry" and c != "_wf_positive" and line_clean[c].dtype == object:
                line_clean[c] = line_clean[c].astype(str)

        for _, row in line_clean.iterrows():
            is_blue = row.get("_wf_positive", False)
            well_color = "#2196F3" if is_blue else "black"
            well_weight = 1.5 if is_blue else 0.5
            well_opacity = 1.0 if is_blue else 0.8

            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _, _c=well_color, _w=well_weight, _o=well_opacity: {
                    "color": _c, "weight": _w, "opacity": _o
                },
            ).add_to(well_fg)

            ep = endpoint_of_geom(row.geometry)
            if ep:
                folium.CircleMarker(
                    [ep.y, ep.x],
                    radius=2 if is_blue else 1,
                    color=well_color,
                    fill=True,
                    fill_color=well_color,
                    fill_opacity=1.0 if is_blue else 0.8,
                    weight=1,
                ).add_to(well_fg)

    for _, row in point_wells.iterrows():
        is_blue = row.get("_wf_positive", False)
        well_color = "#2196F3" if is_blue else "black"

        tip = "<br>".join(
            f"<b>{c}:</b> {fmt_val(c, row[c]) if isinstance(row[c], (int, float)) else row[c]}"
            for c in well_tip_fields if c in row.index and pd.notna(row[c])
        )
        folium.CircleMarker(
            [row.geometry.y, row.geometry.x],
            radius=3 if is_blue else 2,
            color=well_color,
            fill=True,
            fill_color=well_color,
            fill_opacity=1.0 if is_blue else 0.9,
            weight=1,
            tooltip=folium.Tooltip(tip, sticky=True, style=TOOLTIP_STYLE),
        ).add_to(well_fg)
    well_fg.add_to(m)

    # ── Prospect Buffers ──
    buf_fg = folium.FeatureGroup(name="Prospect Buffers")
    buffer_gdf["_bstyle"] = "fail"
    buffer_gdf.loc[buffer_gdf["_passes_filter"], "_bstyle"] = "pass"
    buffer_gdf.loc[buffer_gdf["_no_proximal"], "_bstyle"] = "noprox"
    buffer_gdf.loc[buffer_gdf["_is_custom"], "_bstyle"] = "custom"

    _BSTYLES = {
        "pass":   {"fillOpacity": 0, "color": "#000", "weight": 1.2, "opacity": 0.6, "dashArray": "6 4"},
        "fail":   {"fillOpacity": 0, "color": "#000", "weight": 0.8, "opacity": 0.25, "dashArray": "6 4"},
        "noprox": {"fillOpacity": 0, "color": "#000", "weight": 0.8, "opacity": 0.3, "dashArray": "4 6"},
        "custom": {"fillOpacity": 0.04, "fillColor": "#ff00ff", "color": "#ff00ff", "weight": 1.5, "opacity": 0.7, "dashArray": "4 4"},
    }

    folium.GeoJson(
        buffer_gdf[["_bstyle", "geometry"]].to_json(),
        style_function=lambda feat: _BSTYLES.get(
            feat["properties"].get("_bstyle", "fail"), _BSTYLES["fail"]
        ),
    ).add_to(buf_fg)
    buf_fg.add_to(m)

    # ── Prospect Wells ──
    prospect_fg = folium.FeatureGroup(name="Prospect Wells", show=True)

    for _, row in p_lines_4326.iterrows():
        lc = row["_line_color"]
        tip = row["_tooltip"]
        is_custom = row.get("_is_custom", False)
        line_weight = 5 if is_custom else 3

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, _lc=lc, _w=line_weight: {"color": _lc, "weight": _w, "opacity": 0.9},
            highlight_function=lambda _: {"weight": 7, "color": "#ff4444"},
            tooltip=folium.Tooltip(tip, sticky=True, style="font-size:12px"),
        ).add_to(prospect_fg)

        if is_custom:
            coords = list(row.geometry.coords)
            if len(coords) >= 2:
                folium.RegularPolygonMarker(
                    [coords[0][1], coords[0][0]],
                    number_of_sides=4,
                    radius=6,
                    color=lc,
                    fill=True,
                    fill_color=lc,
                    fill_opacity=0.9,
                    weight=2,
                    rotation=45,
                    tooltip=folium.Tooltip(f"✏️ Heel<br>{tip}", sticky=True, style="font-size:12px"),
                ).add_to(prospect_fg)
                folium.RegularPolygonMarker(
                    [coords[-1][1], coords[-1][0]],
                    number_of_sides=5,
                    radius=8,
                    color=lc,
                    fill=True,
                    fill_color=lc,
                    fill_opacity=0.9,
                    weight=2,
                    tooltip=folium.Tooltip(f"✏️ Toe<br>{tip}", sticky=True, style="font-size:12px"),
                ).add_to(prospect_fg)
        else:
            ep = endpoint_of_geom(row.geometry)
            if ep:
                folium.CircleMarker(
                    [ep.y, ep.x],
                    radius=3,
                    color=lc,
                    fill=True,
                    fill_color=lc,
                    fill_opacity=0.9,
                    weight=1,
                    tooltip=folium.Tooltip(tip, sticky=True, style="font-size:12px"),
                ).add_to(prospect_fg)

    prospect_fg.add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    _ = st_folium(
        m,
        use_container_width=True,
        height=800,
        returned_objects=[],
    )


render_map()

# ==========================================================
# Custom Well Results
# ==========================================================
custom_prospects = p[p["_is_custom"]].copy()
if not custom_prospects.empty:
    st.markdown("---")
    st.subheader("✏️ Custom Well Results")

    for _, cw_row in custom_prospects.iterrows():
        label = cw_row.get("Label", "Custom Well")
        cls = cw_row.get("Classification", None)
        cls_str = cls if pd.notna(cls) else "Unclassified"
        cls_color = COLOR_MAP_CLASS.get(cls, "#888") if pd.notna(cls) else "#888"

        with st.expander(f"**{label}** — {cls_str}", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"Heel: `{cw_row.get('Heel Latitude', '—')}, {cw_row.get('Heel Longitude', '—')}`  \n"
                    f"Toe: `{cw_row.get('BH Latitude', '—')}, {cw_row.get('BH Longitude', '—')}`  \n"
                    f"Proximal Wells: **{int(cw_row.get('Proximal_Count', 0))}**"
                )
            with c2:
                for col in ALL_METRIC_COLS:
                    val = cw_row.get(col, np.nan)
                    prefix = "Σ " if col in SUM_COLS else ""
                    st.markdown(f"{prefix}{col}: **{fmt_val(col, val)}**")
            with c3:
                st.markdown(
                    f"<span style='color:{cls_color};font-size:1.2em;font-weight:bold'>"
                    f"{cls_str}</span>",
                    unsafe_allow_html=True
                )
                pz = cw_row.get("Productivity_Z", np.nan)
                rz = cw_row.get("Resource_Z", np.nan)
                if pd.notna(pz):
                    st.markdown(f"Prod Z: **{pz:.2f}σ**")
                if pd.notna(rz):
                    st.markdown(f"Resource Z: **{rz:.2f}σ**")

# ==========================================================
# Classification Results
# ==========================================================
if classification_ready and "Classification" in p.columns:
    st.markdown("---")
    st.header("📐 Classification — 4-Quadrant View")

    pros_chart = p[p["_passes_filter"] & p["Classification"].notna()]

    if not pros_chart.empty:
        col1, col2 = st.columns(2)
        x_range = np.linspace(field[SECTION_ROIP_COL].min(), field[SECTION_ROIP_COL].max(), 100)

        for y_col, model, target_col in [
            ("Norm EUR", eur_model, col1),
            ("Norm 1Y Cuml", y1_model, col2),
            ("Norm IP90", ip90_model, col1),
        ]:
            with target_col:
                chart_df = pros_chart.copy()
                chart_df["_symbol"] = chart_df["_is_custom"].map({True: "Custom", False: "Prospect"})

                fig = px.scatter(
                    chart_df, x=SECTION_ROIP_COL, y=y_col,
                    color="Classification", color_discrete_map=COLOR_MAP_CLASS,
                    symbol="_symbol",
                    symbol_map={"Prospect": "circle", "Custom": "star"},
                    hover_data=["Label"], title=f"{y_col} vs {SECTION_ROIP_COL}",
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
                for trace in fig.data:
                    if hasattr(trace, 'marker') and getattr(trace.marker, 'symbol', None) == 'star':
                        trace.marker.size = 14
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            chart_df = pros_chart.copy()
            chart_df["_symbol"] = chart_df["_is_custom"].map({True: "Custom", False: "Prospect"})

            fig_quad = px.scatter(
                chart_df, x="Resource_Z", y="Productivity_Z",
                color="Classification", color_discrete_map=COLOR_MAP_CLASS,
                symbol="_symbol",
                symbol_map={"Prospect": "circle", "Custom": "star"},
                hover_data=["Label"],
                title="Productivity Z vs Resource Z",
                labels={
                    "Resource_Z": f"Resource Z ({SECTION_ROIP_COL})",
                    "Productivity_Z": "Productivity Z (Composite)"
                },
            )

            for trace in fig_quad.data:
                if hasattr(trace, 'marker') and getattr(trace.marker, 'symbol', None) == 'star':
                    trace.marker.size = 14

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
                    type="rect",
                    xref="x", yref="y",
                    layer="below",
                    line=dict(width=0),
                    **rect
                )

            fig_quad.add_hline(y=prod_threshold, line_dash="dot", line_color="grey",
                               annotation_text=f"Prod σ = {prod_threshold}")
            fig_quad.add_vline(x=resource_threshold, line_dash="dot", line_color="grey",
                               annotation_text=f"Resource σ = {resource_threshold}")
            fig_quad.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.5)
            fig_quad.add_vline(x=0, line_dash="dash", line_color="black", line_width=0.5)

            st.plotly_chart(fig_quad, use_container_width=True)

        table_cols = [
            "Label", "_is_custom",
            "Heel Latitude", "Heel Longitude", "BH Latitude", "BH Longitude",
            SECTION_OOIP_COL, SECTION_ROIP_COL,
            "Norm EUR", "Norm 1Y Cuml", "Norm IP90",
        ] + SUM_COLS + [
            "Z_EUR", "Z_IP90", "Z_1Y", "Productivity_Z", "Resource_Z", "Classification"
        ]
        table_cols = [c for c in table_cols if c in pros_chart.columns]

        cls_display = pros_chart[table_cols].sort_values(
            "Productivity_Z", ascending=False
        ).reset_index(drop=True)
        cls_display.rename(columns={"_is_custom": "Custom Well"}, inplace=True)
        st.dataframe(cls_display, use_container_width=True)
        st.download_button(
            "📥 Download CSV",
            data=cls_display.to_csv(index=False),
            file_name="classified_prospects.csv",
            mime="text/csv",
        )

# No-proximal
no_prox = p[p["_no_proximal"]]
if not no_prox.empty:
    with st.expander(f"⚠️ {len(no_prox)} prospects with no proximal wells within {buffer_distance}m"):
        st.dataframe(
            no_prox[["Label", "_prospect_type", "_is_custom",
                     "Heel Latitude", "Heel Longitude",
                     "BH Latitude", "BH Longitude"]].rename(
                columns={"_prospect_type": "Type", "_is_custom": "Custom Well"}
            ).reset_index(drop=True),
            use_container_width=True,
        )
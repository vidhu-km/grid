import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from pyproj import Transformer
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, RANSACRegressor
import streamlit.components.v1 as components

# -----------------------------
# UI / Theme
# -----------------------------
st.set_page_config(layout="wide", page_title="Inventory Classifier", page_icon="🛢️")

st.markdown("""
<style>
    [data-testid="stSidebar"] {background-color: #0e1117;}
    [data-testid="stSidebar"] * {color: #fafafa;}
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label {font-size: 0.85rem;}
    .stDataFrame {border: 1px solid #333; border-radius: 6px;}
    div[data-testid="stMetric"] {background: #161b22; padding: 12px 16px; border-radius: 8px; border: 1px solid #30363d;}
    div[data-testid="stMetric"] label {color: #8b949e !important; font-size: 0.8rem;}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {color: #58a6ff !important; font-size: 1.6rem;}
</style>
""", unsafe_allow_html=True)

NULL_STYLE = {"fillColor": "#ffffff", "fillOpacity": 0, "color": "#888", "weight": 0.25}
DEFAULT_BUFFER_M = 900
COLOR_MAP_CLASS = {
    "High Prod / High Resource": "#2ca02c",
    "Low Prod / High Resource": "#ff7f0e",
    "High Prod / Low Resource": "#1f77b4",
    "Low Prod / Low Resource": "#d62728",
}
WELL_COLS = ["Norm EUR", "Norm 1Y Cuml", "Norm IP90"]
SUM_COLS = ["WF", "FOOZ"]
SECTION_OOIP_COL = "SectionOOIP"
SECTION_ROIP_COL = "SectionROIP"
ALL_METRIC_COLS = WELL_COLS + SUM_COLS + [SECTION_OOIP_COL, SECTION_ROIP_COL]


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
    if geom.geom_type == "LineString":
        return geom.interpolate(0.5, normalized=True)
    if geom.geom_type == "MultiLineString":
        return max(geom.geoms, key=lambda g: g.length).interpolate(0.5, normalized=True)
    if geom.geom_type == "Point":
        return geom
    return geom.centroid


def startpoint_of_geom(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return Point(geom.coords[0])
    if geom.geom_type == "MultiLineString":
        return Point(geom.geoms[0].coords[0])
    if geom.geom_type == "Point":
        return geom
    return None


def endpoint_of_geom(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return Point(geom.coords[-1])
    if geom.geom_type == "MultiLineString":
        return Point(geom.geoms[-1].coords[-1])
    if geom.geom_type == "Point":
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


def parse_coord_line(line: str):
    line = line.strip().replace(";", " ").replace(",", " ")
    if not line:
        return None
    parts = [p for p in line.split() if p]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return None


def parse_wells_from_text(text: str):
    wells, current = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                wells.append(current)
                current = []
            continue
        parsed = parse_coord_line(line)
        if parsed:
            current.append(parsed)
    if current:
        wells.append(current)
    return [w for w in wells if len(w) >= 2]


@st.cache_resource(show_spinner="Loading spatial data…")
def load_data():
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
            if fname.endswith(".shp"):
                stem = fname[:-4]
                if not any(os.path.exists(stem + ext) for ext in [".dbf", ".shx", ".prj"]):
                    missing.append(f"{fname}  ({desc})")
            else:
                missing.append(f"{fname}  ({desc})")
    if missing:
        st.error("**Missing required data files:**\n\n" + "\n".join(f"• `{f}`" for f in missing))
        st.stop()

    lines = gpd.read_file("lines.shp")
    points = gpd.read_file("points.shp")
    grid = gpd.read_file("ooipsectiongrid.shp")
    inv = gpd.read_file("inv.shp")
    units = gpd.read_file("Bakken Units.shp")
    land = gpd.read_file("Bakken Land.shp")

    xls = pd.ExcelFile("wells.xlsx")
    well_df = pd.read_excel(xls, sheet_name=0)
    section_df = pd.read_excel(xls, sheet_name=1)

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
        on="Section",
        how="left"
    )
    grid_enriched = grid.merge(section_df, on="Section", how="left")

    section_4326 = grid_enriched.to_crs(4326)
    units_4326 = units.to_crs(4326)

    lines_with_uwi = lines[["UWI", "geometry"]].copy()
    points_only = points[~points["UWI"].isin(lines_with_uwi["UWI"])][["UWI", "geometry"]].copy()

    existing_wells = gpd.GeoDataFrame(
        pd.concat([lines_with_uwi, points_only], ignore_index=True),
        geometry="geometry",
        crs=lines.crs
    )

    proximal_wells = gpd.GeoDataFrame(
        existing_wells.merge(well_df_out, on="UWI", how="inner"),
        geometry="geometry",
        crs=existing_wells.crs
    )
    proximal_wells["_midpoint"] = proximal_wells.geometry.apply(midpoint_of_geom)

    return (lines, points, grid, inv, land, well_df_out, section_df,
            sec_numeric_cols, section_4326, units_4326, proximal_wells, units)


(_lines_gdf, _points_gdf, grid_gdf, inv_gdf, land_gdf, well_df, section_df,
 SEC_NUMERIC_COLS, section_enriched_4326, units_gdf, proximal_wells, units_gdf_raw) = load_data()

# -----------------------------
# Custom Wells in sidebar
# -----------------------------
if "drawn_wells" not in st.session_state:
    st.session_state.drawn_wells = []
if "drawn_coords_set" not in st.session_state:
    st.session_state.drawn_coords_set = set()

st.sidebar.title("⚙️ Settings")

buffer_distance = st.sidebar.slider("Buffer Distance (m)", 100, 2000, DEFAULT_BUFFER_M, step=50)
st.sidebar.markdown("---")
section_gradient = st.sidebar.selectbox("Section Grid Colour", ["None"] + SEC_NUMERIC_COLS)

st.sidebar.markdown("---")
st.sidebar.subheader("✏️ Custom Wells")
st.sidebar.markdown(
    '<a href="https://vidhu-km.github.io/invdraw/" target="_blank">'
    '<button style="width:100%;padding:10px;background:linear-gradient(135deg,#4a90d9,#357abd);color:white;'
    'border:none;border-radius:6px;font-size:14px;cursor:pointer;font-weight:600;">'
    '🔗 Open Coordinate Selector</button></a>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Paste EPSG:4326 `lon,lat` per line. Blank line separates wells.")

coords_text = st.sidebar.text_area(
    "Coordinates",
    height=150,
    placeholder="-103.2345,48.0123\n-103.2330,48.0130\n\n-103.2200,48.0200\n-103.2150,48.0220",
    key="coords_text",
)

if st.sidebar.button("➕ Add Custom Wells", type="primary"):
    wells = parse_wells_from_text(coords_text)
    if not wells:
        st.sidebar.warning("No valid wells found. Need ≥2 points per well.")
    else:
        tf = Transformer.from_crs("EPSG:4326", "EPSG:26913", always_xy=True)
        added = 0
        for wcoords in wells:
            coord_list = [(float(lon), float(lat)) for lon, lat in wcoords]
            key = _coords_key(coord_list)
            if key not in st.session_state.drawn_coords_set:
                st.session_state.drawn_coords_set.add(key)
                st.session_state.drawn_wells.append({"coords": coord_list, "label": None})
                added += 1
        if added:
            st.toast(f"{added} well(s) added.", icon="✅")
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

# -----------------------------
# Build prospects from custom wells
# -----------------------------
tf_to_proj = Transformer.from_crs("EPSG:4326", "EPSG:26913", always_xy=True)
prospects_rows = []

if st.session_state.drawn_wells:
    for i, cw in enumerate(st.session_state.drawn_wells):
        coords_proj = [tf_to_proj.transform(lon, lat) for lon, lat in cw["coords"]]
        if len(coords_proj) >= 2:
            prospects_rows.append({
                "geometry": LineString(coords_proj),
                "_prospect_type": "Custom",
                "_is_custom": True,
                "Label": cw.get("label") or f"Custom-{i+1}",
            })

if not prospects_rows:
    st.title("🛢️ Inventory Classifier")
    st.info("Add at least one custom well (2+ points) via the sidebar to begin analysis.")
    st.stop()

prospects = gpd.GeoDataFrame(prospects_rows, geometry="geometry", crs="EPSG:26913")
prospects["_is_custom"] = True
prospects["_label_is_section"] = False
if "Label" not in prospects.columns:
    prospects["Label"] = ""

# assign section label by endpoint
ep_series = prospects.geometry.apply(endpoint_of_geom)
valid_ep = ep_series.notna()
if valid_ep.any():
    ep_gdf = gpd.GeoDataFrame(
        {"_pidx": ep_series[valid_ep].index, "geometry": ep_series[valid_ep].values},
        crs=prospects.crs
    )
    joined = gpd.sjoin(ep_gdf, grid_gdf[["Section", "geometry"]], how="left", predicate="within")
    for _, r in joined.dropna(subset=["Section"]).iterrows():
        prospects.at[r["_pidx"], "Label"] = str(r["Section"]).strip()
        prospects.at[r["_pidx"], "_label_is_section"] = True

prospects["Label"] = prospects["Label"].fillna("")

# -----------------------------
# Analysis functions
# -----------------------------
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
    buffer_gdf = gpd.GeoDataFrame({"_pidx": pros.index, "geometry": pros["_buffer"]}, crs=pros.crs)

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
    proximal_uwis = well_hits.groupby("index_right")["UWI"].apply(lambda x: ", ".join(x.astype(str))).reindex(
        pros.index, fill_value=""
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


# compute metrics for prospects
prospect_metrics = analyze_prospects(prospects, proximal_wells, grid_gdf, buffer_distance)
for c in prospect_metrics.columns:
    prospects[c] = prospect_metrics[c].values

for col in ALL_METRIC_COLS:
    if col in prospects.columns:
        prospects[col] = prospects[col].replace([np.inf, -np.inf], np.nan)

_tf_wgs = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)

for prefix, geom_func in [("BH", endpoint_of_geom), ("Heel", startpoint_of_geom)]:
    pts = prospects.geometry.apply(geom_func)
    px = pts.apply(lambda pt: pt.x if pt else np.nan)
    py = pts.apply(lambda pt: pt.y if pt else np.nan)
    valid = px.notna()

    lon_arr, lat_arr = np.full(len(prospects), np.nan), np.full(len(prospects), np.nan)
    if valid.any():
        lon_arr[valid], lat_arr[valid] = _tf_wgs.transform(px[valid].values, py[valid].values)

    prospects[f"{prefix} Latitude"] = np.round(lat_arr, 6)
    prospects[f"{prefix} Longitude"] = np.round(lon_arr, 6)

# -----------------------------
# Classification settings
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Classification")

cw_eur = st.sidebar.number_input("EUR weight %", 0, 100, 34, key="cw_eur")
cw_1y = st.sidebar.number_input("1Y weight %", 0, 100, 33, key="cw_1y")
cw_ip90 = st.sidebar.number_input("IP90 weight %", 0, 100, 33, key="cw_ip90")
cw_sum = cw_eur + cw_1y + cw_ip90

classification_ready = False

if cw_sum != 100:
    st.sidebar.error(f"Weights sum to {cw_sum}% — must equal 100%")
else:
    prod_threshold = st.sidebar.slider("Productivity Z threshold (σ)", -1.0, 2.0, 0.0, 0.05, key="prod_thresh")
    resource_threshold = st.sidebar.slider("Resource Z threshold (σ)", -1.0, 2.0, 0.0, 0.05, key="res_thresh")

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
                    pros_cls[tag] = (pros_cls[src] - pros_cls[pred_col]) / std if std and std > 0 else 0.0

                pros_cls["Productivity_Z"] = (cw_eur / 100) * pros_cls["Z_EUR"] + (cw_1y / 100) * pros_cls["Z_1Y"] + (cw_ip90 / 100) * pros_cls["Z_IP90"]
                pros_cls["Resource_Z"] = (pros_cls[SECTION_ROIP_COL] - field_roip_mean) / field_roip_std if field_roip_std and field_roip_std > 0 else 0.0

                pros_cls["Classification"] = pros_cls.apply(
                    lambda r: classify_quadrant(r["Productivity_Z"], r["Resource_Z"], prod_threshold, resource_threshold),
                    axis=1
                )

                for col in ["Classification", "Productivity_Z", "Resource_Z", "Z_EUR", "Z_IP90", "Z_1Y"]:
                    if col not in prospects.columns:
                        prospects[col] = np.nan
                    prospects.loc[pros_cls.index, col] = pros_cls[col].values

                classification_ready = True
    else:
        st.sidebar.warning("Need ≥2 wells with valid ROIP & metrics to classify.")

# -----------------------------
# Filters
# -----------------------------
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

transformer_to_4326 = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)

p["_line_color"] = "red"
if classification_ready and "Classification" in p.columns:
    p["_line_color"] = p["Classification"].map(COLOR_MAP_CLASS).fillna("red")

if "Classification" in p.columns:
    custom_no_cls = p["_is_custom"] & p["Classification"].isna()
else:
    custom_no_cls = p["_is_custom"].copy()
p.loc[custom_no_cls, "_line_color"] = "#ff00ff"


def _build_tooltip_html(row):
    parts = []
    if bool(row.get("_is_custom", False)):
        parts.append("✏️ <b>CUSTOM WELL</b>")

    label = row.get("Label", "")
    if label:
        tag = "Section" if row.get("_label_is_section", False) else "Label"
        parts.append(f"<b>{tag}:</b> {label}")

    parts.append(f"Proximal Wells: {row.get('Proximal_Count', '—')}")

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

p_lines_4326 = gpd.GeoDataFrame({
    "_tooltip": p["_tooltip"].values,
    "_line_color": p["_line_color"].values,
    "_passes_filter": p["_passes_filter"].values,
    "_is_custom": p["_is_custom"].values,
    "geometry": p.geometry,
}, crs=p.crs).to_crs(4326)

buffer_gdf_display = gpd.GeoDataFrame({
    "_passes_filter": p["_passes_filter"].values,
    "_no_proximal": p["_no_proximal"].values,
    "_is_custom": p["_is_custom"].values,
    "geometry": buffer_geoms,
}, crs=p.crs).to_crs(4326)

inv_4326 = inv_gdf.to_crs(4326)

# Map center
bounds = p.total_bounds
cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
clon, clat = transformer_to_4326.transform(cx, cy)


def safe_geojson_obj(gdf, simplify_tol=30):
    """Return a GeoJSON dict (not string)."""
    try:
        gdf = gdf.copy()
        if simplify_tol is not None:
            gdf["geometry"] = gdf.geometry.simplify(simplify_tol, preserve_topology=True)
        gj = gdf.to_json()
        import json
        return json.loads(gj)
    except Exception:
        return None


@st.fragment
def render_map():
    payload = {
        "center": [float(clat), float(clon)],
        "lines_geo": safe_geojson_obj(_lines_gdf.to_crs(4326), 30),
        "points_geo": safe_geojson_obj(_points_gdf.to_crs(4326), 0),
        "inv_geo": safe_geojson_obj(inv_4326, 60),
        "land_geo": safe_geojson_obj(land_gdf.to_crs(4326), 100),
        "units_geo": safe_geojson_obj(units_gdf.to_crs(4326), 50),
        "section_geo": safe_geojson_obj(section_enriched_4326, 80),
        "buffers_geo": safe_geojson_obj(buffer_gdf_display, None),
        "prospects_geo": safe_geojson_obj(p_lines_4326, None),
    }

    import json
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <style>
        #map {{ width: 100%; height: 750px; border-radius: 8px; overflow: hidden; }}
      </style>
    </head>
    <body>
      <div id="map"></div>
      <script>
        const payload = {json.dumps(payload)};

        const map = L.map('map', {{ zoomControl: true }});
        const center = payload.center;

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}.png', {{
          attribution: '&copy; OpenStreetMap contributors'
        }}).addTo(map);

        map.setView(center, 11);

        const layers = {{
          existingWells: L.layerGroup(),
          inventory: L.layerGroup(),
          land: L.layerGroup(),
          units: L.layerGroup(),
          sectionGrid: L.layerGroup(),
          buffers: L.layerGroup(),
          prospects: L.layerGroup(),
        }};

        function addGeojsonToLayer(geo, layer, styleFn, onEachFeature=null) {{
          if (!geo) return;
          const gj = L.geoJSON(geo, {{
            style: styleFn,
            onEachFeature: onEachFeature
          }});
          gj.addTo(layer);
        }}

        // Existing wells
        addGeojsonToLayer(
          payload.lines_geo,
          layers.existingWells,
          function() {{ return {{ color: '#d62728', weight: 2, opacity: 0.9 }}; }}
        );
        addGeojsonToLayer(
          payload.points_geo,
          layers.existingWells,
          function() {{ return {{ color: '#1f77b4', weight: 1, opacity: 0.95, fillOpacity: 0.9 }}; }}
        );

        // Inventory
        addGeojsonToLayer(
          payload.inv_geo,
          layers.inventory,
          function() {{ return {{ color: '#999', weight: 1, fillOpacity: 0.06 }}; }}
        );

        // Land
        addGeojsonToLayer(
          payload.land_geo,
          layers.land,
          function() {{ return {{ fillColor: '#fff9c4', color: '#fff9c4', weight: 0.5, fillOpacity: 0.2 }}; }}
        );

        // Units
        addGeojsonToLayer(
          payload.units_geo,
          layers.units,
          function() {{ return {{ color: 'black', weight: 2, fillOpacity: 0 }}; }}
        );

        // Section grid
        addGeojsonToLayer(
          payload.section_geo,
          layers.sectionGrid,
          function() {{ return {{ fillOpacity: 0, color: '#999', weight: 0.3 }}; }}
        );

        // Buffers (dashed)
        addGeojsonToLayer(
          payload.buffers_geo,
          layers.buffers,
          function() {{
            return {{ color: '#000', weight: 1, opacity: 0.4, dashArray: '5,5', fillOpacity: 0 }};
          }}
        );

        // Prospects
        function prospectsStyle(feature) {{
          const p = feature.properties || {{}};
          return {{
            color: p._line_color || 'red',
            weight: (p._is_custom ? 5 : 3),
            opacity: 0.9
          }};
        }}

        addGeojsonToLayer(
          payload.prospects_geo,
          layers.prospects,
          prospectsStyle,
          function(feature, layer) {{
            const p = feature.properties || {{}};
            if (p._tooltip) {{
              layer.bindTooltip(p._tooltip, {{ sticky: true, direction: 'auto' }});
            }}
          }}
        );

        // Add defaults (match your earlier behavior)
        layers.existingWells.addTo(map);
        layers.inventory.addTo(map);
        layers.land.addTo(map);
        layers.units.addTo(map);
        layers.sectionGrid.addTo(map); // set off here if you want it hidden by default
        layers.buffers.addTo(map);     // set off here if you want it hidden by default
        layers.prospects.addTo(map);

        const overlayMaps = {{
          "Existing Wells": layers.existingWells,
          "Inventory": layers.inventory,
          "Land": layers.land,
          "Units": layers.units,
          "Section Grid": layers.sectionGrid,
          "Buffers": layers.buffers,
          "Prospects": layers.prospects,
        }};

        L.control.layers(null, overlayMaps, {{ collapsed: false }}).addTo(map);
      </script>
    </body>
    </html>
    """

    components.html(html, height=800, scrolling=False)


# -----------------------------
# Page Title + Metrics
# -----------------------------
st.title("🛢️ Inventory Classifier")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Total Prospects", n_total)
col_m2.metric("Passing Filters", n_passing)
col_m3.metric("Buffer Distance", f"{buffer_distance}m")
col_m4.metric(
    "Classified",
    int(prospects["Classification"].notna().sum()) if "Classification" in prospects.columns else 0
)

render_map()

# -----------------------------
# Classification charts
# -----------------------------
if classification_ready:
    st.markdown("---")
    st.subheader("Classification Distribution")
    cls_counts = prospects["Classification"].dropna().value_counts()
    if not cls_counts.empty:
        fig = go.Figure(go.Bar(
            x=cls_counts.index,
            y=cls_counts.values,
            marker_color=[COLOR_MAP_CLASS.get(c, "#888") for c in cls_counts.index],
            text=cls_counts.values,
            textposition="auto",
        ))
        fig.update_layout(
            xaxis_title="Classification",
            yaxis_title="Count",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(l=40, r=40, t=20, b=60),
            font=dict(size=12),
        )
        st.plotly_chart(fig, use_container_width=True)

    if "Productivity_Z" in prospects.columns and "Resource_Z" in prospects.columns:
        scatter_df = prospects.dropna(subset=["Productivity_Z", "Resource_Z"]).copy()
        if not scatter_df.empty:
            st.subheader("Productivity vs Resource")
            fig2 = go.Figure()
            for cls_name, color in COLOR_MAP_CLASS.items():
                subset = scatter_df[scatter_df.get("Classification") == cls_name]
                if not subset.empty:
                    fig2.add_trace(go.Scatter(
                        x=subset["Resource_Z"],
                        y=subset["Productivity_Z"],
                        mode="markers",
                        name=cls_name,
                        marker=dict(color=color, size=10, line=dict(width=1, color="#333")),
                        text=subset["Label"],
                        hovertemplate="<b>%{text}</b><br>Resource Z: %{x:.2f}<br>Prod Z: %{y:.2f}<extra></extra>",
                    ))
            fig2.add_hline(y=prod_threshold, line_dash="dash", line_color="#666", opacity=0.6)
            fig2.add_vline(x=resource_threshold, line_dash="dash", line_color="#666", opacity=0.6)
            fig2.update_layout(
                xaxis_title="Resource Z (σ)",
                yaxis_title="Productivity Z (σ)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=450,
                margin=dict(l=40, r=40, t=20, b=60),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Results table
# -----------------------------
st.markdown("---")
st.header("📋 Prospect Results")

p_display = p[p["_passes_filter"]].copy()
if p_display.empty:
    st.info("No prospects pass current filters.")
else:
    table_cols = [
        "Label", "_is_custom",
        "Heel Latitude", "Heel Longitude", "BH Latitude", "BH Longitude",
        SECTION_OOIP_COL, SECTION_ROIP_COL,
        "Norm EUR", "Norm 1Y Cuml", "Norm IP90", "WF", "FOOZ",
        "Productivity_Z", "Resource_Z", "Classification",
    ]
    table_cols = [c for c in table_cols if c in p_display.columns]

    df = p_display[table_cols].sort_values(
        "Proximal_Count" if "Proximal_Count" in p_display.columns else table_cols[0],
        ascending=False
    ).reset_index(drop=True)

    df.rename(columns={"_is_custom": "Custom Well"}, inplace=True)

    st.dataframe(df, use_container_width=True, height=400)
    st.download_button(
        "📥 Download CSV",
        data=df.to_csv(index=False),
        file_name="prospects_classified.csv",
        mime="text/csv",
    )

no_prox = p[p["_no_proximal"]].copy()
if not no_prox.empty:
    with st.expander(f"⚠️ {len(no_prox)} prospects with no proximal wells within {buffer_distance}m"):
        show_cols = [c for c in ["Label", "_prospect_type", "_is_custom", "Heel Latitude", "Heel Longitude", "BH Latitude", "BH Longitude"] if c in no_prox.columns]
        st.dataframe(
            no_prox[show_cols].rename(columns={"_prospect_type": "Type", "_is_custom": "Custom Well"}).reset_index(drop=True),
            use_container_width=True,
        )
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import MiniMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import branca.colormap as cm  # (kept, even if unused)
from shapely.geometry import Point, LineString
from pyproj import Transformer
from sklearn.linear_model import LinearRegression, RANSACRegressor
import string
import traceback

# ==========================================================
# Page config & Constants
# ==========================================================
st.set_page_config(layout="wide", page_title="Inventory Classifier", page_icon="🛢️")

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

TOOLTIP_STYLE = (
    "font-size:11px;padding:3px 6px;background:rgba(255,255,255,0.92);"
    "border:1px solid #333;border-radius:3px;"
)

# ==========================================================
# Helpers
# ==========================================================
def safe_range(series: pd.Series):
    vals = series.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        if lo == 0:
            return 0.0, 1.0
        return lo - abs(lo) * 0.1, lo + abs(lo) * 0.1
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


def startpoint_of_geom(geom):
    if geom is None or geom.is_empty:
        return None
    gt = geom.geom_type
    if gt == "LineString":
        return Point(geom.coords[0])
    if gt == "MultiLineString":
        return Point(geom.geoms[0].coords[0])
    if gt == "Point":
        return geom
    return None


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


def fit_trend(x: pd.Series, y: pd.Series):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    X = x[mask].values.reshape(-1, 1)
    Y = y[mask].values
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
    if (not hp) and hr:
        return "Low Prod / High Resource"
    if hp and (not hr):
        return "High Prod / Low Resource"
    return "Low Prod / Low Resource"


def fmt_val(col, v):
    if pd.isna(v):
        return "—"
    if col in SUM_COLS:
        try:
            return f"{int(v)}"
        except Exception:
            return "—"
    try:
        v = float(v)
    except Exception:
        return "—"
    return f"{v:,.0f}" if abs(v) > 100 else f"{v:.3f}"


def _alpha_combos(length: int):
    if length == 1:
        return list(string.ascii_uppercase)
    return [b + c for b in _alpha_combos(length - 1) for c in string.ascii_uppercase]


def _suffix_generator():
    n = 1
    while True:
        if n == 1:
            yield from string.ascii_uppercase
        else:
            for combo in _alpha_combos(n):
                yield combo
        n += 1


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
    wells, current = [], []
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
    return [w for w in wells if len(w) >= 2]


# ==========================================================
# Load data (cached)
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
    well_df["FOOZ"] = pd.to_numeric(well_df.get("FOOZ", np.nan), errors="coerce")

    section_df["Section"] = section_df["Section"].astype(str).str.strip()
    section_df[SECTION_OOIP_COL] = pd.to_numeric(section_df[SECTION_OOIP_COL], errors="coerce")
    section_df[SECTION_ROIP_COL] = pd.to_numeric(section_df[SECTION_ROIP_COL], errors="coerce")

    sec_numeric_cols = [
        c
        for c in section_df.columns
        if c != "Section" and pd.api.types.is_numeric_dtype(section_df[c])
    ]

    lines["UWI"] = lines["UWI"].astype(str).str.strip()
    points["UWI"] = points["UWI"].astype(str).str.strip()

    well_df_out = well_df.merge(
        section_df[["Section", SECTION_OOIP_COL, SECTION_ROIP_COL]],
        on="Section",
        how="left",
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
        geometry="geometry",
        crs=lines.crs,
    )

    proximal_wells = gpd.GeoDataFrame(
        existing_wells.merge(well_df_out, on="UWI", how="inner"),
        geometry="geometry",
        crs=existing_wells.crs,
    )
    proximal_wells["_midpoint"] = proximal_wells.geometry.apply(midpoint_of_geom)

    return (
        lines,
        points,
        grid,
        units,
        infills,
        lease_lines,
        merged,
        land,
        well_df_out,
        section_df,
        sec_numeric_cols,
        grid_enriched,
        section_4326,
        units_4326,
        land_4326,
        land_json,
        units_json,
        proximal_wells,
        well_df,  # keep original field data for trend fit later
    )


(
    lines_gdf,
    points_gdf,
    grid_gdf,
    units_gdf,
    infills_gdf,
    lease_lines_gdf,
    merged_gdf,
    land_gdf,
    well_df_out,
    section_df,
    SEC_NUMERIC_COLS,
    section_enriched,
    section_4326,
    units_4326,
    land_4326,
    land_json,
    units_json,
    proximal_wells,
    well_df_field,  # original for classification model
) = load_data()

LAYER_GDFS = {"Infill": infills_gdf, "Lease Line": lease_lines_gdf, "Merged": merged_gdf}

# ==========================================================
# Session state (custom wells)
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

show_layers = {
    "Infill": st.sidebar.checkbox("Unit Infills", value=True),
    "Lease Line": st.sidebar.checkbox("Unit Lease Lines", value=True),
    "Merged": st.sidebar.checkbox("Mosaic Merged Inventory", value=True),
}

st.sidebar.markdown("---")
st.sidebar.subheader("✏️ Custom Wells")
st.sidebar.caption("Paste lon/lat coordinates below. Each well is a polyline (>=2 points).")

st.sidebar.subheader("📌 Paste Coordinates (lon,lat)")
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
        tf_to_proj = Transformer.from_crs("EPSG:4326", "EPSG:26913", always_xy=True)
        for i, wcoords in enumerate(wells):
            coord_list = [(float(lon), float(lat)) for lon, lat in wcoords]
            key = _coords_key(coord_list)
            if key in st.session_state.drawn_coords_set:
                continue
            st.session_state.drawn_coords_set.add(key)
            st.session_state.drawn_wells.append({"coords": coord_list, "label": None})
        st.toast("Custom well(s) added.", icon="✅")
        st.rerun()

# Edit/delete custom wells
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
for name, enabled in show_layers.items():
    if enabled:
        f = LAYER_GDFS[name].copy()
        f["_prospect_type"] = name
        f["_is_custom"] = False
        prospect_frames.append(f)

if st.session_state.drawn_wells:
    tf_to_proj = Transformer.from_crs("EPSG:4326", "EPSG:26913", always_xy=True)
    custom_rows = []
    for i, cw in enumerate(st.session_state.drawn_wells):
        coords_proj = [tf_to_proj.transform(lon, lat) for lon, lat in cw["coords"]]
        if len(coords_proj) >= 2:
            custom_rows.append(
                {
                    "geometry": LineString(coords_proj),
                    "_prospect_type": "Custom",
                    "_is_custom": True,
                    "Label": cw.get("label") or f"Custom-{i+1}",
                }
            )
    if custom_rows:
        prospect_frames.append(gpd.GeoDataFrame(custom_rows, crs="EPSG:26913"))

if not prospect_frames:
    st.error("Enable at least one prospect layer.")
    st.stop()

prospects = gpd.GeoDataFrame(pd.concat(prospect_frames, ignore_index=True), geometry="geometry")
prospects.set_crs(infills_gdf.crs, inplace=True)

if "_is_custom" not in prospects.columns:
    prospects["_is_custom"] = False
prospects["_is_custom"] = prospects["_is_custom"].fillna(False).astype(bool)

# ==========================================================
# Label prospects
# ==========================================================
custom_mask_label = prospects["_is_custom"]

if "Label" not in prospects.columns:
    prospects["Label"] = np.nan
prospects["_label_is_section"] = False

# Prefer existing Label for non-custom if present; otherwise use UWI where available
if "UWI" in prospects.columns:
    non_custom_mask = ~custom_mask_label
    uwi_vals = (
        prospects.loc[non_custom_mask, "UWI"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    )
    # keep existing Label only if non-empty
    existing = prospects.loc[non_custom_mask, "Label"]
    keep = existing.notna() & (existing.astype(str).str.strip() != "")
    prospects.loc[non_custom_mask, "Label"] = np.where(keep, existing, uwi_vals)

unnamed_mask = prospects["Label"].isna() & ~custom_mask_label
if unnamed_mask.any():
    ep_series = prospects.loc[unnamed_mask, "geometry"].apply(endpoint_of_geom).dropna()
    if not ep_series.empty:
        ep_gdf = gpd.GeoDataFrame({"_pidx": ep_series.index, "geometry": ep_series.values}, crs=prospects.crs)

        joined = gpd.sjoin(
            ep_gdf,
            grid_gdf[["Section", "geometry"]],
            how="left",
            predicate="within",
        )

        # fallback intersects
        still_missing = joined["Section"].isna()
        if still_missing.any():
            missing_ep = ep_gdf.loc[still_missing[still_missing].index]
            if not missing_ep.empty:
                fallback = gpd.sjoin(
                    missing_ep,
                    grid_gdf[["Section", "geometry"]],
                    how="left",
                    predicate="intersects",
                )
                fb_first = fallback.dropna(subset=["Section"]).groupby("_pidx").first()
                for pidx, row in fb_first.iterrows():
                    joined.loc[joined["_pidx"] == pidx, "Section"] = row["Section"]

        valid_rows = joined.dropna(subset=["Section"])
        for _, row in valid_rows.iterrows():
            idx = row["_pidx"]
            prospects.at[idx, "Label"] = str(row["Section"]).strip()
            prospects.at[idx, "_label_is_section"] = True

        # disambiguate duplicate section labels
        section_labeled = prospects[prospects["_label_is_section"]]
        dupe_sections = section_labeled["Label"].value_counts()
        dupe_sections = dupe_sections[dupe_sections > 1].index.tolist()
        for section_name in dupe_sections:
            idxs = section_labeled[section_labeled["Label"] == section_name].index
            suffix_gen = _suffix_generator()
            for pidx in idxs:
                prospects.at[pidx, "Label"] = f"{section_name}-{next(suffix_gen)}"

prospects["Label"] = prospects["Label"].fillna("").astype(str)

# ==========================================================
# Analyse prospects
# ==========================================================
def idw_for_column(hits: pd.DataFrame, col: str, pros_index: pd.Index):
    valid = hits.loc[hits[col].notna() & hits["_w"].notna()]
    if valid.empty:
        return pd.Series(np.nan, index=pros_index)

    wv = valid[col] * valid["_w"]
    g = (
        pd.DataFrame({"_wv": wv, "_w": valid["_w"], "ir": valid["index_right"]})
        .groupby("ir")
        .sum()
    )
    out = (g["_wv"] / g["_w"]).reindex(pros_index)
    return out


def analyze_prospects(pros: gpd.GeoDataFrame, prox: gpd.GeoDataFrame, sections: gpd.GeoDataFrame, buffer_m: float):
    pros = pros.copy()
    pros["_midpoint"] = pros.geometry.apply(midpoint_of_geom)
    pros["_buffer"] = pros.geometry.buffer(buffer_m, cap_style=2)

    buffer_gdf = gpd.GeoDataFrame({"_pidx": pros.index, "geometry": pros["_buffer"]}, crs=pros.crs)
    midpt_gdf = prox.loc[prox["_midpoint"].notna()].copy()
    midpt_gdf = midpt_gdf.set_geometry(gpd.GeoSeries(midpt_gdf["_midpoint"], crs=pros.crs))

    # wells whose midpoints fall within prospect buffer
    well_hits = gpd.sjoin(midpt_gdf, buffer_gdf, how="inner", predicate="within")

    # Assign hit point (prospect midpoint) for each index_right
    px_mp = well_hits["index_right"].map(pros["_midpoint"])
    hit_x = well_hits["_midpoint"].apply(lambda pt: pt.x)
    hit_y = well_hits["_midpoint"].apply(lambda pt: pt.y)
    px_x = px_mp.apply(lambda pt: pt.x if pt else np.nan)
    px_y = px_mp.apply(lambda pt: pt.y if pt else np.nan)

    dist = np.sqrt((hit_x - px_x) ** 2 + (hit_y - px_y) ** 2).replace(0, 1.0)
    well_hits["_dist"] = dist
    well_hits["_w"] = 1.0 / (dist**2)

    # IDW for well metrics
    out = pd.DataFrame(index=pros.index)
    out["_prospect_type"] = pros["_prospect_type"].values
    out["Proximal_Count"] = well_hits.groupby("index_right").size().reindex(pros.index, fill_value=0).values

    proximal_uwis = (
        well_hits.groupby("index_right")["UWI"].apply(lambda x: ", ".join(x.astype(str))).reindex(pros.index, fill_value="")
    )
    out["_proximal_uwis"] = proximal_uwis.values

    for col in WELL_COLS:
        out[col] = idw_for_column(well_hits, col, pros.index).values

    # Sum metrics intersect buffers
    for sc in SUM_COLS:
        if sc not in prox.columns:
            out[sc] = 0.0
            continue

        sc_wells = prox.loc[prox[sc].notna(), ["UWI", sc, "geometry"]].copy()
        if sc_wells.empty:
            out[sc] = 0.0
            continue

        sc_hits = gpd.sjoin(sc_wells, buffer_gdf, how="inner", predicate="intersects")
        out[sc] = sc_hits.groupby("index_right")[sc].sum().reindex(pros.index, fill_value=0).values

    # Section means intersect buffers
    sec_join = gpd.sjoin(
        sections[["geometry", SECTION_OOIP_COL, SECTION_ROIP_COL]],
        buffer_gdf,
        how="inner",
        predicate="intersects",
    )
    out[SECTION_OOIP_COL] = sec_join.groupby("index_right")[SECTION_OOIP_COL].mean().reindex(pros.index).values
    out[SECTION_ROIP_COL] = sec_join.groupby("index_right")[SECTION_ROIP_COL].mean().reindex(pros.index).values

    return out


prospect_metrics = analyze_prospects(prospects, proximal_wells, section_enriched, buffer_distance)
for c in prospect_metrics.columns:
    prospects[c] = prospect_metrics[c].values

# Clean inf
for col in ALL_METRIC_COLS:
    if col in prospects.columns:
        prospects[col] = prospects[col].replace([np.inf, -np.inf], np.nan)

# ==========================================================
# Coordinates (to 4326)
# ==========================================================
tf_to_4326 = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)

endpoints = prospects.geometry.apply(endpoint_of_geom)
ep_x = endpoints.apply(lambda pt: pt.x if pt else np.nan)
ep_y = endpoints.apply(lambda pt: pt.y if pt else np.nan)

valid_ep = ep_x.notna()
lon_bh = np.full(len(prospects), np.nan)
lat_bh = np.full(len(prospects), np.nan)
if valid_ep.any():
    lon_bh[valid_ep.values] = np.round(tf_to_4326.transform(ep_x[valid_ep].values, ep_y[valid_ep].values)[0], 6)
    lat_bh[valid_ep.values] = np.round(tf_to_4326.transform(ep_x[valid_ep].values, ep_y[valid_ep].values)[1], 6)

prospects["BH Latitude"] = lat_bh
prospects["BH Longitude"] = lon_bh

startpoints = prospects.geometry.apply(startpoint_of_geom)
sp_x = startpoints.apply(lambda pt: pt.x if pt else np.nan)
sp_y = startpoints.apply(lambda pt: pt.y if pt else np.nan)

valid_sp = sp_x.notna()
lon_heel = np.full(len(prospects), np.nan)
lat_heel = np.full(len(prospects), np.nan)
if valid_sp.any():
    lon_heel[valid_sp.values] = np.round(tf_to_4326.transform(sp_x[valid_sp].values, sp_y[valid_sp].values)[0], 6)
    lat_heel[valid_sp.values] = np.round(tf_to_4326.transform(sp_x[valid_sp].values, sp_y[valid_sp].values)[1], 6)

prospects["Heel Latitude"] = lat_heel
prospects["Heel Longitude"] = lon_heel

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

field = pd.DataFrame()
prod_threshold = resource_threshold = None

if cw_sum != 100:
    st.sidebar.error(f"Weights sum to {cw_sum}%, must be 100%")
else:
    prod_threshold = st.sidebar.slider("Productivity Z threshold (σ)", -1.0, 2.0, 0.0, 0.05, key="prod_thresh")
    resource_threshold = st.sidebar.slider("Resource Z threshold (σ)", -1.0, 2.0, 0.0, 0.05, key="res_thresh")

    field = well_df_field.dropna(subset=[SECTION_ROIP_COL] + WELL_COLS).copy()
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
                resid = field[src] - model.predict(field[SECTION_ROIP_COL].values.reshape(-1, 1))
                field[f"{tag}_resid"] = resid
                resid_std[tag] = float(field[f"{tag}_resid"].std())

            field_roip_mean = float(field[SECTION_ROIP_COL].mean())
            field_roip_std = float(field[SECTION_ROIP_COL].std())

            pros_cls = prospects.dropna(subset=[SECTION_ROIP_COL] + WELL_COLS).copy()
            pros_cls = pros_cls[pros_cls[SECTION_ROIP_COL] > 0]
            if not pros_cls.empty:
                roip_vals = pros_cls[SECTION_ROIP_COL].values.reshape(-1, 1)
                pros_cls["EUR_pred"] = eur_model.predict(roip_vals)
                pros_cls["IP90_pred"] = ip90_model.predict(roip_vals)
                pros_cls["Y1_pred"] = y1_model.predict(roip_vals)

                pros_cls["Z_EUR"] = (pros_cls["Norm EUR"] - pros_cls["EUR_pred"]) / resid_std["EUR"] if resid_std["EUR"] > 0 else 0
                pros_cls["Z_IP90"] = (pros_cls["Norm IP90"] - pros_cls["IP90_pred"]) / resid_std["IP90"] if resid_std["IP90"] > 0 else 0
                pros_cls["Z_1Y"] = (pros_cls["Norm 1Y Cuml"] - pros_cls["Y1_pred"]) / resid_std["Y1"] if resid_std["Y1"] > 0 else 0

                pros_cls["Productivity_Z"] = (
                    (cw_eur / 100) * pros_cls["Z_EUR"] +
                    (cw_1y / 100) * pros_cls["Z_1Y"] +
                    (cw_ip90 / 100) * pros_cls["Z_IP90"]
                )

                pros_cls["Resource_Z"] = (
                    (pros_cls[SECTION_ROIP_COL] - field_roip_mean) / field_roip_std if field_roip_std > 0 else 0.0
                )

                pros_cls["Classification"] = pros_cls.apply(
                    lambda r: classify_quadrant(r["Productivity_Z"], r["Resource_Z"], prod_threshold, resource_threshold),
                    axis=1,
                )

                for col in ["Classification", "Productivity_Z", "Resource_Z", "Z_EUR", "Z_IP90", "Z_1Y"]:
                    prospects[col] = np.nan
                    prospects.loc[pros_cls.index, col] = pros_cls[col]

                classification_ready = True

# ==========================================================
# Filters
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

p = prospects

has_proximal = p["Proximal_Count"].fillna(0) > 0
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
n_custom = int(p["_is_custom"].sum())

# ==========================================================
# Prepare display
# ==========================================================
existing_display = proximal_wells.to_crs(4326)

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
            tag = "Section" if bool(row.get("_label_is_section", False)) else "UWI"
            parts.append(f"<b>{tag}:</b> {label}")

    pc = row.get("Proximal_Count", "—")
    parts.append(f"Proximal Wells: {pc}")

    for col in ALL_METRIC_COLS:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"{col}: {fmt_val(col, row[col])}")

    cls = row.get("Classification", None)
    if pd.notna(cls):
        parts.append(f"<b>Class:</b> {cls}")

    if pd.notna(row.get("Productivity_Z", np.nan)):
        parts.append(f"Prod Z: {row.get('Productivity_Z'):.2f}")
    if pd.notna(row.get("Resource_Z", np.nan)):
        parts.append(f"Resource Z: {row.get('Resource_Z'):.2f}")

    return "<br>".join(parts) if parts else "Prospect"


p["_tooltip"] = p.apply(_build_tooltip_html, axis=1)

if classification_ready and "Classification" in p.columns:
    p["_line_color"] = p["Classification"].map(COLOR_MAP_CLASS).fillna("red")
else:
    p["_line_color"] = "red"

custom_no_cls = p["_is_custom"] & (p.get("Classification").isna() if "Classification" in p.columns else True)
p.loc[custom_no_cls, "_line_color"] = "#ff00ff"

buffer_geoms = p.geometry.buffer(buffer_distance, cap_style=2)

# 4326 layers for Folium
buffer_gdf = gpd.GeoDataFrame(
    {
        "_passes_filter": p["_passes_filter"].values,
        "_no_proximal": p["_no_proximal"].values,
        "_is_custom": p["_is_custom"].values,
        "geometry": buffer_geoms,
    },
    crs=p.crs,
).to_crs(4326)

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

# ==========================================================
# Title
# ==========================================================
st.title("🛢️ Inventory Classifier")
st.caption(
    f"**{n_passing}** / {n_total} prospects pass filters · Buffer: {buffer_distance}m"
    + (f" · ✏️ {n_custom} custom well(s)" if n_custom else "")
)

# ==========================================================
# MAP (JSON-safe, fixes point/serialization)
# ==========================================================
try:
    if len(p) > 0 and np.isfinite(p.total_bounds).all():
        bounds = p.total_bounds
        cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        tf_to_4326_center = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)
        clon, clat = tf_to_4326_center.transform(cx, cy)
    else:
        clon, clat = -103.2, 47.5
except Exception:
    clon, clat = -103.2, 47.5

map_key = f"map_{section_gradient}_{buffer_distance}_{n_custom}"

@st.cache_resource(show_spinner=False)
def build_folium_map_base(section_gradient_local: str, land_json_local: str, units_json_local: str, section_4326_local_json: str, existing_display_local_json: str):
    m = folium.Map(location=[clat, clon], zoom_start=11, tiles="CartoDB positron", prefer_canvas=True)
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)

    land_fg = folium.FeatureGroup(name="Bakken Land", show=True)
    folium.GeoJson(
        land_json_local,
        style_function=lambda _: {"fillColor": "#fff9c4", "color": "#fff9c4", "weight": 0.5, "fillOpacity": 0.2},
    ).add_to(land_fg)
    land_fg.add_to(m)

    units_fg = folium.FeatureGroup(name="Units", show=True)
    folium.GeoJson(
        units_json_local,
        style_function=lambda _: {"color": "black", "weight": 2, "fillOpacity": 0, "interactive": False},
    ).add_to(units_fg)
    units_fg.add_to(m)

    section_fg_show = (section_gradient_local != "None")
    section_fg = folium.FeatureGroup(name="Section Grid", show=section_fg_show)

    sec_style = lambda _: NULL_STYLE
    if section_fg_show:
        sec_style = lambda _: {"fillOpacity": 0.2, "color": "white", "weight": 0.3}

    sec_tip_fields = [c for c in section_4326_local_json if False]  # placeholder, overridden below

    # NOTE: section_4326 is a GeoDataFrame; for tooltip fields we use the local df keys
    # Keep simple: tooltip for everything except geometry via the already-serialized json layer.
    section_df_fields = [c for c in section_4326.columns if c != "geometry"]

    folium.GeoJson(
        section_4326_local_json,
        style_function=sec_style,
        tooltip=folium.GeoJsonTooltip(
            fields=section_df_fields,
            aliases=[f"{f}:" for f in section_df_fields],
            localize=True,
            sticky=True,
            style=TOOLTIP_STYLE,
        ),
    ).add_to(section_fg)

    section_fg.add_to(m)

    well_fg = folium.FeatureGroup(name="Existing Wells")
    folium.GeoJson(
        existing_display_local_json,
        style_function=lambda _: {"color": "black", "weight": 0.5, "opacity": 0.8},
    ).add_to(well_fg)
    well_fg.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    return m


section_4326_json = section_4326.to_json()
existing_display_json = existing_display.to_json()

try:
    base_map = build_folium_map_base(
        section_gradient,
        land_json,
        units_json,
        section_4326_json,
        existing_display_json,
    )
except Exception as e:
    st.error(f"Base map build failed: {e}")
    st.text(traceback.format_exc())
    base_map = folium.Map(location=[clat, clon], zoom_start=11, tiles="CartoDB positron")

# Dynamic layers
m = base_map
try:
    # Prospect buffers
    buf_fg = folium.FeatureGroup(name="Prospect Buffers")
    if len(buffer_gdf) > 0:
        buffer_gdf2 = buffer_gdf.copy()
        buffer_gdf2["_bstyle"] = "fail"
        buffer_gdf2.loc[buffer_gdf2["_passes_filter"], "_bstyle"] = "pass"
        buffer_gdf2.loc[buffer_gdf2["_no_proximal"], "_bstyle"] = "noprox"
        buffer_gdf2.loc[buffer_gdf2["_is_custom"], "_bstyle"] = "custom"

        _BSTYLES = {
            "pass": {"fillOpacity": 0, "color": "#000", "weight": 1.2, "opacity": 0.6, "dashArray": "6 4"},
            "fail": {"fillOpacity": 0, "color": "#000", "weight": 0.8, "opacity": 0.25, "dashArray": "6 4"},
            "noprox": {"fillOpacity": 0, "color": "#000", "weight": 0.8, "opacity": 0.3, "dashArray": "4 6"},
            "custom": {"fillOpacity": 0.04, "fillColor": "#ff00ff", "color": "#ff00ff", "weight": 1.5, "opacity": 0.7, "dashArray": "4 4"},
        }

        # Only keep what we need for styling
        buffer_gdf2 = buffer_gdf2[["_bstyle", "geometry"]]

        gj_json = buffer_gdf2.to_json()
        def style_function(feat):
            key = feat["properties"].get("_bstyle", "fail")
            return _BSTYLES.get(key, _BSTYLES["fail"])

        folium.GeoJson(gj_json, style_function=style_function).add_to(buf_fg)
    buf_fg.add_to(m)

    # Prospect wells (lines)
    prospect_fg = folium.FeatureGroup(name="Prospect Wells", show=True)

    if not p_lines_4326.empty:
        g = p_lines_4326.copy()
        g["tooltip"] = g["_tooltip"].astype(str)
        g["line_color"] = g["_line_color"].astype(str)
        g["is_custom"] = g["_is_custom"].fillna(False).astype(bool)
        g = g[["tooltip", "line_color", "is_custom", "geometry"]]

        gj_json = g.to_json()

        def style_function_lines(feat):
            lc = feat["properties"].get("line_color", "red")
            is_custom = feat["properties"].get("is_custom", False)
            w = 5 if is_custom else 3
            return {"color": lc, "weight": w, "opacity": 0.9}

        folium.GeoJson(
            gj_json,
            style_function=style_function_lines,
            highlight_function=lambda _: {"weight": 7, "color": "#ff4444"},
            tooltip=folium.GeoJsonTooltip(
                fields=["tooltip"],
                aliases=[""],
                localize=False,
                sticky=True,
                style="font-size:12px",
            ),
        ).add_to(prospect_fg)

        # Custom heel/toe markers (avoid JSON issues: use python coords)
        for _, row in p_lines_4326.iterrows():
            if not bool(row.get("_is_custom", False)):
                continue
            coords = list(row.geometry.coords) if row.geometry is not None else []
            if len(coords) >= 2:
                lc = row["_line_color"]
                tip = str(row["_tooltip"])
                # folium expects [lat, lon]
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

    prospect_fg.add_to(m)

except Exception as e:
    st.error(f"Dynamic map layer build failed: {e}")
    st.text(traceback.format_exc())

try:
    st_folium(m, use_container_width=True, height=800, returned_objects=[], key=map_key)
except Exception as e:
    st.error(f"st_folium failed: {e}")
    st.text(traceback.format_exc())

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
                    f"<span style='color:{cls_color};font-size:1.2em;font-weight:bold'>{cls_str}</span>",
                    unsafe_allow_html=True,
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

    pros_chart = p[p["_passes_filter"] & p["Classification"].notna()].copy()

    if not pros_chart.empty:
        col1, col2 = st.columns(2)

        x_range = np.linspace(field[SECTION_ROIP_COL].min(), field[SECTION_ROIP_COL].max(), 100)

        # EUR / Y1 / IP90 models
        for y_col, model, target_col in [
            ("Norm EUR", eur_model, col1),
            ("Norm 1Y Cuml", y1_model, col2),
            ("Norm IP90", ip90_model, col1),
        ]:
            with target_col:
                chart_df = pros_chart.copy()
                chart_df["_symbol"] = chart_df["_is_custom"].map({True: "Custom", False: "Prospect"})

                fig = px.scatter(
                    chart_df,
                    x=SECTION_ROIP_COL,
                    y=y_col,
                    color="Classification",
                    color_discrete_map=COLOR_MAP_CLASS,
                    symbol="_symbol",
                    symbol_map={"Prospect": "circle", "Custom": "star"},
                    hover_data=["Label"],
                    title=f"{y_col} vs {SECTION_ROIP_COL}",
                )

                fig.add_trace(
                    go.Scatter(
                        x=field[SECTION_ROIP_COL],
                        y=field[y_col],
                        mode="markers",
                        name="Field UWIs",
                        marker=dict(color="lightgrey", size=4, opacity=0.5),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=model.predict(x_range.reshape(-1, 1)),
                        mode="lines",
                        name="Trend",
                        line=dict(color="black", dash="dash"),
                    )
                )

                # make custom markers bigger
                for trace in fig.data:
                    if hasattr(trace, "marker") and getattr(trace.marker, "symbol", None) == "star":
                        trace.marker.size = 14

                st.plotly_chart(fig, use_container_width=True)

        # Quadrant
        with col2:
            chart_df = pros_chart.copy()
            chart_df["_symbol"] = chart_df["_is_custom"].map({True: "Custom", False: "Prospect"})

            fig_quad = px.scatter(
                chart_df,
                x="Resource_Z",
                y="Productivity_Z",
                color="Classification",
                color_discrete_map=COLOR_MAP_CLASS,
                symbol="_symbol",
                symbol_map={"Prospect": "circle", "Custom": "star"},
                hover_data=["Label"],
                title="Productivity Z vs Resource Z",
                labels={
                    "Resource_Z": f"Resource Z ({SECTION_ROIP_COL})",
                    "Productivity_Z": "Productivity Z (Composite)",
                },
            )

            for trace in fig_quad.data:
                if hasattr(trace, "marker") and getattr(trace.marker, "symbol", None) == "star":
                    trace.marker.size = 14

            rx_min = min(pros_chart["Resource_Z"].min(), -2) - 0.5
            rx_max = max(pros_chart["Resource_Z"].max(), 2) + 0.5
            ry_min = min(pros_chart["Productivity_Z"].min(), -2) - 0.5
            ry_max = max(pros_chart["Productivity_Z"].max(), 2) + 0.5

            for rect in [
                dict(x0=resource_threshold, x1=rx_max, y0=prod_threshold, y1=ry_max, fillcolor=COLOR_MAP_CLASS["High Prod / High Resource"], opacity=0.07),
                dict(x0=resource_threshold, x1=rx_max, y0=ry_min, y1=prod_threshold, fillcolor=COLOR_MAP_CLASS["Low Prod / High Resource"], opacity=0.07),
                dict(x0=rx_min, x1=resource_threshold, y0=prod_threshold, y1=ry_max, fillcolor=COLOR_MAP_CLASS["High Prod / Low Resource"], opacity=0.07),
                dict(x0=rx_min, x1=resource_threshold, y0=ry_min, y1=prod_threshold, fillcolor=COLOR_MAP_CLASS["Low Prod / Low Resource"], opacity=0.07),
            ]:
                fig_quad.add_shape(
                    type="rect",
                    xref="x",
                    yref="y",
                    layer="below",
                    line=dict(width=0),
                    **rect,
                )

            fig_quad.add_hline(y=prod_threshold, line_dash="dot", line_color="grey", annotation_text=f"Prod σ = {prod_threshold}")
            fig_quad.add_vline(x=resource_threshold, line_dash="dot", line_color="grey", annotation_text=f"Resource σ = {resource_threshold}")
            fig_quad.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.5)
            fig_quad.add_vline(x=0, line_dash="dash", line_color="black", line_width=0.5)

            st.plotly_chart(fig_quad, use_container_width=True)

        table_cols = [
            "Label",
            "_is_custom",
            "Heel Latitude",
            "Heel Longitude",
            "BH Latitude",
            "BH Longitude",
            SECTION_OOIP_COL,
            SECTION_ROIP_COL,
            "Norm EUR",
            "Norm 1Y Cuml",
            "Norm IP90",
        ] + SUM_COLS + ["Z_EUR", "Z_IP90", "Z_1Y", "Productivity_Z", "Resource_Z", "Classification"]
        table_cols = [c for c in table_cols if c in pros_chart.columns]

        cls_display = pros_chart[table_cols].sort_values("Productivity_Z", ascending=False).reset_index(drop=True)
        cls_display.rename(columns={"_is_custom": "Custom Well"}, inplace=True)

        st.dataframe(cls_display, use_container_width=True)
        st.download_button(
            "📥 Download CSV",
            data=cls_display.to_csv(index=False),
            file_name="classified_prospects.csv",
            mime="text/csv",
        )

# ==========================================================
# No-proximal
# ==========================================================
no_prox = p[p["_no_proximal"]].copy()
if not no_prox.empty:
    st.markdown("---")
    with st.expander(f"⚠️ {len(no_prox)} prospects with no proximal wells within {buffer_distance}m"):
        st.dataframe(
            no_prox[
                ["Label", "_prospect_type", "_is_custom", "Heel Latitude", "Heel Longitude", "BH Latitude", "BH Longitude"]
            ].rename(columns={"_prospect_type": "Type", "_is_custom": "Custom Well"}).reset_index(drop=True),
            use_container_width=True,
        )
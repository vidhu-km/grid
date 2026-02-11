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
from plotly.subplots import make_subplots
from pyproj import Transformer

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
CONFIDENCE_HIGH = 8

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


def confidence_tier(count):
    if count >= 8:
        return "🟢 High"
    elif count >= 4:
        return "🟡 Medium"
    elif count >= 2:
        return "🟠 Low"
    else:
        return "🔴 Insufficient"


# ==========================================================
# Data loading (cached)
# ==========================================================

@st.cache_resource(show_spinner="Loading spatial data …")
def load_data():
    lines = gpd.read_file("lines.shp")
    points = gpd.read_file("points.shp")
    grid = gpd.read_file("ooipsectiongrid.shp")
    infills = gpd.read_file("2M_Infills_plyln.shp")
    lease_lines = gpd.read_file("2M_LL_plyln.shp")
    units = gpd.read_file("Bakken Units.shp")

    prod_in = pd.read_excel("well.xlsx", sheet_name="inunit")
    prod_out = pd.read_excel("well.xlsx", sheet_name="outunit")

    all_gdfs = [lines, points, grid, units, infills, lease_lines]
    for gdf in all_gdfs:
        if gdf.crs is None:
            gdf.set_crs(epsg=26913, inplace=True)
        gdf.to_crs(epsg=26913, inplace=True)

    grid["Section"] = grid["Section"].astype(str).str.strip()
    grid["OOIP"] = pd.to_numeric(grid["OOIP"], errors="coerce")

    PROD_NUMERIC = ["Cuml", "EUR", "IP90", "1YCuml", "Wcut"]
    for df in [prod_in, prod_out]:
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

    return lines, points, grid, units, infills, lease_lines, prod_in, prod_out


(
    lines_gdf, points_gdf, grid_gdf, units_gdf,
    infills_gdf, lease_lines_gdf, prod_in_df, prod_out_df,
) = load_data()

# ==========================================================
# Sidebar — proximal Well Source
# ==========================================================
st.sidebar.title("Map Settings")

st.sidebar.subheader("📂 proximal Well Source")
show_in_unit = st.sidebar.checkbox("In-Unit Wells", value=True)
show_out_unit = st.sidebar.checkbox("Out-of-Unit Wells", value=False)

if not show_in_unit and not show_out_unit:
    st.sidebar.error("Select at least one well source.")
    st.stop()

st.sidebar.subheader("🎯 Prospect Type")
show_infills = st.sidebar.checkbox("2M Infills", value=True)
show_lease_lines = st.sidebar.checkbox("2M Lease Lines", value=False)
if not show_infills and not show_lease_lines:
    st.sidebar.error("Select at least one prospect type.")
    st.stop()

# ==========================================================
# Build the proximal well pool
# ==========================================================
frames = []
if show_in_unit:
    frames.append(prod_in_df)
if show_out_unit:
    frames.append(prod_out_df)

prod_pool = pd.concat(frames, ignore_index=True)
prod_pool = prod_pool.drop_duplicates(subset="UWI", keep="first")

lines_with_uwi = lines_gdf[["UWI", "geometry"]].copy()
points_with_uwi = points_gdf[["UWI", "geometry"]].copy()
points_only = points_with_uwi[~points_with_uwi["UWI"].isin(lines_with_uwi["UWI"])]
existing_wells = pd.concat([lines_with_uwi, those_mids := points_only], ignore_index=True)
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
# PROSPECT ANALYSIS (DEFAULT MODE)
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

prospects = pd.concat(prospect_frames, ignore_index=True)
prospects = gpd.GeoDataFrame(prospects, geometry="geometry", crs=infills_gdf.crs)

# ---- Prospect analysis with IDW ----
@st.cache_data(show_spinner="Analysing prospects (IDW²) …")
def analyze_prospects_idw(_prospects, _proximal_wells, _section_enriched, _buffer_m):
    pros = _prospects.copy()
    proximal = _proximal_wells.copy()
    sections = _section_enriched.copy()

    results = []
    for idx, prospect in pros.iterrows():
        geom = prospect.geometry
        record = {"_idx": idx, "_prospect_type": prospect["_prospect_type"]}

        prospect_mid = midpoint_of_geom(geom)
        if prospect_mid is None:
            for col in WELL_LEVEL_METRICS + SECTION_LEVEL_METRICS:
                record[col] = np.nan
            record["proximal_Count"] = 0
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
        record["_section_label"] = str(sec_hit.iloc[0]["Section"]) if not sec_hit.empty else "Unknown"

        buffer_geom = geom.buffer(_buffer_m)
        buffer_gdf = gpd.GeoDataFrame([{"geometry": buffer_geom}], crs=pros.crs)
        hits = gpd.sjoin(proximal, buffer_gdf, how="inner", predicate="intersects")

        record["proximal_Count"] = len(hits)
        record["_proximal_uwis"] = ",".join(hits["UWI"].tolist()) if len(hits) > 0 else ""

        if len(hits) > 0:
            hit_mids = hits["_midpoint"].apply(lambda mp: prospect_mid.distance(mp) if mp is not None else np.nan).replace(0, 1.0)
            weights = (1.0 / (hit_mids ** 2)).replace([np.inf, -np.inf], np.nan)
            valid_w = weights.dropna()

            if valid_w.sum() > 0:
                for col in WELL_LEVEL_METRICS:
                    col_vals = hits.loc[valid_w.index, col]
                    mask = col_vals.notna() & valid_w.notna()
                    record[col] = (col_vals[mask] * valid_w[mask]).sum() / valid_w[mask].sum() if mask.sum() > 0 else np.nan
            else:
                for col in WELL_LEVEL_METRICS:
                    record[col] = hits[col].mean()

            record["EUR_median"] = hits["EUR"].median()
            record["EUR_p10"] = hits["EUR"].quantile(0.9)
            record["EUR_p90"] = hits["EUR"].quantile(0.1)
        else:
            for col in WELL_LEVEL_METRICS: record[col] = np.nan
            record["EUR_median"] = record["EUR_p10"] = record["EUR_p90"] = np.nan

        overlaps = gpd.overlay(sections[["Section", "OOIP", "RFTD", "URF", "geometry"]], gpd.GeoDataFrame(geometry=[buffer_geom], crs=pros.crs), how="intersection")
        for col in SECTION_LEVEL_METRICS:
            record[col] = overlaps[col].dropna().mean() if not overlaps.empty else np.nan

        results.append(record)
    return pd.DataFrame(results).set_index("_idx")

prospect_metrics = analyze_prospects_idw(prospects, proximal_wells, section_enriched, buffer_distance)
prospects = prospects.join(prospect_metrics.drop(columns=["_prospect_type"], errors="ignore"))
prospects["Label"] = prospects["_section_label"]
prospects["Confidence"] = prospects["proximal_Count"].fillna(0).apply(confidence_tier)

# ---- Filters ----
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Prospect Filters")
p = prospects.copy()

f_wcut = st.sidebar.slider("Max Water Cut (%)", *safe_range(p["Wcut"]), float(safe_range(p["Wcut"])[1]), step=1.0)
f_ooip = st.sidebar.slider("Min OOIP (bbl)", *safe_range(p["OOIP"]), float(safe_range(p["OOIP"])[0]), step=1.0)
f_eur = st.sidebar.slider("Min EUR (bbl)", *safe_range(p["EUR"]), float(safe_range(p["EUR"])[0]), step=1.0)
f_ip90 = st.sidebar.slider("Min IP90 (bbl/d)", *safe_range(p["IP90"]), float(safe_range(p["IP90"])[0]), step=1.0)

filter_mask = (p["proximal_Count"] > 0) & \
              ((p["Wcut"] <= f_wcut) | p["Wcut"].isna()) & \
              ((p["OOIP"] >= f_ooip) | p["OOIP"].isna()) & \
              ((p["EUR"] >= f_eur) | p["EUR"].isna()) & \
              ((p["IP90"] >= f_ip90) | p["IP90"].isna())

p["_passes_filter"] = filter_mask
n_passing = int(filter_mask.sum())
st.sidebar.markdown(f"**{n_passing}** / {len(p)} prospects pass filters")

# ---- Ranking ----
selected_metric = st.sidebar.selectbox("Rank prospects by", ["EUR", "IP90", "1YCuml", "Wcut", "OOIP", "URF", "RFTD"])
ascending = selected_metric in ["Wcut", "URF", "RFTD"]

# ---- Display ----
st.title("🛢️ Bakken Prospect Analyzer")
col_map, col_rank = st.columns([7, 4])

with col_map:
    # Build Map
    bounds = p.total_bounds
    transformer = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)
    centre_lon, centre_lat = transformer.transform((bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2)
    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=11, tiles="CartoDB positron")
    
    # Sections
    if section_gradient != "None":
        grad_col = GRADIENT_COL_MAP[section_gradient]
        grad_vals = section_enriched[grad_col].dropna()
        if not grad_vals.empty:
            colormap = cm.LinearColormap(colors=["#ffffcc", "#006837"], vmin=grad_vals.min(), vmax=grad_vals.max()).to_step(n=7)
            m.add_child(colormap)
            folium.GeoJson(section_enriched.to_crs(4326), style_function=lambda f: {"fillColor": colormap(f['properties'][grad_col]) if f['properties'][grad_col] else "#ffffff", "fillOpacity": 0.4, "weight": 0.3}).add_to(m)

    # Prospect Lines
    folium.GeoJson(p.to_crs(4326), style_function=lambda _: {"color": "red", "weight": 3}).add_to(m)
    st_folium(m, use_container_width=True, height=700)

with col_rank:
    st.header("📊 Prospect Ranking")
    rank_df = p[p["_passes_filter"]].sort_values(selected_metric, ascending=ascending)
    st.dataframe(rank_df[["Label", "Confidence", selected_metric]], use_container_width=True)
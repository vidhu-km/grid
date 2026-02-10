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

NULL_STYLE = {
    "fillColor": "#ffffff",
    "fillOpacity": 0,
    "color": "#888",
    "weight": 0.25,
}

FILTERED_OUT_STYLE = {
    "fillColor": "#d3d3d3",
    "fillOpacity": 0.35,
    "color": "#aaa",
    "weight": 0.25,
}

WELL_LEVEL_METRICS = ["EUR", "IP90", "1YCuml", "Wcut"]
SECTION_LEVEL_METRICS = ["OOIP", "RFTD", "URF"]

# Prospect line styles
NO_ANALOG_STYLE = {"color": "orange", "weight": 3, "dashArray": "5 5"}
PASSING_STYLE   = {"color": "#2196F3", "weight": 3}
FAILING_STYLE   = {"color": "#d3d3d3", "weight": 2}

DEFAULT_TOP_N = 5
BUFFER_DISTANCE_M = 800

# Confidence tiers based on analog count
CONFIDENCE_THRESHOLDS = {"High": 8, "Medium": 4, "Low": 2}


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


def confidence_tier(count):
    if count >= CONFIDENCE_THRESHOLDS["High"]:
        return "🟢 High"
    elif count >= CONFIDENCE_THRESHOLDS["Medium"]:
        return "🟡 Medium"
    elif count >= CONFIDENCE_THRESHOLDS["Low"]:
        return "🟠 Low"
    else:
        return "🔴 Insufficient"


def build_radar_chart(row: pd.Series, all_passing: pd.DataFrame, mode="prospect") -> go.Figure:
    if mode == "prospect":
        categories = ["EUR", "IP90", "1YCuml", "OOIP"]
        invert = {"Wcut", "URF", "RFTD"}
        radar_cats = categories + ["Wcut", "URF", "RFTD"]
    else:
        categories = ["Avg EUR", "Avg IP90", "Avg 1YCuml", "OOIP"]
        invert = {"Avg Wcut", "URF", "RFTD"}
        radar_cats = categories + ["Avg Wcut", "URF", "RFTD"]

    values = []
    for c in radar_cats:
        col_vals = all_passing[c].dropna()
        if col_vals.empty or pd.isna(row.get(c)):
            values.append(50)
            continue
        pct = (col_vals < row[c]).sum() / len(col_vals) * 100
        if c in invert:
            pct = 100 - pct
        values.append(pct)

    values.append(values[0])
    labels = radar_cats + [radar_cats[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            fillcolor="rgba(33,150,243,0.25)",
            line_color="#2196F3",
            name=str(row.get("Label", row.get("Section", ""))),
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
# Sidebar — Analysis Mode Toggle
# ==========================================================
st.sidebar.title("Map Settings")

st.sidebar.subheader("⚙️ Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Rank by",
    ["Prospect (Buffer Approach)", "Section (Grid Analysis)"],
    index=0,
)
is_section_mode = analysis_mode.startswith("Section")

# ==========================================================
# Sidebar — Analog Well Source
# ==========================================================
st.sidebar.subheader("📂 Analog Well Source")
show_in_unit = st.sidebar.checkbox("In-Unit Wells", value=True)
show_out_unit = st.sidebar.checkbox("Out-of-Unit Wells", value=False)

if not show_in_unit and not show_out_unit:
    st.sidebar.error("Select at least one well source.")
    st.stop()

# --- Prospect type selection (only in prospect mode) ---
if not is_section_mode:
    st.sidebar.subheader("🎯 Prospect Type")
    show_infills = st.sidebar.checkbox("2M Infills", value=True)
    show_lease_lines = st.sidebar.checkbox("2M Lease Lines", value=False)
    if not show_infills and not show_lease_lines:
        st.sidebar.error("Select at least one prospect type.")
        st.stop()
else:
    show_infills = True
    show_lease_lines = True


# ==========================================================
# Build the analog well pool
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
existing_wells = pd.concat([lines_with_uwi, points_only], ignore_index=True)
existing_wells = gpd.GeoDataFrame(existing_wells, geometry="geometry", crs=lines_gdf.crs)

analog_wells = existing_wells.merge(prod_pool, on="UWI", how="inner")
analog_wells = gpd.GeoDataFrame(analog_wells, geometry="geometry", crs=existing_wells.crs)


# ==========================================================
# Assign wells to sections by last endpoint
# ==========================================================

@st.cache_data(show_spinner="Assigning wells to sections …")
def assign_wells_to_sections(_analog_wells, _grid):
    aw = _analog_wells.copy()
    g = _grid[["Section", "geometry"]].copy()

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
        crs=_analog_wells.crs,
    )

    joined = gpd.sjoin(endpoint_gdf, g, how="left", predicate="within")
    joined = joined.rename(columns={"Section_right": "Assigned_Section"})
    if "Section_left" in joined.columns:
        joined = joined.rename(columns={"Section_left": "Section_orig"})
    elif "Section" in joined.columns:
        joined = joined.rename(columns={"Section": "Assigned_Section"})

    # Handle the column naming from sjoin
    sec_col = None
    for c in joined.columns:
        if c.startswith("Section") and c != "Section_orig":
            sec_col = c
            break
    if sec_col and sec_col != "Assigned_Section":
        joined = joined.rename(columns={sec_col: "Assigned_Section"})

    keep_cols = [c for c in ["UWI", "Assigned_Section", "EUR", "IP90", "1YCuml",
                             "Wcut", "Cuml"] if c in joined.columns]
    result = joined[keep_cols].copy()
    result = result.drop_duplicates(subset="UWI", keep="first")
    return result


wells_by_section = assign_wells_to_sections(analog_wells, grid_gdf)


# ==========================================================
# Compute section-level metrics (enriched with well averages)
# ==========================================================

@st.cache_data(show_spinner="Computing section metrics …")
def compute_section_metrics(_wells_by_section, _grid_df):
    ws = _wells_by_section.copy()
    g = _grid_df[["Section", "OOIP", "geometry"]].copy()

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
            EUR_median=("EUR", "median"),
        )
        .reset_index()
        .rename(columns={"Assigned_Section": "Section"})
    )

    # Add P10/P90
    p10 = ws.groupby("Assigned_Section")["EUR"].quantile(0.9).reset_index()
    p10.columns = ["Section", "EUR_p10"]
    p90 = ws.groupby("Assigned_Section")["EUR"].quantile(0.1).reset_index()
    p90.columns = ["Section", "EUR_p90"]
    section_agg = section_agg.merge(p10, on="Section", how="left")
    section_agg = section_agg.merge(p90, on="Section", how="left")

    g = g.merge(section_agg, on="Section", how="left")

    ooip_safe = g["OOIP"].replace(0, np.nan)
    g["RFTD"] = g["Section_Cuml"] / ooip_safe
    g["URF"] = g["Section_EUR"] / ooip_safe

    for col in ["RFTD", "URF"]:
        g[col] = g[col].replace([np.inf, -np.inf], np.nan)

    return g


section_enriched = compute_section_metrics(wells_by_section, grid_gdf)


# ==========================================================
# SECTION MODE — filters, ranking, display
# ==========================================================
if is_section_mode:

    # ---- Section gradient selector ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗺️ Section Grid Gradient")
    section_gradient = st.sidebar.selectbox(
        "Colour sections by",
        ["None", "OOIP", "Avg EUR", "Avg IP90", "Avg 1YCuml", "Avg Wcut",
         "Section Cuml", "RFTD", "URF"],
    )

    GRADIENT_COL_MAP = {
        "OOIP": "OOIP",
        "Avg EUR": "Avg_EUR",
        "Avg IP90": "Avg_IP90",
        "Avg 1YCuml": "Avg_1YCuml",
        "Avg Wcut": "Avg_Wcut",
        "Section Cuml": "Section_Cuml",
        "RFTD": "RFTD",
        "URF": "URF",
    }

    # ---- Section filters ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Section Filters")

    s = section_enriched.copy()

    _s_ooip_min, _s_ooip_max = safe_range(s["OOIP"])
    s_filter_min_ooip = st.sidebar.slider(
        "Min OOIP (bbl)", _s_ooip_min, _s_ooip_max, _s_ooip_min, step=1.0,
        format="%.2f", key="s_ooip",
    )

    _s_eur_min, _s_eur_max = safe_range(s["Avg_EUR"])
    s_filter_min_eur = st.sidebar.slider(
        "Min Avg EUR (bbl)", _s_eur_min, _s_eur_max, _s_eur_min, step=1.0,
        format="%.0f", key="s_eur",
    )

    _s_ip90_min, _s_ip90_max = safe_range(s["Avg_IP90"])
    s_filter_min_ip90 = st.sidebar.slider(
        "Min Avg IP90 (bbl/d)", _s_ip90_min, _s_ip90_max, _s_ip90_min, step=1.0,
        format="%.0f", key="s_ip90",
    )

    _s_1y_min, _s_1y_max = safe_range(s["Avg_1YCuml"])
    s_filter_min_1y = st.sidebar.slider(
        "Min Avg 1Y Cuml (bbl)", _s_1y_min, _s_1y_max, _s_1y_min, step=1.0,
        format="%.0f", key="s_1y",
    )

    _s_wcut_min, _s_wcut_max = safe_range(s["Avg_Wcut"])
    s_filter_max_wcut = st.sidebar.slider(
        "Max Avg Wcut (%)", _s_wcut_min, _s_wcut_max, _s_wcut_max, step=1.0,
        key="s_wcut",
    )

    _s_urf_min, _s_urf_max = safe_range(s["URF"])
    s_filter_max_urf = st.sidebar.slider(
        "Max URF", _s_urf_min, _s_urf_max, _s_urf_max, step=0.01,
        format="%.2f", key="s_urf",
    )

    _s_rftd_min, _s_rftd_max = safe_range(s["RFTD"])
    s_filter_max_rftd = st.sidebar.slider(
        "Max RFTD", _s_rftd_min, _s_rftd_max, _s_rftd_max, step=0.01,
        format="%.2f", key="s_rftd",
    )

    # Apply filters
    has_wells = s["Well_Count"].fillna(0) > 0
    passes = (
        has_wells
        & ((s["OOIP"] >= s_filter_min_ooip) | s["OOIP"].isna())
        & ((s["Avg_EUR"] >= s_filter_min_eur) | s["Avg_EUR"].isna())
        & ((s["Avg_IP90"] >= s_filter_min_ip90) | s["Avg_IP90"].isna())
        & ((s["Avg_1YCuml"] >= s_filter_min_1y) | s["Avg_1YCuml"].isna())
        & ((s["Avg_Wcut"] <= s_filter_max_wcut) | s["Avg_Wcut"].isna())
        & ((s["URF"] <= s_filter_max_urf) | s["URF"].isna())
        & ((s["RFTD"] <= s_filter_max_rftd) | s["RFTD"].isna())
    )

    s["_passes_filter"] = passes
    s["_no_wells"] = ~has_wells

    n_total_s = int(has_wells.sum())
    n_passing_s = int(passes.sum())

    st.sidebar.markdown(
        f"**{n_passing_s}** / {n_total_s} sections with wells pass filters "
        f"({n_passing_s / max(n_total_s, 1) * 100:.0f}%)"
    )

    # ---- Ranking metric ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Ranking Metric")

    SECTION_RANKING_OPTIONS = [
        "Avg EUR", "Avg IP90", "Avg 1YCuml", "Avg Wcut",
        "OOIP", "RFTD", "URF", "Section Cuml", "High-Grade Score",
    ]
    SECTION_METRIC_COL_MAP = {
        "Avg EUR": "Avg_EUR",
        "Avg IP90": "Avg_IP90",
        "Avg 1YCuml": "Avg_1YCuml",
        "Avg Wcut": "Avg_Wcut",
        "OOIP": "OOIP",
        "RFTD": "RFTD",
        "URF": "URF",
        "Section Cuml": "Section_Cuml",
        "High-Grade Score": "HighGradeScore",
    }

    selected_metric = st.sidebar.selectbox(
        "Rank sections by", SECTION_RANKING_OPTIONS, key="s_metric",
    )

    st.sidebar.subheader("🏆 Top-N Highlight")
    top_n = st.sidebar.slider(
        "Highlight top N sections on map",
        min_value=0, max_value=20, value=DEFAULT_TOP_N, step=1, key="s_topn",
    )

    # ---- High-Grade Score weights ----
    if selected_metric == "High-Grade Score":
        st.sidebar.markdown("---")
        st.sidebar.subheader("High-Grade Score Weights")
        st.sidebar.caption("Must total 100 %")

        c1, c2 = st.sidebar.columns(2)
        w_eur = c1.number_input("Avg EUR", 0, 100, 20, key="sw_eur")
        w_ip90 = c2.number_input("Avg IP90", 0, 100, 15, key="sw_ip90")
        w_1ycuml = c1.number_input("Avg 1Y Cuml", 0, 100, 15, key="sw_1y")
        w_wcut = c2.number_input("Avg Wcut", 0, 100, 10, key="sw_wcut")
        w_ooip = c1.number_input("OOIP", 0, 100, 20, key="sw_ooip")
        w_urf = c2.number_input("URF", 0, 100, 10, key="sw_urf")
        w_rftd = c1.number_input("RFTD", 0, 100, 10, key="sw_rftd")

        total_weight = w_eur + w_ip90 + w_1ycuml + w_wcut + w_ooip + w_urf + w_rftd

        if total_weight == 100:
            st.sidebar.success(f"Total Weight: {total_weight}%")
        elif total_weight > 100:
            st.sidebar.error(f"Total Weight: {total_weight}% (Over by {total_weight - 100}%)")
        else:
            st.sidebar.warning(f"Total Weight: {total_weight}% ({100 - total_weight}% remaining)")
        st.sidebar.progress(min(total_weight / 100, 1.0))
    else:
        total_weight = 0

    # ---- Compute HG Score for sections ----
    if selected_metric == "High-Grade Score" and total_weight == 100:
        passing_s = s[s["_passes_filter"]].copy()

        z_eur = zscore(passing_s["Avg_EUR"])
        z_ip90 = zscore(passing_s["Avg_IP90"])
        z_1y = zscore(passing_s["Avg_1YCuml"])
        z_ooip = zscore(passing_s["OOIP"])
        z_wcut = -zscore(passing_s["Avg_Wcut"])
        z_urf = -zscore(passing_s["URF"])
        z_rftd = -zscore(passing_s["RFTD"])

        hgs = (
            (w_eur / 100) * z_eur
            + (w_ip90 / 100) * z_ip90
            + (w_1ycuml / 100) * z_1y
            + (w_wcut / 100) * z_wcut
            + (w_ooip / 100) * z_ooip
            + (w_urf / 100) * z_urf
            + (w_rftd / 100) * z_rftd
        )

        s["HighGradeScore"] = np.nan
        s.loc[passing_s.index, "HighGradeScore"] = hgs
    else:
        s["HighGradeScore"] = np.nan

    # ---- Confidence tier ----
    s["Confidence"] = s["Well_Count"].fillna(0).apply(confidence_tier)

    # ---- Prepare display ----
    s_display = s.copy().to_crs(4326)
    units_display = units_gdf.copy().to_crs(4326)

    existing_display_cols = ["UWI", "geometry"] + [
        c for c in ["EUR", "IP90", "1YCuml", "Wcut", "Section"] if c in analog_wells.columns
    ]
    existing_display = analog_wells[existing_display_cols].copy().to_crs(4326)

    # ---- Top-N sections ----
    metric_col = SECTION_METRIC_COL_MAP.get(selected_metric, selected_metric)
    ascending = selected_metric in ["Avg Wcut", "URF", "RFTD"]

    top_n_sections = set()
    if top_n > 0:
        rank_pool = s[s["_passes_filter"]].dropna(subset=[metric_col])
        if not rank_pool.empty:
            top_subset = rank_pool.sort_values(metric_col, ascending=ascending).head(top_n)
            top_n_sections = set(top_subset["Section"].tolist())

    # ================================================================
    # EXECUTIVE SUMMARY
    # ================================================================
    st.title("🛢️ Bakken Section Analyzer")

    if n_passing_s > 0:
        best_section = s[s["_passes_filter"]].dropna(subset=[metric_col])
        if not best_section.empty:
            best_row = best_section.sort_values(metric_col, ascending=ascending).iloc[0]
            best_name = best_row["Section"]
            best_val = best_row[metric_col]

            avg_well_count = s[s["_passes_filter"]]["Well_Count"].mean()
            high_conf = s[s["_passes_filter"] & (s["Well_Count"] >= CONFIDENCE_THRESHOLDS["High"])]

            st.success(
                f"**{n_passing_s}** of {n_total_s} sections with wells pass your filters. "
                f"Top section by **{selected_metric}** is **{best_name}** "
                f"({metric_col} = {best_val:,.2f}). "
                f"Average well count per passing section is **{avg_well_count:.1f}** — "
                f"**{len(high_conf)}** sections have high-confidence analog support (≥{CONFIDENCE_THRESHOLDS['High']} wells)."
            )
        else:
            st.info(f"**{n_passing_s}** sections pass filters but none have valid {selected_metric} data.")
    else:
        st.warning("No sections pass the current filters. Try relaxing your criteria.")

    col_map, col_rank = st.columns([8, 3])

    # ----------------------------------------------------------
    # LEFT COLUMN — Map (Section Mode)
    # ----------------------------------------------------------
    with col_map:
        bounds = s_display.total_bounds
        centre = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

        m = folium.Map(location=centre, zoom_start=11, tiles="CartoDB positron")
        MiniMap(toggle_display=True, position="bottomleft").add_to(m)

        # ---- Section grid gradient ----
        if section_gradient != "None":
            grad_col = GRADIENT_COL_MAP.get(section_gradient)
            grad_vals = s_display[grad_col].dropna()

            # Determine if lower is better for coloring
            lower_is_better = section_gradient in ["Avg Wcut", "RFTD", "URF"]

            if not grad_vals.empty:
                if lower_is_better:
                    colors = ["#006837", "#78c679", "#ffffcc"]
                else:
                    colors = ["#ffffcc", "#78c679", "#006837"]

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
                    return {
                        "fillColor": colormap(val),
                        "fillOpacity": 0.55,
                        "color": "white",
                        "weight": 0.3,
                    }
            else:
                def section_style(feature):
                    return NULL_STYLE
        else:
            def section_style(feature):
                return NULL_STYLE

        section_tooltip_fields = ["Section", "OOIP", "Well_Count",
                                  "Avg_EUR", "Avg_IP90", "Avg_1YCuml", "Avg_Wcut",
                                  "RFTD", "URF"]
        section_tooltip_aliases = ["Section:", "OOIP:", "Wells:",
                                   "Avg EUR:", "Avg IP90:", "Avg 1Y Cuml:", "Avg Wcut:",
                                   "RFTD:", "URF:"]

        folium.GeoJson(
            s_display.to_json(),
            name="Section Grid",
            style_function=section_style,
            highlight_function=lambda _: {"weight": 2, "color": "black", "fillOpacity": 0.6},
            tooltip=folium.GeoJsonTooltip(
                fields=section_tooltip_fields,
                aliases=section_tooltip_aliases,
                localize=True, sticky=True,
                style=("font-size: 12px; padding: 4px 8px; "
                       "background-color: rgba(255,255,255,0.9); "
                       "border: 1px solid #333; border-radius: 3px;"),
            ),
        ).add_to(m)

        # ---- Units ----
        folium.GeoJson(
            units_gdf.copy().to_crs(4326).to_json(),
            name="Units",
            style_function=lambda _: {"color": "black", "weight": 2, "fillOpacity": 0, "interactive": False},
        ).add_to(m)

        # ---- Existing wells ----
        point_wells = existing_display[existing_display.geometry.type == "Point"]
        line_wells = existing_display[existing_display.geometry.type != "Point"]

        well_line_fg = folium.FeatureGroup(name="Existing Well Lines")
        if not line_wells.empty:
            wl_fields = ["UWI"]
            wl_aliases = ["UWI:"]
            for wf, wa in [("EUR", "EUR:"), ("IP90", "IP90:"), ("Section", "Section:")]:
                if wf in line_wells.columns:
                    wl_fields.append(wf)
                    wl_aliases.append(wa)
            folium.GeoJson(
                line_wells.to_json(),
                style_function=lambda _: {"color": "red", "weight": 1.5, "opacity": 0.8},
                highlight_function=lambda _: {"weight": 3, "color": "yellow"},
                tooltip=folium.GeoJsonTooltip(
                    fields=wl_fields, aliases=wl_aliases,
                    localize=True, sticky=True,
                    style=("font-size: 11px; padding: 3px 6px; "
                           "background-color: rgba(255,255,255,0.92); "
                           "border: 1px solid #c00; border-radius: 3px;"),
                ),
            ).add_to(well_line_fg)
        well_line_fg.add_to(m)

        well_point_fg = folium.FeatureGroup(name="Existing Well Points")
        for _, row in point_wells.iterrows():
            tip_parts = [f"<b>UWI:</b> {row.get('UWI', '—')}"]
            if "EUR" in row and pd.notna(row["EUR"]):
                tip_parts.append(f"<b>EUR:</b> {row['EUR']:,.0f}")
            if "IP90" in row and pd.notna(row["IP90"]):
                tip_parts.append(f"<b>IP90:</b> {row['IP90']:,.0f}")
            if "1YCuml" in row and pd.notna(row.get("1YCuml")):
                tip_parts.append(f"<b>1Y Cuml:</b> {row['1YCuml']:,.0f}")
            if "Wcut" in row and pd.notna(row.get("Wcut")):
                tip_parts.append(f"<b>Wcut:</b> {row['Wcut']:.1f}%")
            tip_text = "<br>".join(tip_parts)
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5, color="red", fill=True, fill_color="red",
                fill_opacity=0.8, weight=1,
                tooltip=folium.Tooltip(tip_text, sticky=True,
                    style=("font-size: 11px; padding: 3px 6px; "
                           "background-color: rgba(255,255,255,0.92); "
                           "border: 1px solid #c00; border-radius: 3px;")),
            ).add_to(well_point_fg)
        well_point_fg.add_to(m)

        # ---- Top-N section highlight ----
        if top_n_sections:
            top_secs = s_display[s_display["Section"].isin(top_n_sections)]
            if not top_secs.empty:
                top_fg = folium.FeatureGroup(name="⭐ Top Sections")
                folium.GeoJson(
                    top_secs.to_json(),
                    style_function=lambda _: {
                        "color": "gold", "weight": 4, "opacity": 0.9,
                        "fillColor": "gold", "fillOpacity": 0.15,
                    },
                    highlight_function=lambda _: {"weight": 6, "color": "#FFD700"},
                    tooltip=folium.GeoJsonTooltip(
                        fields=["Section"], aliases=["⭐ Top Section:"],
                        localize=True, sticky=True,
                        style=("font-size: 13px; font-weight: bold; padding: 5px 10px; "
                               "background-color: rgba(255,215,0,0.2); "
                               "border: 2px solid gold; border-radius: 4px;"),
                    ),
                ).add_to(top_fg)
                top_fg.add_to(m)

        folium.LayerControl(collapsed=True).add_to(m)
        st_folium(m, use_container_width=True, height=900, returned_objects=[])

    # ----------------------------------------------------------
    # RIGHT COLUMN — Section Ranking Table
    # ----------------------------------------------------------
    with col_rank:
        st.header("📊 Section Ranking")

        if selected_metric == "High-Grade Score" and total_weight != 100:
            st.warning("Adjust weights to total 100 % to see rankings.")
        else:
            rank_df = s[s["_passes_filter"]].copy()

            display_cols = [
                "Section", "Well_Count", "Confidence",
                "Avg_EUR", "Avg_IP90", "Avg_1YCuml", "Avg_Wcut",
                "OOIP", "Section_Cuml", "RFTD", "URF",
                "EUR_median", "EUR_p10", "EUR_p90",
            ]
            if selected_metric == "High-Grade Score":
                display_cols.append("HighGradeScore")
            if metric_col not in display_cols:
                display_cols.append(metric_col)

            # Only keep columns that exist
            display_cols = [c for c in display_cols if c in rank_df.columns]

            rank_df = rank_df[display_cols].copy()
            rank_df = rank_df.dropna(subset=[metric_col])

            if rank_df.empty:
                st.warning(f"No valid data for **{selected_metric}**.")
            else:
                rank_df["Percentile"] = (
                    rank_df[metric_col].rank(pct=True, ascending=(not ascending)) * 100
                )
                rank_df = rank_df.sort_values(metric_col, ascending=ascending).reset_index(drop=True)
                rank_df.index = rank_df.index + 1
                rank_df.index.name = "Rank"

                rename_map = {
                    "Well_Count": "Wells",
                    "Avg_EUR": "Avg EUR",
                    "Avg_IP90": "Avg IP90",
                    "Avg_1YCuml": "Avg 1YCuml",
                    "Avg_Wcut": "Avg Wcut",
                    "Section_Cuml": "Sec Cuml",
                    "EUR_median": "EUR Med",
                    "EUR_p10": "EUR P10",
                    "EUR_p90": "EUR P90",
                }
                rank_df = rank_df.rename(columns=rename_map)

                # Update metric_col name if it was renamed
                display_metric = rename_map.get(metric_col, metric_col)

                st.caption(f"Ranked by **{selected_metric}** · {len(rank_df)} sections")

                summary_cols = st.columns(3)
                is_score = "Score" in selected_metric
                with summary_cols[0]:
                    st.metric("Best", f"{rank_df[display_metric].iloc[0]:,.2f}" if is_score else f"{rank_df[display_metric].iloc[0]:,.0f}")
                with summary_cols[1]:
                    med = rank_df[display_metric].median()
                    st.metric("Median", f"{med:,.2f}" if is_score else f"{med:,.0f}")
                with summary_cols[2]:
                    st.metric("Worst", f"{rank_df[display_metric].iloc[-1]:,.2f}" if is_score else f"{rank_df[display_metric].iloc[-1]:,.0f}")

                fmt = {
                    "Avg EUR": "{:,.0f}", "Avg IP90": "{:,.0f}", "Avg 1YCuml": "{:,.0f}",
                    "Avg Wcut": "{:.1f}", "OOIP": "{:,.0f}", "Sec Cuml": "{:,.0f}",
                    "RFTD": "{:.3f}", "URF": "{:.3f}", "Percentile": "{:.0f}%",
                    "EUR Med": "{:,.0f}", "EUR P10": "{:,.0f}", "EUR P90": "{:,.0f}",
                    "Wells": "{:.0f}",
                }
                if "HighGradeScore" in rank_df.columns:
                    fmt["HighGradeScore"] = "{:.2f}"

                if not ascending:
                    gmap_vals = rank_df[display_metric]
                else:
                    gmap_vals = -rank_df[display_metric]

                st.dataframe(
                    rank_df.style.background_gradient(
                        subset=[display_metric], cmap="YlGn", gmap=gmap_vals,
                    ).background_gradient(
                        subset=["Percentile"], cmap="RdYlGn",
                    ).format(fmt),
                    use_container_width=True,
                    height=600,
                )

                csv = rank_df.to_csv().encode("utf-8")
                st.download_button("⬇️ Download Rankings (CSV)", data=csv,
                                   file_name="bakken_section_rankings.csv", mime="text/csv")

                # ---- Section Detail ----
                st.markdown("---")
                st.subheader("🔬 Section Detail")
                detail_section = st.selectbox("Select a section", rank_df["Section"].tolist())

                if detail_section:
                    detail_row = rank_df[rank_df["Section"] == detail_section].iloc[0]

                    dc1, dc2, dc3 = st.columns(3)
                    dc1.metric("Avg EUR", f"{detail_row['Avg EUR']:,.0f}" if pd.notna(detail_row.get("Avg EUR")) else "—")
                    dc2.metric("Avg IP90", f"{detail_row['Avg IP90']:,.0f}" if pd.notna(detail_row.get("Avg IP90")) else "—")
                    dc3.metric("Avg Wcut", f"{detail_row['Avg Wcut']:.1f}%" if pd.notna(detail_row.get("Avg Wcut")) else "—")

                    dc4, dc5, dc6 = st.columns(3)
                    dc4.metric("OOIP", f"{detail_row['OOIP']:,.0f}" if pd.notna(detail_row.get("OOIP")) else "—")
                    dc5.metric("URF", f"{detail_row['URF']:.3f}" if pd.notna(detail_row.get("URF")) else "—")
                    dc6.metric("RFTD", f"{detail_row['RFTD']:.3f}" if pd.notna(detail_row.get("RFTD")) else "—")

                    dc7, dc8 = st.columns(2)
                    dc7.metric("Wells", f"{detail_row['Wells']:.0f}" if pd.notna(detail_row.get("Wells")) else "—")
                    dc8.metric("Confidence", detail_row.get("Confidence", "—"))

                    if pd.notna(detail_row.get("EUR P10")) and pd.notna(detail_row.get("EUR P90")):
                        st.caption(
                            f"EUR range: P90 = {detail_row['EUR P90']:,.0f} · "
                            f"Median = {detail_row.get('EUR Med', 0):,.0f} · "
                            f"P10 = {detail_row['EUR P10']:,.0f}"
                        )

                    radar_fig = build_radar_chart(detail_row, rank_df, mode="section")
                    st.plotly_chart(radar_fig, use_container_width=True)

        # ---- Sections with no wells ----
        no_well_secs = s[s["_no_wells"] & s["OOIP"].notna()].copy()
        if not no_well_secs.empty:
            st.markdown("---")
            st.subheader("⚠️ Sections With No Wells")
            st.caption("These sections have OOIP data but no assigned analog wells.")
            st.dataframe(
                no_well_secs[["Section", "OOIP"]].sort_values("OOIP", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )


# ==========================================================
# PROSPECT MODE — original behavior with enhancements
# ==========================================================
else:

    # ---- Build prospect GeoDataFrame ----
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

    # ---- Per-prospect spatial analysis ----
    @st.cache_data(show_spinner="Analysing prospects …")
    def analyze_prospects(_prospects, _analog_wells, _section_enriched):
        prospects = _prospects.copy()
        analog = _analog_wells.copy()
        sections = _section_enriched.copy()

        results = []
        for idx, prospect in prospects.iterrows():
            geom = prospect.geometry
            record = {"_idx": idx, "_prospect_type": prospect["_prospect_type"]}

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

            buffer_geom = geom.buffer(BUFFER_DISTANCE_M)
            buffer_gdf = gpd.GeoDataFrame([{"geometry": buffer_geom}], crs=prospects.crs)
            hits = gpd.sjoin(analog, buffer_gdf, how="inner", predicate="intersects")

            record["Analog_Count"] = len(hits)
            record["_analog_uwis"] = ",".join(hits["UWI"].tolist()) if len(hits) > 0 else ""

            if len(hits) > 0:
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

    prospects = prospects.join(
        prospect_metrics.drop(columns=["_prospect_type"], errors="ignore")
    )
    prospects["Label"] = prospects["_section_label"]

    for col in WELL_LEVEL_METRICS + SECTION_LEVEL_METRICS:
        if col in prospects.columns:
            prospects[col] = prospects[col].replace([np.inf, -np.inf], np.nan)

    # ---- Confidence tier ----
    prospects["Confidence"] = prospects["Analog_Count"].fillna(0).apply(confidence_tier)

    # ---- Sidebar: gradient ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗺️ Section Grid Gradient")

    PROSPECT_GRADIENT_OPTIONS = [
        "None", "OOIP", "Avg EUR", "Avg IP90", "Avg 1YCuml", "Avg Wcut",
        "Section Cuml", "RFTD", "URF",
    ]
    PROSPECT_GRADIENT_COL_MAP = {
        "OOIP": "OOIP",
        "Avg EUR": "Avg_EUR",
        "Avg IP90": "Avg_IP90",
        "Avg 1YCuml": "Avg_1YCuml",
        "Avg Wcut": "Avg_Wcut",
        "Section Cuml": "Section_Cuml",
        "RFTD": "RFTD",
        "URF": "URF",
    }

    section_gradient = st.sidebar.selectbox(
        "Colour sections by", PROSPECT_GRADIENT_OPTIONS, key="p_gradient",
    )

    # ---- Sidebar: prospect filters ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Prospect Filters")
    st.sidebar.caption("Prospects that fail are greyed out and excluded from rankings.")

    p = prospects.copy()

    _wcut_min, _wcut_max = safe_range(p["Wcut"])
    filter_max_wcut = st.sidebar.slider("Max Water Cut (%)", _wcut_min, _wcut_max, _wcut_max, step=1.0, key="p_wcut")

    _ooip_min, _ooip_max = safe_range(p["OOIP"])
    filter_min_ooip = st.sidebar.slider("Min OOIP (bbl)", _ooip_min, _ooip_max, _ooip_min, step=1.0, format="%.2f", key="p_ooip")

    _eur_min, _eur_max = safe_range(p["EUR"])
    filter_min_eur = st.sidebar.slider("Min EUR (bbl)", _eur_min, _eur_max, _eur_min, step=1.0, format="%.0f", key="p_eur")

    _ip90_min, _ip90_max = safe_range(p["IP90"])
    filter_min_ip90 = st.sidebar.slider("Min IP90 (bbl/d)", _ip90_min, _ip90_max, _ip90_min, step=1.0, format="%.0f", key="p_ip90")

    _1y_min, _1y_max = safe_range(p["1YCuml"])
    filter_min_1ycuml = st.sidebar.slider("Min 1Y Cuml (bbl)", _1y_min, _1y_max, _1y_min, step=1.0, format="%.0f", key="p_1y")

    _urf_min, _urf_max = safe_range(p["URF"])
    filter_max_urf = st.sidebar.slider("Max URF", _urf_min, _urf_max, _urf_max, step=0.01, format="%.2f", key="p_urf")

    _rftd_min, _rftd_max = safe_range(p["RFTD"])
    filter_max_rftd = st.sidebar.slider("Max RFTD", _rftd_min, _rftd_max, _rftd_max, step=0.01, format="%.2f", key="p_rftd")

    # Apply filters
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

    # ---- Sidebar: ranking metric ----
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Ranking Metric")

    selected_metric = st.sidebar.selectbox(
        "Rank prospects by",
        ["EUR", "IP90", "1YCuml", "Wcut", "OOIP", "URF", "RFTD", "High-Grade Score"],
        key="p_metric",
    )

    st.sidebar.subheader("🏆 Top-N Highlight")
    top_n = st.sidebar.slider(
        "Highlight top N prospects on map",
        min_value=0, max_value=20, value=DEFAULT_TOP_N, step=1, key="p_topn",
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
            st.sidebar.success(f"Total Weight: {total_weight}%")
        elif total_weight > 100:
            st.sidebar.error(f"Total Weight: {total_weight}% (Over by {total_weight - 100}%)")
        else:
            st.sidebar.warning(f"Total Weight: {total_weight}% ({100 - total_weight}% remaining)")
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

    # ---- Prepare display data ----
    p_display = p.copy().to_crs(4326)
    # Section grid for gradient in prospect mode uses section_enriched
    section_display = section_enriched.copy().to_crs(4326)
    units_display = units_gdf.copy().to_crs(4326)

    existing_display_cols = ["UWI", "geometry"] + [
        c for c in ["EUR", "IP90", "1YCuml", "Wcut", "Section"] if c in analog_wells.columns
    ]
    existing_display = analog_wells[existing_display_cols].copy().to_crs(4326)

    # ---- Top-N prospects ----
    metric_col = "HighGradeScore" if selected_metric == "High-Grade Score" else selected_metric
    ascending = selected_metric in ["Wcut", "URF", "RFTD"]

    top_n_labels = set()
    if top_n > 0:
        rank_pool = p[p["_passes_filter"]].dropna(subset=[metric_col])
        if not rank_pool.empty:
            top_subset = rank_pool.sort_values(metric_col, ascending=ascending).head(top_n)
            top_n_labels = set(top_subset["Label"].tolist())

    # ================================================================
    # EXECUTIVE SUMMARY
    # ================================================================
    st.title("🛢️ Bakken Prospect Analyzer")

    if n_passing > 0:
        best_prospects = p[p["_passes_filter"]].dropna(subset=[metric_col])
        if not best_prospects.empty:
            best_row = best_prospects.sort_values(metric_col, ascending=ascending).iloc[0]
            best_name = best_row["Label"]
            best_val = best_row[metric_col]

            avg_analogs = p[p["_passes_filter"]]["Analog_Count"].mean()
            high_conf_count = int((p[p["_passes_filter"]]["Analog_Count"] >= CONFIDENCE_THRESHOLDS["High"]).sum())
            low_conf_count = int((p[p["_passes_filter"]]["Analog_Count"] < CONFIDENCE_THRESHOLDS["Low"]).sum())

            summary_parts = [
                f"**{n_passing}** of {n_total} prospects pass your filters.",
                f"Top prospect by **{selected_metric}** is **{best_name}** "
                f"({metric_col} = {best_val:,.2f}).",
                f"Average analog count is **{avg_analogs:.1f}** wells/prospect — "
                f"**{high_conf_count}** have high confidence (≥{CONFIDENCE_THRESHOLDS['High']} analogs).",
            ]
            if low_conf_count > 0:
                summary_parts.append(
                    f"⚠️ **{low_conf_count}** passing prospects have fewer than "
                    f"{CONFIDENCE_THRESHOLDS['Low']} analogs — treat with caution."
                )
            if n_no_analogs > 0:
                summary_parts.append(
                    f"🟠 **{n_no_analogs}** prospects have **no analog wells** nearby and are excluded."
                )

            st.success(" ".join(summary_parts))
        else:
            st.info(f"**{n_passing}** prospects pass filters but none have valid {selected_metric} data.")
    else:
        st.warning("No prospects pass the current filters. Try relaxing your criteria.")

    col_map, col_rank = st.columns([8, 3])

    # ----------------------------------------------------------
    # LEFT COLUMN — Map (Prospect Mode)
    # ----------------------------------------------------------
    with col_map:
        bounds = p_display.total_bounds
        centre = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

        m = folium.Map(location=centre, zoom_start=11, tiles="CartoDB positron")
        MiniMap(toggle_display=True, position="bottomleft").add_to(m)

        # ---- Section grid gradient ----
        if section_gradient != "None":
            grad_col = PROSPECT_GRADIENT_COL_MAP.get(section_gradient)
            grad_vals = section_display[grad_col].dropna()

            lower_is_better = section_gradient in ["Avg Wcut", "RFTD", "URF"]

            if not grad_vals.empty:
                if lower_is_better:
                    colors = ["#006837", "#78c679", "#ffffcc"]
                else:
                    colors = ["#ffffcc", "#78c679", "#006837"]

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
                    return {
                        "fillColor": colormap(val),
                        "fillOpacity": 0.55,
                        "color": "white",
                        "weight": 0.3,
                    }
            else:
                def section_style(feature):
                    return NULL_STYLE
        else:
            def section_style(feature):
                return NULL_STYLE

        section_tooltip_fields = ["Section", "OOIP", "Well_Count",
                                  "Avg_EUR", "Avg_IP90", "Avg_1YCuml", "Avg_Wcut",
                                  "RFTD", "URF"]
        section_tooltip_aliases = ["Section:", "OOIP:", "Wells:",
                                   "Avg EUR:", "Avg IP90:", "Avg 1Y Cuml:", "Avg Wcut:",
                                   "RFTD:", "URF:"]

        folium.GeoJson(
            section_display.to_json(),
            name="Section Grid",
            style_function=section_style,
            highlight_function=lambda _: {"weight": 2, "color": "black", "fillOpacity": 0.6},
            tooltip=folium.GeoJsonTooltip(
                fields=section_tooltip_fields,
                aliases=section_tooltip_aliases,
                localize=True, sticky=True,
                style=("font-size: 12px; padding: 4px 8px; "
                       "background-color: rgba(255,255,255,0.9); "
                       "border: 1px solid #333; border-radius: 3px;"),
            ),
        ).add_to(m)

        # ---- Units ----
        folium.GeoJson(
            units_display.to_json(),
            name="Units",
            style_function=lambda _: {"color": "black", "weight": 2, "fillOpacity": 0, "interactive": False},
        ).add_to(m)

        # ---- Existing wells ----
        point_wells = existing_display[existing_display.geometry.type == "Point"]
        line_wells = existing_display[existing_display.geometry.type != "Point"]

        well_line_fg = folium.FeatureGroup(name="Existing Well Lines")
        if not line_wells.empty:
            wl_fields = ["UWI"]
            wl_aliases = ["UWI:"]
            for wf, wa in [("EUR", "EUR:"), ("IP90", "IP90:"), ("Section", "Section:")]:
                if wf in line_wells.columns:
                    wl_fields.append(wf)
                    wl_aliases.append(wa)
            folium.GeoJson(
                line_wells.to_json(),
                style_function=lambda _: {"color": "red", "weight": 1.5, "opacity": 0.8},
                highlight_function=lambda _: {"weight": 3, "color": "yellow"},
                tooltip=folium.GeoJsonTooltip(
                    fields=wl_fields, aliases=wl_aliases,
                    localize=True, sticky=True,
                    style=("font-size: 11px; padding: 3px 6px; "
                           "background-color: rgba(255,255,255,0.92); "
                           "border: 1px solid #c00; border-radius: 3px;"),
                ),
            ).add_to(well_line_fg)
        well_line_fg.add_to(m)

        well_point_fg = folium.FeatureGroup(name="Existing Well Points")
        for _, row in point_wells.iterrows():
            tip_parts = [f"<b>UWI:</b> {row.get('UWI', '—')}"]
            if "EUR" in row and pd.notna(row["EUR"]):
                tip_parts.append(f"<b>EUR:</b> {row['EUR']:,.0f}")
            if "IP90" in row and pd.notna(row["IP90"]):
                tip_parts.append(f"<b>IP90:</b> {row['IP90']:,.0f}")
            if "1YCuml" in row and pd.notna(row.get("1YCuml")):
                tip_parts.append(f"<b>1Y Cuml:</b> {row['1YCuml']:,.0f}")
            if "Wcut" in row and pd.notna(row.get("Wcut")):
                tip_parts.append(f"<b>Wcut:</b> {row['Wcut']:.1f}%")
            tip_text = "<br>".join(tip_parts)
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5, color="red", fill=True, fill_color="red",
                fill_opacity=0.8, weight=1,
                tooltip=folium.Tooltip(tip_text, sticky=True,
                    style=("font-size: 11px; padding: 3px 6px; "
                           "background-color: rgba(255,255,255,0.92); "
                           "border: 1px solid #c00; border-radius: 3px;")),
            ).add_to(well_point_fg)
        well_point_fg.add_to(m)

        # ---- Prospects ----
        prospect_tooltip_fields = [
            "Label", "_prospect_type", "Analog_Count", "Confidence",
            "EUR", "IP90", "1YCuml", "Wcut",
            "OOIP", "RFTD", "URF",
        ]
        prospect_tooltip_aliases = [
            "Prospect:", "Type:", "Analogs:", "Confidence:",
            "Avg EUR:", "Avg IP90:", "Avg 1Y Cuml:", "Avg Wcut:",
            "OOIP:", "RFTD:", "URF:",
        ]

        if selected_metric == "High-Grade Score" and total_weight == 100:
            prospect_tooltip_fields.append("HighGradeScore")
            prospect_tooltip_aliases.append("HG Score:")

        def prospect_style(feature):
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
                localize=True, sticky=True,
                style=("font-size: 12px; padding: 5px 10px; "
                       "background-color: rgba(255,255,255,0.95); "
                       "border: 1px solid #2196F3; border-radius: 4px; "
                       "box-shadow: 2px 2px 6px rgba(0,0,0,0.15);"),
            ),
        ).add_to(prospect_fg)
        prospect_fg.add_to(m)

        # ---- Top-N highlight ----
        if top_n_labels:
            top_prospects = p_display[p_display["Label"].isin(top_n_labels)]
            if not top_prospects.empty:
                top_fg = folium.FeatureGroup(name="⭐ Top Prospects")
                folium.GeoJson(
                    top_prospects.to_json(),
                    style_function=lambda _: {
                        "color": "gold", "weight": 5, "opacity": 0.9, "dashArray": "",
                    },
                    highlight_function=lambda _: {"weight": 7, "color": "#FFD700"},
                    tooltip=folium.GeoJsonTooltip(
                        fields=["Label"], aliases=["⭐ Top Prospect:"],
                        localize=True, sticky=True,
                        style=("font-size: 13px; font-weight: bold; padding: 5px 10px; "
                               "background-color: rgba(255,215,0,0.2); "
                               "border: 2px solid gold; border-radius: 4px;"),
                    ),
                ).add_to(top_fg)
                top_fg.add_to(m)

        folium.LayerControl(collapsed=True).add_to(m)
        st_folium(m, use_container_width=True, height=900, returned_objects=[])

    # ----------------------------------------------------------
    # RIGHT COLUMN — Prospect Ranking Table
    # ----------------------------------------------------------
    with col_rank:
        st.header("📊 Prospect Ranking")

        if selected_metric == "High-Grade Score" and total_weight != 100:
            st.warning("Adjust weights to total 100 % to see rankings.")
        else:
            rank_df = p[p["_passes_filter"]].copy()

            display_cols = [
                "Label", "_prospect_type", "Analog_Count", "Confidence",
                "EUR", "IP90", "1YCuml", "Wcut",
                "OOIP", "RFTD", "URF",
                "EUR_median", "EUR_p10", "EUR_p90",
            ]
            if selected_metric == "High-Grade Score":
                display_cols.append("HighGradeScore")
            if metric_col not in display_cols:
                display_cols.append(metric_col)

            display_cols = [c for c in display_cols if c in rank_df.columns]

            rank_df = rank_df[display_cols].copy()
            rank_df = rank_df.dropna(subset=[metric_col])

            if rank_df.empty:
                st.warning(f"No valid data for **{selected_metric}**.")
            else:
                rank_df["Percentile"] = (
                    rank_df[metric_col].rank(pct=True, ascending=(not ascending)) * 100
                )
                rank_df = rank_df.sort_values(metric_col, ascending=ascending).reset_index(drop=True)
                rank_df.index = rank_df.index + 1
                rank_df.index.name = "Rank"

                rank_df = rank_df.rename(columns={
                    "_prospect_type": "Type",
                    "Analog_Count": "Analogs",
                    "EUR_median": "EUR Med",
                    "EUR_p10": "EUR P10",
                    "EUR_p90": "EUR P90",
                })

                st.caption(f"Ranked by **{selected_metric}** · {len(rank_df)} prospects")

                summary_cols = st.columns(3)
                is_score = "Score" in selected_metric
                with summary_cols[0]:
                    st.metric("Best", f"{rank_df[metric_col].iloc[0]:,.2f}" if is_score else f"{rank_df[metric_col].iloc[0]:,.0f}")
                with summary_cols[1]:
                    med = rank_df[metric_col].median()
                    st.metric("Median", f"{med:,.2f}" if is_score else f"{med:,.0f}")
                with summary_cols[2]:
                    st.metric("Worst", f"{rank_df[metric_col].iloc[-1]:,.2f}" if is_score else f"{rank_df[metric_col].iloc[-1]:,.0f}")

                fmt = {
                    "EUR": "{:,.0f}", "IP90": "{:,.0f}", "1YCuml": "{:,.0f}",
                    "Wcut": "{:.1f}", "OOIP": "{:,.0f}",
                    "RFTD": "{:.3f}", "URF": "{:.3f}", "Percentile": "{:.0f}%",
                    "EUR Med": "{:,.0f}", "EUR P10": "{:,.0f}", "EUR P90": "{:,.0f}",
                }
                if "HighGradeScore" in rank_df.columns:
                    fmt["HighGradeScore"] = "{:.2f}"

                if not ascending:
                    gmap_vals = rank_df[metric_col]
                else:
                    gmap_vals = -rank_df[metric_col]

                st.dataframe(
                    rank_df.style.background_gradient(
                        subset=[metric_col], cmap="YlGn", gmap=gmap_vals,
                    ).background_gradient(
                        subset=["Percentile"], cmap="RdYlGn",
                    ).format(fmt),
                    use_container_width=True,
                    height=600,
                )

                csv = rank_df.to_csv().encode("utf-8")
                st.download_button("⬇️ Download Rankings (CSV)", data=csv,
                                   file_name="bakken_prospect_rankings.csv", mime="text/csv")

                # ---- Prospect Detail ----
                st.markdown("---")
                st.subheader("🔬 Prospect Detail")
                detail_label = st.selectbox("Select a prospect", rank_df["Label"].tolist())

                if detail_label:
                    detail_row = rank_df[rank_df["Label"] == detail_label].iloc[0]

                    dc1, dc2, dc3 = st.columns(3)
                    dc1.metric("EUR", f"{detail_row['EUR']:,.0f}" if pd.notna(detail_row.get("EUR")) else "—")
                    dc2.metric("IP90", f"{detail_row['IP90']:,.0f}" if pd.notna(detail_row.get("IP90")) else "—")
                    dc3.metric("Wcut", f"{detail_row['Wcut']:.1f}%" if pd.notna(detail_row.get("Wcut")) else "—")

                    dc4, dc5, dc6 = st.columns(3)
                    dc4.metric("OOIP", f"{detail_row['OOIP']:,.0f}" if pd.notna(detail_row.get("OOIP")) else "—")
                    dc5.metric("URF", f"{detail_row['URF']:.3f}" if pd.notna(detail_row.get("URF")) else "—")
                    dc6.metric("RFTD", f"{detail_row['RFTD']:.3f}" if pd.notna(detail_row.get("RFTD")) else "—")

                    dc7, dc8 = st.columns(2)
                    dc7.metric("Analogs", f"{detail_row['Analogs']:.0f}" if pd.notna(detail_row.get("Analogs")) else "—")
                    dc8.metric("Confidence", detail_row.get("Confidence", "—"))

                    if pd.notna(detail_row.get("EUR P10")) and pd.notna(detail_row.get("EUR P90")):
                        st.caption(
                            f"EUR range: P90 = {detail_row['EUR P90']:,.0f} · "
                            f"Median = {detail_row.get('EUR Med', 0):,.0f} · "
                            f"P10 = {detail_row['EUR P10']:,.0f}"
                        )

                    radar_fig = build_radar_chart(detail_row, rank_df, mode="prospect")
                    st.plotly_chart(radar_fig, use_container_width=True)

        # ---- No-analog prospects ----
        no_analog_prospects = p[p["_no_analogs"]].copy()
        if not no_analog_prospects.empty:
            st.markdown("---")
            st.subheader("⚠️ No Analogs Found")
            st.caption("These prospects have no analog wells within the buffer zone.")
            st.dataframe(
                no_analog_prospects[["Label", "_prospect_type"]]
                .rename(columns={"_prospect_type": "Type"})
                .reset_index(drop=True),
                use_container_width=True,
            )
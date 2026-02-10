import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import branca.colormap as cm
from shapely.geometry import Point

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(layout="wide", page_title="Bakken Prospect Analyzer")

# --------------------------------------------------
# Constants
# --------------------------------------------------
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
ALL_RANKING_METRICS = WELL_LEVEL_METRICS + SECTION_LEVEL_METRICS + ["High-Grade Score"]

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading spatial data...")
def load_data():
    lines = gpd.read_file("lines.shp")
    points = gpd.read_file("points.shp")
    grid = gpd.read_file("ooipsectiongrid.shp")
    infills = gpd.read_file("2M_Infills_plyln.shp")
    lease_lines = gpd.read_file("2M_LL_plyln.shp")
    units = gpd.read_file("Bakken Units.shp")

    prod_in = pd.read_excel("well.xlsx", sheet_name="inunit")
    prod_out = pd.read_excel("well.xlsx", sheet_name="outunit")

    # --- CRS normalization (all to 26913 for spatial ops) ---
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

    # --- Clean well UWIs ---
    lines["UWI"] = lines["UWI"].astype(str).str.strip()
    points["UWI"] = points["UWI"].astype(str).str.strip()

    # --- Simplify grid for display ---
    grid["geometry"] = grid.geometry.simplify(50, preserve_topology=True)  # meters in 26913

    return lines, points, grid, units, infills, lease_lines, prod_in, prod_out


(
    lines_gdf, points_gdf, grid_gdf, units_gdf,
    infills_gdf, lease_lines_gdf, prod_in_df, prod_out_df,
) = load_data()

# --------------------------------------------------
# Sidebar — Section Source Selection
# --------------------------------------------------
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

# --------------------------------------------------
# Build analog pool
# --------------------------------------------------
frames = []
if show_in_unit:
    frames.append(prod_in_df)
if show_out_unit:
    frames.append(prod_out_df)

prod_pool = pd.concat(frames, ignore_index=True)
prod_pool = prod_pool.drop_duplicates(subset="UWI", keep="first")

# --- Combine existing well geometries (lines take priority) ---
lines_with_uwi = lines_gdf[["UWI", "geometry"]].copy()
points_with_uwi = points_gdf[["UWI", "geometry"]].copy()

# Keep line geometry if UWI exists in both
points_only = points_with_uwi[~points_with_uwi["UWI"].isin(lines_with_uwi["UWI"])]
existing_wells = pd.concat([lines_with_uwi, points_only], ignore_index=True)
existing_wells = gpd.GeoDataFrame(existing_wells, geometry="geometry", crs=lines_gdf.crs)

# Join production data to existing wells
analog_wells = existing_wells.merge(prod_pool, on="UWI", how="inner")
analog_wells = gpd.GeoDataFrame(analog_wells, geometry="geometry", crs=existing_wells.crs)

# --------------------------------------------------
# Section-level metrics (pre-compute)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_section_metrics(_analog_wells_df, _grid_df):
    """Compute RFTD and URF at the section level."""
    aw = _analog_wells_df.copy()
    g = _grid_df[["Section", "OOIP", "geometry"]].copy()

    section_sums = aw.groupby("Section").agg(
        Section_Cuml=("Cuml", "sum"),
        Section_EUR=("EUR", "sum"),
    ).reset_index()

    g = g.merge(section_sums, on="Section", how="left")

    ooip_safe = g["OOIP"].replace(0, np.nan)
    g["RFTD"] = g["Section_Cuml"] / ooip_safe
    g["URF"] = g["Section_EUR"] / ooip_safe

    # Clean inf
    for col in ["RFTD", "URF"]:
        g[col] = g[col].replace([np.inf, -np.inf], np.nan)

    return g


section_enriched = compute_section_metrics(analog_wells, grid_gdf)

# --------------------------------------------------
# Build prospect list
# --------------------------------------------------
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

# --------------------------------------------------
# Per-prospect analysis
# --------------------------------------------------
@st.cache_data(show_spinner="Analyzing prospects...")
def analyze_prospects(_prospects, _analog_wells, _section_enriched):
    """For each prospect: buffer → find analogs → compute metrics."""
    prospects = _prospects.copy()
    analog = _analog_wells.copy()
    sections = _section_enriched.copy()

    # Ensure all in 26913
    results = []

    for idx, prospect in prospects.iterrows():
        geom = prospect.geometry
        record = {"_idx": idx, "_prospect_type": prospect["_prospect_type"]}

        # --- Label: section at endpoint ---
        if geom.geom_type == "MultiLineString":
            endpoint = Point(list(geom.geoms[-1].coords)[-1])
        else:
            endpoint = Point(list(geom.coords)[-1])

        endpoint_gdf = gpd.GeoDataFrame(
            [{"geometry": endpoint}], crs=prospects.crs
        )
        section_hit = gpd.sjoin(endpoint_gdf, sections, how="left", predicate="within")

        if not section_hit.empty and pd.notna(section_hit.iloc[0].get("Section")):
            record["_section_label"] = str(section_hit.iloc[0]["Section"])
        else:
            record["_section_label"] = "Unknown"

        # --- Buffer (800m each side = 1600m total width) ---
        buffer_geom = geom.buffer(800)

        # --- Well-level analogs ---
        buffer_gdf = gpd.GeoDataFrame(
            [{"geometry": buffer_geom}], crs=prospects.crs
        )
        hits = gpd.sjoin(analog, buffer_gdf, how="inner", predicate="intersects")

        record["Analog_Count"] = len(hits)

        if len(hits) > 0:
            for col in ["EUR", "IP90", "1YCuml", "Wcut"]:
                record[col] = hits[col].mean()
        else:
            for col in ["EUR", "IP90", "1YCuml", "Wcut"]:
                record[col] = np.nan

        # --- Section-level: area-weighted OOIP, RFTD, URF ---
        buffer_series = gpd.GeoSeries([buffer_geom], crs=prospects.crs)
        buffer_clip_gdf = gpd.GeoDataFrame(geometry=buffer_series)

        overlaps = gpd.overlay(
            sections[["Section", "OOIP", "RFTD", "URF", "geometry"]],
            buffer_clip_gdf,
            how="intersection",
        )

        if not overlaps.empty and overlaps.geometry.area.sum() > 0:
            overlaps["_area"] = overlaps.geometry.area
            total_area = overlaps["_area"].sum()
            overlaps["_weight"] = overlaps["_area"] / total_area

            for col in ["OOIP", "RFTD", "URF"]:
                valid = overlaps.dropna(subset=[col])
                if not valid.empty:
                    # Re-normalize weights for non-null rows
                    w = valid["_area"] / valid["_area"].sum()
                    record[col] = (valid[col] * w).sum()
                else:
                    record[col] = np.nan
        else:
            for col in ["OOIP", "RFTD", "URF"]:
                record[col] = np.nan

        results.append(record)

    results_df = pd.DataFrame(results)

    # --- Deduplicate labels (add suffix) ---
    label_counts = results_df["_section_label"].value_counts()
    dup_labels = label_counts[label_counts > 1].index

    for label in dup_labels:
        mask = results_df["_section_label"] == label
        indices = results_df[mask].index
        for i, row_idx in enumerate(indices, 1):
            results_df.loc[row_idx, "_section_label"] = f"{label}-{i}"

    # --- Merge back to prospects ---
    results_df = results_df.set_index("_idx")

    return results_df


prospect_metrics = analyze_prospects(prospects, analog_wells, section_enriched)

# Attach metrics to prospect geometries
prospects = prospects.join(prospect_metrics.drop(columns=["_prospect_type"], errors="ignore"))
prospects["Label"] = prospects["_section_label"]

# Replace inf with NaN
for col in WELL_LEVEL_METRICS + SECTION_LEVEL_METRICS:
    if col in prospects.columns:
        prospects[col] = prospects[col].replace([np.inf, -np.inf], np.nan)

# --------------------------------------------------
# Sidebar — Filters
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Prospect Filters")
st.sidebar.caption("Prospects failing filters are greyed out and excluded from rankings.")


def _safe_range(series):
    vals = series.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        return (0.0, 1.0) if lo == 0.0 else (lo, lo + abs(lo) * 0.1)
    return lo, hi


# --- Wcut (max) ---
_wcut_min, _wcut_max = _safe_range(prospects["Wcut"])
filter_max_wcut = st.sidebar.slider(
    "Max Water Cut (%)", _wcut_min, _wcut_max, _wcut_max, step=1.0,
)

# --- OOIP (min) ---
_ooip_min, _ooip_max = _safe_range(prospects["OOIP"])
filter_min_ooip = st.sidebar.slider(
    "Min OOIP (bbl)", _ooip_min, _ooip_max, _ooip_min, step=1.0, format="%.2f",
)

# --- EUR (min) ---
_eur_min, _eur_max = _safe_range(prospects["EUR"])
filter_min_eur = st.sidebar.slider(
    "Min EUR (bbl)", _eur_min, _eur_max, _eur_min, step=1.0, format="%.0f",
)

# --- IP90 (min) ---
_ip90_min, _ip90_max = _safe_range(prospects["IP90"])
filter_min_ip90 = st.sidebar.slider(
    "Min IP90 (bbl/d)", _ip90_min, _ip90_max, _ip90_min, step=1.0, format="%.0f",
)

# --- 1YCuml (min) ---
_1y_min, _1y_max = _safe_range(prospects["1YCuml"])
filter_min_1ycuml = st.sidebar.slider(
    "Min 1Y Cuml (bbl)", _1y_min, _1y_max, _1y_min, step=1.0, format="%.0f",
)

# --- URF (max) ---
_urf_min, _urf_max = _safe_range(prospects["URF"])
filter_max_urf = st.sidebar.slider(
    "Max URF", _urf_min, _urf_max, _urf_max, step=0.01, format="%.2f",
)

# --- RFTD (max) ---
_rftd_min, _rftd_max = _safe_range(prospects["RFTD"])
filter_max_rftd = st.sidebar.slider(
    "Max RFTD", _rftd_min, _rftd_max, _rftd_max, step=0.01, format="%.2f",
)

# --------------------------------------------------
# Apply Filters
# --------------------------------------------------
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

# --------------------------------------------------
# Sidebar — Gradient & Ranking Metric
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Ranking Metric")

selected_metric = st.sidebar.selectbox(
    "Rank prospects by",
    ["EUR", "IP90", "1YCuml", "Wcut", "OOIP", "URF", "RFTD", "High-Grade Score"],
)

st.sidebar.subheader("🗺️ Section Grid Gradient")
section_gradient = st.sidebar.selectbox(
    "Section grid color", ["None", "OOIP"],
)

# --------------------------------------------------
# High-Grade Score
# --------------------------------------------------
if selected_metric == "High-Grade Score":
    st.sidebar.markdown("---")
    st.sidebar.subheader("High-Grade Score Weights")
    st.sidebar.caption("Must total 100%")

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
        st.sidebar.error(f"Total Weight: {total_weight}% (Over by {total_weight - 100}%)")
    else:
        st.sidebar.warning(f"Total Weight: {total_weight}% ({100 - total_weight}% remaining)")
    st.sidebar.progress(min(total_weight / 100, 1.0))
else:
    total_weight = 0


def zscore(s):
    vals = s.replace([np.inf, -np.inf], np.nan)
    std = vals.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0, index=s.index)
    return (vals - vals.mean()) / std


if selected_metric == "High-Grade Score" and total_weight == 100:
    passing = p[p["_passes_filter"]].copy()

    z_eur = zscore(passing["EUR"])
    z_ip90 = zscore(passing["IP90"])
    z_1y = zscore(passing["1YCuml"])
    z_wcut = -zscore(passing["Wcut"])
    z_ooip = zscore(passing["OOIP"])
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

# --------------------------------------------------
# Prepare display data (reproject to 4326)
# --------------------------------------------------
p_display = p.copy().to_crs(4326)
section_display = section_enriched.copy().to_crs(4326)
units_display = units_gdf.copy().to_crs(4326)
existing_display = analog_wells[["UWI", "geometry"]].copy().to_crs(4326)

# --------------------------------------------------
# MAP SECTION – updated (points as circles, units non‑blocking)
# --------------------------------------------------
st.title("Bakken Prospect Analyzer")
col_map, col_rank = st.columns([8, 3])

with col_map:
    # --------------------------------------------------
    # 1️⃣  Initialise the base Folium map
    # --------------------------------------------------
    bounds = p_display.total_bounds                     # [xmin, ymin, xmax, ymax]
    centre = [(bounds[1] + bounds[3]) / 2,
              (bounds[0] + bounds[2]) / 2]               # [lat, lng]

    m = folium.Map(
        location=centre,
        zoom_start=11,
        tiles="CartoDB positron"
    )

    # --------------------------------------------------
    # 2️⃣  SECTION GRID (optional colour gradient)
    # --------------------------------------------------
    if section_gradient == "OOIP":
        ooip_vals = section_display["OOIP"].dropna()
        if not ooip_vals.empty:
            colormap = cm.LinearColormap(
                colors=["#ffffcc", "#78c679", "#006837"],
                vmin=ooip_vals.min(),
                vmax=ooip_vals.max(),
            ).to_step(n=7)
            colormap.caption = "OOIP (bbl)"
            m.add_child(colormap)

            def section_style(feature):
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
            section_style = lambda f: NULL_STYLE
    else:
        section_style = lambda f: NULL_STYLE

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
            fields=["Section", "OOIP", "RFTD", "URF"],
            aliases=["Section:", "OOIP:", "RFTD:", "URF:"],
            localize=True,
            sticky=True,
        ),
    ).add_to(m)

    # --------------------------------------------------
    # 3️⃣  UNITS – drawn *first* and made non‑interactive
    # --------------------------------------------------
    folium.GeoJson(
        units_display.to_json(),
        name="Units",
        style_function=lambda _: {
            "color": "black",
            "weight": 2,
            "fillOpacity": 0,          # invisible fill
            "interactive": False,      # prevents the layer from capturing mouse events
        },
    ).add_to(m)

    # --------------------------------------------------
    # 4️⃣  EXISTING WELLS – points become CircleMarkers (Option A)
    # --------------------------------------------------
    # Separate point‑type and line‑type geometries
    point_wells = existing_display[
        existing_display.geometry.type == "Point"
    ]
    line_wells = existing_display[
        existing_display.geometry.type != "Point"
    ]

    # ---- 4a)  Draw well lines (unchanged) -----------------------
    folium.GeoJson(
        line_wells.to_json(),
        name="Existing Well Lines",
        style_function=lambda _: {"color": "red", "weight": 0.5},
    ).add_to(m)

    # ---- 4b)  Draw well points as CircleMarkers ---------------
    for _, row in point_wells.iterrows():
        # You can customise radius / colour based on any column here
        tooltip = folium.Tooltip(f"UWI: {row.get('UWI', '‑')}")
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,                     # pixel radius – adjust as you like
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.8,
            weight=1,
            tooltip=tooltip,
        ).add_to(m)

    # --------------------------------------------------
    # 5️⃣  PROSPECTS – added *after* units & wells so they sit on top
    # --------------------------------------------------
    NO_ANALOG_STYLE = {"color": "orange", "weight": 3, "dashArray": "5 5"}
    PASSING_STYLE   = {"color": "#2196F3", "weight": 3}
    FAILING_STYLE   = {"color": "#d3d3d3", "weight": 2}

    def prospect_style(feature):
        props = feature["properties"]
        if props.get("_no_analogs", False):
            return NO_ANALOG_STYLE
        if not props.get("_passes_filter", True):
            return FAILING_STYLE
        return PASSING_STYLE

    folium.GeoJson(
        p_display.to_json(),
        name="Prospects",
        style_function=prospect_style,
        highlight_function=lambda _: {"weight": 5, "color": "yellow"},
        tooltip=folium.GeoJsonTooltip(
            fields=prospect_tooltip_fields,
            aliases=prospect_tooltip_aliases,
            localize=True,
            sticky=True,
        ),
    ).add_to(m)

    # --------------------------------------------------
    # 6️⃣  LAYER CONTROL & render the map in Streamlit
    # --------------------------------------------------
    folium.LayerControl(collapsed=True).add_to(m)

    st_folium(
        m,
        use_container_width=True,
        height=900,
        returned_objects=[],
    )

# --------------------------------------------------
# Ranking Table
# --------------------------------------------------
with col_rank:
    st.header("📊 Prospect Ranking")

    if selected_metric == "High-Grade Score" and total_weight != 100:
        st.warning("Adjust weights to total 100% to see rankings.")
    else:
        metric_col = "HighGradeScore" if selected_metric == "High-Grade Score" else selected_metric
        ascending = selected_metric in ["Wcut", "URF", "RFTD"]

        rank_df = p[p["_passes_filter"]].copy()

        display_cols = [
            "Label", "_prospect_type", "Analog_Count",
            "EUR", "IP90", "1YCuml", "Wcut",
            "OOIP", "RFTD", "URF",
        ]
        if selected_metric == "High-Grade Score":
            display_cols.append("HighGradeScore")

        rank_df = rank_df[display_cols + [metric_col] if metric_col not in display_cols else display_cols].copy()
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

            # Rename for display
            rank_df = rank_df.rename(columns={
                "_prospect_type": "Type",
                "Analog_Count": "Analogs",
            })

            # Format dict
            fmt = {
                "EUR": "{:,.0f}",
                "IP90": "{:,.0f}",
                "1YCuml": "{:,.0f}",
                "Wcut": "{:.1f}",
                "OOIP": "{:,.0f}",
                "RFTD": "{:.3f}",
                "URF": "{:.3f}",
                "Percentile": "{:.0f}%",
            }
            if "HighGradeScore" in rank_df.columns:
                fmt["HighGradeScore"] = "{:.2f}"

            st.caption(
                f"Ranked by **{selected_metric}** · {len(rank_df)} prospects"
            )

            st.dataframe(
                rank_df.style.background_gradient(
                    subset=[metric_col], cmap="YlGn",
                    gmap=rank_df[metric_col] if not ascending else -rank_df[metric_col],
                ).format(fmt),
                use_container_width=True,
                height=750,
            )

            csv = rank_df.to_csv().encode("utf-8")
            st.download_button(
                "⬇️ Download Rankings",
                data=csv,
                file_name="bakken_prospect_rankings.csv",
                mime="text/csv",
            )

    # --- Flagged: no analogs ---
    no_analog_prospects = p[p["_no_analogs"]].copy()
    if not no_analog_prospects.empty:
        st.markdown("---")
        st.subheader("⚠️ No Analogs Found")
        st.dataframe(
            no_analog_prospects[["Label", "_prospect_type"]].rename(
                columns={"_prospect_type": "Type"}
            ).reset_index(drop=True),
            use_container_width=True,
        )
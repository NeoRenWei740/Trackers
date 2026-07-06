import streamlit as st
from spacetrack import SpaceTrackClient
import spacetrack.operators as op
from skyfield.api import EarthSatellite, load, wgs84
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import math
import bisect
import pandas as pd
import numpy as np

# =========================================================================
# Math & Spatial Helpers
# =========================================================================
def euclidean_km(p1, p2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def parse_tle_string(tle_string, ts):
    """Parses a block of TLE strings into chronologically sorted EarthSatellite objects."""
    lines = tle_string.strip().split('\n')
    entries = []
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
        l1, l2 = lines[i].strip(), lines[i + 1].strip()
        try:
            sat = EarthSatellite(l1, l2, "sat", ts)
            epoch_dt = sat.epoch.utc_datetime()
            entries.append((epoch_dt, sat))
        except Exception:
            continue
    entries.sort(key=lambda x: x[0])
    return entries


def best_tle(entries: list, t_dt: datetime):
    """Finds the closest TLE in time for accurate relative propagation."""
    if not entries:
        return None
    epochs = [e[0] for e in entries]
    idx = bisect.bisect_left(epochs, t_dt)
    if idx == 0:
        _, cand_sat = entries[0]
    elif idx >= len(entries):
        _, cand_sat = entries[-1]
    else:
        be, bs = entries[idx - 1]
        ae, as_ = entries[idx]
        if abs((ae - t_dt).total_seconds()) < abs((be - t_dt).total_seconds()):
            _, cand_sat = ae, as_
        else:
            _, cand_sat = be, bs
    return cand_sat


def build_time_grid(start_date, end_date, step_hours=3.0, max_frames=300):
    """Builds an evenly spaced UTC datetime grid across the selected range."""
    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)
    total_hours = max(1.0, (end_dt - start_dt).total_seconds() / 3600.0)

    effective_step = step_hours
    if total_hours / effective_step > max_frames:
        effective_step = total_hours / max_frames

    times, cur = [], start_dt
    while cur <= end_dt:
        times.append(cur)
        cur += timedelta(hours=effective_step)
    return times, effective_step


@st.cache_resource(show_spinner=False)
def get_client(user, pw):
    return SpaceTrackClient(identity=user, password=pw)


# =========================================================================
# Page Config
# =========================================================================
st.set_page_config(page_title="Orbital Tracker", layout="wide", page_icon="🛰️")

st.title("🛰️ Satellite TLE Data Explorer")
st.caption("Extract historical orbital elements, animate satellites on a 3D globe, "
           "and track proximity between two objects.")

if 'data_ready' not in st.session_state:
    st.session_state['data_ready'] = False

# =========================================================================
# Sidebar — Credentials (shared across tabs)
# =========================================================================
st.sidebar.header("🔑 Space-Track.org Login")
ST_USER = st.sidebar.text_input("Username (Email)")
ST_PASS = st.sidebar.text_input("Password", type="password")
if not ST_USER or not ST_PASS:
    st.sidebar.info("Enter your Space-Track credentials to enable data fetching.")

st.sidebar.markdown("---")

tab_tracker, tab_search = st.tabs(["📈 Orbital Tracker", "🔍 Satellite Search"])

# =========================================================================
# TAB 1: ORBITAL TRACKER
# =========================================================================
with tab_tracker:
    st.sidebar.header("Tracking Parameters")

    orbit_regime = st.sidebar.radio("Orbit Regime", ["GEO (Geosynchronous)", "LEO (Low Earth Orbit)"])

    if "GEO" in orbit_regime:
        default_cands = "39210, 28737"
        default_ref = "39209"
    else:
        default_cands = "43013, 48274, 40069"
        default_ref = "25544"

    sat_input = st.sidebar.text_input("Candidate NORAD IDs (comma separated)", value=default_cands)
    ref_sat_input = st.sidebar.text_input("Reference Satellite NORAD ID", value=default_ref)

    min_allowed_date = datetime(2003, 1, 1).date()
    max_allowed_date = datetime.now().date()

    start_date = st.sidebar.date_input("Start Date", value=max_allowed_date - timedelta(days=7),
                                        min_value=min_allowed_date, max_value=max_allowed_date)
    end_date = st.sidebar.date_input("End Date", value=max_allowed_date,
                                      min_value=min_allowed_date, max_value=max_allowed_date)

    run_button = st.sidebar.button("🚀 Fetch & Generate Graphs", use_container_width=True)

    if run_button:
        if not ST_USER or not ST_PASS:
            st.error("Please enter your Space-Track credentials in the sidebar.")
        elif start_date > end_date:
            st.error("Start Date must be before End Date.")
        else:
            try:
                with st.spinner("Fetching TLEs from Space-Track..."):
                    st_client = get_client(ST_USER, ST_PASS)
                    sat_list = [s.strip() for s in sat_input.split(",") if s.strip()]
                    ref_id = ref_sat_input.strip()

                    if not sat_list or not ref_id:
                        st.error("Please provide at least one candidate NORAD ID and a reference NORAD ID.")
                        st.stop()

                    drange = op.inclusive_range(start_date, end_date)
                    tle_data = st_client.gp_history(norad_cat_id=sat_list, epoch=drange, format='tle')
                    ref_tle_data = st_client.gp_history(norad_cat_id=ref_id, epoch=drange, format='tle')

                if not tle_data or not ref_tle_data:
                    st.warning("No TLE data found for these satellites/date range. Try widening the "
                               "date range or double-checking the NORAD IDs.")
                else:
                    ts = load.timescale()
                    ref_tles = parse_tle_string(ref_tle_data, ts)
                    cand_tles = {} 

                    plot_data = {sat: {'epoch': [], 'inc': [], 'raan': [], 'ecc': [], 'arg_pe': [],
                                        'mean_anom': [], 'mean_mo': [], 'semi_major': [], 'lon': [], 'dist': []}
                                 for sat in sat_list}

                    lines = tle_data.strip().split('\n')
                    for i in range(0, len(lines), 2):
                        if i + 1 >= len(lines):
                            break
                        l1, l2 = lines[i].strip(), lines[i + 1].strip()
                        try:
                            nid = str(int(l1[2:7]))
                        except ValueError:
                            continue
                        if nid not in plot_data:
                            continue

                        sat_obj = EarthSatellite(l1, l2, nid, ts)
                        t = sat_obj.epoch
                        t_dt = t.utc_datetime()

                        plot_data[nid]['epoch'].append(t_dt)
                        plot_data[nid]['inc'].append(math.degrees(sat_obj.model.inclo))
                        plot_data[nid]['raan'].append(math.degrees(sat_obj.model.nodeo))
                        plot_data[nid]['ecc'].append(sat_obj.model.ecco)
                        plot_data[nid]['arg_pe'].append(math.degrees(sat_obj.model.argpo))
                        plot_data[nid]['mean_anom'].append(math.degrees(sat_obj.model.mo))
                        plot_data[nid]['mean_mo'].append(sat_obj.model.no_kozai * 1440 / (2 * math.pi))
                        plot_data[nid]['lon'].append(sat_obj.at(t).subpoint().longitude.degrees)
                        
                        # Calculate Semi-Major Axis derived from Mean Motion using Kepler's 3rd Law
                        n_rad_s = sat_obj.model.no_kozai / 60.0
                        if n_rad_s > 0:
                            semi_major_km = (398600.4418 / (n_rad_s ** 2)) ** (1/3)
                        else:
                            semi_major_km = None
                        plot_data[nid]['semi_major'].append(semi_major_km)

                        ref_sat_best = best_tle(ref_tles, t_dt)
                        if ref_sat_best:
                            plot_data[nid]['dist'].append(
                                euclidean_km(tuple(sat_obj.at(t).position.km),
                                             tuple(ref_sat_best.at(t).position.km)))
                        else:
                            plot_data[nid]['dist'].append(None)

                        cand_tles.setdefault(nid, []).append((t_dt, sat_obj))

                    for nid in cand_tles:
                        cand_tles[nid].sort(key=lambda x: x[0])

                    all_sats_data = []
                    for sat in sat_list:
                        if plot_data[sat]['epoch']:
                            df_sat = pd.DataFrame(plot_data[sat])
                            df_sat['NORAD_ID'] = sat
                            all_sats_data.append(df_sat)

                    csv_data = None
                    if all_sats_data:
                        combined_df = pd.concat(all_sats_data, ignore_index=True)
                        cols = ['NORAD_ID', 'epoch', 'dist', 'lon', 'inc', 'raan', 'ecc', 'arg_pe',
                                'mean_anom', 'mean_mo', 'semi_major']
                        csv_data = combined_df[cols].to_csv(index=False).encode('utf-8')

                    st.session_state['plot_data'] = plot_data
                    st.session_state['sat_list'] = sat_list
                    st.session_state['ref_id'] = ref_id
                    st.session_state['csv_data'] = csv_data
                    st.session_state['ref_tles'] = ref_tles
                    st.session_state['cand_tles'] = cand_tles
                    st.session_state['slider_min'] = datetime.combine(
                        start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                    st.session_state['slider_max'] = datetime.combine(
                        end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
                    st.session_state['start_date'] = start_date
                    st.session_state['end_date'] = end_date
                    st.session_state['data_ready'] = True

            except Exception as e:
                st.error(f"Error: {e}")

    # ---------------------------------------------------------------
    # Rendering (only when data is ready)
    # ---------------------------------------------------------------
    if st.session_state.get('data_ready'):
        plot_data = st.session_state['plot_data']
        sat_list = st.session_state['sat_list']
        ref_id = st.session_state['ref_id']
        csv_data = st.session_state['csv_data']
        ref_tles = st.session_state['ref_tles']
        cand_tles = st.session_state['cand_tles']

        sat_colors = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD',
                      '#17BECF', '#E377C2', '#BCBD22', '#8C564B', '#FF9896']

        globe_tab, charts_tab, data_tab = st.tabs(["🌍 3D Globe (Animated)", "📊 Orbital Element Charts", "📥 Raw Data"])

        # -------------------- 3D GLOBE TAB --------------------
        with globe_tab:
            st.markdown(f"Animated 3D playback for candidates vs. **reference {ref_id}**. "
                        "The Earth is rendered as a translucent sphere so satellites on the far side "
                        "stay visible instead of being hidden behind the globe. "
                        "Use the slider or Play button to step through time, and drag to rotate/zoom.")

            focus_sat = st.selectbox("Highlight distance-to-reference for:", sat_list, key="focus_sat")

            interval_options = {"1 hour": 1.0, "3 hours": 3.0, "6 hours": 6.0, "12 hours": 12.0, "24 hours": 24.0}
            interval_label = st.selectbox("Animation time step", list(interval_options.keys()), index=1)
            step_hours = interval_options[interval_label]

            globe_opacity = st.slider("Globe translucency", 0.10, 0.80, 0.35, 0.05)

            if not ref_tles:
                st.warning("No reference satellite TLEs available to build the globe animation.")
            else:
                ts = load.timescale()
                times, effective_step = build_time_grid(st.session_state['start_date'],
                                                          st.session_state['end_date'],
                                                          step_hours=step_hours)
                
                sat_frames = {sat: {'x': [], 'y': [], 'z': [], 'dist': []} for sat in sat_list}
                ref_frames = {'x': [], 'y': [], 'z': []}

                for t_dt in times:
                    t_sf = ts.from_datetime(t_dt)
                    ref_best = best_tle(ref_tles, t_dt)
                    if ref_best:
                        rx, ry, rz = ref_best.at(t_sf).position.km
                        ref_frames['x'].append(rx)
                        ref_frames['y'].append(ry)
                        ref_frames['z'].append(rz)
                        ref_pos = (rx, ry, rz)
                    else:
                        ref_frames['x'].append(None)
                        ref_frames['y'].append(None)
                        ref_frames['z'].append(None)
                        ref_pos = None

                    for sat in sat_list:
                        best = best_tle(cand_tles.get(sat, []), t_dt)
                        if best:
                            sx, sy, sz = best.at(t_sf).position.km
                            sat_frames[sat]['x'].append(sx)
                            sat_frames[sat]['y'].append(sy)
                            sat_frames[sat]['z'].append(sz)
                            if ref_pos:
                                sat_frames[sat]['dist'].append(euclidean_km((sx, sy, sz), ref_pos))
                            else:
                                sat_frames[sat]['dist'].append(None)
                        else:
                            sat_frames[sat]['x'].append(None)
                            sat_frames[sat]['y'].append(None)
                            sat_frames[sat]['z'].append(None)
                            sat_frames[sat]['dist'].append(None)

                EARTH_RADIUS_KM = 6371.0
                _u = np.linspace(0, 2 * np.pi, 60)
                _v = np.linspace(0, np.pi, 60)
                _ex = EARTH_RADIUS_KM * np.outer(np.cos(_u), np.sin(_v))
                _ey = EARTH_RADIUS_KM * np.outer(np.sin(_u), np.sin(_v))
                _ez = EARTH_RADIUS_KM * np.outer(np.ones_like(_u), np.cos(_v))

                def make_earth():
                    return go.Surface(
                        x=_ex, y=_ey, z=_ez, opacity=globe_opacity,
                        colorscale=[[0, "#2b6cb0"], [1, "#2b6cb0"]],
                        showscale=False, hoverinfo='skip', name='Earth',
                    )

                def make_point(x, y, z, name, color, symbol='circle', size=6):
                    return go.Scatter3d(
                        x=[x] if x is not None else [], y=[y] if y is not None else [],
                        z=[z] if z is not None else [], mode='markers+text', text=[name], 
                        textposition='top center', name=name,
                        marker=dict(size=size, color=color, symbol=symbol, line=dict(width=1, color='white')),
                    )

                def make_link(x1, y1, z1, x2, y2, z2):
                    if None in (x1, y1, z1, x2, y2, z2):
                        return go.Scatter3d(x=[], y=[], z=[], mode='lines', name='Distance link', showlegend=False)
                    return go.Scatter3d(
                        x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                        line=dict(width=6, color='rgba(255, 80, 80, 0.9)', dash='dot'),
                        name='Distance link', showlegend=False,
                    )

                base_data = [make_earth(),
                             make_point(ref_frames['x'][0], ref_frames['y'][0], ref_frames['z'][0],
                                        f"REF {ref_id}", 'yellow', 'diamond', 9)]
                for idx, sat in enumerate(sat_list):
                    base_data.append(make_point(sat_frames[sat]['x'][0], sat_frames[sat]['y'][0],
                                                 sat_frames[sat]['z'][0], f"Sat {sat}",
                                                 sat_colors[idx % len(sat_colors)]))
                base_data.append(make_link(
                    sat_frames[focus_sat]['x'][0], sat_frames[focus_sat]['y'][0], sat_frames[focus_sat]['z'][0],
                    ref_frames['x'][0], ref_frames['y'][0], ref_frames['z'][0]
                ))

                frames = []
                for f_idx, t_dt in enumerate(times):
                    frame_data = [make_earth(),
                                  make_point(ref_frames['x'][f_idx], ref_frames['y'][f_idx], ref_frames['z'][f_idx],
                                             f"REF {ref_id}", 'yellow', 'diamond', 9)]
                    for idx, sat in enumerate(sat_list):
                        frame_data.append(make_point(sat_frames[sat]['x'][f_idx], sat_frames[sat]['y'][f_idx],
                                                      sat_frames[sat]['z'][f_idx], f"Sat {sat}",
                                                      sat_colors[idx % len(sat_colors)]))
                    frame_data.append(make_link(
                        sat_frames[focus_sat]['x'][f_idx], sat_frames[focus_sat]['y'][f_idx],
                        sat_frames[focus_sat]['z'][f_idx],
                        ref_frames['x'][f_idx], ref_frames['y'][f_idx], ref_frames['z'][f_idx]
                    ))

                    d = sat_frames[focus_sat]['dist'][f_idx]
                    dist_label = f"Distance {focus_sat} ↔ {ref_id}: {d:,.0f} km" if d is not None else "Distance: N/A"

                    frames.append(go.Frame(
                        data=frame_data, name=str(f_idx),
                        layout=go.Layout(
                            annotations=[dict(text=f"{t_dt.strftime('%Y-%m-%d %H:%M UTC')}  |  {dist_label}",
                                               showarrow=False, x=0.5, y=1.06, xref='paper', yref='paper',
                                               font=dict(size=14, color='white'))]
                        )
                    ))

                fig_globe = go.Figure(data=base_data, frames=frames)

                fig_globe.update_layout(
                    template="plotly_dark", height=700, margin=dict(t=60, b=10, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                        aspectmode='data', bgcolor="rgba(0,0,0,0)",
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5),
                    updatemenus=[dict(
                        type="buttons", showactive=False, y=0, x=0.05, xanchor="left", yanchor="top",
                        buttons=[
                            dict(label="▶ Play", method="animate",
                                 args=[None, dict(frame=dict(duration=400, redraw=True),
                                                   fromcurrent=True, transition=dict(duration=0))]),
                            dict(label="⏸ Pause", method="animate",
                                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                        ]
                    )],
                    sliders=[dict(
                        active=0, y=0, x=0.15, len=0.8, xanchor="left",
                        currentvalue=dict(prefix="Time: ", font=dict(size=12), visible=True, xanchor="right"),
                        steps=[dict(method="animate",
                                    args=[[str(i)], dict(mode="immediate",
                                                          frame=dict(duration=0, redraw=True),
                                                          transition=dict(duration=0))],
                                    label=t_dt.strftime('%m-%d %H:%M'))
                               for i, t_dt in enumerate(times)]
                    )]
                )

                st.plotly_chart(fig_globe, use_container_width=True)

        # -------------------- ORBITAL ELEMENT CHARTS TAB --------------------
        with charts_tab:
            st.markdown("### 🔎 Zoom Control")
            selected_range = st.slider(
                "Drag the handles to zoom in on a specific timeframe:",
                min_value=st.session_state['slider_min'],
                max_value=st.session_state['slider_max'],
                value=(st.session_state['slider_min'], st.session_state['slider_max']),
                format="YYYY-MM-DD",
                label_visibility="collapsed",
                key="chart_zoom_slider",
            )

            # --- 3-Axis Graph (Time, Distance, Longitude) ---
            st.markdown(f"### 🔀 Distance & Longitude Correlation (vs. {ref_id})")
            
            fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
            for idx, sat in enumerate(sat_list):
                if not plot_data[sat]['epoch']:
                    continue
                current_color = sat_colors[idx % len(sat_colors)]
                
                # Plot Distance (Left Y-Axis)
                fig_combo.add_trace(go.Scatter(
                    x=plot_data[sat]['epoch'], y=plot_data[sat]['dist'],
                    name=f"Sat {sat} (Distance)", mode='lines+markers', 
                    line=dict(color=current_color)
                ), secondary_y=False)
                
                # Plot Longitude (Right Y-Axis)
                fig_combo.add_trace(go.Scatter(
                    x=plot_data[sat]['epoch'], y=plot_data[sat]['lon'],
                    name=f"Sat {sat} (Longitude)", mode='lines+markers', 
                    line=dict(color=current_color, dash='dot'), opacity=0.6
                ), secondary_y=True)

            # Increased top margin (t=60) to give the title more room
            fig_combo.update_layout(
                template="plotly_dark", height=450, hovermode="x unified",
                margin=dict(t=60, b=30, l=50, r=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            fig_combo.update_xaxes(range=[selected_range[0], selected_range[1]], 
                                   showline=True, linewidth=1, linecolor='gray', mirror=True)
            fig_combo.update_yaxes(title_text=f"Distance to {ref_id} (km)", secondary_y=False, showgrid=False)
            fig_combo.update_yaxes(title_text="Longitude (°)", secondary_y=True, showgrid=False)
            
            st.plotly_chart(fig_combo, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📊 Complete Orbital Elements Breakdown")
            
            # Subplot reduced to 9 rows, increased vertical spacing, reordered titles
            fig = make_subplots(
                rows=9, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                subplot_titles=(
                    "Inclination (°)", "RAAN (°)", "Eccentricity", "Arg of Perigee (°)",
                    "Mean Anomaly (°)", "Mean Motion (rev/day)", "Longitude (°)", 
                    f"3D Distance to {ref_id} (km)", "Semi-Major Axis (km)"
                )
            )

            for idx, sat in enumerate(sat_list):
                if not plot_data[sat]['epoch']:
                    continue
                current_color = sat_colors[idx % len(sat_colors)]
                
                # Reordered to move semi-major axis to the 9th row
                params = [('inc', 1), ('raan', 2), ('ecc', 3), ('arg_pe', 4),
                          ('mean_anom', 5), ('mean_mo', 6), ('lon', 7), 
                          ('dist', 8), ('semi_major', 9)]

                for p_key, row in params:
                    if p_key == 'dist' and all(v is None for v in plot_data[sat][p_key]):
                        continue
                    fig.add_trace(go.Scatter(
                        x=plot_data[sat]['epoch'], y=plot_data[sat][p_key],
                        name=f"Sat {sat}", legendgroup=f"group_{sat}", showlegend=(row == 1),
                        mode='lines+markers', line=dict(color=current_color), marker=dict(color=current_color)
                    ), row=row, col=1)

            # Updated layout height, and increased top margin (t=80) to fix title clipping
            fig.update_layout(height=2300, hovermode="x unified", template="plotly_dark",
                               margin=dict(t=80, b=50, l=50, r=50))
            fig.update_xaxes(range=[selected_range[0], selected_range[1]],
                              showline=True, linewidth=1, linecolor='gray', mirror=True)
            fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
            fig.update_annotations(yshift=15)

            st.plotly_chart(fig, use_container_width=True)

        # -------------------- RAW DATA TAB --------------------
        with data_tab:
            st.markdown("### Combined orbital element table")
            if csv_data:
                st.download_button(
                    label="📥 Download Orbital Data (CSV)",
                    data=csv_data,
                    file_name="orbital_data_export.csv",
                    mime="text/csv",
                )
                preview_rows = []
                for sat in sat_list:
                    if plot_data[sat]['epoch']:
                        df_sat = pd.DataFrame(plot_data[sat])
                        df_sat['NORAD_ID'] = sat
                        preview_rows.append(df_sat)
                if preview_rows:
                    st.dataframe(pd.concat(preview_rows, ignore_index=True), use_container_width=True, height=500)
            else:
                st.info("No data available to display.")
    else:
        st.info("👈 Enter your credentials and tracking parameters in the sidebar, then click "
                "**Fetch & Generate Graphs** to get started.")

# =========================================================================
# TAB 2: SATELLITE SEARCH
# =========================================================================
with tab_search:
    st.markdown("### 🔍 Search the Space-Track Satellite Catalog")
    st.caption("Look up NORAD Catalog IDs by object name, country, launch year, or object type — "
               "handy for finding IDs to paste into the Orbital Tracker sidebar.")

    col1, col2, col3 = st.columns(3)
    with col1:
        name_query = st.text_input("Object name contains", placeholder="e.g. STARLINK, ISS, GPS")
    with col2:
        norad_query = st.text_input("NORAD ID(s), comma separated", placeholder="e.g. 25544, 43013")
    with col3:
        country_query = st.text_input("Country code", placeholder="e.g. US, PRC, CIS")

    col4, col5 = st.columns(2)
    with col4:
        object_type = st.selectbox("Object type", ["Any", "PAYLOAD", "ROCKET BODY", "DEBRIS"])
    with col5:
        max_results = st.number_input("Max results", min_value=10, max_value=1000, value=100, step=10)

    search_button = st.button("🔎 Search Catalog", use_container_width=True)

    if search_button:
        if not ST_USER or not ST_PASS:
            st.error("Please enter your Space-Track credentials in the sidebar.")
        elif not (name_query or norad_query or country_query):
            st.warning("Enter at least one search field (name, NORAD ID, or country).")
        else:
            try:
                with st.spinner("Querying Space-Track satellite catalog..."):
                    st_client = get_client(ST_USER, ST_PASS)
                    kwargs = {"orderby": "NORAD_CAT_ID", "limit": int(max_results)}

                    if name_query:
                        kwargs["satname"] = op.like(f"%{name_query.strip().upper()}%")
                    if norad_query:
                        ids = [s.strip() for s in norad_query.split(",") if s.strip()]
                        kwargs["norad_cat_id"] = ids
                    if country_query:
                        kwargs["country"] = country_query.strip().upper()
                    if object_type != "Any":
                        kwargs["object_type"] = object_type

                    results = st_client.satcat(**kwargs)

                if not results:
                    st.warning("No matching satellites found. Try broadening your search.")
                else:
                    df = pd.DataFrame(results)
                    keep_cols = [c for c in ["NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_TYPE", "COUNTRY",
                                              "LAUNCH", "DECAY", "PERIOD", "INCLINATION", "APOGEE",
                                              "PERIGEE", "CURRENT"] if c in df.columns]
                    df = df[keep_cols] if keep_cols else df
                    st.session_state['search_results'] = df
                    st.success(f"Found {len(df)} matching object(s).")
            except Exception as e:
                st.error(f"Search error: {e}")

    if st.session_state.get('search_results') is not None:
        df = st.session_state['search_results']
        st.markdown("#### Results")
        filter_text = st.text_input("Filter results (client-side, matches any column)", key="client_filter")
        display_df = df
        if filter_text:
            mask = df.apply(lambda row: row.astype(str).str.contains(filter_text, case=False, na=False).any(), axis=1)
            display_df = df.loc[mask]

        st.dataframe(display_df, use_container_width=True, height=450)

        csv_bytes = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Search Results (CSV)", data=csv_bytes,
                            file_name="satellite_search_results.csv", mime="text/csv")

        if "NORAD_CAT_ID" in display_df.columns and len(display_df) > 0:
            ids_str = ", ".join(display_df["NORAD_CAT_ID"].astype(str).tolist())
            st.text_area("Copy NORAD IDs for use in the Orbital Tracker tab:", value=ids_str, height=80)
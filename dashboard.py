# dashboard.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, time as dtime

st.set_page_config(page_title="Energiedashboard", layout="wide")
st.title("Laadprofielen, capaciteit & grid (15-min)")

# Bestandsnamen (zelfde map als dit script)
FILE_CAP = "gridpowertarget.csv"   # Beschikbaar vermogen per kwartier (kW)
FILE_BATT = "r_580_3.csv"            # SAM-export: rij1 runs (1..4), rij2 scaling, rijen 3-5 n.v.t., vanaf rij6 data

# ---------- Helpers ----------
def read_csv_robust(path: Path) -> pd.DataFrame:
    """Robuust CSV inlezen (vangt delimiter/quoting varianten af)."""
    if not path.exists():
        st.error(f"Bestand niet gevonden: {path.resolve()}")
        return None
    trials = [
        dict(engine="python"),
        dict(engine="python", sep=";"),
        dict(engine="python", sep=","),
        dict(engine="python", sep="\t"),
    ]
    for kw in trials:
        try:
            return pd.read_csv(path, **kw)
        except Exception:
            continue
    try:
        import csv
        with open(path, "r", encoding="utf-8") as f:
            sample = f.read(4096)
            dialect = csv.Sniffer().sniff(sample)
        return pd.read_csv(path, engine="python", sep=dialect.delimiter)
    except Exception as e:
        st.error(f"Kon {path.name} niet lezen: {e}")
        return None

def sam_parse_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verwacht kolommen: 'run', '1','2','3','4'
    Parameters starten vanaf rij 6 (index 5).
    """
    df.columns = [str(c).strip() for c in df.columns]
    if "run" not in df.columns:
        first = df.columns[0]
        df = df.rename(columns={first: "run"})
    needed = {"run", "1", "2", "3", "4"}
    if not needed.issubset(set(df.columns)):
        missing = needed - set(df.columns)
        st.error(f"r_580.csv mist kolommen: {missing}")
        return None
    if len(df) < 6:
        st.error("r_580.csv bevat minder dan 6 rijen; verwacht parameters vanaf rij 6.")
        return None
    data = df.iloc[5:].reset_index(drop=True).copy()
    data["run"] = data["run"].astype(str)
    for c in ["1", "2", "3", "4"]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data

def pick_default(params, *hints):
    for h in hints:
        for p in params:
            if h.lower() in p.lower():
                return p
    return params[0] if params else None

# ---------- Load data ----------
cap_df = read_csv_robust(Path(FILE_CAP))
bat_raw = read_csv_robust(Path(FILE_BATT))
if cap_df is None or bat_raw is None:
    st.stop()

data_df = sam_parse_data(bat_raw)
if data_df is None:
    st.stop()

params = data_df["run"].unique().tolist()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Mapping & instellingen")

    # Beschikbaar vermogen keuze
    st.subheader("Beschikbaar vermogen (kW)")
    cap_col = st.selectbox(
        "Kolom in gridpowertarget.csv",
        options=list(cap_df.columns),
        index=0
    )

    # Jaarladen-keuze -> map naar kolom positie 1..4
    st.subheader("Laden (MWh/jaar) → kolom 1..4")
    chosen_mwh = st.selectbox("Kies jaarladen (MWh)", [200, 400, 600, 800], index=0)
    col_map = {200: "1", 400: "2", 600: "3", 800: "4"}
    scale_col = col_map[chosen_mwh]

    # Parameters voor flows naar load
    st.subheader("Flows naar Load (kW)")
    default_batt = pick_default(params, "batt_to_load", "Electricity to load from battery")
    default_grid = pick_default(params, "grid_to_load", "Electricity to load from grid")
    default_sys  = pick_default(params, "system_to_load", "pv_to_load", "Electricity to load from system")
    batt_param = st.selectbox("Battery → Load", options=params, index=params.index(default_batt) if default_batt in params else 0)
    grid_param = st.selectbox("Grid → Load",    options=params, index=params.index(default_grid) if default_grid in params else 0)
    sys_param  = st.selectbox("System → Load",  options=params, index=params.index(default_sys)  if default_sys  in params else 0)

    # Battery SOC parameter
    st.subheader("Battery State of Charge (%)")
    soc_candidates = [p for p in params if "batt_SOC" in p.lower() or "state of charge" in p.lower()]
    if not soc_candidates:
        soc_candidates = [p for p in params if "soc" in p.lower()]
    default_soc = pick_default(soc_candidates, "batt_SOC", "state of charge", "soc")
    soc_param = st.selectbox("Battery SOC (%)", options=soc_candidates,
                            index=soc_candidates.index(default_soc) if default_soc in soc_candidates else 0)

    # Parameter voor grid reeks (kW per kwartier, negatief=inkoop, positief=verkoop)
    st.subheader("Grid power reeks (kW/kwartier) voor cap-vergelijk")
    candidates = [p for p in params if "grid_power" in p.lower()]
    if not candidates:
        candidates = [p for p in params if "grid" in p.lower()]
    default_grid_power = pick_default(candidates, "grid_power", "grid", "power")
    grid_power_param = st.selectbox("grid_power (negatief=inkoop, positief=verkoop)", options=candidates,
                                   index=candidates.index(default_grid_power) if default_grid_power in candidates else 0)

    # Tijdstap & shift
    st.subheader("Tijd & resolutie")
    step_minutes = st.number_input("Tijdstap (minuten)", min_value=1, max_value=60, value=15)
    treat_values_as = st.radio("Flows-eenheid", ["kW (vermogen per kwartier)", "kWh per kwartier"], index=0)

    st.subheader("Align grid-hourly met capaciteit")
    shift_n = st.number_input(
        "Shift grid-hourly (x-1 = 1)", min_value=0, max_value=10, value=1,
        help="Positieve shift 1 = gebruik waarde van vorige timestep (x-1) om te matchen met capaciteit."
    )

    # X-as als datum/tijd op basis van kwartierstappen
    st.subheader("X-as tijdinstellingen")
    start_date = st.date_input("Startdatum", value=pd.Timestamp("2025-01-01").date())
    start_time = st.time_input("Starttijd", value=dtime(0, 0))

# ---------- Extract ----------
def get_series(param_name: str) -> pd.Series:
    s = data_df.loc[data_df["run"] == param_name, scale_col].reset_index(drop=True)
    return pd.to_numeric(s, errors="coerce")

cap_series = pd.to_numeric(cap_df[cap_col], errors="coerce").reset_index(drop=True)
batt_series = get_series(batt_param)
grid_series = get_series(grid_param)
sys_series  = get_series(sys_param)
soc_series  = get_series(soc_param)  # Battery SOC in %

# grid_power is kW per kwartier (negatief=inkoop, positief=verkoop):
grid_power_kw = get_series(grid_power_param)  # kW per kwartier
# Validatie
if grid_power_kw.isna().all() or len(grid_power_kw) == 0:
    st.error("De gekozen 'grid power' regel bevat geen numerieke waarden. Kies een andere regel in de sidebar.")
    st.stop()

# Shift toepassen indien gewenst
if shift_n > 0:
    grid_power_kw = grid_power_kw.shift(shift_n).bfill().fillna(0)

# Converteer logica: negatief=inkoop, positief=verkoop -> voor capaciteit nemen we absolute waarde
# Maar eerst * -1 om de logica om te draaien: negatief wordt positief (inkoop)
grid_import_kw = (grid_power_kw * -1).clip(lower=0).fillna(0)  # alleen inkoop (oorspronkelijk negatief)

# ---- Flows naar kW normaliseren voor plotting/ruimte ----
hours_per_step = step_minutes / 60.0
def to_kw(series: pd.Series) -> pd.Series:
    return series if treat_values_as.startswith("kW") else (series / hours_per_step)

batt_kw = to_kw(batt_series)
grid_kw = to_kw(grid_series)
sys_kw  = to_kw(sys_series)
cap_kw  = cap_series  # capaciteit is al kW

# ---- Lengtes gelijk trekken (grid_import_kw is al op kwartierbasis) ----
n = min(len(cap_kw), len(batt_kw), len(grid_kw), len(sys_kw), len(grid_import_kw), len(soc_series))
cap_kw         = cap_kw.iloc[:n].reset_index(drop=True).fillna(0)
batt_kw        = batt_kw.iloc[:n].reset_index(drop=True)
grid_kw        = grid_kw.iloc[:n].reset_index(drop=True)
sys_kw         = sys_kw.iloc[:n].reset_index(drop=True)
grid_import_kw = grid_import_kw.iloc[:n].reset_index(drop=True).fillna(0)
soc_pct        = soc_series.iloc[:n].reset_index(drop=True).fillna(0)  # SOC in %

# Som van flows naar load (kW)
sum_flows_kw = (batt_kw.fillna(0) + grid_kw.fillna(0) + sys_kw.fillna(0))

# Datetime x-as
start_dt = datetime.combine(start_date, start_time)
t = pd.date_range(start=start_dt, periods=n, freq=f"{step_minutes}min")

# Ruimte op aansluiting (kW) = beschikbaar - grid inkoop
ruimte_kw = (cap_kw - grid_import_kw).fillna(0)
ruimte_neg = ruimte_kw.where(ruimte_kw < 0, 0)  # alleen negatieve delen voor rood

# ---------- Grafiek 1: Flows naar Load (gestapeld) + SOC ----------
st.subheader("Flows naar Load (gestapeld) + somlijn + Battery SOC")
fig1 = go.Figure()
# Onderste laag: System → Load (verborgen maar in stackgroup)
fig1.add_trace(go.Scatter(x=t, y=sys_kw.clip(lower=0),   name="System → Load",  mode="lines", stackgroup="one", visible="legendonly"))
# Midden: Battery → Load (zichtbaar)
fig1.add_trace(go.Scatter(x=t, y=batt_kw.clip(lower=0),  name="Battery → Load", mode="lines", stackgroup="one"))
# Boven: Grid → Load (verborgen maar in stackgroup)
fig1.add_trace(go.Scatter(x=t, y=grid_kw.clip(lower=0),  name="Grid → Load",    mode="lines", stackgroup="one", visible="legendonly"))
# Somlijn (verborgen en niet gestapeld)
fig1.add_trace(go.Scatter(x=t, y=sum_flows_kw,           name="Som flows → Load",
                          mode="lines", line=dict(dash="dash"), visible="legendonly"))
# SOC op secundaire y-as (zichtbaar)
fig1.add_trace(go.Scatter(x=t, y=soc_pct, name="Battery SOC (%)", 
                          mode="lines", yaxis="y2", line=dict(color="orange", width=3)))

# Layout met secundaire y-as voor SOC
fig1.update_layout(
    xaxis_title="Tijd", 
    yaxis_title="kW",
    yaxis2=dict(title="SOC (%)", overlaying="y", side="right", range=[0, 100]),
    legend=dict(orientation="h", y=1.05),
    xaxis=dict(range=["2025-06-01", "2025-06-08"])  # Zoom in op eerste week van juni
)
st.plotly_chart(fig1, use_container_width=True, key="flows_stacked")

# ---------- Sankey: geaggregeerde energiestromen naar Load (met %) ----------
st.subheader("Sankey: geaggregeerde energiestromen naar Load (%)")
batt_energy = batt_kw.clip(lower=0).sum() * hours_per_step
grid_energy = grid_kw.clip(lower=0).sum() * hours_per_step
sys_energy  = sys_kw.clip(lower=0).sum()  * hours_per_step
total_energy = batt_energy + grid_energy + sys_energy
if total_energy <= 0:
    total_energy = 1e-9

labels = ["Battery", "Grid", "System", "Load"]
sources = [0, 1, 2]
targets = [3, 3, 3]
values  = [batt_energy, grid_energy, sys_energy]
percents = [v / total_energy * 100 for v in values]
link_labels = [
    f"Battery → Load ({percents[0]:.1f}%)",
    f"Grid → Load ({percents[1]:.1f}%)",
    f"System → Load ({percents[2]:.1f}%)",
]

fig_sk = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(label=labels, pad=15, thickness=20),
    link=dict(source=sources, target=targets, value=values, label=link_labels),
)])
fig_sk.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_sk, use_container_width=True, key="sankey_flows_pct")

# ---------- Grafiek 2: Capaciteit vs netto inkoop + ruimte op aansluiting ----------
st.subheader("Capaciteit vs netto inkoop + ‘ruimte op aansluiting’")

cap_vs_grid = go.Figure()
# Lijnen: capaciteit en grid inkoop
cap_vs_grid.add_trace(go.Scatter(x=t, y=cap_kw,         mode="lines", name=f"Beschikbaar ({cap_col})"))
cap_vs_grid.add_trace(go.Scatter(x=t, y=grid_import_kw, mode="lines", name="Grid inkoop (kW)"))
# Som-lijn: aanwezig maar standaard uit
cap_vs_grid.add_trace(go.Scatter(
    x=t, y=sum_flows_kw, mode="lines",
    name="Som batt+grid+system → Load",
    line=dict(dash="dot"),
    visible="legendonly"
))
# Ruimte op aansluiting (positief/negatief)
cap_vs_grid.add_trace(go.Scatter(x=t, y=ruimte_kw, mode="lines", name="Ruimte op aansluiting"))
# Rood vlak voor overschrijding (negatieve ruimte)
cap_vs_grid.add_trace(go.Scatter(
    x=t, y=ruimte_neg, mode="lines", name="Overschrijding (negatief)",
    fill="tozeroy"
))
# SOC op secundaire y-as
cap_vs_grid.add_trace(go.Scatter(x=t, y=soc_pct, name="Battery SOC (%)", 
                                mode="lines", yaxis="y2", line=dict(color="orange", width=3)))

cap_vs_grid.update_layout(
    xaxis_title="Tijd", 
    yaxis_title="kW",
    yaxis2=dict(title="SOC (%)", overlaying="y", side="right", range=[0, 100]),
    legend=dict(orientation="h", y=1.05),
    xaxis=dict(range=["2025-06-01", "2025-06-08"])  # Zoom in op eerste week van juni
)
st.plotly_chart(cap_vs_grid, use_container_width=True, key="capacity_vs_grid_room")

# ---------- Metrics & Debug ----------
neg_pct = (ruimte_kw < 0).mean() * 100 if len(ruimte_kw) else 0.0
st.metric("Tijd met overschrijding", f"{neg_pct:.1f}%")
st.metric("Min. ruimte", f"{ruimte_kw.min():.1f} kW" if len(ruimte_kw) else "—")
st.metric("P95 grid inkoop", f"{np.percentile(grid_import_kw, 95):.1f} kW" if len(grid_import_kw) else "—")

with st.expander("Debug: eerste 10 waardes"):
    st.write(pd.DataFrame({
        "cap_kw": cap_kw[:10].values,
        "grid_import_kw": grid_import_kw[:10].values,
        "ruimte_kw": ruimte_kw[:10].values,
    }))

# ---------- ROBUUSTE EXPORT ----------
def to_series(x):
    if isinstance(x, pd.Series):
        return x.reset_index(drop=True)
    return pd.Series(x)

cols = {
    "beschikbaar_kw": cap_kw,
    "grid_inkoop_kw": grid_import_kw,
    "ruimte_op_aansluiting_kw": ruimte_kw,
    "batt_to_load_kw": batt_kw,
    "grid_to_load_kw": grid_kw,
    "system_to_load_kw": sys_kw,
    "som_flows_kw": sum_flows_kw,
    "battery_soc_pct": soc_pct,
}
cols = {k: to_series(v) for k, v in cols.items()}
lengths = {k: len(v) for k, v in cols.items()}
n_common = min(lengths.values())

if n_common == 0:
    st.error("Geen data om te exporteren (n_common == 0). Controleer de mappings/filters.")
else:
    t_common = pd.date_range(start=start_dt, periods=n_common, freq=f"{step_minutes}min")
    aligned = {k: v.iloc[:n_common].values for k, v in cols.items()}
    out = pd.DataFrame(aligned, index=t_common)
    out.index.name = "datetime"

    st.download_button(
        "Download gecombineerde resultaten (CSV)",
        data=out.to_csv(index=True).encode("utf-8"),
        file_name="flows_capaciteit_grid.csv",
        mime="text/csv",
    )

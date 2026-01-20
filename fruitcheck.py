import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Residue Monitoring Dashboard", layout="wide")

EXPECTED = {
    "datum", "proben_id", "produktgruppe", "produkt", "herkunft", "wirkstoff",
    "methode", "labor", "loq_mgkg", "mrl_mgkg", "ergebnis_mgkg", "befundflag"
}

@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None, path=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(path)
    # Normalize column names (just in case)
    df.columns = [c.strip() for c in df.columns]
    # Parse date
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    return df

st.title("Residue Monitoring (Pestizidrückstände) — Dashboard")

with st.sidebar:
    st.header("📁 Datenquelle")
    uploaded = st.file_uploader("CSV hochladen", type=["csv"])
    use_example = st.toggle("Beispieldatei verwenden", value=False, help="Nutze lokale Datei, wenn vorhanden.")

    path = None
    if use_example:
        path = "fruitmonitoring_dez2025_simulated_realistic_mrl.csv"

df = None
if uploaded is not None:
    df = load_data(uploaded_file=uploaded)
elif use_example:
    try:
        df = load_data(path=path)
    except Exception as e:
        st.error(f"Konnte Beispiel-Datei nicht laden: {e}")

if df is None:
    st.info("Bitte CSV hochladen (Sidebar) oder Beispieldatei aktivieren.")
    st.stop()

missing = EXPECTED - set(df.columns)
if missing:
    st.warning(f"Fehlende Spalten: {sorted(missing)}. Einige Plots funktionieren evtl. nicht.")

# Flags (robust)
if "befundflag" in df.columns:
    df["detected"] = df["befundflag"].isin(["quantifiziert", ">MRL"])
    df["exceedance"] = df["befundflag"].eq(">MRL")
else:
    df["detected"] = False
    df["exceedance"] = False

# Numeric conversions
for col in ["loq_mgkg", "mrl_mgkg", "ergebnis_mgkg"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Sidebar filters ---
with st.sidebar:
    st.header("Filter")

    # Date filter
    if "datum" in df.columns and df["datum"].notna().any():
        dmin, dmax = df["datum"].min(), df["datum"].max()
        date_range = st.date_input("Datum", value=(dmin.date(), dmax.date()))
    else:
        date_range = None

    def multiselect_if(col, label):
        if col in df.columns:
            opts = sorted([x for x in df[col].dropna().unique()])
            return st.multiselect(label, opts, default=[])
        return []

    sel_pg = multiselect_if("produktgruppe", "Produktgruppe")
    sel_origin = multiselect_if("herkunft", "Herkunft")
    sel_sub = multiselect_if("wirkstoff", "Wirkstoff")
    sel_lab = multiselect_if("labor", "Labor")

# Apply filters
f = df.copy()
if date_range and "datum" in f.columns:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    f = f[(f["datum"] >= start) & (f["datum"] <= end)]

def apply_ms(col, values):
    global f
    if values and col in f.columns:
        f = f[f[col].isin(values)]

apply_ms("produktgruppe", sel_pg)
apply_ms("herkunft", sel_origin)
apply_ms("wirkstoff", sel_sub)
apply_ms("labor", sel_lab)

# --- KPIs ---
c1, c2, c3, c4, c5 = st.columns(5)
n_results = len(f)
n_samples = f["proben_id"].nunique() if "proben_id" in f.columns else np.nan
det_rate = float(f["detected"].mean()) if n_results else 0.0
exc_rate = float(f["exceedance"].mean()) if n_results else 0.0

c1.metric("Ergebnisse", f"{n_results:,}")
c2.metric("Proben (unique)", f"{int(n_samples):,}" if pd.notna(n_samples) else "—")
c3.metric("Nachweisrate", f"{det_rate*100:.1f}%")
c4.metric("Überschreitungsrate", f"{exc_rate*100:.2f}%")
if "mrl_mgkg" in f.columns and "ergebnis_mgkg" in f.columns:
    ratio = (f["ergebnis_mgkg"] / f["mrl_mgkg"]).replace([np.inf, -np.inf], np.nan)
    c5.metric("Median Ergebnis/MRL", f"{np.nanmedian(ratio):.3f}" if np.isfinite(np.nanmedian(ratio)) else "—")
else:
    c5.metric("Median Ergebnis/MRL", "—")

st.divider()

# --- Layout: Plots ---
left, right = st.columns((1.15, 0.85))

# 1) Trend: Nachweisrate pro Tag
with left:
    st.subheader("Trend: Nachweisrate pro Tag")
    if "datum" in f.columns and f["datum"].notna().any():
        daily = (f.groupby(f["datum"].dt.date)["detected"]
                   .mean()
                   .reset_index(name="nachweisrate"))
        fig = px.line(daily, x="datum", y="nachweisrate", markers=True)
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Keine gültige Datumsspalte für Trendplot.")

# 2) Top substances (detections)
with right:
    st.subheader("Top-Wirkstoffe (Nachweise)")
    if "wirkstoff" in f.columns:
        top = (f[f["detected"]]
               .groupby("wirkstoff")
               .size()
               .reset_index(name="n")
               .sort_values("n", ascending=False)
               .head(12))
        fig = px.bar(top, x="n", y="wirkstoff", orientation="h")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Spalte 'wirkstoff' fehlt.")

st.divider()

# 3) Heatmap: Überschreitungsrate Produktgruppe × Herkunft
st.subheader("Heatmap: Überschreitungsrate (Produktgruppe × Herkunft)")
if all(c in f.columns for c in ["produktgruppe", "herkunft"]):
    heat = (f.pivot_table(index="produktgruppe", columns="herkunft",
                          values="exceedance", aggfunc="mean", fill_value=0.0))
    # plotly heatmap
    fig = px.imshow(heat, aspect="auto", origin="lower")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_coloraxes(colorbar_title="Rate")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Für Heatmap werden 'produktgruppe' und 'herkunft' benötigt.")

# 4) Treemap: wo kommen Ergebnisse her? (Produktgruppe → Produkt → Herkunft)
st.subheader("Struktur: Produktgruppe → Produkt → Herkunft (Treemap)")
if all(c in f.columns for c in ["produktgruppe", "produkt", "herkunft"]):
    tre = (f.groupby(["produktgruppe", "produkt", "herkunft"])
             .size()
             .reset_index(name="n"))
    fig = px.treemap(tre, path=["produktgruppe", "produkt", "herkunft"], values="n")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Für Treemap werden 'produktgruppe', 'produkt', 'herkunft' benötigt.")

# 5) Exceedances: Verteilung Ergebnis/MRL (nur >MRL)
st.subheader("Überschreitungen: Ergebnis/MRL-Verteilung")
if all(c in f.columns for c in ["ergebnis_mgkg", "mrl_mgkg"]) and f["exceedance"].any():
    ex = f[f["exceedance"]].copy()
    ex["exceed_factor"] = ex["ergebnis_mgkg"] / ex["mrl_mgkg"]
    fig = px.histogram(ex, x="exceed_factor", nbins=30)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Top combinations table
    st.caption("Top Kombinationen bei Überschreitungen")
    cols = [c for c in ["produktgruppe", "produkt", "herkunft", "wirkstoff", "labor"] if c in ex.columns]
    top_ex = (ex.groupby(cols).size().reset_index(name="n_exceed")
                .sort_values("n_exceed", ascending=False).head(20))
    st.dataframe(top_ex, use_container_width=True)
else:
    st.info("Keine (oder zu wenige) Überschreitungen im aktuellen Filterbereich.")

st.divider()
st.caption("Hinweis: Dashboard arbeitet mit Monitoring-/Exportdaten. MRLs sind rechtliche Grenzwerte (kein toxikologischer Grenzwert).")
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Fruit Monitoring Dashboard",
    layout="wide",
)

st.title("Rückstandsmonitoring – Dashboard")
st.caption("CSV Upload | Produkte · Herkunft · Wirkstoffe · MRL-Überschreitungen")


# -----------------------------
# Robust column mapping
# -----------------------------
# Mapping keys are compared lowercased.
COLUMN_MAP = {
    # Produkt
    "produkt": "produkt",
    "product": "produkt",
    "commodity": "produkt",
    "item": "produkt",

    # Herkunft
    "herkunftsland": "herkunftsland",
    "origin": "herkunftsland",
    "country": "herkunftsland",
    "country_of_origin": "herkunftsland",
    "origin_country": "herkunftsland",
    "herkunft": "herkunftsland",

    # Wirkstoff
    "wirkstoff": "wirkstoff",
    "pesticide": "wirkstoff",
    "active_substance": "wirkstoff",
    "active substance": "wirkstoff",
    "substance": "wirkstoff",

    # Datum
    "datum": "datum",
    "date": "datum",
    "sampling_date": "datum",
    "sampling date": "datum",
    "proben_datum": "datum",
    "sample_date": "datum",

    # Messwert / Ergebnis
    "result": "result",
    "ergebnis": "result",
    "messwert": "result",
    "wert": "result",
    "value": "result",
    "concentration": "result",
    "measured_value": "result",
    "measured value": "result",

    # MRL
    "mrl": "mrl",
    "mrl_value": "mrl",
    "mrl value": "mrl",
    "mrl_wert": "mrl",
    "grenzwert": "mrl",
    "limit": "mrl",

    # LOQ
    "loq": "loq",
    "loq_value": "loq",
    "loq value": "loq",
    "loq_wert": "loq",
    "bestimmungsgrenze": "loq",
    "quant_limit": "loq",
    "quantification_limit": "loq",

    # Optional
    "einheit": "einheit",
    "unit": "einheit",
    "labor": "labor",
    "lab": "labor",
    "befund": "befund",
    "status": "befund",
}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find a column by exact match or substring match (case-insensitive)."""
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}

    # exact match
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    # substring match
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c

    return None


def normalize_df(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # 1) Apply basic rename using COLUMN_MAP on lowercase
    rename_dict = {}
    for c in df.columns:
        mapped = COLUMN_MAP.get(c.lower())
        if mapped and mapped != c:
            rename_dict[c] = mapped
    if rename_dict:
        df = df.rename(columns=rename_dict)

    # 2) Fallback detection for key columns if still missing
    if "produkt" not in df.columns:
        c = _find_col(df, ["produkt", "product", "commodity", "item", "artikel", "warenart"])
        if c and c != "produkt":
            df = df.rename(columns={c: "produkt"})

    if "herkunftsland" not in df.columns:
        c = _find_col(df, ["herkunftsland", "country", "origin", "country_of_origin", "origin_country", "herkunft"])
        if c and c != "herkunftsland":
            df = df.rename(columns={c: "herkunftsland"})

    if "wirkstoff" not in df.columns:
        c = _find_col(df, ["wirkstoff", "pesticide", "active", "active_substance", "substance"])
        if c and c != "wirkstoff":
            df = df.rename(columns={c: "wirkstoff"})

    if "datum" not in df.columns:
        c = _find_col(df, ["datum", "date", "sampling_date", "sample_date", "proben_datum"])
        if c and c != "datum":
            df = df.rename(columns={c: "datum"})

    if "result" not in df.columns:
        c = _find_col(df, ["result", "ergebnis", "messwert", "wert", "value", "concentration"])
        if c and c != "result":
            df = df.rename(columns={c: "result"})

    if "mrl" not in df.columns:
        c = _find_col(df, ["mrl", "mrl_value", "mrl_wert", "limit", "grenzwert"])
        if c and c != "mrl":
            df = df.rename(columns={c: "mrl"})

    if "loq" not in df.columns:
        c = _find_col(df, ["loq", "loq_value", "loq_wert", "bestimmungsgrenze", "quantification_limit"])
        if c and c != "loq":
            df = df.rename(columns={c: "loq"})

    # 3) Parse date & helper columns
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
        df["monat"] = df["datum"].dt.to_period("M").astype(str)
        df["jahr"] = df["datum"].dt.year.astype("Int64")
        df["kw"] = df["datum"].dt.isocalendar().week.astype("Int64")
    else:
        df["datum"] = pd.NaT
        df["monat"] = "Unbekannt"
        df["jahr"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        df["kw"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # 4) Coerce numeric columns if present
    for col in ["result", "mrl", "loq"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5) Exceedance flag (only if possible)
    if "result" in df.columns and "mrl" in df.columns:
        df["mrl_ratio"] = df["result"] / df["mrl"]
        df["is_exceedance"] = (df["result"] > df["mrl"]) & df["result"].notna() & df["mrl"].notna()
    else:
        df["mrl_ratio"] = np.nan
        df["is_exceedance"] = False

    # 6) Clean strings
    for col in ["produkt", "herkunftsland", "wirkstoff", "labor", "einheit", "befund"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "n_rows": int(n),
        "n_products": int(df["produkt"].nunique()) if "produkt" in df.columns else None,
        "n_countries": int(df["herkunftsland"].nunique()) if "herkunftsland" in df.columns else None,
        "n_actives": int(df["wirkstoff"].nunique()) if "wirkstoff" in df.columns else None,
        "exceed_rate": float(df["is_exceedance"].mean()) if "is_exceedance" in df.columns and n else 0.0,
    }


def top_table(df: pd.DataFrame, group_col: str, n: int = 12) -> pd.DataFrame:
    if group_col not in df.columns or "is_exceedance" not in df.columns:
        return pd.DataFrame()

    g = df.groupby(group_col, dropna=False)
    out = pd.DataFrame({
        "n": g.size(),
        "exceedances": g["is_exceedance"].sum(),
    })
    out["exceed_rate"] = out["exceedances"] / out["n"]
    out = out.sort_values(["exceedances", "exceed_rate", "n"], ascending=False).head(n).reset_index()
    return out


def plot_trend_exceedance(df: pd.DataFrame):
    if "monat" not in df.columns or "is_exceedance" not in df.columns:
        return None

    trend = (
        df.groupby("monat", dropna=False)["is_exceedance"]
        .mean()
        .reset_index()
        .rename(columns={"is_exceedance": "exceed_rate"})
        .sort_values("monat")
    )

    fig = px.line(
        trend,
        x="monat",
        y="exceed_rate",
        markers=True,
        title="Trend: MRL-Überschreitungsrate pro Monat",
        labels={"monat": "Monat", "exceed_rate": "Überschreitungsrate"},
    )
    fig.update_layout(template="plotly_white")
    fig.update_yaxes(tickformat=".0%")
    return fig


def seaborn_heatmap_product_x_country(df: pd.DataFrame, top_products: int = 12, top_countries: int = 12):
    needed = {"produkt", "herkunftsland", "is_exceedance"}
    if not needed.issubset(df.columns):
        return None

    prods = df["produkt"].value_counts().head(top_products).index
    countries = df["herkunftsland"].value_counts().head(top_countries).index
    sub = df[df["produkt"].isin(prods) & df["herkunftsland"].isin(countries)].copy()

    pivot = (
        sub.pivot_table(
            index="produkt",
            columns="herkunftsland",
            values="is_exceedance",
            aggfunc="mean",
        )
        .fillna(0)
        .sort_index()
    )

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, ax=ax, cmap="viridis")
    ax.set_title("Heatmap: Überschreitungsrate nach Produkt × Herkunftsland")
    ax.set_xlabel("Herkunftsland")
    ax.set_ylabel("Produkt")
    plt.tight_layout()
    return fig


# -----------------------------
# Choropleth helpers (DE -> EN -> ISO3)
# -----------------------------
ALIASES = {
    # Deutsch → Englisch (pycountry-kompatibel)
    "deutschland": "Germany",
    "belgien": "Belgium",
    "italien": "Italy",
    "spanien": "Spain",
    "niederlande": "Netherlands",
    "polen": "Poland",
    "türkei": "Turkey",
    "marokko": "Morocco",
    "österreich": "Austria",
    "schweiz": "Switzerland",
    "frankreich": "France",
    "griechenland": "Greece",
    "portugal": "Portugal",
    "ungarn": "Hungary",
    "rumänien": "Romania",
    "bulgarien": "Bulgaria",
    "kroatien": "Croatia",
    "tschechien": "Czechia",
    "slowakei": "Slovakia",
    "slowenien": "Slovenia",
    "türkei": "Türkiye",

    # Englische Sonderfälle / Abkürzungen
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "czech republic": "Czechia",
    "ivory coast": "Côte d'Ivoire",
    "russia": "Russian Federation",
    "south korea": "Korea, Republic of",
    "north korea": "Korea, Democratic People's Republic of",
    "vietnam": "Viet Nam",
}


def choropleth_agg(filtered: pd.DataFrame):
    import pycountry

    if not {"herkunftsland", "is_exceedance"}.issubset(filtered.columns):
        return None, "Benötigt Spalten: herkunftsland, is_exceedance"

    agg = (
        filtered.groupby("herkunftsland", dropna=False)
        .agg(
            n=("herkunftsland", "size"),
            exceedances=("is_exceedance", "sum"),
        )
        .reset_index()
        .rename(columns={"herkunftsland": "country_name"})
    )
    agg["exceed_rate"] = np.where(agg["n"] > 0, agg["exceedances"] / agg["n"], 0.0)

    def to_iso3(name: str):
        if pd.isna(name):
            return None
        name_clean = str(name).strip().lower()
        name_clean = ALIASES.get(name_clean, name_clean)  # DE/alias -> EN
        try:
            return pycountry.countries.lookup(name_clean).alpha_3
        except Exception:
            return None

    agg["iso_a3"] = agg["country_name"].apply(to_iso3)

    unmatched = (
        agg.loc[agg["iso_a3"].isna(), "country_name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    warn = None
    if unmatched:
        warn = "Nicht gematchte Länder (pycountry): " + ", ".join(unmatched[:12]) + (" ..." if len(unmatched) > 12 else "")

    return agg, warn


def plot_choropleth(agg: pd.DataFrame, metric: str):
    if metric == "Probenanzahl":
        col = "n"
        label = "Proben"
    elif metric == "Überschreitungen":
        col = "exceedances"
        label = "Überschreitungen"
    else:
        col = "exceed_rate"
        label = "Überschreitungsrate"

    fig = px.choropleth(
        agg,
        locations="iso_a3",
        color=col,
        hover_name="country_name",
        hover_data={
            "n": True,
            "exceedances": True,
            "exceed_rate": ":.1%",
            "iso_a3": False,
        },
        color_continuous_scale="Viridis",
        title=f"Weltkarte: {label}",
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=50, b=0),
        height=520,
    )

    fig.update_geos(
        showcountries=True,
        showcoastlines=False,
        showland=True,
    )

    if metric == "Überschreitungsrate":
        fig.update_coloraxes(colorbar_tickformat=".0%")

    return fig


# -----------------------------
# Sidebar: upload + filters
# -----------------------------
with st.sidebar:
    st.header("Daten-Upload")
    uploaded_file = st.file_uploader(
        "CSV hochladen (Drag & Drop)",
        type=["csv"],
        help="Beispiel: fruitmonitoring_okt_nov2025_simulated_realistic_mrl.csv",
    )

    st.caption("Optional: Fallback, wenn du eine Demo-Datei im Repo unter /data ablegst.")
    use_demo = st.checkbox("Demo-Daten verwenden", value=False)

    st.divider()
    st.header("Filter")


# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def _read_demo_csv() -> pd.DataFrame:
    return pd.read_csv("data/fruitmonitoring_okt_nov2025_simulated_realistic_mrl.csv")


if uploaded_file is not None:
    raw = pd.read_csv(uploaded_file)
    df = normalize_df(raw)
elif use_demo:
    raw = _read_demo_csv()
    df = normalize_df(raw)
else:
    st.info("Bitte lade eine CSV hoch, um zu starten.")
    st.stop()


# -----------------------------
# Debug panel
# -----------------------------
with st.expander("Debug: Spalten & Beispielzeilen", expanded=False):
    st.write("Spalten in der hochgeladenen CSV (raw):")
    st.write(list(raw.columns))
    st.write("Spalten nach Normalisierung:")
    st.write(list(df.columns))
    st.write("Head (raw):")
    st.dataframe(raw.head(5), use_container_width=True)
    st.write("Head (normalized):")
    st.dataframe(df.head(5), use_container_width=True)


# -----------------------------
# Filters
# -----------------------------
def multiselect(col: str, label: str):
    if col not in df.columns:
        return []
    options = sorted(df[col].dropna().unique().tolist())
    return st.sidebar.multiselect(label, options, default=[])


sel_products = multiselect("produkt", "Produkte")
sel_countries = multiselect("herkunftsland", "Herkunftsländer")
sel_actives = multiselect("wirkstoff", "Wirkstoffe")
sel_months = multiselect("monat", "Monate")

filtered = df.copy()
if sel_products:
    filtered = filtered[filtered["produkt"].isin(sel_products)]
if sel_countries:
    filtered = filtered[filtered["herkunftsland"].isin(sel_countries)]
if sel_actives:
    filtered = filtered[filtered["wirkstoff"].isin(sel_actives)]
if sel_months:
    filtered = filtered[filtered["monat"].isin(sel_months)]


# -----------------------------
# KPIs
# -----------------------------
k = compute_kpis(filtered)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Zeilen", f"{k['n_rows']:,}")
c2.metric("Produkte", f"{k['n_products']:,}" if k["n_products"] is not None else "—")
c3.metric("Länder", f"{k['n_countries']:,}" if k["n_countries"] is not None else "—")
c4.metric("Wirkstoffe", f"{k['n_actives']:,}" if k["n_actives"] is not None else "—")
c5.metric("MRL-Überschreitungsrate", f"{k['exceed_rate']*100:.1f}%")

st.divider()


# -----------------------------
# Trend + Top tables
# -----------------------------
left, right = st.columns([1.4, 1])

with left:
    fig = plot_trend_exceedance(filtered)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Kein Monats-Trend möglich: Datum/Monat oder Überschreitungs-Flag fehlt.")

with right:
    st.subheader("Top: Überschreitungen")
    tabs = st.tabs(["Produkt", "Herkunft", "Wirkstoff"])
    with tabs[0]:
        st.dataframe(top_table(filtered, "produkt", n=12), use_container_width=True)
    with tabs[1]:
        st.dataframe(top_table(filtered, "herkunftsland", n=12), use_container_width=True)
    with tabs[2]:
        st.dataframe(top_table(filtered, "wirkstoff", n=12), use_container_width=True)

st.divider()


# -----------------------------
# Muster & Hotspots
# -----------------------------
st.subheader("Muster & Hotspots")
hm_col, map_col = st.columns([1.2, 1])

with hm_col:
    fig_hm = seaborn_heatmap_product_x_country(filtered)
    if fig_hm is not None:
        st.pyplot(fig_hm, clear_figure=True)
    else:
        st.warning(
            "Heatmap nicht verfügbar. "
            "Benötigt Spalten: produkt, herkunftsland, is_exceedance. "
            "Öffne den Debug-Expander, um deine Spaltennamen zu sehen."
        )

with map_col:
    st.subheader("Weltkarte (Choropleth, Plotly)")

    agg, warn = choropleth_agg(filtered)

    if agg is None:
        st.warning(warn)
    else:
        metric = st.radio(
            "Metrik",
            ["Probenanzahl", "Überschreitungen", "Überschreitungsrate"],
            horizontal=True
        )

        fig_map = plot_choropleth(agg, metric)
        st.plotly_chart(fig_map, use_container_width=True)

        if warn:
            st.info(warn)

        matched = agg["iso_a3"].notna().sum()
        total = len(agg)
        st.caption(f"Länder gemappt: {matched}/{total} ({matched/total:.0%})")

        top_geo = agg.sort_values("exceedances", ascending=False).head(10)
        st.caption("Top 10 Länder nach Überschreitungen")
        st.dataframe(top_geo[["country_name", "n", "exceedances", "exceed_rate"]], use_container_width=True)

st.divider()


# -----------------------------
# Data preview + download
# -----------------------------
st.subheader("Daten (Preview)")
st.dataframe(filtered.head(500), use_container_width=True)

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Gefilterte Daten als CSV herunterladen",
    data=csv_bytes,
    file_name="filtered_fruitmonitoring.csv",
    mime="text/csv",
)
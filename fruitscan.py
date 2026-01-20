from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# =============================
# Page config + global theme
# =============================
st.set_page_config(page_title="Rückstandsmonitoring — Modern Data Analysis", layout="wide")

px.defaults.template = "plotly_dark"
px.defaults.width = None
px.defaults.height = None

st.markdown(
    """
    <style>
      /* ---- Page background ---- */
      .stApp {
        background: radial-gradient(1200px 600px at 20% 0%, rgba(255,255,255,0.08), rgba(0,0,0,0)) ,
                    radial-gradient(900px 500px at 80% 10%, rgba(0,180,255,0.10), rgba(0,0,0,0)),
                    linear-gradient(180deg, #0b0f17 0%, #070a10 100%);
        color: rgba(255,255,255,0.92);
      }

      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px; }

      /* ---- KPI Grid ---- */
      .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.75rem;
        margin: 0.75rem 0 0.75rem;
      }
      @media (max-width: 1100px){
        .kpi-grid { grid-template-columns: repeat(2, 1fr); }
      }

      /* ---- KPI Cards (glass) ---- */
      .kpi {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 14px 14px;
        background: rgba(255,255,255,0.06);
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        backdrop-filter: blur(10px);
        transition: transform .12s ease, border-color .12s ease;
      }
      .kpi:hover {
        transform: translateY(-2px);
        border-color: rgba(0,180,255,0.35);
      }
      .kpi .label { font-size: 0.85rem; color: rgba(255,255,255,0.70); }
      .kpi .value { font-size: 1.65rem; font-weight: 800; margin-top: 6px; color: rgba(255,255,255,0.95); letter-spacing: -0.02em; }
      .kpi .delta { font-size: 0.85rem; color: rgba(255,255,255,0.60); margin-top: 4px; }

      /* ---- Pills / Badges ---- */
      .pill {
        display:inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.06);
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 8px;
        color: rgba(255,255,255,0.85);
      }

      /* ---- Card containers ---- */
      .card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 14px 14px;
        background: rgba(255,255,255,0.06);
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        backdrop-filter: blur(10px);
      }

      /* ---- Dataframe tweaks ---- */
      div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.10);
      }

      button[data-baseweb="tab"] { font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Rückstandsmonitoring — Data Analysis — Modern Overview")
st.caption("CSV Upload | Produkte · Herkunft · Wirkstoffe · Nachweise & MRL-Überschreitungen")


# =============================
# Plot helper
# =============================
def polish_plotly(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    return fig


# =============================
# Schema detection + normalization
# =============================
EXPECTED_EXPORT = {
    "datum", "proben_id", "produktgruppe", "produkt", "herkunft", "wirkstoff",
    "methode", "labor", "loq_mgkg", "mrl_mgkg", "ergebnis_mgkg", "befundflag"
}

COLUMN_MAP = {
    "produkt": "produkt", "product": "produkt", "commodity": "produkt", "item": "produkt",
    "herkunftsland": "herkunftsland", "origin": "herkunftsland", "country": "herkunftsland",
    "country_of_origin": "herkunftsland", "origin_country": "herkunftsland", "herkunft": "herkunftsland",
    "wirkstoff": "wirkstoff", "pesticide": "wirkstoff", "active_substance": "wirkstoff", "substance": "wirkstoff",
    "datum": "datum", "date": "datum", "sampling_date": "datum", "sample_date": "datum",
    "result": "result", "ergebnis": "result", "messwert": "result", "wert": "result", "value": "result",
    "mrl": "mrl", "mrl_value": "mrl", "grenzwert": "mrl", "limit": "mrl",
    "loq": "loq", "loq_value": "loq", "bestimmungsgrenze": "loq",
    "labor": "labor", "lab": "labor", "befund": "befund", "status": "befund",
}


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None


def normalize_generic(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_dict = {}
    for c in df.columns:
        mapped = COLUMN_MAP.get(c.lower())
        if mapped and mapped != c:
            rename_dict[c] = mapped
    if rename_dict:
        df = df.rename(columns=rename_dict)

    if "produkt" not in df.columns:
        c = _find_col(df, ["produkt", "product", "commodity", "item"])
        if c:
            df = df.rename(columns={c: "produkt"})
    if "herkunftsland" not in df.columns:
        c = _find_col(df, ["herkunftsland", "herkunft", "country", "origin"])
        if c:
            df = df.rename(columns={c: "herkunftsland"})
    if "wirkstoff" not in df.columns:
        c = _find_col(df, ["wirkstoff", "pesticide", "active", "substance"])
        if c:
            df = df.rename(columns={c: "wirkstoff"})
    if "datum" not in df.columns:
        c = _find_col(df, ["datum", "date", "sampling_date", "sample_date"])
        if c:
            df = df.rename(columns={c: "datum"})
    if "result" not in df.columns:
        c = _find_col(df, ["result", "ergebnis", "messwert", "wert", "value"])
        if c:
            df = df.rename(columns={c: "result"})
    if "mrl" not in df.columns:
        c = _find_col(df, ["mrl", "grenzwert", "limit"])
        if c:
            df = df.rename(columns={c: "mrl"})
    if "loq" not in df.columns:
        c = _find_col(df, ["loq", "bestimmungsgrenze"])
        if c:
            df = df.rename(columns={c: "loq"})

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce") if "datum" in df.columns else pd.NaT
    df["monat"] = df["datum"].dt.to_period("M").astype(str) if df["datum"].notna().any() else "Unbekannt"

    for col in ["result", "mrl", "loq"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "result" in df.columns and "mrl" in df.columns:
        df["mrl_ratio"] = df["result"] / df["mrl"]
        df["is_exceedance"] = (df["result"] > df["mrl"]) & df["result"].notna() & df["mrl"].notna()
    else:
        df["mrl_ratio"] = np.nan
        df["is_exceedance"] = False

    if "loq" in df.columns and "result" in df.columns:
        df["detected"] = df["result"].notna() & (df["result"] >= df["loq"])
    else:
        df["detected"] = df["result"].notna() if "result" in df.columns else False

    if "produktgruppe" not in df.columns:
        df["produktgruppe"] = "—"
    if "labor" not in df.columns:
        df["labor"] = "—"
    if "proben_id" not in df.columns:
        df["proben_id"] = pd.Series(range(1, len(df) + 1)).map(lambda x: f"SAMPLE-{x:07d}")

    for col in ["produkt", "herkunftsland", "wirkstoff", "produktgruppe", "labor"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def normalize_export(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce") if "datum" in df.columns else pd.NaT
    df["monat"] = df["datum"].dt.to_period("M").astype(str) if df["datum"].notna().any() else "Unbekannt"

    if "herkunft" in df.columns:
        df = df.rename(columns={"herkunft": "herkunftsland"})
    if "ergebnis_mgkg" in df.columns:
        df = df.rename(columns={"ergebnis_mgkg": "result"})
    if "mrl_mgkg" in df.columns:
        df = df.rename(columns={"mrl_mgkg": "mrl"})
    if "loq_mgkg" in df.columns:
        df = df.rename(columns={"loq_mgkg": "loq"})

    for col in ["result", "mrl", "loq"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "befundflag" in df.columns:
        df["detected"] = df["befundflag"].isin(["quantifiziert", ">MRL"])
        df["is_exceedance"] = df["befundflag"].eq(">MRL")
    else:
        df["detected"] = df["result"].notna()
        df["is_exceedance"] = (df["result"] > df["mrl"]) if ("result" in df.columns and "mrl" in df.columns) else False

    df["mrl_ratio"] = (df["result"] / df["mrl"]) if ("result" in df.columns and "mrl" in df.columns) else np.nan

    for c, default in [
        ("produktgruppe", "—"),
        ("labor", "—"),
        ("proben_id", pd.Series(range(1, len(df) + 1)).map(lambda x: f"SAMPLE-{x:07d}")),
    ]:
        if c not in df.columns:
            df[c] = default

    for col in ["produktgruppe", "produkt", "herkunftsland", "wirkstoff", "labor"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


@st.cache_data(show_spinner=False)
def load_any(uploaded_file=None, path: str | None = None) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_csv(path)

    raw.columns = [c.strip() for c in raw.columns]
    is_export = EXPECTED_EXPORT.issubset(set(raw.columns))

    if is_export:
        df = normalize_export(raw)
        schema = "Export-Schema (befundflag/…)"
    else:
        df = normalize_generic(raw)
        schema = "Generisch (normalisiert)"

    return df, schema, raw


# =============================
# URL params + session state (bookmarkable filters)
# =============================
def _csv_param(values: list[str]) -> str:
    return ",".join([str(v) for v in values]) if values else ""


def _parse_csv_param(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def get_query_params():
    try:
        return st.query_params  # Streamlit >= 1.30
    except Exception:
        return {}


def set_query_params(params: dict):
    try:
        st.query_params.update(params)
    except Exception:
        st.experimental_set_query_params(**params)


def ensure_state(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


def seed_ms_from_url(key: str, url_values: list[str], options: list[str]):
    # Seed only once (first run). After that, the widget owns the state.
    if key not in st.session_state:
        st.session_state[key] = [v for v in url_values if v in options]


# =============================
# Sidebar: data source
# =============================
with st.sidebar:
    st.header("📁 Daten")
    uploaded = st.file_uploader("CSV hochladen", type=["csv"])
    use_demo = st.toggle("Demo-Datei verwenden", value=False)

    demo_path = "data/fruitmonitoring_okt_nov_dez2025_simulated_realistic_mrl.csv"  # optional
    st.caption("Tipp: Lege eine Demo-Datei in /data ab, wenn du ohne Upload starten willst.")
    st.divider()
    st.header("Filter")


if uploaded is None and not use_demo:
    st.info("Bitte CSV hochladen (Sidebar) oder Demo-Datei aktivieren.")
    st.stop()

try:
    if uploaded is not None:
        df, schema, raw = load_any(uploaded_file=uploaded)
    else:
        df, schema, raw = load_any(path=demo_path)
except Exception as e:
    st.error(f"Fehler beim Laden: {e}")
    st.stop()


# =============================
# Read URL params (for seeding widgets ONCE)
# =============================
qp = dict(get_query_params())

url_pg = _parse_csv_param(qp.get("pg", ""))
url_prod = _parse_csv_param(qp.get("prod", ""))
url_country = _parse_csv_param(qp.get("country", ""))
url_sub = _parse_csv_param(qp.get("sub", ""))
url_lab = _parse_csv_param(qp.get("lab", ""))
url_month = _parse_csv_param(qp.get("month", ""))

url_date = qp.get("date", "")
url_date_tuple = None
if url_date and "|" in str(url_date) and df["datum"].notna().any():
    try:
        a, b = str(url_date).split("|", 1)
        url_date_tuple = (pd.to_datetime(a).date(), pd.to_datetime(b).date())
    except Exception:
        url_date_tuple = None

# date_range state (widget owns it, but we seed once)
if df["datum"].notna().any():
    dmin, dmax = df["datum"].min().date(), df["datum"].max().date()
    ensure_state("date_range", url_date_tuple if url_date_tuple else (dmin, dmax))
else:
    ensure_state("date_range", None)

# drilldown state
ensure_state("dd_country", "(alle)")
ensure_state("dd_product", "(alle)")
ensure_state("dd_active", "(alle)")


# =============================
# Sidebar filter widgets (FIXED: no default + no double-setting)
# =============================
def ms_widget(col: str, label: str, state_key: str, url_values: list[str]) -> list[str]:
    if col not in df.columns:
        st.sidebar.caption(f"{label}: — (Spalte fehlt)")
        st.session_state[state_key] = []
        return []

    opts = sorted(df[col].dropna().astype(str).str.strip().unique().tolist())

    # seed only once from URL -> session_state
    seed_ms_from_url(state_key, url_values, opts)

    # IMPORTANT: no default= here (widget owns st.session_state[state_key])
    st.sidebar.multiselect(label, opts, key=state_key)
    return st.session_state[state_key]


date_range = None
if df["datum"].notna().any() and st.session_state["date_range"] is not None:
    date_range = st.sidebar.date_input("Zeitraum", key="date_range")

sel_pg = ms_widget("produktgruppe", "Produktgruppe", "sel_pg", url_pg)
sel_prod = ms_widget("produkt", "Produkt", "sel_prod", url_prod)
sel_country = ms_widget("herkunftsland", "Herkunft (Land)", "sel_country", url_country)
sel_sub = ms_widget("wirkstoff", "Wirkstoff", "sel_sub", url_sub)
sel_lab = ms_widget("labor", "Labor", "sel_lab", url_lab)
sel_month = ms_widget("monat", "Monat", "sel_month", url_month)


# =============================
# Apply filters
# =============================
f = df.copy()

if date_range and "datum" in f.columns and f["datum"].notna().any():
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    f = f[(f["datum"] >= start) & (f["datum"] <= end)]

for col, values in [
    ("produktgruppe", sel_pg),
    ("produkt", sel_prod),
    ("herkunftsland", sel_country),
    ("wirkstoff", sel_sub),
    ("labor", sel_lab),
    ("monat", sel_month),
]:
    if values and col in f.columns:
        f = f[f[col].isin(values)]


# =============================
# Write back URL params (shareable links)
# =============================
params_out = {
    "pg": _csv_param(sel_pg),
    "prod": _csv_param(sel_prod),
    "country": _csv_param(sel_country),
    "sub": _csv_param(sel_sub),
    "lab": _csv_param(sel_lab),
    "month": _csv_param(sel_month),
}
if date_range:
    params_out["date"] = f"{date_range[0]}|{date_range[1]}"
else:
    params_out["date"] = ""

set_query_params(params_out)


# =============================
# KPI cards
# =============================
def kpi_cards(df_: pd.DataFrame, schema_label: str):
    n_rows = len(df_)
    n_samples = df_["proben_id"].nunique() if "proben_id" in df_.columns else np.nan
    n_products = df_["produkt"].nunique() if "produkt" in df_.columns else np.nan
    n_countries = df_["herkunftsland"].nunique() if "herkunftsland" in df_.columns else np.nan
    det_rate = float(df_["detected"].mean()) if n_rows and "detected" in df_.columns else 0.0
    exc_rate = float(df_["is_exceedance"].mean()) if n_rows and "is_exceedance" in df_.columns else 0.0
    med_ratio = np.nanmedian(df_["mrl_ratio"]) if "mrl_ratio" in df_.columns and df_["mrl_ratio"].notna().any() else np.nan

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi">
            <div class="label">Ergebnisse</div>
            <div class="value">{n_rows:,}</div>
            <div class="delta">Zeilen nach Filter</div>
          </div>
          <div class="kpi">
            <div class="label">Proben (unique)</div>
            <div class="value">{int(n_samples):,}</div>
            <div class="delta">IDs</div>
          </div>
          <div class="kpi">
            <div class="label">Produkte</div>
            <div class="value">{int(n_products):,}</div>
            <div class="delta">Distinct</div>
          </div>
          <div class="kpi">
            <div class="label">Länder</div>
            <div class="value">{int(n_countries):,}</div>
            <div class="delta">Distinct</div>
          </div>
          <div class="kpi">
            <div class="label">Nachweis / MRL</div>
            <div class="value">{det_rate*100:.1f}% / {exc_rate*100:.2f}%</div>
            <div class="delta">detected / exceed</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    min_d = df_["datum"].min().date() if "datum" in df_.columns and df_["datum"].notna().any() else "—"
    max_d = df_["datum"].max().date() if "datum" in df_.columns and df_["datum"].notna().any() else "—"

    st.markdown(
        f"""
        <span class="pill">Schema: {schema_label}</span>
        <span class="pill">Median Ergebnis/MRL: {med_ratio:.3f}</span>
        <span class="pill">Filter: {min_d} → {max_d}</span>
        """,
        unsafe_allow_html=True,
    )


kpi_cards(f, schema)


# =============================
# Auto-Insights Box
# =============================
def auto_insights(df_: pd.DataFrame) -> list[str]:
    insights = []
    if len(df_) == 0:
        return ["Keine Daten im aktuellen Filterbereich."]

    if all(c in df_.columns for c in ["herkunftsland", "is_exceedance"]):
        agg = df_.groupby("herkunftsland").agg(
            n=("herkunftsland", "size"),
            exc=("is_exceedance", "sum"),
            exc_rate=("is_exceedance", "mean"),
        ).reset_index()
        agg = agg[agg["n"] >= 50] if len(agg) else agg
        if len(agg) and agg["exc"].sum() > 0:
            best = agg.sort_values(["exc_rate", "exc", "n"], ascending=False).iloc[0]
            insights.append(
                f"**Top Risiko-Land:** {best['herkunftsland']} "
                f"(Überschreitungsrate {best['exc_rate']*100:.2f}%, n={int(best['n'])})."
            )

    if all(c in df_.columns for c in ["wirkstoff", "is_exceedance"]):
        a = df_[df_["is_exceedance"]].groupby("wirkstoff").size().sort_values(ascending=False)
        if len(a):
            w = a.index[0]
            insights.append(f"**Auffälligster Wirkstoff:** {w} ({int(a.iloc[0])} Überschreitungen).")

    if all(c in df_.columns for c in ["monat", "is_exceedance"]):
        m = df_.groupby("monat")["is_exceedance"].mean().reset_index(name="exc_rate").sort_values("monat")
        if len(m) >= 2:
            last = m.iloc[-1]
            prev = m.iloc[-2]
            delta = (last["exc_rate"] - prev["exc_rate"]) * 100
            arrow = "⬆" if delta > 0 else ("⬇" if delta < 0 else "➡️")
            insights.append(
                f"**MoM Change (Überschreitungsrate):** {prev['monat']} → {last['monat']} "
                f"{arrow} {delta:+.2f} %-Punkte."
            )

    if "herkunftsland" in df_.columns:
        insights.append(f"**Coverage:** {df_['herkunftsland'].nunique()} Herkunftsländer im Filter.")

    return insights[:6]


with st.expander("Auto-Insights (kurz & erklärbar)", expanded=True):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    for line in auto_insights(f):
        st.markdown(f"- {line}")
    st.markdown("</div>", unsafe_allow_html=True)


# =============================
# Helpers
# =============================
def top_table(df_: pd.DataFrame, group_col: str, n: int = 12) -> pd.DataFrame:
    if group_col not in df_.columns:
        return pd.DataFrame()

    g = df_.groupby(group_col, dropna=False)
    out = pd.DataFrame({
        "n": g.size(),
        "detections": g["detected"].sum() if "detected" in df_.columns else 0,
        "exceedances": g["is_exceedance"].sum() if "is_exceedance" in df_.columns else 0,
    })
    out["det_rate"] = out["detections"] / out["n"].replace(0, np.nan)
    out["exc_rate"] = out["exceedances"] / out["n"].replace(0, np.nan)
    out = out.sort_values(["exceedances", "detections", "n"], ascending=False).head(n).reset_index()
    return out


def data_quality_panel(df_: pd.DataFrame):
    issues = []
    if "datum" in df_.columns:
        invalid_dates = int(df_["datum"].isna().sum())
        if invalid_dates:
            issues.append(("Datum nicht parsebar", invalid_dates))
    if "result" in df_.columns:
        na_result = int(df_["result"].isna().sum())
        if na_result:
            issues.append(("Fehlende Messwerte (result)", na_result))
    if "mrl" in df_.columns:
        na_mrl = int(df_["mrl"].isna().sum())
        if na_mrl:
            issues.append(("Fehlende MRL-Werte", na_mrl))
    if "loq" in df_.columns:
        na_loq = int(df_["loq"].isna().sum())
        if na_loq:
            issues.append(("Fehlende LOQ-Werte", na_loq))
    if "mrl" in df_.columns and "loq" in df_.columns:
        weird = int((df_["loq"] > df_["mrl"]).sum())
        if weird:
            issues.append(("LOQ > MRL (prüfen)", weird))

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Data Quality Checks**")
    if not issues:
        st.success("Keine offensichtlichen Issues gefunden (Basics).")
    else:
        for label, cnt in issues:
            st.warning(f"{label}: {cnt:,}")
    st.markdown("</div>", unsafe_allow_html=True)


# =============================
# Choropleth (manual ISO3)
# =============================
ISO3_MAP = {
    "deutschland": "DEU", "belgien": "BEL", "italien": "ITA", "spanien": "ESP",
    "niederlande": "NLD", "polen": "POL", "türkei": "TUR", "türkiye": "TUR",
    "marokko": "MAR", "österreich": "AUT", "schweiz": "CHE", "frankreich": "FRA",
    "griechenland": "GRC", "portugal": "PRT", "ungarn": "HUN", "rumänien": "ROU",
    "bulgarien": "BGR", "kroatien": "HRV", "tschechien": "CZE", "slowakei": "SVK", "slowenien": "SVN",
    "germany": "DEU", "belgium": "BEL", "italy": "ITA", "spain": "ESP",
    "netherlands": "NLD", "poland": "POL", "turkey": "TUR", "morocco": "MAR",
    "austria": "AUT", "switzerland": "CHE", "france": "FRA", "greece": "GRC", "portugal": "PRT",
}


def choropleth_agg(df_: pd.DataFrame):
    if not {"herkunftsland", "is_exceedance", "detected"}.issubset(df_.columns):
        return None, "Benötigt Spalten: herkunftsland, detected, is_exceedance"

    agg = (
        df_.groupby("herkunftsland", dropna=False)
        .agg(
            n=("herkunftsland", "size"),
            exceedances=("is_exceedance", "sum"),
            detections=("detected", "sum"),
        )
        .reset_index()
        .rename(columns={"herkunftsland": "country_name"})
    )
    agg["exc_rate"] = np.where(agg["n"] > 0, agg["exceedances"] / agg["n"], 0.0)
    agg["det_rate"] = np.where(agg["n"] > 0, agg["detections"] / agg["n"], 0.0)
    agg["iso_a3"] = agg["country_name"].apply(lambda x: ISO3_MAP.get(str(x).strip().lower()) if pd.notna(x) else None)

    unmatched = agg.loc[agg["iso_a3"].isna(), "country_name"].dropna().astype(str).unique().tolist()
    warn = None
    if unmatched:
        warn = "Nicht gematchte Länder: " + ", ".join(unmatched[:12]) + (" ..." if len(unmatched) > 12 else "")
    return agg, warn


def plot_choropleth(agg: pd.DataFrame, metric: str):
    if metric == "Probenanzahl":
        col, label, tick = "n", "Proben", None
    elif metric == "Nachweisrate":
        col, label, tick = "det_rate", "Nachweisrate", ".0%"
    elif metric == "Überschreitungsrate":
        col, label, tick = "exc_rate", "Überschreitungsrate", ".0%"
    else:
        col, label, tick = "exceedances", "Überschreitungen", None

    fig = px.choropleth(
        agg,
        locations="iso_a3",
        color=col,
        hover_name="country_name",
        hover_data={"n": True, "detections": True, "exceedances": True, "det_rate": ":.1%", "exc_rate": ":.1%", "iso_a3": False},
        color_continuous_scale="Viridis",
        title=f"Weltkarte: {label}",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=520)
    fig.update_geos(showcoastlines=False, showcountries=True, showland=True)
    if tick:
        fig.update_coloraxes(colorbar_tickformat=tick)

    return polish_plotly(fig)


# =============================
# Tabs
# =============================
tab_overview, tab_trends, tab_hotspots, tab_geo, tab_drill, tab_data = st.tabs(
    ["Übersicht", "Trends", "Hotspots", "Geo", "Drilldown", "Daten"]
)


# -----------------------------
# TAB: Übersicht
# -----------------------------
with tab_overview:
    st.markdown("### Quick Insights")

    c1, c2 = st.columns((1.2, 0.8), gap="large")

    with c1:
        st.subheader("Top: Überschreitungen")
        tprod, tland, twirk = st.tabs(["Produkt", "Herkunft", "Wirkstoff"])
        with tprod:
            st.dataframe(top_table(f, "produkt", 15), use_container_width=True)
        with tland:
            st.dataframe(top_table(f, "herkunftsland", 15), use_container_width=True)
        with twirk:
            st.dataframe(top_table(f, "wirkstoff", 15), use_container_width=True)

    with c2:
        st.subheader("Pareto: Wo entsteht Risiko?")
        if "herkunftsland" in f.columns and "is_exceedance" in f.columns:
            pareto = (
                f.groupby("herkunftsland")["is_exceedance"]
                .sum()
                .sort_values(ascending=False)
                .reset_index(name="exceedances")
            )
            pareto = pareto.head(15)
            fig = px.bar(pareto, x="herkunftsland", y="exceedances")
            fig.update_layout(height=320)
            st.plotly_chart(polish_plotly(fig), use_container_width=True)
        else:
            st.info("Spalte herkunftsland/is_exceedance fehlt.")

    st.divider()

    c3, c4 = st.columns((1, 1), gap="large")
    with c3:
        st.subheader("MRL-Faktor (Ergebnis/MRL) — Verteilung")
        if "mrl_ratio" in f.columns and f["mrl_ratio"].notna().any():
            tmp = f.copy()
            tmp["mrl_ratio"] = tmp["mrl_ratio"].replace([np.inf, -np.inf], np.nan)
            fig = px.histogram(tmp, x="mrl_ratio", nbins=40)
            fig.update_layout(height=320)
            st.plotly_chart(polish_plotly(fig), use_container_width=True)
        else:
            st.info("Keine mrl_ratio verfügbar (benötigt result + mrl).")

    with c4:
        data_quality_panel(f)


# -----------------------------
# TAB: Trends
# -----------------------------
with tab_trends:
    st.markdown("### Zeitliche Entwicklung")
    left, right = st.columns((1.1, 0.9), gap="large")

    with left:
        st.subheader("Nachweisrate pro Tag")
        if f["datum"].notna().any() and "detected" in f.columns:
            daily = f.groupby(f["datum"].dt.date)["detected"].mean().reset_index(name="nachweisrate")
            fig = px.line(daily, x="datum", y="nachweisrate", markers=True)
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(height=360)
            st.plotly_chart(polish_plotly(fig), use_container_width=True)
        else:
            st.info("Kein Datum/detected verfügbar.")

    with right:
        st.subheader("Überschreitungsrate pro Monat")
        if "monat" in f.columns and "is_exceedance" in f.columns:
            m = f.groupby("monat")["is_exceedance"].mean().reset_index(name="exc_rate").sort_values("monat")
            fig = px.line(m, x="monat", y="exc_rate", markers=True)
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(height=360)
            st.plotly_chart(polish_plotly(fig), use_container_width=True)
        else:
            st.info("Kein Monat/is_exceedance verfügbar.")

    st.divider()

    st.subheader("Top-Wirkstoffe (Nachweise)")
    if "wirkstoff" in f.columns and "detected" in f.columns:
        top = f[f["detected"]].groupby("wirkstoff").size().reset_index(name="n").sort_values("n", ascending=False).head(20)
        fig = px.bar(top, x="n", y="wirkstoff", orientation="h")
        fig.update_layout(height=520)
        st.plotly_chart(polish_plotly(fig), use_container_width=True)
    else:
        st.info("Spalte wirkstoff oder detected fehlt.")


# -----------------------------
# TAB: Hotspots
# -----------------------------
with tab_hotspots:
    st.markdown("### Hotspots & Muster")

    if all(c in f.columns for c in ["produktgruppe", "herkunftsland", "is_exceedance"]) and (f["produktgruppe"] != "—").any():
        st.subheader("Heatmap: Überschreitungsrate (Produktgruppe × Herkunft)")
        heat = f.pivot_table(index="produktgruppe", columns="herkunftsland", values="is_exceedance", aggfunc="mean", fill_value=0.0)
        fig = px.imshow(heat, aspect="auto", origin="lower")
        fig.update_layout(height=520)
        fig.update_coloraxes(colorbar_title="Rate", colorbar_tickformat=".0%")
        st.plotly_chart(polish_plotly(fig), use_container_width=True)

    elif all(c in f.columns for c in ["produkt", "herkunftsland", "is_exceedance"]):
        st.subheader("Heatmap: Überschreitungsrate (Produkt × Herkunft)")
        heat = f.pivot_table(index="produkt", columns="herkunftsland", values="is_exceedance", aggfunc="mean", fill_value=0.0)
        fig = px.imshow(heat, aspect="auto", origin="lower")
        fig.update_layout(height=520)
        fig.update_coloraxes(colorbar_title="Rate", colorbar_tickformat=".0%")
        st.plotly_chart(polish_plotly(fig), use_container_width=True)
    else:
        st.info("Für Heatmap werden produkt(gruppe), herkunftsland und is_exceedance benötigt.")

    st.divider()

    st.subheader("Treemap: Produktgruppe → Produkt → Herkunft")
    if all(c in f.columns for c in ["produktgruppe", "produkt", "herkunftsland"]) and (f["produktgruppe"] != "—").any():
        tre = f.groupby(["produktgruppe", "produkt", "herkunftsland"]).size().reset_index(name="n")
        fig = px.treemap(tre, path=["produktgruppe", "produkt", "herkunftsland"], values="n")
        fig.update_layout(height=560)
        st.plotly_chart(polish_plotly(fig), use_container_width=True)
    elif all(c in f.columns for c in ["produkt", "herkunftsland"]):
        tre = f.groupby(["produkt", "herkunftsland"]).size().reset_index(name="n")
        fig = px.treemap(tre, path=["produkt", "herkunftsland"], values="n")
        fig.update_layout(height=560)
        st.plotly_chart(polish_plotly(fig), use_container_width=True)
    else:
        st.info("Für Treemap werden produkt(gruppe) und herkunftsland benötigt.")

    st.divider()

    st.subheader("Seaborn Heatmap (optional, classic analytics vibe)")
    if all(c in f.columns for c in ["produkt", "herkunftsland", "is_exceedance"]):
        top_products = st.slider("Top Produkte", 5, 30, 12)
        top_countries = st.slider("Top Länder", 5, 30, 12)

        prods = f["produkt"].value_counts().head(top_products).index
        countries = f["herkunftsland"].value_counts().head(top_countries).index
        sub = f[f["produkt"].isin(prods) & f["herkunftsland"].isin(countries)].copy()

        pivot = sub.pivot_table(index="produkt", columns="herkunftsland", values="is_exceedance", aggfunc="mean").fillna(0)
        sns.set_theme(style="white")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, ax=ax, cmap="viridis")
        ax.set_title("Überschreitungsrate nach Produkt × Herkunft (Top-Auswahl)")
        ax.set_xlabel("Herkunft")
        ax.set_ylabel("Produkt")
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("Für Seaborn-Heatmap werden produkt, herkunftsland und is_exceedance benötigt.")


# -----------------------------
# TAB: Geo
# -----------------------------
with tab_geo:
    st.markdown("### Geo-Ansicht")

    agg, warn = choropleth_agg(f)
    if agg is None:
        st.info("Für Choropleth werden herkunftsland, detected und is_exceedance benötigt.")
    else:
        metric = st.radio("Metrik", ["Probenanzahl", "Nachweisrate", "Überschreitungen", "Überschreitungsrate"], horizontal=True)
        fig = plot_choropleth(agg, metric)
        st.plotly_chart(fig, use_container_width=True)

        if warn:
            st.info(warn)

        matched = agg["iso_a3"].notna().sum()
        total = len(agg)
        st.caption(f"Länder gemappt: {matched}/{total} ({matched/total:.0%})")

        st.subheader("Top 15 Länder (nach Überschreitungen)")
        top_geo = agg.sort_values("exceedances", ascending=False).head(15)
        st.dataframe(top_geo[["country_name", "n", "detections", "exceedances", "det_rate", "exc_rate"]], use_container_width=True)


# -----------------------------
# TAB: Drilldown (session state) FIXED
# -----------------------------
with tab_drill:
    st.markdown("### Drilldown: Land → Produkt → Wirkstoff")

    if "herkunftsland" not in f.columns:
        st.info("Drilldown benötigt herkunftsland.")
    else:
        dcol1, dcol2, dcol3 = st.columns((1, 1, 1), gap="large")

        countries = sorted([x for x in f["herkunftsland"].dropna().unique()])
        country_options = ["(alle)"] + countries
        country_index = country_options.index(st.session_state["dd_country"]) if st.session_state["dd_country"] in country_options else 0

        dd_country = dcol1.selectbox("Land", country_options, index=country_index, key="dd_country")

        f1 = f.copy()
        if dd_country != "(alle)":
            f1 = f1[f1["herkunftsland"] == dd_country]

        products = sorted([x for x in f1["produkt"].dropna().unique()]) if "produkt" in f1.columns else []
        product_options = ["(alle)"] + products
        product_index = product_options.index(st.session_state["dd_product"]) if st.session_state["dd_product"] in product_options else 0

        dd_product = dcol2.selectbox("Produkt", product_options, index=product_index, key="dd_product")

        f2 = f1.copy()
        if dd_product != "(alle)" and "produkt" in f2.columns:
            f2 = f2[f2["produkt"] == dd_product]

        actives = sorted([x for x in f2["wirkstoff"].dropna().unique()]) if "wirkstoff" in f2.columns else []
        active_options = ["(alle)"] + actives
        active_index = active_options.index(st.session_state["dd_active"]) if st.session_state["dd_active"] in active_options else 0

        dd_active = dcol3.selectbox("Wirkstoff", active_options, index=active_index, key="dd_active")

        f3 = f2.copy()
        if dd_active != "(alle)" and "wirkstoff" in f3.columns:
            f3 = f3[f3["wirkstoff"] == dd_active]

        st.divider()
        st.markdown("**Drilldown KPIs**")
        kpi_cards(f3, schema)

        st.divider()
        l, r = st.columns((1.1, 0.9), gap="large")

        with l:
            st.subheader("Zeitverlauf: Überschreitungsrate (Monat)")
            if "monat" in f3.columns and "is_exceedance" in f3.columns:
                tr = f3.groupby("monat")["is_exceedance"].mean().reset_index(name="exc_rate").sort_values("monat")
                fig = px.line(tr, x="monat", y="exc_rate", markers=True)
                fig.update_yaxes(tickformat=".0%")
                fig.update_layout(height=340)
                st.plotly_chart(polish_plotly(fig), use_container_width=True)
            else:
                st.info("Kein Monat/is_exceedance verfügbar.")

        with r:
            st.subheader("Top Kombinationen (Überschreitungen)")
            if "is_exceedance" in f3.columns and f3["is_exceedance"].any():
                ex = f3[f3["is_exceedance"]].copy()
                cols = [c for c in ["produktgruppe", "produkt", "herkunftsland", "wirkstoff", "labor"] if c in ex.columns]
                top_ex = ex.groupby(cols).size().reset_index(name="n_exceed").sort_values("n_exceed", ascending=False).head(15)
                st.dataframe(top_ex, use_container_width=True)
            else:
                st.info("Keine Überschreitungen im Drilldown-Filter.")

        st.divider()
        st.subheader("Verteilung: Ergebnis/MRL (nur Überschreitungen)")
        if "mrl_ratio" in f3.columns and "is_exceedance" in f3.columns and f3["is_exceedance"].any():
            ex = f3[f3["is_exceedance"]].copy()
            ex["exceed_factor"] = ex["mrl_ratio"].replace([np.inf, -np.inf], np.nan)
            fig = px.histogram(ex, x="exceed_factor", nbins=25)
            fig.update_layout(height=340)
            st.plotly_chart(polish_plotly(fig), use_container_width=True)
        else:
            st.info("Keine Exceedance-Verteilung verfügbar.")


# -----------------------------
# TAB: Daten
# -----------------------------
with tab_data:
    st.markdown("### Daten & Export")
    c1, c2 = st.columns((1, 1), gap="large")

    with c1:
        st.subheader("Preview (gefiltert)")
        st.dataframe(f.head(500), use_container_width=True)
        csv_bytes = f.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Gefilterte Daten als CSV herunterladen",
            data=csv_bytes,
            file_name="filtered_fruitmonitoring.csv",
            mime="text/csv",
        )

    with c2:
        st.subheader("Raw (optional)")
        st.caption("Praktisch fürs Debugging/Schema-Check.")
        st.dataframe(raw.head(200), use_container_width=True)
        raw_bytes = raw.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Raw-Daten als CSV herunterladen",
            data=raw_bytes,
            file_name="raw_uploaded.csv",
            mime="text/csv",
        )

    st.divider()
    st.caption("Hinweis: MRLs sind rechtliche Höchstgehalte (keine toxikologischen Grenzwerte).")

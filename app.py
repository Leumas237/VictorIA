"""
app.py – VictorIA Streamlit Web UI
Sports Match Prediction AI with ML-powered analysis.
"""
import sys
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from training.pipeline import models_are_ready, train_models
from data.data_fetcher import DataFetcher
from data.weather_fetcher import WeatherFetcher
from ui.performance import render_performance_page


@st.cache_resource(show_spinner="Chargement des modèles …")
def get_predictor(_app_version: str):
    """Cached predictor; _app_version busts cache after code updates."""
    from predictor import MatchPredictor
    return MatchPredictor(force_retrain=False)

# ── Page config (must be first Streamlit call) ─────────────
st.set_page_config(
    page_title="VictorIA – Prédiction Sportive IA",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
/* Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp { background: #0a0c10; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d111a 0%, #111827 100%);
    border-right: 1px solid #1f2937;
}
[data-testid="stSidebar"] * { color: #e5e7eb !important; }

/* Cards */
.card {
    background: linear-gradient(135deg, #111827 0%, #1a2035 100%);
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin: 1rem 0;
}
.card-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #9ca3af;
    margin-bottom: 0.6rem;
}

/* Big probability badges */
.prob-badge {
    display: inline-block;
    padding: 0.5rem 1.2rem;
    border-radius: 99px;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}

/* Metric row */
.metric-row {
    display: flex; gap: 0.8rem; flex-wrap: wrap;
}
.metric-box {
    flex: 1; min-width: 110px;
    background: #0d111a;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-value { font-size: 1.35rem; font-weight: 700; color: #f0f1f5; }
.metric-label { font-size: 0.68rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.08em; }

/* Factor tags */
.factor-tag {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem;
}

/* Outcome pulse */
@keyframes glow {
    0%   { box-shadow: 0 0 8px rgba(99,179,237,0.3); }
    50%  { box-shadow: 0 0 24px rgba(99,179,237,0.6); }
    100% { box-shadow: 0 0 8px rgba(99,179,237,0.3); }
}
.outcome-card {
    animation: glow 3s ease-in-out infinite;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    background: linear-gradient(135deg, #1a2540 0%, #0f1929 100%);
    border: 1px solid #374151;
}

h1,h2,h3,h4 { color: #f0f1f5 !important; }
p { color: #d5d9e3; }
.stTextInput input, .stSelectbox select {
    background: #111827 !important;
    color: #f9fafb !important;
    border-color: #374151 !important;
    border-radius: 10px !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    color: white;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    width: 100%;
    transition: all 0.2s;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.4);
}

/* Divider */
hr { border-color: #1f2937 !important; }

.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f0f1f5;
    margin: 0.25rem 0 0.75rem;
}
</style>
""", unsafe_allow_html=True)


def render_disclaimer():
    st.markdown(
        """
        <div style="text-align:center;padding:1.5rem 0 0.5rem;color:#4b5563;font-size:0.72rem">
        ⚠️ VictorIA fournit des analyses statistiques à titre informatif uniquement.
        Ce n'est pas un conseil de pari. Les résultats sportifs restent imprévisibles.
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_visual(conf: float) -> tuple[str, str, str]:
    if conf >= 85:
        return "🟢 Très Haute", "#22c55e", "Très haute confiance sur les signaux actuels."
    if conf >= 70:
        return "🟡 Haute", "#fbbf24", "Bonne robustesse, mais quelques incertitudes subsistent."
    if conf >= 50:
        return "🟠 Moyenne", "#fb923c", "Prédiction équilibrée, prudence recommandée."
    return "🔴 Basse", "#ef4444", "Scénario très incertain."


_data_probe = DataFetcher()
_data_badge = (
    "📡 Données API (football-data.org)"
    if _data_probe.use_real_data
    else "🔬 Mode démo (données synthétiques)"
)
_data_color = "#69db7c" if _data_probe.use_real_data else "#fbbf24"


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ **VictorIA**")
    st.markdown("<p style='color:#6b7280;font-size:0.85rem;'>Prédiction Sportive par IA</p>",
                unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🔮 Prédiction", "📈 Performance"],
        label_visibility="collapsed",
    )

    st.markdown(
        f"<div style='background:#111827;border:1px solid #374151;border-radius:8px;"
        f"padding:0.5rem 0.75rem;font-size:0.75rem;color:{_data_color};margin:0.5rem 0'>"
        f"{_data_badge}</div>",
        unsafe_allow_html=True,
    )

    models_ok = models_are_ready()
    status_color = "#69db7c" if models_ok else "#ff6b6b"
    status_text = "✅ Modèles prêts" if models_ok else "⏳ Modèles non entraînés"
    st.markdown(
        f"<div style='font-size:0.72rem;color:{status_color}'>{status_text}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 1️⃣ Compétition")
    competition = st.selectbox("🏆 Compétition", [
        "Premier League", "La Liga", "Ligue 1", "Serie A",
        "Bundesliga", "Champions League", "Coupe du Monde"
    ])

    st.markdown("### 2️⃣ Équipes")
    home_team = st.text_input("🏠 Équipe à domicile", placeholder="ex: Paris SG")
    away_team = st.text_input("✈️ Équipe à l'extérieur", placeholder="ex: Marseille")

    st.markdown("### 3️⃣ Lancer la prédiction")
    predict_btn = st.button("🔮 Prédire", use_container_width=True)

    with st.expander("⚙️ Options avancées"):
        force_retrain = st.checkbox(
            "🔄 Réentraîner les modèles",
            value=False,
            help="Relance python train.py (1–2 min). À utiliser rarement.",
        )
        show_model_breakdown = st.checkbox("📊 Détails des modèles", value=False)
        show_raw_features = st.checkbox("🔢 Afficher les features brutes", value=False)

    if st.button("🔄 Vider le cache", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='color:#4b5563;font-size:0.72rem;line-height:1.6'>
    <b>Modèles utilisés :</b><br>
    • XGBoost (55%)<br>
    • Random Forest (45%)<br><br>
    <b>Entraînement :</b><br>
    <code>python train.py</code><br><br>
    <b>27 features analysées</b><br>
    • Forme récente<br>
    • Statistiques de buts<br>
    • Elo rating proxy<br>
    • Historique H2H<br>
    • Avantage domicile
    </div>
    """, unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 9])
with col_title:
    st.markdown("# 🧠 VictorIA")
    st.markdown("<p style='color:#6b7280;margin-top:-12px'>Prédiction de matchs par intelligence artificielle · Analyses ML explicables</p>",
                unsafe_allow_html=True)

st.markdown("---")

# ── Performance page ───────────────────────────────────────
if page.startswith("📈"):
    render_performance_page()
    render_disclaimer()
    st.stop()

if force_retrain and models_are_ready():
    with st.spinner("🔄 Réentraînement sur données réelles (plusieurs minutes) …"):
        train_models(force=True, refresh_data=True)
    st.cache_resource.clear()
    st.success("Modèles réentraînés. Vous pouvez prédire.")
    st.rerun()


# ── Helper: donut chart ────────────────────────────────────
def make_donut(probs: dict, outcome_key: str) -> go.Figure:
    colors_map = {
        "HomeWin":  ["#3b82f6", "#1f2937", "#1f2937"],
        "Draw":     ["#1f2937", "#fbbf24", "#1f2937"],
        "AwayWin":  ["#1f2937", "#1f2937", "#ec4899"],
    }
    colors = colors_map.get(outcome_key, ["#3b82f6", "#fbbf24", "#ec4899"])
    values = [probs["home_win"], probs["draw"], probs["away_win"]]
    labels = ["Domicile", "Nul", "Extérieur"]

    fig = go.Figure(go.Pie(
        values=values, labels=labels,
        hole=0.68,
        marker=dict(colors=colors, line=dict(color="#0a0c10", width=3)),
        textinfo="label+percent",
        textfont=dict(size=13, color="white"),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        annotations=[dict(
            text=f"<b>{max(values):.0f}%</b>",
            x=0.5, y=0.5, font_size=26, font_color="white",
            showarrow=False
        )]
    )
    return fig


# ── Helper: bar comparison ─────────────────────────────────
def make_comparison_bar(label, home_val, away_val,
                         home_name, away_name, format_str="{:.2f}") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=home_name, x=[home_val], y=[label],
        orientation="h",
        marker_color="#3b82f6",
        text=[format_str.format(home_val)],
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name=away_name, x=[-away_val], y=[label],
        orientation="h",
        marker_color="#ec4899",
        text=[format_str.format(away_val)],
        textposition="inside",
    ))
    fig.update_layout(
        barmode="relative",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=90,
        margin=dict(t=5, b=5, l=5, r=5),
        xaxis=dict(showticklabels=False, zeroline=True,
                   zerolinecolor="#374151", zerolinewidth=1),
        yaxis=dict(showticklabels=False),
        font=dict(color="white"),
    )
    return fig


# ── Main content ───────────────────────────────────────────
if predict_btn:
    if not home_team.strip() or not away_team.strip():
        st.error("⚠️ Veuillez entrer les deux équipes.")
        st.stop()

    if not models_are_ready():
        st.error(
            "⚠️ Les modèles ne sont pas encore entraînés. "
            "Exécutez dans le terminal : `python train.py`"
        )
        if st.button("🚀 Entraîner maintenant (1–2 min)", type="primary"):
            with st.spinner("Entraînement sur données réelles (plusieurs minutes) …"):
                train_models(force=True, refresh_data=True)
            st.cache_resource.clear()
            st.rerun()
        st.stop()

    predictor = get_predictor(config.APP_VERSION)

    with st.spinner(f"🔍 Analyse de **{home_team}** vs **{away_team}** …"):
        t0 = time.time()
        report = predictor.predict(home_team, away_team, competition)
        elapsed = time.time() - t0

    # ── Executive summary ────────────────────────────────
    probs = report["probabilities"]
    outcome = report["outcome"]
    conf = report["confidence"]
    exact = report.get("exact_score", {})
    live_stats = report.get("live_stats", {})
    weather_fetcher = WeatherFetcher()
    weather = weather_fetcher.get_weather(report["home_stats"]["team"])
    weather_adjustment = weather.get("confidence_adjustment", 0.0) if weather.get("available") else 0.0
    adjusted_conf = float(np.clip(conf + weather_adjustment, 0, 100))
    conf_level, conf_color, conf_hint = confidence_visual(adjusted_conf)

    outcome_colors = {
        "HomeWin": "#3b82f6", "Draw": "#fbbf24", "AwayWin": "#ec4899"
    }
    oc = outcome_colors.get(report["outcome_key"], "#3b82f6")
    weather_line = (
        f"{weather.get('emoji', '🌤️')} {weather.get('temperature_c', '—')}°C · "
        f"{weather.get('conditions', 'Conditions indisponibles')}"
        if weather.get("available")
        else "🌤️ Météo indisponible (clé OpenWeatherMap manquante)"
    )
    adjustment_sign = "+" if weather_adjustment > 0 else ""

    st.markdown(f"""
    <div class="outcome-card" style="padding:1.8rem 1.8rem 1.6rem">
        <div style='color:#6b7280;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.15em'>
            {report['competition']} · {report['date']}
            {'&nbsp;&nbsp;🔬 données synthétiques' if report['synthetic'] else '&nbsp;&nbsp;📡 données réelles'}
        </div>
        <div style='font-size:2.3rem;font-weight:800;margin:0.8rem 0;color:#f0f1f5'>
            {report['home_stats']['team']} <span style='color:#374151'>vs</span> {report['away_stats']['team']}
        </div>
        <div style='font-size:0.95rem;color:#c2c8d7;margin-bottom:0.8rem'>{weather_line}</div>
        <div style='font-size:2.9rem;font-weight:900;color:{oc}'>
            {report['outcome_emoji']} {outcome}
        </div>
        <div style='font-size:1.05rem;font-weight:700;color:#facc15;margin-top:0.7rem'>
            ⚽ Score exact probable : {exact.get('scoreline', '1-1')}
            <span style='font-size:0.85rem;color:#9ca3af'>(xG {exact.get('xg_home', 1.5)} - {exact.get('xg_away', 1.2)})</span>
        </div>
        <div style='margin-top:1rem;display:flex;gap:0.5rem;justify-content:center;align-items:center;flex-wrap:wrap'>
            <span style='background:{conf_color}22;color:{conf_color};padding:0.3rem 0.9rem;border-radius:99px;font-size:0.9rem;font-weight:700;border:1px solid {conf_color}44'>{conf_level}</span>
            <span style='color:#d5d9e3;font-size:0.9rem'>{adjusted_conf:.1f}% confiance ajustée météo ({adjustment_sign}{weather_adjustment:.1f}%)</span>
        </div>
        <div style='height:8px;background:#1f2937;border-radius:99px;margin-top:0.8rem;max-width:420px;margin-left:auto;margin-right:auto;'>
            <div style='height:8px;width:{adjusted_conf}%;background:{conf_color};border-radius:99px'></div>
        </div>
        <div style='margin-top:0.7rem;color:#9ca3af;font-size:0.82rem'>Analyse en {elapsed:.1f}s · {conf_hint}</div>
    </div>
    """, unsafe_allow_html=True)

    quality = report.get("data_quality", {})
    if quality.get("warning"):
        st.warning(quality["warning"])
    elif quality.get("home_resolved") or quality.get("away_resolved"):
        st.caption(
            "Équipes API résolues : "
            f"{quality.get('home_query')} → {quality.get('home_resolved')} · "
            f"{quality.get('away_query')} → {quality.get('away_resolved')}"
        )
    if report.get("empirical_adjusted"):
        st.info(
            "Prédiction recalibrée avec les statistiques réelles (forme, buts, confrontations)."
        )
    if (
        report["home_stats"]["win_rate_pct"] == report["away_stats"]["win_rate_pct"]
        and report["home_stats"]["record"] == report["away_stats"]["record"]
        and abs(probs["home_win"] - 49.7) < 0.2
    ):
        st.error(
            "⚠️ Résultat suspect (données identiques). Cliquez sur **Vider le cache** "
            "dans la barre latérale, puis relancez la prédiction."
        )

    tab_pred, tab_weather, tab_analysis, tab_h2h = st.tabs(
        ["📊 Prédiction", "🌦️ Météo", "📈 Analyse", "⚔️ H2H"]
    )

    with tab_pred:
        st.markdown('<div class="section-title">Lecture rapide du pronostic</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        prob_data = [
            (col1, "🏠 Victoire Domicile", probs["home_win"], "#3b82f6"),
            (col2, "🤝 Match Nul", probs["draw"], "#fbbf24"),
            (col3, "✈️ Victoire Extérieur", probs["away_win"], "#ec4899"),
        ]
        for col, label, prob, color in prob_data:
            with col:
                st.markdown(f"""
                <div class="card" style="text-align:center;border-color:{color}44">
                    <div class="card-title">{label}</div>
                    <div class="prob-badge" style="color:{color}">{prob:.1f}%</div>
                    <div style="height:7px;background:#1f2937;border-radius:3px;margin-top:0.8rem">
                        <div style="height:7px;width:{prob}%;background:{color};border-radius:3px;transition:width 0.5s"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        comp_rows = [
            ("Taux victoire", report["home_stats"]["win_rate_pct"], report["away_stats"]["win_rate_pct"], "%"),
            ("Buts/match", report["home_stats"]["avg_goals_scored"], report["away_stats"]["avg_goals_scored"], ""),
            ("Diff. buts", report["home_stats"]["goal_diff"], report["away_stats"]["goal_diff"], ""),
        ]
        rows_html = ""
        for label, h_val, a_val, suffix in comp_rows:
            h_badge = "✅" if h_val >= a_val else "❌"
            a_badge = "✅" if a_val > h_val else "❌"
            rows_html += (
                f"<tr>"
                f"<td style='padding:0.4rem;color:#dbe1ee'>{h_val:.1f}{suffix} {h_badge}</td>"
                f"<td style='padding:0.4rem;color:#9ca3af'>{label}</td>"
                f"<td style='padding:0.4rem;color:#dbe1ee'>{a_badge} {a_val:.1f}{suffix}</td>"
                f"</tr>"
            )
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📊 Comparaison côte à côte</div>
            <table style="width:100%;font-size:0.92rem;border-collapse:collapse">
                <thead>
                    <tr style="color:#f0f1f5;font-weight:700">
                        <th style="text-align:left;padding:0.35rem 0.4rem">{report['home_stats']['team']}</th>
                        <th style="text-align:left;padding:0.35rem 0.4rem">Stat</th>
                        <th style="text-align:left;padding:0.35rem 0.4rem">{report['away_stats']['team']}</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        col_exact, col_live = st.columns(2)
        with col_exact:
            top_scorelines = exact.get("top_scorelines", [])
            rows = "".join(
                f"<div style='display:flex;justify-content:space-between;margin-bottom:0.45rem'>"
                f"<span style='color:#dbe1ee'>{idx+1}. {s['scoreline']}</span>"
                f"<span style='color:#facc15;font-weight:700'>{s['probability_pct']:.2f}%</span>"
                f"</div>"
                for idx, s in enumerate(top_scorelines[:5])
            )
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🎯 Top scores exacts probables</div>
                {rows if rows else "<span style='color:#9ca3af'>Aucune estimation disponible.</span>"}
            </div>
            """, unsafe_allow_html=True)

        with col_live:
            momentum_map = {"home": f"Avantage {home_team}", "away": f"Avantage {away_team}", "balanced": "Équilibré"}
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📡 Stats en temps réel</div>
                <div style='color:#9ca3af;font-size:0.82rem;margin-bottom:0.8rem'>
                    Source : {live_stats.get('source', 'n/a')} · MAJ {live_stats.get('updated_at', 'n/a')}
                </div>
                <div style='display:flex;justify-content:space-between;margin-bottom:0.4rem'><span style='color:#b7bfd1'>Δ Forme</span><span style='color:#f0f1f5'>{live_stats.get('form_delta', 0):+.3f}</span></div>
                <div style='display:flex;justify-content:space-between;margin-bottom:0.4rem'><span style='color:#b7bfd1'>Δ Attaque</span><span style='color:#f0f1f5'>{live_stats.get('attack_delta', 0):+.3f}</span></div>
                <div style='display:flex;justify-content:space-between;margin-bottom:0.4rem'><span style='color:#b7bfd1'>Δ Défense</span><span style='color:#f0f1f5'>{live_stats.get('defense_delta', 0):+.3f}</span></div>
                <div style='display:flex;justify-content:space-between;margin-bottom:0.4rem'><span style='color:#b7bfd1'>H2H Edge</span><span style='color:#f0f1f5'>{live_stats.get('h2h_home_edge', 0):+.3f}</span></div>
                <div style='margin-top:0.8rem;color:#69db7c;font-size:0.92rem'>⚡ Momentum : {momentum_map.get(live_stats.get('momentum', 'balanced'), 'Équilibré')}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab_weather:
        st.markdown('<div class="section-title">Conditions météo & impact prédictif</div>', unsafe_allow_html=True)
        if weather.get("available"):
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🌦️ Météo du stade (approx. domicile)</div>
                <div style='font-size:1.6rem;color:#f0f1f5;font-weight:700'>{weather['emoji']} {weather['conditions']}</div>
                <div style='font-size:0.95rem;color:#9ca3af;margin-top:0.25rem'>{weather['location']}</div>
                <div style='display:grid;grid-template-columns:repeat(2,minmax(160px,1fr));gap:0.7rem;margin-top:1rem'>
                    <div class='metric-box'><div class='metric-value'>{weather['temperature_c']:.1f}°C</div><div class='metric-label'>Température</div></div>
                    <div class='metric-box'><div class='metric-value'>{weather['humidity_pct']:.0f}%</div><div class='metric-label'>Humidité</div></div>
                    <div class='metric-box'><div class='metric-value'>{weather['wind_kmh']:.1f} km/h</div><div class='metric-label'>Vent</div></div>
                    <div class='metric-box'><div class='metric-value'>{weather['rain_mm']:.1f} mm/h</div><div class='metric-label'>Pluie</div></div>
                </div>
                <div style='margin-top:1rem;color:#d5d9e3'>Ajustement confiance: <b style='color:{conf_color}'>{adjustment_sign}{weather_adjustment:.1f}%</b></div>
                <div style='margin-top:0.35rem;color:#d5d9e3'>Score météo: <b style='color:#60a5fa'>{weather['weather_confidence_score']:.1f}/100</b></div>
            </div>
            """, unsafe_allow_html=True)
            for note in weather.get("impact_notes", []):
                st.markdown(f"- {note}")
        else:
            st.info(
                "Météo indisponible. Ajoutez `OPENWEATHER_API_KEY` pour activer "
                "les conditions en direct et l'ajustement de confiance."
            )
            if weather.get("reason"):
                st.caption(f"Détail: {weather['reason']}")

    with tab_analysis:
        col_donut, col_shap = st.columns([1, 2])
        with col_donut:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🎯 Distribution des probabilités</div>', unsafe_allow_html=True)
            fig_donut = make_donut(probs, report["outcome_key"])
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
            agreement_pct = report["agreement"] * 100
            agr_color = "#69db7c" if agreement_pct > 80 else "#ffd43b" if agreement_pct > 60 else "#ff6b6b"
            st.markdown(f"<div style='text-align:center;margin-top:-10px;color:{agr_color};font-size:0.9rem;font-weight:600'>🤝 Consensus modèles : {agreement_pct:.0f}%</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_shap:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🔍 Vue SHAP globale</div>', unsafe_allow_html=True)
            try:
                fig_shap = predictor.get_shap_figure(report)
                st.pyplot(fig_shap, use_container_width=True)
            except Exception as e:
                st.warning(f"SHAP indisponible: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card" style="border-left:3px solid #6366f1">
            <div class="card-title">💬 Analyse IA</div>
            <p style="margin:0;line-height:1.75;color:#dbe1ee">{report['summary']}</p>
        </div>
        """, unsafe_allow_html=True)

        factors = report["top_factors"]
        if factors:
            st.markdown('<div class="section-title">🔑 Facteurs SHAP hiérarchisés</div>', unsafe_allow_html=True)
            max_abs = max(abs(f["shap_value"]) for f in factors) or 1.0
            for i, f in enumerate(factors, 1):
                color = "#00c896" if f["shap_value"] > 0 else "#ff4d6d"
                icon = "📈" if f["shap_value"] > 0 else "📉"
                width = max(10, int(abs(f["shap_value"]) / max_abs * 100))
                st.markdown(f"""
                <div class="card" style="padding:1rem 1.2rem;margin-bottom:0.45rem;border-left:3px solid {color}">
                    <div style="display:flex;justify-content:space-between;gap:0.7rem;align-items:center">
                        <div style="color:#f0f1f5;font-weight:600">{i}. {icon} {f['display_name']}</div>
                        <div style="color:{color};font-weight:700">{f['shap_value']:+.4f}</div>
                    </div>
                    <div style="height:7px;background:#1f2937;border-radius:99px;margin:0.55rem 0 0.35rem">
                        <div style="height:7px;width:{width}%;background:{color};border-radius:99px"></div>
                    </div>
                    <div style="font-size:0.82rem;color:#9ca3af">Effet {f['direction']} sur le résultat prédit</div>
                </div>
                """, unsafe_allow_html=True)

        if show_model_breakdown:
            with st.expander("🤖 Détail par modèle", expanded=False):
                bd = report["model_breakdown"]
                cv = report.get("cv_scores", {})
                cols = st.columns(max(len(bd), 1))
                model_colors = {"XGBoost": "#f59e0b", "RandomForest": "#10b981", "NeuralNet": "#8b5cf6"}
                for col, (model_name, model_probs) in zip(cols, bd.items()):
                    color = model_colors.get(model_name, "#6b7280")
                    cv_str = f"{cv.get(model_name, 0):.1%}" if cv.get(model_name) else "—"
                    rows_html = ""
                    for k, v in model_probs.items():
                        rows_html += f"""
                        <div style="display:flex;justify-content:space-between;margin-bottom:0.35rem">
                            <span style="color:#9ca3af;font-size:0.8rem">{k}</span>
                            <span style="color:#f0f1f5;font-weight:600;font-size:0.9rem">{v:.1f}%</span>
                        </div>
                        <div style="height:4px;background:#1f2937;border-radius:2px;margin-bottom:0.6rem">
                            <div style="height:4px;width:{v}%;background:{color};border-radius:2px"></div>
                        </div>
                        """
                    with col:
                        st.markdown(f"""
                        <div class="card" style="border-top:2px solid {color}">
                            <div class="card-title">{model_name}</div>
                            <div style="font-size:0.75rem;color:#9ca3af;margin-bottom:0.6rem">CV Acc: {cv_str}</div>
                            {rows_html}
                        </div>
                        """, unsafe_allow_html=True)

        if show_raw_features and "feature_df" in report:
            with st.expander("🔢 Features brutes", expanded=False):
                df = report["feature_df"].T.reset_index()
                df.columns = ["Feature", "Valeur"]
                df["Valeur"] = df["Valeur"].round(4)
                st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_h2h:
        h2h = report["h2h"]
        with st.expander("⚔️ Historique des confrontations", expanded=True):
            st.markdown(f"""
            <div class="card">
                <div class="card-title">Statistiques H2H ({h2h['total']} matchs)</div>
                <div style="display:flex;align-items:center;gap:1rem">
                    <div style="text-align:center;flex:1">
                        <div style="font-size:2rem;font-weight:800;color:#3b82f6">{h2h['home_wins']}</div>
                        <div style="color:#9ca3af;font-size:0.75rem">Victoires<br>{home_team}</div>
                    </div>
                    <div style="flex:3">
                        <div style="height:12px;background:#1f2937;border-radius:6px;display:flex;overflow:hidden">
                            <div style="width:{h2h['home_win_pct']}%;background:#3b82f6"></div>
                            <div style="width:{h2h['draw_pct']}%;background:#fbbf24"></div>
                            <div style="width:{h2h['away_win_pct']}%;background:#ec4899"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;margin-top:0.4rem;font-size:0.75rem;color:#9ca3af">
                            <span>{h2h['home_win_pct']}%</span>
                            <span>Nuls : {h2h['draws']} ({h2h['draw_pct']}%)</span>
                            <span>{h2h['away_win_pct']}%</span>
                        </div>
                    </div>
                    <div style="text-align:center;flex:1">
                        <div style="font-size:2rem;font-weight:800;color:#ec4899">{h2h['away_wins']}</div>
                        <div style="color:#9ca3af;font-size:0.75rem">Victoires<br>{away_team}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    render_disclaimer()

else:
    # ── Welcome screen ─────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;">
        <div style="font-size:4rem">⚽</div>
        <h2 style="color:white;margin-top:1rem">Prêt à prédire votre match ?</h2>
        <p style="color:#6b7280;max-width:520px;margin:0 auto;line-height:1.8">
            Entrez les deux équipes dans la barre latérale, sélectionnez la compétition,
            puis cliquez sur <b>🔮 Prédire</b> pour obtenir une analyse complète
            alimentée par <b>XGBoost + Random Forest</b> et des explications SHAP.
        </p>
        <p style="color:#fbbf24;font-size:0.85rem;margin-top:1rem">
            Première utilisation ? Lancez <code style="background:#1f2937;padding:2px 8px;border-radius:6px">python train.py</code>
        </p>
        <div style="display:flex;justify-content:center;gap:1.5rem;margin-top:2.5rem;flex-wrap:wrap">
            <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.4rem 2rem;min-width:160px">
                <div style="font-size:1.8rem">🧠</div>
                <div style="color:white;font-weight:600;margin-top:0.5rem">2 Modèles ML</div>
                <div style="color:#4b5563;font-size:0.8rem">XGBoost + Random Forest</div>
            </div>
            <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.4rem 2rem;min-width:160px">
                <div style="font-size:1.8rem">🔍</div>
                <div style="color:white;font-weight:600;margin-top:0.5rem">Explications SHAP</div>
                <div style="color:#4b5563;font-size:0.8rem">Décisions transparentes</div>
            </div>
            <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.4rem 2rem;min-width:160px">
                <div style="font-size:1.8rem">📊</div>
                <div style="color:white;font-weight:600;margin-top:0.5rem">27 Features</div>
                <div style="color:#4b5563;font-size:0.8rem">Forme, Elo, H2H…</div>
            </div>
            <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:1.4rem 2rem;min-width:160px">
                <div style="font-size:1.8rem">⚡</div>
                <div style="color:white;font-weight:600;margin-top:0.5rem">Analyse rapide</div>
                <div style="color:#4b5563;font-size:0.8rem">Résultats en &lt; 5s</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    render_disclaimer()

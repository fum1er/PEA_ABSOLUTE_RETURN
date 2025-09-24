import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Ajouter le répertoire parent au path pour importer nos modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regime_detection.market_regime_detector import MarketRegimeDetector

# Configuration de la page
st.set_page_config(
    page_title="PEA Absolute Return - Market Regime Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🎯 PEA Absolute Return - Market Regime Dashboard")
st.markdown("---")

# Sidebar pour les contrôles
st.sidebar.header("⚙️ Paramètres")
auto_refresh = st.sidebar.checkbox("Actualisation automatique", value=False)
lookback_days = st.sidebar.slider("Période d'analyse (jours)", 60, 500, 252)

@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def get_regime_data(lookback):
    """Récupère les données de régime avec cache"""
    detector = MarketRegimeDetector()
    return detector.detect_regime()

def create_regime_indicators_heatmap(indicators):
    """Crée une heatmap des indicateurs de régime"""
    # Sélectionner les indicateurs principaux
    key_indicators = {k: v for k, v in indicators.items() 
                     if k in ['Price_vs_SMA', 'RSI', 'VIX_level', 'Realized_vol', 
                             'Price_momentum_3M', 'Price_momentum_6M']}
    
    # Normaliser les valeurs pour la heatmap (-1 à 1)
    normalized = {}
    for key, value in key_indicators.items():
        if key == 'RSI':
            normalized[key] = (value - 50) / 50  # RSI centré sur 50
        elif key == 'VIX_level':
            normalized[key] = min(1, (value - 15) / 15)  # VIX normalisé
        elif 'momentum' in key.lower():
            normalized[key] = max(-1, min(1, value / 20))  # Momentum en %
        else:
            normalized[key] = max(-1, min(1, value / 10))
    
    if not normalized:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=[list(normalized.values())],
        x=list(normalized.keys()),
        y=['Signal Strength'],
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{v:.2f}" for v in normalized.values()]],
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Indicateurs de Régime - Heatmap",
        height=200,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_regime_gauge(confidence):
    """Crée une jauge pour la confiance"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Niveau de Confiance"},
        delta = {'reference': 75},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def get_regime_color(regime):
    """Retourne la couleur associée au régime"""
    colors = {
        'early_bull': '#28a745',   # Vert foncé
        'mid_bull': '#90EE90',     # Vert clair
        'late_bull': '#ffc107',    # Orange
        'early_bear': '#fd7e14',   # Orange foncé  
        'deep_bear': '#dc3545',    # Rouge
        'recovery': '#17a2b8',     # Bleu
        'sideways': '#6c757d'      # Gris
    }
    return colors.get(regime, '#6c757d')

def get_regime_emoji(regime):
    """Retourne l'emoji associé au régime"""
    emojis = {
        'early_bull': '🐂',
        'mid_bull': '📈', 
        'late_bull': '⚠️',
        'early_bear': '🔴',
        'deep_bear': '💥',
        'recovery': '🔄',
        'sideways': '↔️'
    }
    return emojis.get(regime, '❓')

# Interface principale
try:
    # Récupération des données
    with st.spinner('Analyse du régime de marché en cours...'):
        regime_data = get_regime_data(lookback_days)
    
    # Section principale - Régime actuel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        regime = regime_data['regime']
        emoji = get_regime_emoji(regime)
        st.metric(
            "Régime Actuel", 
            f"{emoji} {regime.replace('_', ' ').title()}", 
            delta=f"Confiance: {regime_data['confidence']:.0%}"
        )
    
    with col2:
        if 'VIX_level' in regime_data['indicators']:
            vix_level = regime_data['indicators']['VIX_level']
            vix_delta = regime_data['indicators'].get('VIX_change', 0)
            st.metric(
                "Niveau VIX", 
                f"{vix_level:.1f}", 
                delta=f"{vix_delta:.1f}% (20j)"
            )
    
    with col3:
        if 'Price_vs_SMA' in regime_data['indicators']:
            trend = regime_data['indicators']['Price_vs_SMA']
            st.metric(
                "Tendance vs SMA200", 
                f"{trend:+.1f}%",
                delta="Bullish" if trend > 0 else "Bearish"
            )
    
    # Jauge de confiance et heatmap
    col1, col2 = st.columns([1, 2])
    
    with col1:
        gauge_fig = create_regime_gauge(regime_data['confidence'])
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    with col2:
        heatmap_fig = create_regime_indicators_heatmap(regime_data['indicators'])
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Facteurs clés
    st.subheader("🎯 Facteurs Clés du Régime")
    if regime_data['key_drivers']:
        cols = st.columns(len(regime_data['key_drivers']))
        for i, driver in enumerate(regime_data['key_drivers']):
            with cols[i]:
                st.info(f"**{driver}**")
    else:
        st.warning("Aucun facteur clé identifié")
    
    # Détails des indicateurs
    st.subheader("📊 Détails des Indicateurs")
    
    # Créer un DataFrame pour afficher les indicateurs
    indicators_df = pd.DataFrame([
        {"Indicateur": k, "Valeur": f"{v:.2f}" if isinstance(v, (int, float)) else str(v)}
        for k, v in regime_data['indicators'].items()
    ])
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(indicators_df, use_container_width=True)
    
    with col2:
        # Recommandations stratégiques basées sur le régime
        st.subheader("💡 Recommandations Stratégiques")
        
        recommendations = {
            'early_bull': {
                'allocation': '85% Actions',
                'style': 'Value + Cycliques',
                'sectors': 'Financials, Industrials, Materials'
            },
            'mid_bull': {
                'allocation': '90% Actions',
                'style': 'Growth + Momentum', 
                'sectors': 'Technology, Consumer Disc'
            },
            'late_bull': {
                'allocation': '75% Actions',
                'style': 'Quality + Low Vol',
                'sectors': 'Healthcare, Utilities'
            },
            'early_bear': {
                'allocation': '60% Actions',
                'style': 'Quality + Defensive',
                'sectors': 'Consumer Staples, Healthcare'
            },
            'deep_bear': {
                'allocation': '30% Actions',
                'style': 'Cash + Bonds',
                'sectors': 'Defensive seulement'
            },
            'recovery': {
                'allocation': '70% Actions',
                'style': 'Value contrarian',
                'sectors': 'Financials, REITs'
            },
            'sideways': {
                'allocation': '80% Actions',
                'style': 'Rotation sectorielle',
                'sectors': 'Momentum relatif'
            }
        }
        
        current_reco = recommendations.get(regime, {})
        if current_reco:
            st.success(f"**Allocation:** {current_reco.get('allocation', 'N/A')}")
            st.info(f"**Style:** {current_reco.get('style', 'N/A')}")
            st.info(f"**Secteurs:** {current_reco.get('sectors', 'N/A')}")
    
    # Section historique et timing
    st.subheader("⏱️ Timing et Historique")
    st.write(f"**Dernière mise à jour:** {regime_data['timestamp']}")
    
    # Scores détaillés
    with st.expander("🔍 Scores Détaillés"):
        scores_df = pd.DataFrame([
            {"Dimension": k.replace('_', ' ').title(), "Score": v}
            for k, v in regime_data['scores'].items()
        ])
        st.dataframe(scores_df, use_container_width=True)
    
    # Auto-refresh
    if auto_refresh:
        st.rerun()

except Exception as e:
    st.error(f"Erreur lors du chargement des données: {str(e)}")
    st.info("Vérifiez votre connexion internet et réessayez.")

# Footer
st.markdown("---")
st.markdown("**PEA Absolute Return** - Système d'adaptation automatique aux cycles de marché")
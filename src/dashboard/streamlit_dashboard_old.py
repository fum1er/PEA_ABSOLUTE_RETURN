import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Ajouter le répertoire parent au path pour importer nos modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from regime_detection.market_regime_detector import MarketRegimeDetector
except ImportError:
    st.error("Impossible d'importer MarketRegimeDetector. Vérifiez la structure des dossiers.")
    st.stop()

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
    try:
        detector = MarketRegimeDetector()
        return detector.detect_regime()
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données: {e}")
        # Retourner des données par défaut
        return {
            "regime": "sideways",
            "confidence": 0.5,
            "key_drivers": ["Données indisponibles"],
            "scores": {"trend_score": 0, "momentum_score": 0, "volatility_score": 0, "risk_appetite_score": 0},
            "indicators": {"VIX_level": 20, "Price_vs_SMA": 0, "RSI": 50},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def create_regime_indicators_heatmap(indicators):
    """Crée une heatmap des indicateurs de régime"""
    try:
        # Sélectionner les indicateurs principaux
        key_indicators = {}
        for key in ['Price_vs_SMA', 'RSI', 'VIX_level', 'Realized_vol', 'Price_momentum_3M', 'Price_momentum_6M']:
            if key in indicators:
                key_indicators[key] = indicators[key]
        
        if not key_indicators:
            return None
        
        # Normaliser les valeurs pour la heatmap (-1 à 1)
        normalized = {}
        for key, value in key_indicators.items():
            try:
                if pd.isna(value):
                    normalized[key] = 0
                elif key == 'RSI':
                    normalized[key] = (value - 50) / 50  # RSI centré sur 50
                elif key == 'VIX_level':
                    normalized[key] = min(1, (value - 15) / 15)  # VIX normalisé
                elif 'momentum' in key.lower():
                    normalized[key] = max(-1, min(1, value / 20))  # Momentum en %
                else:
                    normalized[key] = max(-1, min(1, value / 10))
            except:
                normalized[key] = 0
        
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
    except Exception as e:
        st.error(f"Erreur lors de la création de la heatmap: {e}")
        return None

def create_regime_gauge(confidence):
    """Crée une jauge pour la confiance"""
    try:
        # S'assurer que confidence est un nombre valide
        if pd.isna(confidence) or confidence is None:
            confidence = 0.5
        
        confidence_pct = float(confidence) * 100
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence_pct,
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
    except Exception as e:
        st.error(f"Erreur lors de la création de la jauge: {e}")
        return None

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

def safe_format_value(value, format_type="float"):
    """Formate une valeur de manière sécurisée"""
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        
        if format_type == "float":
            return f"{float(value):.1f}"
        elif format_type == "percent":
            return f"{float(value):.1f}%"
        elif format_type == "percent_confidence":
            return f"{float(value):.0%}"
        else:
            return str(value)
    except:
        return "N/A"

# Interface principale
try:
    # Récupération des données
    with st.spinner('Analyse du régime de marché en cours...'):
        regime_data = get_regime_data(lookback_days)
    
    if regime_data is None:
        st.error("Impossible de récupérer les données de régime")
        st.stop()
    
    # Section principale - Régime actuel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        regime = regime_data.get('regime', 'unknown')
        emoji = get_regime_emoji(regime)
        confidence = regime_data.get('confidence', 0.5)
        
        st.metric(
            "Régime Actuel", 
            f"{emoji} {regime.replace('_', ' ').title()}", 
            delta=f"Confiance: {safe_format_value(confidence, 'percent_confidence')}"
        )
    
    with col2:
        indicators = regime_data.get('indicators', {})
        if 'VIX_level' in indicators:
            vix_level = indicators['VIX_level']
            vix_change = indicators.get('VIX_change', 0)
            st.metric(
                "Niveau VIX", 
                safe_format_value(vix_level),
                delta=f"{safe_format_value(vix_change, 'percent')} (20j)"
            )
        else:
            st.metric("Niveau VIX", "N/A", delta="N/A")
    
    with col3:
        if 'Price_vs_SMA' in indicators:
            trend = indicators['Price_vs_SMA']
            trend_direction = "Bullish" if trend > 0 else "Bearish" if trend < 0 else "Neutral"
            st.metric(
                "Tendance vs SMA200", 
                f"{safe_format_value(trend, 'percent')}",
                delta=trend_direction
            )
        else:
            st.metric("Tendance vs SMA200", "N/A", delta="N/A")
    
    # Jauge de confiance et heatmap
    col1, col2 = st.columns([1, 2])
    
    with col1:
        gauge_fig = create_regime_gauge(confidence)
        if gauge_fig:
            st.plotly_chart(gauge_fig, use_container_width=True)
        else:
            st.warning("Impossible d'afficher la jauge de confiance")
    
    with col2:
        heatmap_fig = create_regime_indicators_heatmap(indicators)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.warning("Impossible d'afficher la heatmap des indicateurs")
    
    # Facteurs clés
    st.subheader("🎯 Facteurs Clés du Régime")
    key_drivers = regime_data.get('key_drivers', [])
    if key_drivers and isinstance(key_drivers, list):
        if len(key_drivers) > 0:
            cols = st.columns(len(key_drivers))
            for i, driver in enumerate(key_drivers):
                with cols[i]:
                    st.info(f"**{driver}**")
        else:
            st.warning("Aucun facteur clé identifié")
    else:
        st.warning("Aucun facteur clé disponible")
    
    # Détails des indicateurs
    st.subheader("📊 Détails des Indicateurs")
    
    # Créer un DataFrame pour afficher les indicateurs
    if indicators:
        try:
            indicators_list = []
            for k, v in indicators.items():
                formatted_value = safe_format_value(v)
                indicators_list.append({"Indicateur": k, "Valeur": formatted_value})
            
            if indicators_list:
                indicators_df = pd.DataFrame(indicators_list)
            else:
                indicators_df = pd.DataFrame({"Indicateur": ["Aucun"], "Valeur": ["N/A"]})
        except Exception as e:
            st.error(f"Erreur lors de la création du DataFrame: {e}")
            indicators_df = pd.DataFrame({"Indicateur": ["Erreur"], "Valeur": [str(e)]})
    else:
        indicators_df = pd.DataFrame({"Indicateur": ["Aucun"], "Valeur": ["N/A"]})
    
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
        else:
            st.warning("Aucune recommandation disponible pour ce régime")
    
    # Section historique et timing
    st.subheader("⏱️ Timing et Historique")
    timestamp = regime_data.get('timestamp', 'N/A')
    st.write(f"**Dernière mise à jour:** {timestamp}")
    
    # Scores détaillés
    with st.expander("🔍 Scores Détaillés"):
        scores = regime_data.get('scores', {})
        if scores:
            try:
                scores_list = []
                for k, v in scores.items():
                    formatted_score = safe_format_value(v)
                    scores_list.append({"Dimension": k.replace('_', ' ').title(), "Score": formatted_score})
                
                if scores_list:
                    scores_df = pd.DataFrame(scores_list)
                    st.dataframe(scores_df, use_container_width=True)
                else:
                    st.warning("Aucun score disponible")
            except Exception as e:
                st.error(f"Erreur lors de l'affichage des scores: {e}")
        else:
            st.warning("Aucun score détaillé disponible")
    
    # Auto-refresh
    if auto_refresh:
        st.rerun()

except Exception as e:
    st.error(f"Erreur lors du chargement des données: {str(e)}")
    st.info("Vérifiez votre connexion internet et réessayez.")
    
    # Afficher des informations de debug
    with st.expander("🔧 Informations de Debug"):
        st.write("**Structure des dossiers attendue:**")
        st.code("""
pea_absolute_return/
├── src/
│   ├── regime_detection/
│   │   ├── __init__.py
│   │   └── market_regime_detector.py
│   └── dashboard/
│       └── streamlit_dashboard.py
        """)
        st.write("**Commande pour lancer le dashboard:**")
        st.code("streamlit run src/dashboard/streamlit_dashboard.py")

# Footer
st.markdown("---")
st.markdown("**PEA Absolute Return** - Système d'adaptation automatique aux cycles de marché")
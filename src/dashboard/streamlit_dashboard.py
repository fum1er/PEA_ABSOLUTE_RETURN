import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="PEA Dashboard - Analyse Multifactorielle",
    page_icon="📊",
    layout="wide"
)

st.title("📊 PEA Dashboard - Analyse Multifactorielle")
st.markdown("Analyse des actions européennes et ETFs pour stratégie PEA")
st.markdown("---")

# Liste des actifs PEA éligibles
PEA_STOCKS = {
    # CAC 40 - Les plus liquides
    "Total": "TTE.PA",
    "LVMH": "MC.PA",
    "Sanofi": "SAN.PA",
    "L'Oréal": "OR.PA",
    "Schneider": "SU.PA",
    "Air Liquide": "AI.PA",
    "BNP Paribas": "BNP.PA",
    "Vinci": "DG.PA",
    "Safran": "SAF.PA",
    "Stellantis": "STLA.PA",
    
    # Europe - Grandes valeurs
    "ASML": "ASML.AS",
    "SAP": "SAP.DE",
    "Siemens": "SIE.DE",
    "Allianz": "ALV.DE",
    "Unilever": "UNA.AS",
    "Nestlé": "NESN.SW",
    "Roche": "ROG.SW",
    "Novo Nordisk": "NOVO-B.CO",
    "ABB": "ABB.SW",
    "Adidas": "ADS.DE"
}

# ETFs éligibles PEA (répliquant des indices US/Monde)
PEA_ETFS = {
    "S&P 500 PEA": "PE500.PA",  # Amundi S&P 500 PEA
    "NASDAQ PEA": "PUST.PA",     # Lyxor NASDAQ PEA
    "World PEA": "CW8.PA",       # Amundi MSCI World
    "USA PEA": "WNKE.PA",        # Lyxor PEA USA
    "Healthcare": "HLTW.PA",      # Healthcare World
    "Technology": "TNOW.PA",      # Technology
    "Europe": "MEUD.PA",         # STOXX Europe 600
    "Emergents": "PAEEM.PA"      # Emergents PEA
}

# Fonction pour calculer les indicateurs techniques
@st.cache_data(ttl=300)
def calculate_indicators(df):
    """Calcule les indicateurs techniques pour l'analyse multifactorielle"""
    indicators = {}
    
    if len(df) > 0:
        close = df['Close']
        volume = df['Volume']
        high = df['High']
        low = df['Low']
        
        # MOMENTUM
        indicators['RSI'] = calculate_rsi(close)
        indicators['Momentum_1M'] = ((close.iloc[-1] / close.iloc[-22]) - 1) * 100 if len(close) > 22 else 0
        indicators['Momentum_3M'] = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100 if len(close) > 63 else 0
        indicators['Momentum_6M'] = ((close.iloc[-1] / close.iloc[-126]) - 1) * 100 if len(close) > 126 else 0
        
        # TREND
        if len(close) >= 50:
            indicators['SMA_50'] = close.rolling(window=50).mean().iloc[-1]
            indicators['Price_vs_SMA50'] = ((close.iloc[-1] / indicators['SMA_50']) - 1) * 100
        
        if len(close) >= 200:
            indicators['SMA_200'] = close.rolling(window=200).mean().iloc[-1]
            indicators['Price_vs_SMA200'] = ((close.iloc[-1] / indicators['SMA_200']) - 1) * 100
        
        # VOLATILITY
        if len(close) > 20:
            returns = close.pct_change().dropna()
            indicators['Volatility_20D'] = returns.tail(20).std() * np.sqrt(252) * 100
            indicators['Volatility_60D'] = returns.tail(60).std() * np.sqrt(252) * 100 if len(returns) > 60 else 0
        
        # VOLUME
        if len(volume) > 20:
            indicators['Volume_Ratio'] = volume.iloc[-1] / volume.rolling(window=20).mean().iloc[-1]
        
        # VALUE (nécessiterait des données fondamentales, ici on simule)
        indicators['52W_High'] = high.tail(252).max() if len(high) > 252 else high.max()
        indicators['52W_Low'] = low.tail(252).min() if len(low) > 252 else low.min()
        indicators['Distance_52W_High'] = ((close.iloc[-1] / indicators['52W_High']) - 1) * 100
        
    return indicators

def calculate_rsi(prices, period=14):
    """Calcule le RSI"""
    if len(prices) < period:
        return 50
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

@st.cache_data(ttl=300)
def get_stock_data(ticker, period="6mo"):
    """Récupère les données d'un ticker"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        info = stock.info
        return df, info
    except Exception as e:
        st.error(f"Erreur pour {ticker}: {e}")
        return None, None

def calculate_factor_scores(indicators):
    """Calcule les scores factoriels"""
    scores = {}
    
    # Score Momentum (0-100)
    momentum_factors = []
    if 'RSI' in indicators:
        # RSI optimal entre 40 et 70
        rsi = indicators['RSI']
        if 40 <= rsi <= 70:
            momentum_factors.append(100 - abs(rsi - 55))
        else:
            momentum_factors.append(max(0, 50 - abs(rsi - 50)))
    
    if 'Momentum_3M' in indicators:
        # Momentum positif mais pas excessif
        mom = indicators['Momentum_3M']
        if mom > 0:
            momentum_factors.append(min(100, mom * 2))
        else:
            momentum_factors.append(max(0, 50 + mom))
    
    scores['Momentum'] = np.mean(momentum_factors) if momentum_factors else 50
    
    # Score Trend (0-100)
    trend_factors = []
    if 'Price_vs_SMA200' in indicators:
        # Au-dessus de la SMA200 = positif
        pct = indicators['Price_vs_SMA200']
        trend_factors.append(min(100, max(0, 50 + pct * 2)))
    
    if 'Price_vs_SMA50' in indicators:
        # Au-dessus de la SMA50 = positif
        pct = indicators['Price_vs_SMA50']
        trend_factors.append(min(100, max(0, 50 + pct * 3)))
    
    scores['Trend'] = np.mean(trend_factors) if trend_factors else 50
    
    # Score Quality/Low Vol (0-100)
    quality_factors = []
    if 'Volatility_20D' in indicators:
        # Moins de volatilité = mieux
        vol = indicators['Volatility_20D']
        quality_factors.append(max(0, 100 - vol * 2))
    
    if 'Distance_52W_High' in indicators:
        # Proche du plus haut = force
        dist = indicators['Distance_52W_High']
        quality_factors.append(max(0, 100 + dist))
    
    scores['Quality'] = np.mean(quality_factors) if quality_factors else 50
    
    # Score global
    scores['Global'] = np.mean([scores['Momentum'], scores['Trend'], scores['Quality']])
    
    return scores

# Interface utilisateur
st.sidebar.header("⚙️ Paramètres")

# Sélection du type d'actif
asset_type = st.sidebar.radio(
    "Type d'actif",
    ["Actions Européennes", "ETFs PEA", "Comparaison"]
)

# Sélection de la période
period = st.sidebar.select_slider(
    "Période d'analyse",
    options=["1mo", "3mo", "6mo", "1y", "2y"],
    value="6mo"
)

# Sélection des actifs
if asset_type == "Actions Européennes":
    selected_assets = st.sidebar.multiselect(
        "Sélectionner les actions",
        options=list(PEA_STOCKS.keys()),
        default=["LVMH", "Total", "ASML", "L'Oréal", "Schneider"]
    )
    tickers = {k: PEA_STOCKS[k] for k in selected_assets}
elif asset_type == "ETFs PEA":
    selected_assets = st.sidebar.multiselect(
        "Sélectionner les ETFs",
        options=list(PEA_ETFS.keys()),
        default=["S&P 500 PEA", "World PEA", "Europe"]
    )
    tickers = {k: PEA_ETFS[k] for k in selected_assets}
else:  # Comparaison
    selected_stocks = st.sidebar.multiselect(
        "Actions",
        options=list(PEA_STOCKS.keys()),
        default=["LVMH", "Total"]
    )
    selected_etfs = st.sidebar.multiselect(
        "ETFs",
        options=list(PEA_ETFS.keys()),
        default=["S&P 500 PEA"]
    )
    tickers = {}
    tickers.update({k: PEA_STOCKS[k] for k in selected_stocks})
    tickers.update({k: PEA_ETFS[k] for k in selected_etfs})

# Bouton de rafraîchissement
if st.sidebar.button("🔄 Rafraîchir les données"):
    st.cache_data.clear()

# Collecte des données
if tickers:
    data_dict = {}
    indicators_dict = {}
    scores_dict = {}
    
    with st.spinner("Chargement des données..."):
        for name, ticker in tickers.items():
            df, info = get_stock_data(ticker, period)
            if df is not None and len(df) > 0:
                data_dict[name] = df
                indicators = calculate_indicators(df)
                indicators_dict[name] = indicators
                scores_dict[name] = calculate_factor_scores(indicators)
    
    if data_dict:
        # Vue d'ensemble - Tableau récapitulatif
        st.subheader("📈 Vue d'ensemble des actifs")
        
        summary_data = []
        for name in data_dict.keys():
            if name in indicators_dict:
                ind = indicators_dict[name]
                scores = scores_dict[name]
                
                summary_data.append({
                    "Actif": name,
                    "Prix": data_dict[name]['Close'].iloc[-1],
                    "Perf 1M %": ind.get('Momentum_1M', 0),
                    "Perf 3M %": ind.get('Momentum_3M', 0),
                    "RSI": ind.get('RSI', 50),
                    "Volatilité %": ind.get('Volatility_20D', 0),
                    "Score Momentum": scores.get('Momentum', 50),
                    "Score Trend": scores.get('Trend', 50),
                    "Score Quality": scores.get('Quality', 50),
                    "Score Global": scores.get('Global', 50)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Styling du tableau
        def highlight_scores(val):
            try:
                if isinstance(val, (int, float)):
                    if val > 70:
                        return 'background-color: #90EE90'
                    elif val < 30:
                        return 'background-color: #FFB6C1'
            except:
                pass
            return ''
        
        # Formater le DataFrame pour l'affichage
        formatted_df = summary_df.copy()
        formatted_df['Prix'] = formatted_df['Prix'].apply(lambda x: f"{x:.2f}")
        formatted_df['Perf 1M %'] = formatted_df['Perf 1M %'].apply(lambda x: f"{x:.1f}")
        formatted_df['Perf 3M %'] = formatted_df['Perf 3M %'].apply(lambda x: f"{x:.1f}")
        formatted_df['RSI'] = formatted_df['RSI'].apply(lambda x: f"{x:.0f}")
        formatted_df['Volatilité %'] = formatted_df['Volatilité %'].apply(lambda x: f"{x:.1f}")
        formatted_df['Score Momentum'] = formatted_df['Score Momentum'].apply(lambda x: f"{x:.0f}")
        formatted_df['Score Trend'] = formatted_df['Score Trend'].apply(lambda x: f"{x:.0f}")
        formatted_df['Score Quality'] = formatted_df['Score Quality'].apply(lambda x: f"{x:.0f}")
        formatted_df['Score Global'] = formatted_df['Score Global'].apply(lambda x: f"{x:.0f}")
        
        st.dataframe(
            formatted_df.sort_values("Score Global", ascending=False, key=lambda x: pd.to_numeric(x, errors='coerce')),
            use_container_width=True,
            hide_index=True
        )
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique des performances
            st.subheader("📊 Évolution des prix (base 100)")
            
            fig = go.Figure()
            for name, df in data_dict.items():
                normalized = (df['Close'] / df['Close'].iloc[0]) * 100
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=normalized,
                    mode='lines',
                    name=name
                ))
            
            fig.update_layout(
                hovermode='x unified',
                height=400,
                xaxis_title="Date",
                yaxis_title="Performance (base 100)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Radar des scores factoriels
            st.subheader("🎯 Analyse Factorielle")
            
            # Sélection d'un actif pour le radar
            selected_for_radar = st.selectbox(
                "Sélectionner un actif pour l'analyse détaillée",
                options=list(scores_dict.keys())
            )
            
            if selected_for_radar in scores_dict:
                scores = scores_dict[selected_for_radar]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=[scores['Momentum'], scores['Trend'], scores['Quality']],
                    theta=['Momentum', 'Trend', 'Quality/Low Vol'],
                    fill='toself',
                    name=selected_for_radar
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=False,
                    height=400,
                    title=f"Scores factoriels - {selected_for_radar}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Matrice de corrélation
        st.subheader("🔗 Matrice de corrélation")
        
        if len(data_dict) > 1:
            # Calculer les rendements
            returns_data = {}
            for name, df in data_dict.items():
                returns_data[name] = df['Close'].pct_change().dropna()
            
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu',
                range_color=[-1, 1]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top picks basés sur les scores
        st.subheader("🏆 Top Picks - Sélection Multifactorielle")
        
        # S'assurer qu'on a au moins 3 actifs, sinon prendre le max disponible
        n_top = min(3, len(summary_df))
        
        if n_top > 0:
            top_momentum = summary_df.nlargest(n_top, "Score Momentum")[["Actif", "Score Momentum", "Perf 3M %"]]
            top_trend = summary_df.nlargest(n_top, "Score Trend")[["Actif", "Score Trend", "RSI"]]
            top_quality = summary_df.nlargest(n_top, "Score Quality")[["Actif", "Score Quality", "Volatilité %"]]
            top_global = summary_df.nlargest(n_top, "Score Global")[["Actif", "Score Global", "Perf 3M %", "Volatilité %"]]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🚀 Best Momentum", top_momentum.iloc[0]["Actif"])
                # Formater pour l'affichage
                top_momentum_display = top_momentum.copy()
                top_momentum_display['Score Momentum'] = top_momentum_display['Score Momentum'].apply(lambda x: f"{x:.0f}")
                top_momentum_display['Perf 3M %'] = top_momentum_display['Perf 3M %'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(top_momentum_display, hide_index=True)
            
            with col2:
                st.metric("📈 Best Trend", top_trend.iloc[0]["Actif"])
                top_trend_display = top_trend.copy()
                top_trend_display['Score Trend'] = top_trend_display['Score Trend'].apply(lambda x: f"{x:.0f}")
                top_trend_display['RSI'] = top_trend_display['RSI'].apply(lambda x: f"{x:.0f}")
                st.dataframe(top_trend_display, hide_index=True)
            
            with col3:
                st.metric("💎 Best Quality", top_quality.iloc[0]["Actif"])
                top_quality_display = top_quality.copy()
                top_quality_display['Score Quality'] = top_quality_display['Score Quality'].apply(lambda x: f"{x:.0f}")
                top_quality_display['Volatilité %'] = top_quality_display['Volatilité %'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(top_quality_display, hide_index=True)
            
            with col4:
                st.metric("🌟 Best Global", top_global.iloc[0]["Actif"])
                top_global_display = top_global.copy()
                top_global_display['Score Global'] = top_global_display['Score Global'].apply(lambda x: f"{x:.0f}")
                top_global_display['Perf 3M %'] = top_global_display['Perf 3M %'].apply(lambda x: f"{x:.1f}%")
                top_global_display['Volatilité %'] = top_global_display['Volatilité %'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(top_global_display, hide_index=True)
        
        # Indicateurs détaillés
        with st.expander("📋 Voir tous les indicateurs détaillés"):
            detailed_data = []
            for name, ind in indicators_dict.items():
                row = {"Actif": name}
                row.update({k: f"{v:.2f}" if isinstance(v, (int, float)) else v for k, v in ind.items()})
                detailed_data.append(row)
            
            detailed_df = pd.DataFrame(detailed_data)
            st.dataframe(detailed_df, use_container_width=True, hide_index=True)
        
    else:
        st.error("Aucune donnée disponible pour les actifs sélectionnés")
else:
    st.info("👈 Sélectionnez des actifs dans le panneau latéral")

# Footer avec informations
st.markdown("---")
st.markdown("""
**Légende des indicateurs:**
- **RSI**: Relative Strength Index (14 jours) - Surachat >70, Survente <30
- **Momentum**: Performance sur période - Mesure la force du mouvement
- **Volatilité**: Écart-type annualisé des rendements sur 20 jours
- **Score Global**: Moyenne des scores Momentum, Trend et Quality (0-100)

**Stratégie Multifactorielle PEA:**
- Combiner plusieurs facteurs pour sélectionner les meilleures opportunités
- Adapter l'allocation selon les conditions de marché
- Privilégier les actifs avec scores élevés sur plusieurs dimensions
""")
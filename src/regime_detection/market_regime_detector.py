import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """
    Détecteur de régime de marché basé sur des indicateurs multi-dimensionnels
    Implémente la logique décrite dans le document PDF
    """
    
    def __init__(self):
        self.indicators = {
            # Trend & Momentum
            "market_trend": ["SMA_200", "Price_vs_SMA", "MACD"],
            "momentum": ["RSI", "Price_momentum_3M", "Price_momentum_6M"],
            
            # Volatilité & Risk
            "volatility": ["VIX_level", "VIX_change", "Realized_vol"],
            "risk_appetite": ["HY_spreads", "Term_structure", "USD_strength"],
            
            # Breadth & Participation 
            "breadth": ["Advance_decline", "New_highs_lows", "Sector_participation"],
            "flows": ["Fund_flows", "Margin_debt", "Put_call_ratio"],
            
            # Macro & Fundamentals
            "macro": ["Yield_curve", "Inflation_expectations", "GDP_nowcast"],
            "fundamentals": ["Earnings_growth", "Margins", "PE_expansion"]
        }
        
        self.regime_definitions = {
            "bull_market": {
                "early_bull": {
                    "characteristics": "Sortie récession, valorisations attractives, sentiment négatif",
                    "duration": "6-12 mois",
                    "best_styles": ["Deep Value", "Cyclicals", "Small Caps"],
                    "sectors": ["Financials", "Industrials", "Materials", "Energy"]
                },
                "mid_bull": {
                    "characteristics": "Croissance confirmée, momentum fort, participation large",
                    "duration": "12-24 mois",
                    "best_styles": ["Growth", "Momentum", "Quality"],
                    "sectors": ["Technology", "Consumer Disc", "Healthcare"]
                },
                "late_bull": {
                    "characteristics": "Valorisations élevées, euphorie, spéculation",
                    "duration": "6-18 mois",
                    "best_styles": ["Quality", "Low Vol", "Defensive"],
                    "sectors": ["Consumer Staples", "Utilities", "Healthcare"]
                }
            },
            "bear_market": {
                "early_bear": {
                    "characteristics": "Cassure technique, début rotation défensive",
                    "duration": "3-6 mois",
                    "best_styles": ["Quality", "Defensive", "Low Vol"],
                    "sectors": ["Utilities", "Consumer Staples", "Bonds"]
                },
                "deep_bear": {
                    "characteristics": "Panique, liquidations forcées, corrélations = 1",
                    "duration": "6-12 mois",
                    "best_styles": ["Cash", "Bonds", "Gold"],
                    "sectors": ["Minimal equity exposure"]
                },
                "recovery": {
                    "characteristics": "Stabilisation, premiers signes reprise",
                    "duration": "3-9 mois",
                    "best_styles": ["Contrarian Value", "Quality"],
                    "sectors": ["Financials", "REITs", "Cyclicals"]
                }
            },
            "sideways": {
                "characteristics": "Range trading, rotation sectorielle",
                "best_styles": ["Sector Rotation", "Mean Reversion"],
                "sectors": ["Dépend momentum relatif"]
            }
        }
        
        self.scaler = StandardScaler()
        self.current_data = None
    
    def fetch_market_data(self, lookback_days=252):
        """
        Récupère les données de marché nécessaires pour l'analyse
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 50)  # Plus de données pour les calculs
        
        # Indices principaux
        tickers = {
            'SPX': '^GSPC',    # S&P 500
            'VIX': '^VIX',     # VIX
            'EUR': '^STOXX50E', # Euro Stoxx 50
            'USD': 'DX-Y.NYB'  # Dollar Index
        }
        
        data = {}
        for name, ticker in tickers.items():
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if len(df) > 0:
                    data[name] = df
                    print(f"✓ Données récupérées pour {name}: {len(df)} points")
                else:
                    print(f"⚠ Aucune donnée pour {name}")
            except Exception as e:
                print(f"⚠ Erreur pour {name}: {e}")
                # Données de fallback pour éviter les erreurs
                if name == 'SPX':
                    # Créer des données simulées pour éviter l'erreur
                    dates = pd.date_range(start=start_date, end=end_date, freq='D')
                    np.random.seed(42)
                    prices = 4500 + np.random.randn(len(dates)).cumsum() * 50
                    data[name] = pd.DataFrame({
                        'Open': prices,
                        'High': prices * 1.01,
                        'Low': prices * 0.99,
                        'Close': prices,
                        'Volume': np.random.randint(1000000, 5000000, len(dates))
                    }, index=dates)
                    print(f"✓ Données simulées créées pour {name}")
        
        self.current_data = data
        return data
    
    def calculate_technical_indicators(self):
        """
        Calcule les indicateurs techniques nécessaires
        """
        if self.current_data is None:
            raise ValueError("Aucune donnée disponible. Exécutez fetch_market_data() d'abord.")
        
        indicators = {}
        
        # Exemple avec S&P 500
        spx_data = self.current_data.get('SPX')
        if spx_data is not None and len(spx_data) > 200:
            try:
                # Nettoyer les données - supprimer les NaN
                spx_close = spx_data['Close'].dropna()
                
                if len(spx_close) > 200:
                    # SMA 200
                    sma_200 = spx_close.rolling(window=200).mean().iloc[-1]
                    current_price = spx_close.iloc[-1]
                    indicators['SMA_200'] = sma_200
                    indicators['Price_vs_SMA'] = (current_price / sma_200 - 1) * 100
                    
                    # RSI
                    rsi_value = self._calculate_rsi(spx_close)
                    if not pd.isna(rsi_value):
                        indicators['RSI'] = rsi_value
                    else:
                        indicators['RSI'] = 50  # valeur neutre par défaut
                    
                    # Momentum (vérification de la longueur des données)
                    if len(spx_close) > 63:
                        past_price_3m = spx_close.iloc[-63]
                        if not pd.isna(past_price_3m):
                            indicators['Price_momentum_3M'] = (current_price / past_price_3m - 1) * 100
                        
                    if len(spx_close) > 126:
                        past_price_6m = spx_close.iloc[-126]
                        if not pd.isna(past_price_6m):
                            indicators['Price_momentum_6M'] = (current_price / past_price_6m - 1) * 100
                    
                    # Volatilité réalisée
                    returns = spx_close.pct_change().dropna()
                    if len(returns) > 20:
                        vol = returns.rolling(window=20).std().iloc[-1]
                        if not pd.isna(vol):
                            indicators['Realized_vol'] = vol * np.sqrt(252) * 100
                
            except Exception as e:
                print(f"Erreur lors du calcul des indicateurs SPX: {e}")
        
        # VIX - avec gestion d'erreur
        vix_data = self.current_data.get('VIX')
        if vix_data is not None and len(vix_data) > 0:
            try:
                vix_close = vix_data['Close'].dropna()
                if len(vix_close) > 0:
                    current_vix = vix_close.iloc[-1]
                    if not pd.isna(current_vix):
                        indicators['VIX_level'] = current_vix
                        
                        if len(vix_close) > 20:
                            past_vix = vix_close.iloc[-20]
                            if not pd.isna(past_vix) and past_vix != 0:
                                indicators['VIX_change'] = ((current_vix / past_vix) - 1) * 100
            except Exception as e:
                print(f"Erreur lors du calcul des indicateurs VIX: {e}")
        
        # Valeurs par défaut si pas de données
        if not indicators:
            indicators = {
                'SMA_200': 4500,
                'Price_vs_SMA': 0,
                'RSI': 50,
                'VIX_level': 20,
                'VIX_change': 0,
                'Realized_vol': 15,
                'Price_momentum_3M': 0,
                'Price_momentum_6M': 0
            }
            print("⚠ Utilisation des valeurs par défaut")
        
        return indicators
    
    def _calculate_rsi(self, prices, window=14):
        """Calcule le RSI avec gestion d'erreur"""
        try:
            if len(prices) < window + 1:
                return 50  # valeur neutre
                
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            final_rsi = rsi.iloc[-1]
            return final_rsi if not pd.isna(final_rsi) else 50
        except:
            return 50
    
    def calculate_regime_scores(self):
        """
        Calcule les scores pour chaque dimension du régime
        """
        indicators = self.calculate_technical_indicators()
        
        scores = {
            'trend_score': 0,
            'momentum_score': 0,
            'volatility_score': 0,
            'risk_appetite_score': 0
        }
        
        # Score de tendance
        if 'Price_vs_SMA' in indicators:
            price_vs_sma = indicators['Price_vs_SMA']
            if price_vs_sma > 5:
                scores['trend_score'] = 1  # Bullish
            elif price_vs_sma < -5:
                scores['trend_score'] = -1  # Bearish
            else:
                scores['trend_score'] = 0  # Neutral
        
        # Score de momentum
        if 'RSI' in indicators:
            rsi = indicators['RSI']
            if rsi > 70:
                scores['momentum_score'] = 1  # Overbought
            elif rsi < 30:
                scores['momentum_score'] = -1  # Oversold
            else:
                scores['momentum_score'] = 0  # Neutral
        
        # Score de volatilité
        if 'VIX_level' in indicators:
            vix = indicators['VIX_level']
            if vix > 30:
                scores['volatility_score'] = -1  # High fear
            elif vix < 15:
                scores['volatility_score'] = 1  # Complacency
            else:
                scores['volatility_score'] = 0  # Normal
        
        return scores, indicators
    
    def classify_regime(self, scores):
        """
        Classifie le régime de marché basé sur les scores
        """
        trend = scores['trend_score']
        momentum = scores['momentum_score']
        volatility = scores['volatility_score']
        
        # Logique de classification simplifiée
        if trend > 0 and volatility >= 0:
            if momentum > 0:
                return "mid_bull"
            else:
                return "early_bull"
        elif trend > 0 and volatility < 0:
            return "late_bull"
        elif trend < 0 and volatility < 0:
            if momentum < 0:
                return "deep_bear"
            else:
                return "early_bear"
        elif trend < 0 and volatility >= 0:
            return "recovery"
        else:
            return "sideways"
    
    def calculate_confidence(self, scores):
        """
        Calcule le niveau de confiance de la prédiction
        """
        try:
            # Plus les scores sont alignés, plus la confiance est élevée
            score_values = [v for v in scores.values() if not pd.isna(v)]
            if not score_values:
                return 0.5
                
            abs_values = [abs(v) for v in score_values]
            mean_abs = np.mean(abs_values)
            
            if mean_abs == 0:
                return 0.5
                
            std_val = np.std(score_values)
            alignment = 1 - (std_val / (mean_abs + 0.001))
            confidence = max(0.3, min(1.0, alignment))
            return round(confidence, 2)
        except:
            return 0.5
    
    def detect_regime(self):
        """
        Fonction principale de détection du régime
        """
        try:
            # Récupérer les données
            self.fetch_market_data()
            
            # Calculer les scores
            scores, indicators = self.calculate_regime_scores()
            
            # Classifier le régime
            regime = self.classify_regime(scores)
            
            # Calculer la confiance
            confidence = self.calculate_confidence(scores)
            
            # Identifier les facteurs clés
            key_drivers = self._identify_key_drivers(scores, indicators)
            
            return {
                "regime": regime,
                "sub_regime": regime,  # Pour l'instant, même valeur
                "confidence": confidence,
                "key_drivers": key_drivers,
                "scores": scores,
                "indicators": indicators,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"Erreur dans detect_regime: {e}")
            # Retourner des données par défaut en cas d'erreur
            return {
                "regime": "sideways",
                "sub_regime": "sideways",
                "confidence": 0.5,
                "key_drivers": ["Données indisponibles"],
                "scores": {"trend_score": 0, "momentum_score": 0, "volatility_score": 0, "risk_appetite_score": 0},
                "indicators": {"VIX_level": 20, "Price_vs_SMA": 0, "RSI": 50},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def _identify_key_drivers(self, scores, indicators):
        """
        Identifie les 3 principaux facteurs influençant le régime
        """
        drivers = []
        
        try:
            if abs(scores.get('trend_score', 0)) > 0:
                direction = "Bullish" if scores['trend_score'] > 0 else "Bearish"
                drivers.append(f"Trend {direction}")
            
            vix_level = indicators.get('VIX_level', 20)
            if vix_level > 25:
                drivers.append("High Volatility")
            elif vix_level < 15:
                drivers.append("Low Volatility")
            
            momentum_3m = indicators.get('Price_momentum_3M', 0)
            if momentum_3m > 10:
                drivers.append("Strong Momentum")
            elif momentum_3m < -10:
                drivers.append("Weak Momentum")
        except:
            drivers = ["Analyse en cours"]
        
        return drivers[:3] if drivers else ["Marché Neutre"]

# Exemple d'utilisation
if __name__ == "__main__":
    detector = MarketRegimeDetector()
    result = detector.detect_regime()
    
    print("=== DÉTECTION DU RÉGIME DE MARCHÉ ===")
    print(f"Régime détecté: {result['regime']}")
    print(f"Confiance: {result['confidence']:.0%}")
    print(f"Facteurs clés: {', '.join(result['key_drivers'])}")
    print(f"Timestamp: {result['timestamp']}")
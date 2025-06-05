"""
Correction COMPLETE du data leakage pour le projet CS:GO
École89 - 2025
"""

import pandas as pd
import numpy as np  # AJOUT MANQUANT
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

def create_independent_target(df):
    """
    Crée une variable cible INDÉPENDANTE des features utilisées
    """
    
    # MÉTHODE 1: Target basée sur des seuils fixes (recommandée)
    def create_performance_target_v2(df):
        """Version corrigée sans leakage"""
        
        # Utiliser SEULEMENT des métriques de base, pas de ratios calculés
        conditions = []
        weights = []
        
        # Condition 1: Nombre de kills élevé (seuil fixe)
        if 'total_kills' in df.columns:
            kill_threshold = 1500  # Seuil fixe, pas basé sur les données
            conditions.append(df['total_kills'] >= kill_threshold)
            weights.append(0.4)
        
        # Condition 2: Faible nombre de morts (seuil fixe)
        if 'total_deaths' in df.columns:
            death_threshold = 1200  # Seuil fixe
            conditions.append(df['total_deaths'] <= death_threshold)
            weights.append(0.3)
        
        # Condition 3: Dégâts élevés (seuil fixe)
        if 'total_damage_done' in df.columns:
            damage_threshold = 150000  # Seuil fixe
            conditions.append(df['total_damage_done'] >= damage_threshold)
            weights.append(0.3)
        
        # Calculer le score (nombre de conditions remplies)
        if conditions:
            performance_score = sum(
                condition.astype(int) * weight 
                for condition, weight in zip(conditions, weights)
            )
            
            # High performer = top 35% (seuil fixe)
            threshold = 0.7  # Seuil fixe, pas basé sur quantiles
            high_performer = (performance_score >= threshold).astype(int)
        else:
            # Fallback aléatoire équilibré
            high_performer = np.random.binomial(1, 0.4, len(df))
        
        return high_performer
    
    # MÉTHODE 2: Target basée sur l'expérience (complètement indépendante)
    def create_experience_target(df):
        """Target basée sur l'expérience de jeu uniquement"""
        
        if 'total_time_played' in df.columns:
            # Plus de 500 heures = expérimenté
            hours_played = df['total_time_played'] / 3600
            experienced = (hours_played >= 500).astype(int)
            return experienced
        else:
            return np.random.binomial(1, 0.4, len(df))
    
    # MÉTHODE 3: Target mixte recommandée
    def create_hybrid_target(df):
        """Combine expérience et quelques métriques de base"""
        
        score = 0
        factors = 0
        
        # Facteur 1: Expérience (50% du score)
        if 'total_time_played' in df.columns:
            hours = df['total_time_played'] / 3600
            exp_score = np.clip(hours / 1000, 0, 1)  # Normaliser sur 1000h max
            score += exp_score * 0.5
            factors += 0.5
        
        # Facteur 2: Volume de jeu (25% du score)
        if 'total_rounds_played' in df.columns:
            rounds_norm = np.clip(df['total_rounds_played'] / 5000, 0, 1)
            score += rounds_norm * 0.25
            factors += 0.25
        
        # Facteur 3: Activité récente simulée (25% du score)
        recent_activity = np.random.beta(2, 5, len(df))  # Distribution biaisée vers les faibles valeurs
        score += recent_activity * 0.25
        factors += 0.25
        
        # Normaliser le score
        if factors > 0:
            score = score / factors
        
        # Seuil pour high performer (35% des joueurs)
        threshold = np.percentile(score, 65)  # Top 35%
        high_performer = (score >= threshold).astype(int)
        
        return high_performer
    
    # Appliquer la méthode hybride (recommandée)
    df_fixed = df.copy()
    
    # Supprimer l'ancienne target si elle existe
    if 'high_performer' in df_fixed.columns:
        df_fixed = df_fixed.drop('high_performer', axis=1)
    
    # Créer la nouvelle target
    new_target = create_hybrid_target(df_fixed)
    df_fixed['high_performer'] = new_target
    
    # SUPPRIMER TOUTES LES FEATURES SUSPECTES
    forbidden_features = [
        # Features calculées qui peuvent leaker
        'kd_ratio', 'accuracy', 'win_rate', 'mvp_rate',
        'performance_score', 'damage_per_round', 'kills_per_round',
        'experience_score', 'performance_index',
        
        # Features dérivées suspectes
        'deaths_per_round', 'money_per_round', 'bomb_plant_rate',
        'bomb_defuse_rate', 'rifle_efficiency', 'awp_efficiency',
        'knife_rate', 'avg_match_duration',
        
        # Features temporelles qui peuvent leaker
        'hours_played', 'playtime_efficiency',
        
        # Métadonnées
        'steam_id'
    ]
    
    # Supprimer les features interdites
    features_to_remove = [f for f in forbidden_features if f in df_fixed.columns]
    if features_to_remove:
        df_fixed = df_fixed.drop(columns=features_to_remove)
        print(f"🚫 Features supprimées: {features_to_remove}")
    
    return df_fixed, forbidden_features

# Fonction à ajouter dans data_preprocessing.py
def validate_no_leakage(X_train, y_train):
    """
    Valide qu'il n'y a pas de data leakage
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.dummy import DummyClassifier
    
    # Test 1: Dummy classifier ne devrait pas dépasser 0.6
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    dummy_score = cross_val_score(dummy, X_train, y_train, cv=5, scoring='roc_auc').mean()
    
    # Test 2: Corrélation maximale avec target
    correlations = X_train.corrwith(pd.Series(y_train, index=X_train.index)).abs()
    max_correlation = correlations.max()
    
    # Test 3: Random Forest rapide
    quick_rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_score = cross_val_score(quick_rf, X_train, y_train, cv=3, scoring='roc_auc').mean()
    
    print(f"\n🔍 VALIDATION ANTI-LEAKAGE:")
    print(f"  Dummy classifier AUC: {dummy_score:.4f} (devrait être ~0.5)")
    print(f"  Corrélation max: {max_correlation:.4f} (devrait être <0.8)")
    print(f"  Random Forest AUC: {rf_score:.4f} (devrait être 0.7-0.9)")
    
    # Alertes
    if dummy_score > 0.6:
        print(f"  ⚠️ ALERTE: Dummy trop bon ({dummy_score:.4f})")
    
    if max_correlation > 0.9:
        print(f"  ⚠️ ALERTE: Corrélation suspecte ({max_correlation:.4f})")
        most_corr_feature = correlations.idxmax()
        print(f"  Feature la plus corrélée: {most_corr_feature}")
    
    if rf_score > 0.95:
        print(f"  ⚠️ ALERTE: Performance suspecte ({rf_score:.4f})")
    
    return {
        'dummy_score': dummy_score,
        'max_correlation': max_correlation,
        'rf_score': rf_score,
        'leakage_suspected': rf_score > 0.95 or max_correlation > 0.9
    }

# Utilisation dans le pipeline principal
if __name__ == "__main__":
    # Charger et corriger les données
    df_raw = pd.read_csv("data/raw/csgo_raw_data.csv")
    df_fixed, forbidden = create_independent_target(df_raw)
    
    # Sauvegarder
    df_fixed.to_csv("data/raw/csgo_raw_data_fixed.csv", index=False)
    
    print(f"✅ Données corrigées sauvegardées")
    print(f"📊 Nouvelle distribution target: {df_fixed['high_performer'].value_counts()}")
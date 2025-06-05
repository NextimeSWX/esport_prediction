"""
Correction du data leakage - Nouvelle variable cible
École89 - 2025
"""

import pandas as pd
import numpy as np

def create_independent_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée une variable cible INDÉPENDANTE des features utilisées pour la prédiction
    
    Strategy: Utiliser SEULEMENT des variables qui ne seront PAS dans les features
    """
    df_fixed = df.copy()
    
    # ===================================================================
    # NOUVELLE APPROCHE: Variable cible basée sur l'EXPÉRIENCE/TEMPS
    # ===================================================================
    
    # Variables INTERDITES dans les features (car utilisées pour la target)
    forbidden_features = [
        'hours_played',
        'total_time_played', 
        'total_matches_played',
        'account_age_days',  # Si disponible
    ]
    
    # Calcul du score d'expérience/engagement (INDÉPENDANT des performances)
    experience_score = 0
    
    # 1. Temps de jeu (40% du score)
    if 'hours_played' in df.columns:
        hours_norm = np.clip(df['hours_played'] / 3000, 0, 1)  # Cap à 3000h
    else:
        hours_norm = np.clip(df['total_time_played'] / (3000 * 3600), 0, 1)
    
    experience_score += hours_norm * 0.4
    
    # 2. Nombre de matches (30% du score) 
    matches_norm = np.clip(df['total_matches_played'] / 2000, 0, 1)  # Cap à 2000 matches
    experience_score += matches_norm * 0.3
    
    # 3. Ancienneté compte (20% du score) - Si disponible
    if 'account_age_days' in df.columns:
        age_norm = np.clip(df['account_age_days'] / (5 * 365), 0, 1)  # Cap à 5 ans
        experience_score += age_norm * 0.2
    else:
        # Estimation via temps total / intensité
        estimated_intensity = df['total_time_played'] / (df['total_matches_played'] * 60 + 1)
        intensity_norm = np.clip(estimated_intensity / 60, 0, 1)  # Cap à 60min/match
        experience_score += intensity_norm * 0.2
    
    # 4. Diversité des armes (10% du score)
    weapon_cols = [col for col in df.columns if 'total_kills_' in col and col != 'total_kills']
    if len(weapon_cols) >= 3:
        weapon_diversity = (df[weapon_cols] > 0).sum(axis=1) / len(weapon_cols)
        experience_score += weapon_diversity * 0.1
    
    # Classification binaire: Top 40% des joueurs EXPÉRIMENTÉS
    threshold = np.percentile(experience_score, 60)
    df_fixed['high_performer'] = (experience_score > threshold).astype(int)
    
    # Stocker le score pour analyse
    df_fixed['experience_score'] = experience_score
    
    print(f"📊 Nouvelle variable cible créée:")
    print(f"   High performers: {df_fixed['high_performer'].sum()} ({df_fixed['high_performer'].mean():.1%})")
    print(f"   Basée sur l'EXPÉRIENCE, pas la performance")
    
    return df_fixed, forbidden_features

def create_performance_based_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alternative: Variable cible basée sur des RATIOS COMPLEXES
    (Plus difficile à "deviner" par le modèle)
    """
    df_fixed = df.copy()
    
    # Variables que le modèle ne verra PAS
    forbidden_features = [
        'total_wins',
        'total_mvps', 
        'total_planted_bombs',
        'total_defused_bombs'
    ]
    
    # Score complexe avec variables interdites
    complex_score = 0
    
    # 1. Taux de victoire (mais sera retiré des features)
    win_rate = df['total_wins'] / (df['total_matches_played'] + 1)
    complex_score += win_rate * 0.4
    
    # 2. Taux MVP (mais sera retiré des features)  
    mvp_rate = df['total_mvps'] / (df['total_matches_played'] + 1)
    complex_score += mvp_rate * 0.3
    
    # 3. Efficacité objective (bombes)
    bomb_efficiency = (df['total_planted_bombs'] + df['total_defused_bombs']) / (df['total_rounds_played'] / 2 + 1)
    complex_score += np.clip(bomb_efficiency, 0, 1) * 0.3
    
    # Classification
    threshold = np.percentile(complex_score, 65)  # Top 35% 
    df_fixed['high_performer'] = (complex_score > threshold).astype(int)
    
    print(f"📊 Variable cible performance complexe:")
    print(f"   High performers: {df_fixed['high_performer'].sum()} ({df_fixed['high_performer'].mean():.1%})")
    
    return df_fixed, forbidden_features

def remove_forbidden_features(df: pd.DataFrame, forbidden_features: list) -> pd.DataFrame:
    """Retire les features interdites pour éviter le data leakage"""
    
    features_to_remove = []
    for forbidden in forbidden_features:
        if forbidden in df.columns:
            features_to_remove.append(forbidden)
    
    df_clean = df.drop(columns=features_to_remove)
    
    print(f"🚫 Features supprimées pour éviter data leakage:")
    for feature in features_to_remove:
        print(f"   - {feature}")
    
    return df_clean

def fix_data_leakage_complete():
    """
    Fonction principale pour corriger complètement le data leakage
    """
    print("🔧 CORRECTION DU DATA LEAKAGE")
    print("="*50)
    
    # 1. Charger les données brutes
    df_raw = pd.read_csv('data/raw/csgo_raw_data.csv')
    print(f"📁 Données chargées: {len(df_raw)} joueurs")
    
    # 2. Choisir la stratégie de target
    print("\n🎯 Stratégies disponibles:")
    print("1. Experience-based (recommandé)")
    print("2. Performance complexe")
    
    strategy = input("Choisissez (1 ou 2): ").strip()
    
    if strategy == "1":
        df_fixed, forbidden = create_independent_target(df_raw)
        print("✅ Stratégie: Experience-based")
    else:
        df_fixed, forbidden = create_performance_based_target(df_raw)  
        print("✅ Stratégie: Performance complexe")
    
    # 3. Sauvegarder les données corrigées
    output_path = 'data/raw/csgo_raw_data_fixed.csv'
    df_fixed.to_csv(output_path, index=False)
    
    print(f"\n💾 Données corrigées sauvegardées: {output_path}")
    print(f"🚫 Features à éviter dans le ML: {forbidden}")
    
    # 4. Instructions pour la suite
    print(f"\n📋 INSTRUCTIONS POUR LA SUITE:")
    print(f"1. Utilisez {output_path} comme fichier source")
    print(f"2. Modifiez data_preprocessing.py pour exclure: {forbidden}")
    print(f"3. Relancez le pipeline ML")
    
    return df_fixed, forbidden

# Script d'exécution
if __name__ == "__main__":
    df_corrected, forbidden_features = fix_data_leakage_complete()
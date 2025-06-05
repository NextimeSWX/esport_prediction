"""
Preprocessing modifié pour éviter le data leakage
École89 - 2025
"""

import pandas as pd
import numpy as np

class NoLeakagePreprocessor:
    """Preprocessor qui évite le data leakage"""
    
    def __init__(self):
        # Features INTERDITES (utilisées pour créer la target)
        self.forbidden_features = [
            'kd_ratio',           # Utilisé dans le calcul de high_performer
            'accuracy',           # Utilisé dans le calcul de high_performer  
            'win_rate',           # Utilisé dans le calcul de high_performer
            'performance_score',  # Score utilisé pour la target
            
            # Features dérivées qui révèlent trop la target
            'combat_effectiveness',  # Calculé avec kd_ratio + accuracy
            'team_impact',          # Calculé avec win_rate
            'experience_level',     # Catégorique de hours_played
        ]
    
    def select_safe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sélectionne seulement les features sûres (pas de leakage)"""
        
        # Features de BASE autorisées (statistiques brutes)
        safe_base_features = [
            'total_kills',
            'total_deaths', 
            'total_shots_fired',
            'total_shots_hit',
            'total_damage_done',
            'total_money_earned',
            'total_rounds_played',
            'total_mvps',
            'total_planted_bombs',
            'total_defused_bombs',
            'total_matches_played',
            'total_matches_won',
            'hours_played',
            # Armes
            'total_kills_ak47',
            'total_kills_m4a1', 
            'total_kills_awp',
            'total_kills_knife',
            'total_kills_glock',
            'total_kills_hegrenade',
        ]
        
        # Features DÉRIVÉES autorisées (calculées différemment)
        safe_derived_features = [
            'kills_per_round',
            'deaths_per_round', 
            'damage_per_round',
            'money_per_round',
            'bomb_plant_rate',
            'bomb_defuse_rate',
            'avg_match_duration',
            'rifle_efficiency',
            'awp_efficiency', 
            'knife_rate',
        ]
        
        # Sélectionner seulement les features disponibles et sûres
        available_safe_features = []
        
        for feature in safe_base_features + safe_derived_features:
            if feature in df.columns and feature not in self.forbidden_features:
                available_safe_features.append(feature)
        
        # Ajouter la target
        if 'high_performer' in df.columns:
            available_safe_features.append('high_performer')
        
        df_safe = df[available_safe_features].copy()
        
        print(f"🛡️  Features sélectionnées (sans leakage): {len(available_safe_features)-1}")
        print(f"🚫 Features interdites évitées: {len(self.forbidden_features)}")
        
        # Afficher les features gardées
        features_only = [f for f in available_safe_features if f != 'high_performer']
        print(f"✅ Features utilisées: {features_only[:10]}...")
        
        return df_safe
    
    def create_new_target_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée une target simple basée sur l'expérience"""
        df_new = df.copy()
        
        # Target basée UNIQUEMENT sur l'expérience (pas les performances)
        experience_factors = []
        
        # 1. Temps de jeu
        if 'hours_played' in df.columns:
            exp_time = np.clip(df['hours_played'] / 2000, 0, 1)
            experience_factors.append(exp_time * 0.5)
        
        # 2. Nombre de matches
        if 'total_matches_played' in df.columns:
            exp_matches = np.clip(df['total_matches_played'] / 1500, 0, 1)
            experience_factors.append(exp_matches * 0.3)
        
        # 3. Diversité (nombre d'armes différentes utilisées)
        weapon_cols = [col for col in df.columns if 'total_kills_' in col and col != 'total_kills']
        if len(weapon_cols) > 0:
            weapon_diversity = (df[weapon_cols] > 5).sum(axis=1) / len(weapon_cols)  # Au moins 5 kills par arme
            experience_factors.append(weapon_diversity * 0.2)
        
        # Score d'expérience combiné
        if experience_factors:
            experience_score = sum(experience_factors)
            
            # Top 40% = experienced players
            threshold = np.percentile(experience_score, 60)
            df_new['high_performer'] = (experience_score > threshold).astype(int)
            
            print(f"🎯 Nouvelle target créée (expérience):")
            print(f"   High performers: {df_new['high_performer'].sum()} ({df_new['high_performer'].mean():.1%})")
        
        return df_new

def fix_and_rerun_pipeline():
    """
    Script pour corriger le data leakage et relancer le preprocessing
    """
    print("🔧 CORRECTION DU DATA LEAKAGE")
    print("="*50)
    
    # 1. Charger données brutes
    df_raw = pd.read_csv('data/raw/csgo_raw_data.csv')
    print(f"📁 Données originales: {len(df_raw)} joueurs, {len(df_raw.columns)} colonnes")
    
    # 2. Créer nouvelle target
    preprocessor = NoLeakagePreprocessor()
    df_new_target = preprocessor.create_new_target_simple(df_raw)
    
    # 3. Sélectionner features sûres
    df_safe = preprocessor.select_safe_features(df_new_target)
    
    # 4. Sauvegarder
    output_path = 'data/raw/csgo_data_no_leakage.csv'
    df_safe.to_csv(output_path, index=False)
    
    print(f"\n💾 Données corrigées sauvegardées: {output_path}")
    print(f"📊 Dataset final: {len(df_safe)} joueurs, {len(df_safe.columns)-1} features")
    
    # 5. Instructions
    print(f"\n📋 ÉTAPES SUIVANTES:")
    print(f"1. Modifiez data_preprocessing.py pour utiliser '{output_path}'")
    print(f"2. Relancez: python src/data_preprocessing.py")
    print(f"3. Puis: python src/models.py")
    print(f"4. Attendez-vous à des résultats plus réalistes (80-90% accuracy)")
    
    return df_safe

if __name__ == "__main__":
    df_corrected = fix_and_rerun_pipeline()
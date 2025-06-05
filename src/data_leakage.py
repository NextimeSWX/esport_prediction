"""
Module de preprocessing des données CS:GO
École89 - 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.config import (
        RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_STATE, 
        TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE, LOGGER
    )
except ImportError:
    # Configuration de base si config.py incomplet
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DATA_DIR = Path("data/processed")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    RANDOM_STATE = 42
    TRAIN_SIZE = 0.7
    VALIDATION_SIZE = 0.15
    TEST_SIZE = 0.15
    
    import logging
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)

class CSGODataPreprocessor:
    """Classe pour le preprocessing des données CS:GO"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        
        # Features interdites (data leakage)
        self.forbidden_features = [
            'kd_ratio', 'accuracy', 'win_rate', 'performance_score',
            'combat_effectiveness', 'team_impact', 'experience_level',
            'experience_score'  # Score utilisé pour créer la target
        ]
        
    def load_raw_data(self) -> pd.DataFrame:
        """Charge les données brutes depuis le fichier CSV"""
        
        # UTILISER LE FICHIER CORRIGÉ (sans data leakage)
        filename = "csgo_raw_data_fixed.csv"
        
        filepath = self.raw_data_dir / filename
        
        if not filepath.exists():
            # Fallback vers le fichier original si le fixé n'existe pas
            filename = "csgo_raw_data.csv"
            filepath = self.raw_data_dir / filename
            LOGGER.warning(f"⚠️ Fichier fixed non trouvé, utilisation de {filename}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Fichier de données non trouvé: {filepath}")
        
        df = pd.read_csv(filepath)
        LOGGER.info(f"📁 Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
        
        return df
    
    def remove_leakage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les features qui causent le data leakage"""
        
        # Features à supprimer pour éviter le data leakage
        features_to_remove = [f for f in self.forbidden_features if f in df.columns]
        
        if features_to_remove:
            df_clean = df.drop(columns=features_to_remove)
            LOGGER.info(f"🚫 {len(features_to_remove)} features supprimées pour éviter data leakage:")
            for feature in features_to_remove:
                LOGGER.info(f"   - {feature}")
        else:
            df_clean = df.copy()
            LOGGER.info("✅ Aucune feature de leakage détectée")
        
        return df_clean
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie et valide les données"""
        LOGGER.info("🧹 Début du nettoyage des données...")
        
        # 1. Supprimer les features de data leakage AVANT tout autre traitement
        df_no_leakage = self.remove_leakage_features(df)
        
        # 2. Filtrage par minimum d'heures
        min_hours = 50
        if 'total_time_played' in df_no_leakage.columns:
            min_seconds = min_hours * 3600
            initial_count = len(df_no_leakage)
            df_filtered = df_no_leakage[df_no_leakage['total_time_played'] >= min_seconds]
            removed_count = initial_count - len(df_filtered)
            LOGGER.info(f"  Filtrage: minimum {min_hours}h de jeu ({removed_count} joueurs supprimés)")
        else:
            df_filtered = df_no_leakage
            LOGGER.info("  Filtrage: pas de colonne temps disponible")
        
        # 3. Gestion des outliers
        df_clean = self._remove_outliers(df_filtered)
        
        # 4. Corrections de cohérence
        df_consistent = self._fix_data_consistency(df_clean)
        
        LOGGER.info("✅ Nettoyage terminé:")
        LOGGER.info(f"  Lignes supprimées: {len(df) - len(df_consistent)} ({(len(df) - len(df_consistent))/len(df)*100:.1f}%)")
        LOGGER.info(f"  Lignes restantes: {len(df_consistent)}")
        
        return df_consistent
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supprime les outliers avec la méthode IQR"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['high_performer']]
        
        df_clean = df.copy()
        total_outliers = 0
        
        for col in numeric_cols:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    # Plafonner les outliers au lieu de les supprimer
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                    total_outliers += outliers_count
                    
                    if outliers_count > 5:  # Log seulement si beaucoup d'outliers
                        LOGGER.info(f"  Outliers plafonnés pour {col}: {outliers_count}")
        
        if total_outliers > 0:
            LOGGER.info(f"  Total outliers traités: {total_outliers}")
        
        return df_clean
    
    def _fix_data_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrige les incohérences dans les données"""
        
        df_fixed = df.copy()
        corrections = 0
        
        # 1. Vérifier que deaths >= 1
        if 'total_deaths' in df_fixed.columns:
            mask = df_fixed['total_deaths'] < 1
            df_fixed.loc[mask, 'total_deaths'] = 1
            corrections += mask.sum()
        
        # 2. Vérifier cohérence kills/armes
        weapon_cols = [col for col in df_fixed.columns if 'total_kills_' in col and col != 'total_kills']
        if weapon_cols and 'total_kills' in df_fixed.columns:
            weapon_kills_sum = df_fixed[weapon_cols].sum(axis=1)
            # Si la somme des kills par arme dépasse le total, ajuster
            mask = weapon_kills_sum > df_fixed['total_kills'] * 1.1  # 10% de tolérance
            if mask.any():
                for idx in df_fixed[mask].index:
                    total = df_fixed.loc[idx, 'total_kills']
                    weapon_sum = df_fixed.loc[idx, weapon_cols].sum()
                    if weapon_sum > 0:
                        ratio = total / weapon_sum
                        df_fixed.loc[idx, weapon_cols] = (df_fixed.loc[idx, weapon_cols] * ratio).astype(int)
                corrections += mask.sum()
        
        # 3. Vérifier cohérence matches won/played
        if 'total_matches_won' in df_fixed.columns and 'total_matches_played' in df_fixed.columns:
            mask = df_fixed['total_matches_won'] > df_fixed['total_matches_played']
            df_fixed.loc[mask, 'total_matches_won'] = df_fixed.loc[mask, 'total_matches_played']
            corrections += mask.sum()
        
        if corrections > 0:
            LOGGER.info(f"  Corrections de cohérence appliquées: {corrections}")
        
        return df_fixed
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gère les valeurs manquantes"""
        
        missing_count = df.isnull().sum().sum()
        
        if missing_count > 0:
            LOGGER.info(f"⚠️ {missing_count} valeurs manquantes détectées")
            
            # Remplir avec la médiane pour les variables numériques
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_filled = df.copy()
            
            for col in numeric_cols:
                if df_filled[col].isnull().any():
                    median_val = df_filled[col].median()
                    df_filled[col] = df_filled[col].fillna(median_val)
                    LOGGER.info(f"  {col}: rempli avec médiane ({median_val:.2f})")
            
            # Vérification finale
            final_missing = df_filled.isnull().sum().sum()
            LOGGER.info(f"✅ Valeurs manquantes après traitement: {final_missing}")
            
            return df_filled
        else:
            LOGGER.info("  Aucune valeur manquante détectée")
            return df
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée des features dérivées SÛRES (pas de leakage)"""
        LOGGER.info("🔧 Création des features dérivées...")
        
        df_features = df.copy()
        new_features = 0
        
        # 1. Features par round (intensité de jeu)
        if 'total_rounds_played' in df_features.columns:
            if 'total_kills' in df_features.columns:
                df_features['kills_per_round'] = df_features['total_kills'] / (df_features['total_rounds_played'] + 1)
                new_features += 1
            
            if 'total_deaths' in df_features.columns:
                df_features['deaths_per_round'] = df_features['total_deaths'] / (df_features['total_rounds_played'] + 1)
                new_features += 1
            
            if 'total_damage_done' in df_features.columns:
                df_features['damage_per_round'] = df_features['total_damage_done'] / (df_features['total_rounds_played'] + 1)
                new_features += 1
            
            if 'total_money_earned' in df_features.columns:
                df_features['money_per_round'] = df_features['total_money_earned'] / (df_features['total_rounds_played'] + 1)
                new_features += 1
        
        # 2. Features d'objectifs (bombes)
        if 'total_planted_bombs' in df_features.columns and 'total_rounds_played' in df_features.columns:
            df_features['bomb_plant_rate'] = df_features['total_planted_bombs'] / (df_features['total_rounds_played'] / 2 + 1)
            new_features += 1
        
        if 'total_defused_bombs' in df_features.columns and 'total_rounds_played' in df_features.columns:
            df_features['bomb_defuse_rate'] = df_features['total_defused_bombs'] / (df_features['total_rounds_played'] / 2 + 1)
            new_features += 1
        
        # 3. Features d'armes (efficacité relative)
        weapon_cols = [col for col in df_features.columns if 'total_kills_' in col and col != 'total_kills']
        if weapon_cols and 'total_kills' in df_features.columns:
            # Efficacité des rifles
            rifle_cols = [col for col in weapon_cols if 'ak47' in col or 'm4a1' in col]
            if rifle_cols:
                df_features['rifle_kills'] = df_features[rifle_cols].sum(axis=1)
                df_features['rifle_efficiency'] = df_features['rifle_kills'] / (df_features['total_kills'] + 1)
                new_features += 2
            
            # Efficacité AWP
            awp_cols = [col for col in weapon_cols if 'awp' in col]
            if awp_cols:
                df_features['awp_efficiency'] = df_features[awp_cols].sum(axis=1) / (df_features['total_kills'] + 1)
                new_features += 1
            
            # Taux de knife kills
            knife_cols = [col for col in weapon_cols if 'knife' in col]
            if knife_cols:
                df_features['knife_rate'] = df_features[knife_cols].sum(axis=1) / (df_features['total_kills'] + 1)
                new_features += 1
        
        # 4. Features de match (durée moyenne)
        if 'total_time_played' in df_features.columns and 'total_matches_played' in df_features.columns:
            df_features['avg_match_duration'] = df_features['total_time_played'] / (df_features['total_matches_played'] * 60 + 1)
            new_features += 1
        
        # 5. Features de MVP (impact équipe)
        if 'total_mvps' in df_features.columns and 'total_matches_played' in df_features.columns:
            df_features['mvp_rate'] = df_features['total_mvps'] / (df_features['total_matches_played'] + 1)
            new_features += 1
        
        LOGGER.info(f"✅ {new_features} nouvelles features créées")
        
        return df_features
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne les features pour le ML"""
    
    # Features à exclure (STRICTEMENT pour éviter data leakage)
    exclude_features = [
        'steam_id',  # Identifiant
        'total_time_played',  # Utilisé pour créer la target
        'total_matches_played',  # Utilisé pour créer la target
        'hours_played',  # Utilisé pour créer la target (si présent)
        
        # NOUVELLES EXCLUSIONS pour éviter le calcul des ratios
        'total_deaths',      # Permet de calculer KD ratio
        'total_shots_fired', # Permet de calculer accuracy
        'total_shots_hit',   # Permet de calculer accuracy
        'total_wins',        # Permet de calculer win rate
        'total_matches_won', # Permet de calculer win rate
        'total_mvps',        # Très corrélé à la performance
        
    ] + self.forbidden_features
        
        # Garder seulement les features numériques + target
        available_features = []
        
        for col in df.columns:
            if col == 'high_performer':  # Target
                available_features.append(col)
            elif col not in exclude_features:
                if df[col].dtype in ['int64', 'float64']:  # Numérique
                    available_features.append(col)
        
        df_selected = df[available_features].copy()
        
        feature_count = len(available_features) - 1  # -1 pour la target
        LOGGER.info(f"🎯 Features sélectionnées: {feature_count}")
        
        # Afficher quelques features gardées
        features_only = [f for f in available_features if f != 'high_performer']
        LOGGER.info(f"✅ Exemples features: {features_only[:10]}")
        
        return df_selected
    
    def split_data(self, df: pd.DataFrame):
        """Divise les données en train/validation/test"""
        
        # Séparer features et target
        X = df.drop('high_performer', axis=1)
        y = df['high_performer']
        
        # Division en 3 sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Ajuster la taille de validation
        val_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=RANDOM_STATE, stratify=y_temp
        )
        
        LOGGER.info("📊 Division des données:")
        LOGGER.info(f"  Train: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
        LOGGER.info(f"  Validation: {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
        LOGGER.info(f"  Test: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Normalise les features avec StandardScaler"""
        
        # Fit sur train seulement
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform sur val et test
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        LOGGER.info("🔄 Features normalisées avec standard scaler")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def save_processed_data(self, **datasets):
        """Sauvegarde les données preprocessées"""
        
        self.processed_data_dir.mkdir(exist_ok=True)
        
        for name, data in datasets.items():
            if data is not None:
                filepath = self.processed_data_dir / f"{name}.csv"
                data.to_csv(filepath, index=False)
                LOGGER.info(f"💾 {name} sauvegardé: {filepath}")
    
    def get_preprocessing_summary(self, df_original, df_final):
        """Retourne un résumé du preprocessing"""
        
        summary = {
            'original_samples': len(df_original),
            'final_samples': len(df_final),
            'samples_removed': len(df_original) - len(df_final),
            'removal_percentage': (len(df_original) - len(df_final)) / len(df_original) * 100,
            'original_features': len(df_original.columns),
            'final_features': len(df_final.columns) - 1,  # -1 pour la target
            'missing_values': df_final.isnull().sum().sum()
        }
        
        return summary

def main():
    """Fonction principale pour le preprocessing complet"""
    
    print("🧹 " + "="*50)
    print("   PREPROCESSING DES DONNÉES CS:GO")
    print("   École89 - 2025")
    print("="*54)
    
    # Initialisation
    preprocessor = CSGODataPreprocessor()
    
    try:
        # 1. Charger les données brutes
        df_raw = preprocessor.load_raw_data()
        
        # 2. Nettoyage
        df_clean = preprocessor.clean_data(df_raw)
        
        # 3. Gestion des valeurs manquantes
        df_imputed = preprocessor.handle_missing_values(df_clean)
        
        # 4. Création de features dérivées
        df_features = preprocessor.create_derived_features(df_imputed)
        
        # 5. Sélection des features
        df_selected = preprocessor.select_features(df_features)
        
        # 6. Division des données
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_selected)
        
        # 7. Normalisation
        X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_features(
            X_train, X_val, X_test
        )
        
        # 8. Sauvegarde
        preprocessor.save_processed_data(
            X_train=X_train_scaled,
            X_val=X_val_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test
        )
        
        # 9. Résumé
        summary = preprocessor.get_preprocessing_summary(df_raw, df_selected)
        
        LOGGER.info(f"\n📋 RÉSUMÉ DU PREPROCESSING:")
        LOGGER.info(f"  Lignes: {summary['original_samples']} → {summary['final_samples']}")
        LOGGER.info(f"  Features: {summary['original_features']} → {summary['final_features']}")
        LOGGER.info(f"  Qualité: {summary['missing_values']} valeurs manquantes")
        
        print(f"\n✅ PREPROCESSING TERMINÉ AVEC SUCCÈS!")
        print(f"📊 Dataset final: {summary['final_samples']} joueurs, {summary['final_features']} features")
        print(f"🎯 Variable cible: {y_train.sum() + y_val.sum() + y_test.sum()} high performers")
        print(f"💾 Données sauvegardées dans: {preprocessor.processed_data_dir}")
        print(f"🚀 Prochaine étape: python src/models.py")
        
    except Exception as e:
        LOGGER.error(f"❌ Erreur pendant le preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
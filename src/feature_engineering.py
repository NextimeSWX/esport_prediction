"""
Module de feature engineering avancé pour CS:GO
École89 - 2025
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import FEATURES_DATA_DIR, PROCESSED_DATA_DIR, LOGGER

class CSGOFeatureEngineer:
    """Classe pour le feature engineering avancé des données CS:GO"""
    
    def __init__(self):
        self.feature_selector = None
        self.pca = None
        self.poly_features = None
        self.selected_features = []
    
    def load_processed_data(self):
        """Charge les données préprocessées"""
        try:
            X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
            X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val.csv")
            X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
            y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").iloc[:, 0]
            y_val = pd.read_csv(PROCESSED_DATA_DIR / "y_val.csv").iloc[:, 0]
            y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").iloc[:, 0]
            
            LOGGER.info("📁 Données préprocessées chargées avec succès")
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except FileNotFoundError as e:
            LOGGER.error(f"❌ Fichier non trouvé: {e}")
            LOGGER.info("💡 Lancez d'abord data_preprocessing.py")
            raise
    
    def create_advanced_features(self, X_train, X_val, X_test):
        """
        Crée des features avancées à partir des features existantes
        
        Args:
            X_train, X_val, X_test: DataFrames des features
            
        Returns:
            DataFrames avec features avancées
        """
        LOGGER.info("🚀 Création de features avancées...")
        
        def add_advanced_features(df):
            df_advanced = df.copy()
            
            # 1. Features temporelles (si temps de jeu disponible)
            if 'hours_played' in df.columns:
                df_advanced['experience_tier'] = pd.cut(
                    df['hours_played'],
                    bins=[0, 50, 200, 500, 1000, float('inf')],
                    labels=[1, 2, 3, 4, 5]
                ).astype(float)
                
                df_advanced['playtime_efficiency'] = df['total_kills'] / (df['hours_played'] + 1)
            
            # 2. Features de performance relative
            if 'kd_ratio' in df.columns and 'accuracy' in df.columns:
                df_advanced['performance_index'] = (
                    df['kd_ratio'] * 0.4 +
                    df['accuracy'] * 5 * 0.3 +  # Mise à l'échelle de l'accuracy
                    df.get('win_rate', 0.5) * 0.3
                )
                
                # Percentiles de performance
                df_advanced['kd_percentile'] = df['kd_ratio'].rank(pct=True)
                df_advanced['accuracy_percentile'] = df['accuracy'].rank(pct=True)
            
            # 3. Features d'équilibrage gameplay
            weapon_cols = [col for col in df.columns if 'total_kills_' in col and col != 'total_kills']
            if weapon_cols:
                # Diversité des armes (entropie)
                weapon_data = df[weapon_cols].values + 1e-8  # Éviter log(0)
                weapon_props = weapon_data / weapon_data.sum(axis=1, keepdims=True)
                df_advanced['weapon_diversity'] = -np.sum(
                    weapon_props * np.log2(weapon_props), axis=1
                )
                
                # Spécialisation (arme la plus utilisée)
                df_advanced['weapon_specialization'] = weapon_props.max(axis=1)
            
            # 4. Features d'efficacité contextuelle
            if 'total_rounds_played' in df.columns and 'total_damage_done' in df.columns:
                df_advanced['damage_efficiency'] = (
                    df['total_damage_done'] / (df['total_rounds_played'] + 1)
                )
                
                # Ratio damage/kill (plus bas = plus efficace)
                if 'total_kills' in df.columns:
                    df_advanced['damage_per_kill'] = (
                        df['total_damage_done'] / (df['total_kills'] + 1)
                    )
            
            # 5. Features de clutch/pressure situations
            if 'bomb_defuse_rate' in df.columns and 'bomb_plant_rate' in df.columns:
                df_advanced['clutch_factor'] = (
                    df['bomb_defuse_rate'] * 0.6 +  # Défuser = plus difficile
                    df['bomb_plant_rate'] * 0.4
                )
            
            # 6. Features d'interaction entre métriques
            if 'mvp_rate' in df.columns and 'win_rate' in df.columns:
                df_advanced['leadership_index'] = df['mvp_rate'] / (df['win_rate'] + 0.01)
                df_advanced['team_dependency'] = df['win_rate'] - df['mvp_rate']
            
            # 7. Features de régularité/consistance
            performance_features = ['kills_per_round', 'deaths_per_round', 'damage_per_round']
            available_perf_features = [f for f in performance_features if f in df.columns]
            
            if len(available_perf_features) >= 2:
                # Coefficient de variation (stabilité)
                for feature in available_perf_features:
                    mean_val = df[feature].mean()
                    std_val = df[feature].std()
                    df_advanced[f'{feature}_stability'] = 1 - (std_val / (mean_val + 1e-8))
            
            return df_advanced
        
        # Appliquer les transformations
        X_train_advanced = add_advanced_features(X_train)
        X_val_advanced = add_advanced_features(X_val)
        X_test_advanced = add_advanced_features(X_test)
        
        # Nettoyer les valeurs inf/nan
        for df in [X_train_advanced, X_val_advanced, X_test_advanced]:
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(df.median(), inplace=True)
        
        new_features = len(X_train_advanced.columns) - len(X_train.columns)
        LOGGER.info(f"✅ {new_features} features avancées créées")
        
        return X_train_advanced, X_val_advanced, X_test_advanced
    
    def create_polynomial_features(self, X_train, X_val, X_test, degree=2, max_features=50):
        """
        Crée des features polynomiales (interactions)
        
        Args:
            X_train, X_val, X_test: DataFrames des features
            degree: Degré polynomial
            max_features: Nombre max de features à garder
            
        Returns:
            DataFrames avec features polynomiales sélectionnées
        """
        LOGGER.info(f"🔢 Création de features polynomiales (degré {degree})...")
        
        # Sélectionner seulement les features les plus importantes pour éviter explosion
        important_features = [
            'kd_ratio', 'accuracy', 'win_rate', 'kills_per_round', 
            'damage_per_round', 'mvp_rate', 'performance_index'
        ]
        available_features = [f for f in important_features if f in X_train.columns]
        
        if len(available_features) < 3:
            # Prendre les premières features si les importantes ne sont pas là
            available_features = X_train.columns[:min(8, len(X_train.columns))].tolist()
        
        X_train_subset = X_train[available_features]
        X_val_subset = X_val[available_features]
        X_test_subset = X_test[available_features]
        
        # Créer features polynomiales
        self.poly_features = PolynomialFeatures(
            degree=degree, 
            interaction_only=True,  # Seulement interactions, pas puissances
            include_bias=False
        )
        
        X_train_poly = self.poly_features.fit_transform(X_train_subset)
        X_val_poly = self.poly_features.transform(X_val_subset)
        X_test_poly = self.poly_features.transform(X_test_subset)
        
        # Noms des features polynomiales
        poly_feature_names = self.poly_features.get_feature_names_out(available_features)
        
        # Convertir en DataFrame
        X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_feature_names, index=X_train.index)
        X_val_poly_df = pd.DataFrame(X_val_poly, columns=poly_feature_names, index=X_val.index)
        X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_feature_names, index=X_test.index)
        
        # Combiner avec features originales
        X_train_combined = pd.concat([X_train, X_train_poly_df], axis=1)
        X_val_combined = pd.concat([X_val, X_val_poly_df], axis=1)
        X_test_combined = pd.concat([X_test, X_test_poly_df], axis=1)
        
        LOGGER.info(f"✅ {len(poly_feature_names)} features polynomiales créées")
        
        return X_train_combined, X_val_combined, X_test_combined
    
    def select_best_features(self, X_train, X_val, X_test, y_train, method='f_classif', k=50):
        """
        Sélectionne les meilleures features selon un critère statistique
        
        Args:
            X_train, X_val, X_test: DataFrames des features
            y_train: Variable cible
            method: Méthode de sélection ('f_classif' ou 'mutual_info')
            k: Nombre de features à garder
            
        Returns:
            DataFrames avec features sélectionnées
        """
        LOGGER.info(f"🎯 Sélection des {k} meilleures features ({method})...")
        
        # Choisir la fonction de score
        score_func = f_classif if method == 'f_classif' else mutual_info_classif
        
        # Initialiser le sélecteur
        self.feature_selector = SelectKBest(score_func=score_func, k=min(k, X_train.shape[1]))
        
        # Fit sur train
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_val_selected = self.feature_selector.transform(X_val)
        X_test_selected = self.feature_selector.transform(X_test)
        
        # Récupérer les noms des features sélectionnées
        feature_mask = self.feature_selector.get_support()
        self.selected_features = X_train.columns[feature_mask].tolist()
        
        # Convertir en DataFrame
        X_train_selected_df = pd.DataFrame(
            X_train_selected, 
            columns=self.selected_features,
            index=X_train.index
        )
        X_val_selected_df = pd.DataFrame(
            X_val_selected, 
            columns=self.selected_features,
            index=X_val.index
        )
        X_test_selected_df = pd.DataFrame(
            X_test_selected, 
            columns=self.selected_features,
            index=X_test.index
        )
        
        # Afficher les features sélectionnées
        feature_scores = self.feature_selector.scores_[feature_mask]
        feature_ranking = pd.DataFrame({
            'feature': self.selected_features,
            'score': feature_scores
        }).sort_values('score', ascending=False)
        
        LOGGER.info("🏆 Top 10 features sélectionnées:")
        for i, row in feature_ranking.head(10).iterrows():
            LOGGER.info(f"  {row['feature']}: {row['score']:.3f}")
        
        return X_train_selected_df, X_val_selected_df, X_test_selected_df
    
    def apply_pca(self, X_train, X_val, X_test, n_components=0.95):
        """
        Applique une PCA pour réduction de dimensionnalité
        
        Args:
            X_train, X_val, X_test: DataFrames des features
            n_components: Nombre de composantes (ou variance à conserver)
            
        Returns:
            DataFrames transformées par PCA
        """
        LOGGER.info(f"🔄 Application PCA (variance conservée: {n_components})...")
        
        # Initialiser PCA
        self.pca = PCA(n_components=n_components, random_state=42)
        
        # Fit et transform
        X_train_pca = self.pca.fit_transform(X_train)
        X_val_pca = self.pca.transform(X_val)
        X_test_pca = self.pca.transform(X_test)
        
        # Noms des composantes
        n_components_actual = X_train_pca.shape[1]
        component_names = [f'PC{i+1}' for i in range(n_components_actual)]
        
        # Convertir en DataFrame
        X_train_pca_df = pd.DataFrame(X_train_pca, columns=component_names, index=X_train.index)
        X_val_pca_df = pd.DataFrame(X_val_pca, columns=component_names, index=X_val.index)
        X_test_pca_df = pd.DataFrame(X_test_pca, columns=component_names, index=X_test.index)
        
        # Informations sur la variance expliquée
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        LOGGER.info(f"✅ PCA terminée:")
        LOGGER.info(f"  Composantes: {n_components_actual}")
        LOGGER.info(f"  Variance expliquée: {cumulative_variance[-1]:.3f}")
        LOGGER.info(f"  Top 3 composantes: {explained_variance[:3]}")
        
        return X_train_pca_df, X_val_pca_df, X_test_pca_df
    
    def save_engineered_features(self, **datasets):
        """Sauvegarde les features engineered"""
        FEATURES_DATA_DIR.mkdir(exist_ok=True)
        
        for name, data in datasets.items():
            if data is not None:
                filepath = FEATURES_DATA_DIR / f"{name}.csv"
                data.to_csv(filepath, index=False)
                LOGGER.info(f"💾 {name} sauvegardé: {filepath}")
    
    def get_feature_importance_summary(self):
        """Retourne un résumé de l'importance des features"""
        if self.feature_selector is None:
            return None
        
        if hasattr(self.feature_selector, 'scores_'):
            scores = self.feature_selector.scores_[self.feature_selector.get_support()]
            feature_importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance_score': scores
            }).sort_values('importance_score', ascending=False)
            
            return feature_importance
        
        return None

def main():
    """Fonction principale pour tester le feature engineering"""
    
    # Initialisation
    engineer = CSGOFeatureEngineer()
    
    try:
        # 1. Charger les données préprocessées
        X_train, X_val, X_test, y_train, y_val, y_test = engineer.load_processed_data()
        
        LOGGER.info(f"📊 Données chargées:")
        LOGGER.info(f"  Features initiales: {X_train.shape[1]}")
        LOGGER.info(f"  Échantillons train: {X_train.shape[0]}")
        
        # 2. Créer des features avancées
        X_train_adv, X_val_adv, X_test_adv = engineer.create_advanced_features(
            X_train, X_val, X_test
        )
        
        # 3. Créer des features polynomiales (interactions)
        X_train_poly, X_val_poly, X_test_poly = engineer.create_polynomial_features(
            X_train_adv, X_val_adv, X_test_adv, degree=2, max_features=30
        )
        
        LOGGER.info(f"  Features après polynomial: {X_train_poly.shape[1]}")
        
        # 4. Sélectionner les meilleures features
        X_train_selected, X_val_selected, X_test_selected = engineer.select_best_features(
            X_train_poly, X_val_poly, X_test_poly, y_train, 
            method='f_classif', k=40
        )
        
        # 5. Option: Appliquer PCA pour comparaison
        X_train_pca, X_val_pca, X_test_pca = engineer.apply_pca(
            X_train_selected, X_val_selected, X_test_selected, n_components=0.95
        )
        
        # 6. Sauvegarder les différentes versions
        engineer.save_engineered_features(
            # Version avec feature selection
            X_train_engineered=X_train_selected,
            X_val_engineered=X_val_selected,
            X_test_engineered=X_test_selected,
            # Version avec PCA
            X_train_pca=X_train_pca,
            X_val_pca=X_val_pca,
            X_test_pca=X_test_pca,
            # Labels (inchangés)
            y_train_engineered=y_train,
            y_val_engineered=y_val,
            y_test_engineered=y_test
        )
        
        # 7. Résumé de l'importance des features
        feature_importance = engineer.get_feature_importance_summary()
        if feature_importance is not None:
            LOGGER.info("\n🏆 TOP 10 FEATURES LES PLUS IMPORTANTES:")
            for i, row in feature_importance.head(10).iterrows():
                LOGGER.info(f"  {i+1:2d}. {row['feature']:<25} (score: {row['importance_score']:.3f})")
        
        # 8. Statistiques finales
        LOGGER.info(f"\n📈 RÉSUMÉ DU FEATURE ENGINEERING:")
        LOGGER.info(f"  Features initiales: {X_train.shape[1]}")
        LOGGER.info(f"  Features finales (selected): {X_train_selected.shape[1]}")
        LOGGER.info(f"  Features finales (PCA): {X_train_pca.shape[1]}")
        LOGGER.info(f"  Réduction dimensionnalité: {X_train.shape[1] - X_train_selected.shape[1]} features supprimées")
        
        print("\n✅ FEATURE ENGINEERING TERMINÉ AVEC SUCCÈS!")
        print("💡 Les données sont prêtes pour la modélisation")
        
    except Exception as e:
        LOGGER.error(f"❌ Erreur pendant le feature engineering: {e}")
        raise

if __name__ == "__main__":
    main()
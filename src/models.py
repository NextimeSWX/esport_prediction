"""
Module de modélisation ML pour CS:GO
École89 - 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb
import joblib
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))

# Imports avec gestion d'erreur
try:
    from config.config import (
        PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, 
        CV_FOLDS, PRIMARY_METRIC, LOGGER
    )
except ImportError:
    # Configuration de base si config.py incomplet
    PROCESSED_DATA_DIR = Path("data/processed")
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    RANDOM_STATE = 42
    CV_FOLDS = 5
    PRIMARY_METRIC = 'roc_auc'
    
    import logging
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)

class CSGOModelTrainer:
    """Classe pour l'entraînement et l'évaluation des modèles CS:GO"""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Initialisation des modèles
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialise la collection de modèles à tester"""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                n_estimators=100
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                verbosity=0
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=RANDOM_STATE,
                n_estimators=100
            )
        }
        
        LOGGER.info(f"🤖 {len(self.models)} modèles initialisés")
    
    def load_data(self):
        """
        Charge les données pour l'entraînement
        
        Returns:
            Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
            X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val.csv")
            X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
            y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").iloc[:, 0]
            y_val = pd.read_csv(PROCESSED_DATA_DIR / "y_val.csv").iloc[:, 0]
            y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").iloc[:, 0]
            
            LOGGER.info(f"📁 Données chargées:")
            LOGGER.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            LOGGER.info(f"  Features: {X_train.shape[1]}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except FileNotFoundError as e:
            LOGGER.error(f"❌ Fichier non trouvé: {e}")
            LOGGER.info("💡 Lancez d'abord data_preprocessing.py")
            raise
    
    def train_baseline_models(self, X_train, X_val, y_train, y_val):
        """
        Entraîne tous les modèles avec paramètres par défaut
        """
        LOGGER.info("🚀 Entraînement des modèles baseline...")
        
        results = {}
        
        for name, model in self.models.items():
            LOGGER.info(f"  Entraînement: {name}")
            
            try:
                # Entraînement
                model.fit(X_train, y_train)
                
                # Prédictions
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                # Prédictions probabilistes (si disponibles)
                if hasattr(model, 'predict_proba'):
                    y_train_proba = model.predict_proba(X_train)[:, 1]
                    y_val_proba = model.predict_proba(X_val)[:, 1]
                else:
                    # Pour les modèles sans predict_proba
                    y_train_proba = model.decision_function(X_train)
                    y_val_proba = model.decision_function(X_val)
                
                # Calcul des métriques
                train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)
                val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_proba)
                
                # Validation croisée
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                    scoring=PRIMARY_METRIC, n_jobs=-1
                )
                
                results[name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': {
                        'y_train_pred': y_train_pred,
                        'y_val_pred': y_val_pred,
                        'y_train_proba': y_train_proba,
                        'y_val_proba': y_val_proba
                    }
                }
                
                LOGGER.info(f"    ✅ {name}: Val {PRIMARY_METRIC}={val_metrics[PRIMARY_METRIC]:.4f}")
                
            except Exception as e:
                LOGGER.error(f"    ❌ Erreur avec {name}: {e}")
                continue
        
        self.model_results = results
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calcule toutes les métriques de classification"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }
        return metrics
    
    def hyperparameter_tuning(self, X_train, y_train, top_models=2):
        """
        Optimise les hyperparamètres des meilleurs modèles
        """
        LOGGER.info(f"🔧 Optimisation des hyperparamètres (top {top_models} modèles)...")
        
        # Sélectionner les meilleurs modèles basés sur la validation
        model_scores = {
            name: results['val_metrics'][PRIMARY_METRIC] 
            for name, results in self.model_results.items()
        }
        
        best_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:top_models]
        
        LOGGER.info(f"Modèles sélectionnés pour l'optimisation:")
        for name, score in best_models:
            LOGGER.info(f"  {name}: {score:.4f}")
        
        optimized_results = {}
        
        for model_name, _ in best_models:
            LOGGER.info(f"  Optimisation: {model_name}")
            
            try:
                # Récupérer la grille de paramètres
                param_grid = self._get_param_grid(model_name)
                
                if param_grid is None:
                    LOGGER.info(f"    Pas de grille définie pour {model_name}")
                    optimized_results[model_name] = self.model_results[model_name]
                    continue
                
                # Initialiser le modèle
                base_model = self._get_base_model(model_name)
                
                # Grid Search avec validation croisée
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                    scoring=PRIMARY_METRIC,
                    n_jobs=-1,
                    verbose=0
                )
                
                # Entraînement
                grid_search.fit(X_train, y_train)
                
                optimized_results[model_name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_cv_score': grid_search.best_score_,
                    'grid_search': grid_search
                }
                
                LOGGER.info(f"    ✅ Optimisé: {grid_search.best_score_:.4f}")
                LOGGER.info(f"    Meilleurs params: {grid_search.best_params_}")
                
            except Exception as e:
                LOGGER.error(f"    ❌ Erreur optimisation {model_name}: {e}")
                # Garder le modèle original en cas d'erreur
                optimized_results[model_name] = self.model_results[model_name]
        
        return optimized_results
    
    def _get_param_grid(self, model_name):
        """Retourne la grille de paramètres pour un modèle"""
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.2]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2']
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        }
        
        return param_grids.get(model_name)
    
    def _get_base_model(self, model_name):
        """Retourne une instance fraîche du modèle"""
        models = {
            'random_forest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            'xgboost': xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0),
            'logistic_regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            'gradient_boosting': GradientBoostingClassifier(random_state=RANDOM_STATE)
        }
        
        return models[model_name]
    
    def select_best_model(self, optimized_results, X_val, y_val):
        """
        Sélectionne le meilleur modèle basé sur la performance de validation
        """
        LOGGER.info("🏆 Sélection du meilleur modèle...")
        
        model_performances = {}
        
        for name, result in optimized_results.items():
            model = result['model']
            
            # Évaluation sur validation
            y_val_pred = model.predict(X_val)
            
            if hasattr(model, 'predict_proba'):
                y_val_proba = model.predict_proba(X_val)[:, 1]
            else:
                y_val_proba = model.decision_function(X_val)
            
            val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_proba)
            
            model_performances[name] = {
                'model': model,
                'metrics': val_metrics,
                'best_params': result.get('best_params', {}),
                'cv_score': result.get('best_cv_score', result.get('cv_mean', 0))
            }
        
        # Sélectionner le meilleur selon PRIMARY_METRIC
        best_model_name = max(
            model_performances.keys(),
            key=lambda x: model_performances[x]['metrics'][PRIMARY_METRIC]
        )
        
        self.best_model_name = best_model_name
        self.best_model = model_performances[best_model_name]['model']
        
        LOGGER.info(f"🥇 Meilleur modèle: {best_model_name}")
        LOGGER.info(f"   {PRIMARY_METRIC}: {model_performances[best_model_name]['metrics'][PRIMARY_METRIC]:.4f}")
        
        return best_model_name, self.best_model, model_performances
    
    def final_evaluation(self, X_test, y_test, model_performances):
        """
        Évaluation finale sur le test set
        """
        LOGGER.info("📊 Évaluation finale sur le test set...")
        
        if self.best_model is None:
            raise ValueError("Aucun modèle sélectionné. Lancez d'abord select_best_model()")
        
        # Prédictions sur test
        y_test_pred = self.best_model.predict(X_test)
        
        if hasattr(self.best_model, 'predict_proba'):
            y_test_proba = self.best_model.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = self.best_model.decision_function(X_test)
        
        # Métriques finales
        test_metrics = self._calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        # Rapport de classification détaillé
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        final_results = {
            'best_model_name': self.best_model_name,
            'test_metrics': test_metrics,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': {
                'y_test_pred': y_test_pred,
                'y_test_proba': y_test_proba
            }
        }
        
        # Affichage des résultats
        LOGGER.info(f"\n🎯 RÉSULTATS FINAUX - {self.best_model_name}:")
        for metric, value in test_metrics.items():
            LOGGER.info(f"  {metric.upper()}: {value:.4f}")
        
        LOGGER.info(f"\n📋 MATRICE DE CONFUSION:")
        LOGGER.info(f"  TN: {conf_matrix[0,0]:3d} | FP: {conf_matrix[0,1]:3d}")
        LOGGER.info(f"  FN: {conf_matrix[1,0]:3d} | TP: {conf_matrix[1,1]:3d}")
        
        return final_results
    
    def save_best_model(self, model_name=None):
        """Sauvegarde le meilleur modèle"""
        if self.best_model is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        if model_name is None:
            model_name = f"best_model_{self.best_model_name}"
        
        model_path = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(self.best_model, model_path)
        
        LOGGER.info(f"💾 Modèle sauvegardé: {model_path}")
        return model_path
    
    def get_feature_importance(self, feature_names):
        """Récupère l'importance des features si disponible"""
        if self.best_model is None:
            return None
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        elif hasattr(self.best_model, 'coef_'):
            # Pour les modèles linéaires
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(self.best_model.coef_[0])
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None

def main():
    """Fonction principale pour l'entraînement complet"""
    
    print("🤖 " + "="*50)
    print("   ENTRAÎNEMENT MODÈLES CS:GO ML")
    print("   École89 - 2025")
    print("="*54)
    
    # Initialisation
    trainer = CSGOModelTrainer()
    
    try:
        # 1. Charger les données
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()
        
        # 2. Entraînement baseline
        baseline_results = trainer.train_baseline_models(X_train, X_val, y_train, y_val)
        
        # 3. Optimisation des hyperparamètres
        optimized_results = trainer.hyperparameter_tuning(X_train, y_train, top_models=2)
        
        # 4. Sélection du meilleur modèle
        best_name, best_model, model_performances = trainer.select_best_model(
            optimized_results, X_val, y_val
        )
        
        # 5. Évaluation finale
        final_results = trainer.final_evaluation(X_test, y_test, model_performances)
        
        # 6. Sauvegarde
        model_path = trainer.save_best_model()
        
        # 7. Feature importance
        feature_importance = trainer.get_feature_importance(X_train.columns.tolist())
        if feature_importance is not None:
            LOGGER.info(f"\n🔍 TOP 10 FEATURES IMPORTANTES:")
            for i, row in feature_importance.head(10).iterrows():
                LOGGER.info(f"  {i+1:2d}. {row['feature']:<25} ({row['importance']:.4f})")
        
        print(f"\n✅ ENTRAÎNEMENT TERMINÉ!")
        print(f"🏆 Meilleur modèle: {best_name}")
        print(f"📊 Performance test: {final_results['test_metrics'][PRIMARY_METRIC]:.4f}")
        print(f"💾 Modèle sauvé: {model_path}")
        print(f"🚀 Prochaine étape: python src/evaluation.py")
        
    except Exception as e:
        LOGGER.error(f"❌ Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
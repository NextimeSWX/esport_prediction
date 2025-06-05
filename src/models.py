"""
Module de modélisation ML pour CS:GO - Version Complète
École89 - 2025
"""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from config.config import (
        MODELS_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR,
        RANDOM_STATE, CV_FOLDS, PRIMARY_METRIC, LOGGER
    )
except ImportError:
    # Configuration de base si config.py incomplet
    MODELS_DIR = Path("models")
    PROCESSED_DATA_DIR = Path("data/processed")
    FEATURES_DATA_DIR = Path("data/features")
    MODELS_DIR.mkdir(exist_ok=True)
    
    RANDOM_STATE = 42
    CV_FOLDS = 5
    PRIMARY_METRIC = 'roc_auc'
    
    import logging
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)

class CSGOModelTrainer:
    """Classe principale pour l'entraînement des modèles CS:GO"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.cv_results = {}
        
    def load_data(self):
        """Charge les données preprocessées"""
        try:
            # Essayer les données features d'abord
            X_train = pd.read_csv(FEATURES_DATA_DIR / "X_train_engineered.csv")
            X_val = pd.read_csv(FEATURES_DATA_DIR / "X_val_engineered.csv")
            X_test = pd.read_csv(FEATURES_DATA_DIR / "X_test_engineered.csv")
            y_train = pd.read_csv(FEATURES_DATA_DIR / "y_train_engineered.csv").iloc[:, 0]
            y_val = pd.read_csv(FEATURES_DATA_DIR / "y_val_engineered.csv").iloc[:, 0]
            y_test = pd.read_csv(FEATURES_DATA_DIR / "y_test_engineered.csv").iloc[:, 0]
            
            LOGGER.info("📁 Données engineered chargées")
            
        except FileNotFoundError:
            # Fallback vers données processed
            X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
            X_val = pd.read_csv(PROCESSED_DATA_DIR / "X_val.csv")
            X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
            y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").iloc[:, 0]
            y_val = pd.read_csv(PROCESSED_DATA_DIR / "y_val.csv").iloc[:, 0]
            y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").iloc[:, 0]
            
            LOGGER.info("📁 Données processed chargées")
        
        LOGGER.info(f"📊 Données chargées: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_base_models(self):
        """Crée la collection de modèles de base"""
        
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                solver='liblinear'
            ),
            
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_depth=10
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                learning_rate=0.1,
                max_depth=6
            ),
            
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_depth=10
            ),
            
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=RANDOM_STATE,
                C=1.0
            ),
            
            'naive_bayes': GaussianNB(),
            
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=RANDOM_STATE,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        # Ajouter XGBoost si disponible
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                eval_metric='logloss',
                verbosity=0,
                learning_rate=0.1,
                max_depth=6
            )
        
        LOGGER.info(f"🤖 {len(self.models)} modèles de base créés")
        return self.models
    
    def train_baseline_models(self, X_train, X_val, y_train, y_val):
        """Entraîne et évalue les modèles de base"""
        
        LOGGER.info("🎯 Entraînement des modèles baseline...")
        
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        for name, model in self.models.items():
            LOGGER.info(f"  Entraînement: {name}")
            
            try:
                # Validation croisée
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv, scoring=PRIMARY_METRIC, n_jobs=-1
                )
                
                # Entraînement sur train complet
                model.fit(X_train, y_train)
                
                # Évaluation sur validation
                if hasattr(model, 'predict_proba'):
                    y_val_proba = model.predict_proba(X_val)[:, 1]
                else:
                    y_val_proba = model.decision_function(X_val)
                
                y_val_pred = model.predict(X_val)
                
                # Métriques
                val_accuracy = accuracy_score(y_val, y_val_pred)
                val_auc = roc_auc_score(y_val, y_val_proba)
                val_f1 = f1_score(y_val, y_val_pred)
                
                self.results[name] = {
                    'model': model,
                    'cv_score_mean': cv_scores.mean(),
                    'cv_score_std': cv_scores.std(),
                    'val_accuracy': val_accuracy,
                    'val_auc': val_auc,
                    'val_f1': val_f1,
                    'cv_scores': cv_scores
                }
                
                LOGGER.info(f"    ✅ CV AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
            except Exception as e:
                LOGGER.warning(f"    ⚠️ Erreur avec {name}: {e}")
                continue
        
        # Classement des modèles
        ranking = sorted(
            self.results.items(),
            key=lambda x: x[1]['cv_score_mean'],
            reverse=True
        )
        
        LOGGER.info("\n🏆 Classement baseline:")
        for i, (name, results) in enumerate(ranking[:5], 1):
            LOGGER.info(f"  {i}. {name}: {results['cv_score_mean']:.4f}")
        
        return self.results
    
    def hyperparameter_tuning(self, X_train, y_train, top_models=3):
        """Optimise les hyperparamètres des meilleurs modèles"""
        
        LOGGER.info(f"🔧 Optimisation des hyperparamètres (top {top_models})...")
        
        # Sélectionner les meilleurs modèles
        ranking = sorted(
            self.results.items(),
            key=lambda x: x[1]['cv_score_mean'],
            reverse=True
        )
        
        top_model_names = [name for name, _ in ranking[:top_models]]
        optimized_results = {}
        
        # Grilles d'hyperparamètres
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 75)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
        
        # Ajouter XGBoost si disponible
        if XGBOOST_AVAILABLE and 'xgboost' in top_model_names:
            param_grids['xgboost'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        for model_name in top_model_names:
            if model_name not in param_grids:
                # Garder le modèle original si pas de grille définie
                optimized_results[model_name] = self.results[model_name]
                continue
            
            LOGGER.info(f"  Optimisation: {model_name}")
            
            try:
                # Récupérer le modèle de base
                base_model = self.models[model_name]
                
                # Recherche randomisée (plus rapide)
                search = RandomizedSearchCV(
                    base_model,
                    param_grids[model_name],
                    n_iter=20,  # Nombre d'essais
                    cv=cv,
                    scoring=PRIMARY_METRIC,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    verbose=0
                )
                
                search.fit(X_train, y_train)
                
                optimized_results[model_name] = {
                    'model': search.best_estimator_,
                    'best_score': search.best_score_,
                    'best_params': search.best_params_,
                    'cv_results': search.cv_results_
                }
                
                LOGGER.info(f"    ✅ Meilleur score: {search.best_score_:.4f}")
                LOGGER.info(f"    🔧 Meilleurs params: {list(search.best_params_.keys())[:3]}...")
                
            except Exception as e:
                LOGGER.warning(f"    ⚠️ Erreur optimisation {model_name}: {e}")
                # Garder le modèle original
                optimized_results[model_name] = self.results[model_name]
        
        return optimized_results
    
    def select_best_model(self, optimized_results, X_val, y_val):
        """Sélectionne le meilleur modèle"""
        
        LOGGER.info("🎯 Sélection du meilleur modèle...")
        
        model_performances = {}
        
        for name, results in optimized_results.items():
            model = results['model']
            
            # Prédictions sur validation
            if hasattr(model, 'predict_proba'):
                y_val_proba = model.predict_proba(X_val)[:, 1]
            else:
                y_val_proba = model.decision_function(X_val)
            
            y_val_pred = model.predict(X_val)
            
            # Métriques complètes
            performance = {
                'accuracy': accuracy_score(y_val, y_val_pred),
                'precision': precision_score(y_val, y_val_pred),
                'recall': recall_score(y_val, y_val_pred),
                'f1_score': f1_score(y_val, y_val_pred),
                'roc_auc': roc_auc_score(y_val, y_val_proba),
                'model': model
            }
            
            model_performances[name] = performance
        
        # Sélectionner selon la métrique principale
        best_name = max(
            model_performances.keys(),
            key=lambda x: model_performances[x][PRIMARY_METRIC]
        )
        
        self.best_model_name = best_name
        self.best_model = model_performances[best_name]['model']
        
        LOGGER.info(f"🏆 Meilleur modèle: {best_name}")
        LOGGER.info(f"📊 Performance: {model_performances[best_name][PRIMARY_METRIC]:.4f}")
        
        return best_name, self.best_model, model_performances
    
    def final_evaluation(self, X_test, y_test, model_performances):
        """Évaluation finale sur le set de test"""
        
        LOGGER.info("📊 Évaluation finale sur test set...")
        
        model = self.best_model
        
        # Prédictions finales
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = model.decision_function(X_test)
        
        y_test_pred = model.predict(X_test)
        
        # Métriques de test
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1_score': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_proba)
        }
        
        # Rapport de classification
        class_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        
        LOGGER.info(f"🎯 Résultats finaux:")
        LOGGER.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        LOGGER.info(f"  AUC-ROC: {test_metrics['roc_auc']:.4f}")
        LOGGER.info(f"  F1-Score: {test_metrics['f1_score']:.4f}")
        
        return {
            'test_metrics': test_metrics,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': {
                'y_true': y_test,
                'y_pred': y_test_pred,
                'y_proba': y_test_proba
            }
        }
    
    def save_best_model(self, filename=None):
        """Sauvegarde le meilleur modèle"""
        
        if self.best_model is None:
            raise ValueError("Aucun meilleur modèle sélectionné")
        
        if filename is None:
            filename = f"best_model_{self.best_model_name}.pkl"
        
        filepath = MODELS_DIR / filename
        joblib.dump(self.best_model, filepath)
        
        LOGGER.info(f"💾 Meilleur modèle sauvegardé: {filepath}")
        
        # Sauvegarder aussi les métadonnées
        metadata = {
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'features_count': len(self.best_model.feature_names_in_) if hasattr(self.best_model, 'feature_names_in_') else 'unknown',
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = MODELS_DIR / f"metadata_{self.best_model_name}.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath
    
    def create_ensemble_model(self, model_performances, top_n=3):
        """Crée un modèle d'ensemble avec les meilleurs modèles"""
        
        LOGGER.info(f"🔗 Création d'un ensemble des {top_n} meilleurs modèles...")
        
        # Sélectionner les top N modèles
        ranking = sorted(
            model_performances.items(),
            key=lambda x: x[1][PRIMARY_METRIC],
            reverse=True
        )
        
        top_models = ranking[:top_n]
        
        estimators = [
            (name, performance['model'])
            for name, performance in top_models
        ]
        
        # Ensemble voting (soft pour utiliser les probabilités)
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        LOGGER.info(f"📊 Ensemble créé avec:")
        for name, _ in estimators:
            score = model_performances[name][PRIMARY_METRIC]
            LOGGER.info(f"  - {name}: {score:.4f}")
        
        return ensemble

def main():
    """Fonction principale pour l'entraînement des modèles"""
    
    print("🤖 " + "="*50)
    print("   ENTRAÎNEMENT DES MODÈLES CS:GO")
    print("   École89 - 2025")
    print("="*54)
    
    try:
        # Initialisation
        trainer = CSGOModelTrainer()
        
        # 1. Chargement des données
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()
        
        # 2. Création des modèles de base
        trainer.create_base_models()
        
        # 3. Entraînement baseline
        baseline_results = trainer.train_baseline_models(X_train, X_val, y_train, y_val)
        
        # 4. Optimisation des hyperparamètres
        optimized_results = trainer.hyperparameter_tuning(X_train, y_train, top_models=3)
        
        # 5. Sélection du meilleur modèle
        best_name, best_model, model_performances = trainer.select_best_model(
            optimized_results, X_val, y_val
        )
        
        # 6. Évaluation finale
        final_results = trainer.final_evaluation(X_test, y_test, model_performances)
        
        # 7. Sauvegarde
        model_path = trainer.save_best_model()
        
        # 8. Résumé final
        print(f"\n✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print(f"🏆 Meilleur modèle: {best_name}")
        print(f"📊 Performance test:")
        print(f"  Accuracy: {final_results['test_metrics']['accuracy']:.4f}")
        print(f"  AUC-ROC: {final_results['test_metrics']['roc_auc']:.4f}")
        print(f"  F1-Score: {final_results['test_metrics']['f1_score']:.4f}")
        print(f"💾 Modèle sauvegardé: {model_path}")
        print(f"🚀 Prochaine étape: python src/evaluation.py")
        
        return {
            'trainer': trainer,
            'best_model': best_model,
            'best_name': best_name,
            'final_results': final_results,
            'model_path': model_path
        }
        
    except Exception as e:
        LOGGER.error(f"❌ Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
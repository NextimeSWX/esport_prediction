"""
Modèles ML avancés pour maximiser la note du projet
École89 - 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

class AdvancedCSGOModels:
    """Modèles ML avancés pour la prédiction CS:GO"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.ensemble_models = {}
        self.results = {}
    
    def create_base_models(self):
        """Crée une collection étendue de modèles de base"""
        
        self.models = {
            # Modèles linéaires
            'logistic_regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            
            # Modèles basés sur les arbres
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state, eval_metric='logloss', verbosity=0
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.random_state, n_estimators=100
            ),
            
            # Modèles probabilistes
            'naive_bayes': GaussianNB(),
            
            # Support Vector Machine
            'svm_linear': SVC(
                kernel='linear', probability=True, random_state=self.random_state
            ),
            'svm_rbf': SVC(
                kernel='rbf', probability=True, random_state=self.random_state
            ),
            
            # Réseau de neurones
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=self.random_state,
                max_iter=500, early_stopping=True
            )
        }
        
        print(f"🤖 {len(self.models)} modèles de base créés")
        return self.models
    
    def create_ensemble_models(self):
        """Crée des modèles d'ensemble avancés"""
        
        # Modèles de base pour les ensembles
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=self.random_state)),
            ('xgb', xgb.XGBClassifier(n_estimators=50, random_state=self.random_state, verbosity=0)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=self.random_state)),
            ('lr', LogisticRegression(random_state=self.random_state))
        ]
        
        self.ensemble_models = {
            # Voting Classifier (Hard)
            'voting_hard': VotingClassifier(
                estimators=base_models,
                voting='hard'
            ),
            
            # Voting Classifier (Soft) - Utilise les probabilités
            'voting_soft': VotingClassifier(
                estimators=base_models,
                voting='soft'
            ),
            
            # Stacking Classifier
            'stacking': StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(random_state=self.random_state),
                cv=5
            )
        }
        
        print(f"🔗 {len(self.ensemble_models)} modèles d'ensemble créés")
        return self.ensemble_models
    
    def evaluate_all_models(self, X_train, y_train, cv_folds=5):
        """Évalue tous les modèles avec validation croisée"""
        
        print("📊 Évaluation de tous les modèles...")
        
        # Stratified K-Fold pour données déséquilibrées
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        all_models = {**self.models, **self.ensemble_models}
        
        for name, model in all_models.items():
            print(f"  Évaluation: {name}")
            
            try:
                # Validation croisée multiple métriques
                accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                roc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
                f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
                
                self.results[name] = {
                    'accuracy_mean': accuracy_scores.mean(),
                    'accuracy_std': accuracy_scores.std(),
                    'roc_auc_mean': roc_scores.mean(),
                    'roc_auc_std': roc_scores.std(),
                    'f1_mean': f1_scores.mean(),
                    'f1_std': f1_scores.std(),
                    'model': model
                }
                
                print(f"    ✅ AUC: {roc_scores.mean():.4f} (±{roc_scores.std():.4f})")
                
            except Exception as e:
                print(f"    ❌ Erreur: {e}")
                continue
        
        return self.results
    
    def get_model_ranking(self, metric='roc_auc_mean'):
        """Classe les modèles par performance"""
        
        if not self.results:
            raise ValueError("Lancez d'abord evaluate_all_models()")
        
        ranking = sorted(
            self.results.items(),
            key=lambda x: x[1][metric],
            reverse=True
        )
        
        print(f"\n🏆 CLASSEMENT DES MODÈLES ({metric}):")
        print("-" * 50)
        
        for i, (name, results) in enumerate(ranking, 1):
            score = results[metric]
            std = results.get(f"{metric.split('_')[0]}_std", 0)
            print(f"{i:2d}. {name:<20} {score:.4f} (±{std:.4f})")
        
        return ranking
    
    def create_meta_ensemble(self, top_n=3):
        """Crée un ensemble des meilleurs modèles"""
        
        ranking = self.get_model_ranking()
        top_models = ranking[:top_n]
        
        # Préparer les estimateurs pour le méta-ensemble
        estimators = [
            (name, results['model']) 
            for name, results in top_models
        ]
        
        meta_ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'
        )
        
        print(f"\n🚀 Méta-ensemble créé avec les {top_n} meilleurs modèles:")
        for name, _ in estimators:
            print(f"  - {name}")
        
        return meta_ensemble
    
    def perform_statistical_tests(self, X_train, y_train):
        """Effectue des tests statistiques entre modèles"""
        from scipy import stats
        
        print("\n📈 Tests statistiques de comparaison...")
        
        # Comparer les 3 meilleurs modèles
        ranking = self.get_model_ranking()
        top_3 = ranking[:3]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        model_scores = {}
        for name, results in top_3:
            scores = cross_val_score(results['model'], X_train, y_train, cv=cv, scoring='roc_auc')
            model_scores[name] = scores
        
        # Tests de significativité (t-test pairé)
        models = list(model_scores.keys())
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                model1, model2 = models[i], models[j]
                scores1, scores2 = model_scores[model1], model_scores[model2]
                
                # T-test pairé
                t_stat, p_value = stats.ttest_rel(scores1, scores2)
                
                diff = scores1.mean() - scores2.mean()
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                
                print(f"  {model1} vs {model2}:")
                print(f"    Différence: {diff:+.4f} (p={p_value:.4f}) {significance}")
        
        return model_scores
    
    def generate_model_report(self):
        """Génère un rapport complet des modèles"""
        
        print("\n" + "="*60)
        print("RAPPORT COMPLET DES MODÈLES ML")
        print("="*60)
        
        if not self.results:
            print("❌ Aucun résultat disponible")
            return
        
        # Statistiques générales
        n_models = len(self.results)
        best_model = max(self.results.items(), key=lambda x: x[1]['roc_auc_mean'])
        worst_model = min(self.results.items(), key=lambda x: x[1]['roc_auc_mean'])
        
        print(f"\n📊 STATISTIQUES GÉNÉRALES:")
        print(f"  Nombre de modèles testés: {n_models}")
        print(f"  Meilleur modèle: {best_model[0]} (AUC: {best_model[1]['roc_auc_mean']:.4f})")
        print(f"  Pire modèle: {worst_model[0]} (AUC: {worst_model[1]['roc_auc_mean']:.4f})")
        
        # Distribution des performances
        auc_scores = [r['roc_auc_mean'] for r in self.results.values()]
        print(f"  AUC moyen: {np.mean(auc_scores):.4f}")
        print(f"  Écart-type AUC: {np.std(auc_scores):.4f}")
        
        # Catégorisation des modèles
        excellent = [name for name, r in self.results.items() if r['roc_auc_mean'] > 0.9]
        good = [name for name, r in self.results.items() if 0.8 <= r['roc_auc_mean'] <= 0.9]
        moderate = [name for name, r in self.results.items() if r['roc_auc_mean'] < 0.8]
        
        print(f"\n🎯 CATÉGORISATION DES PERFORMANCES:")
        print(f"  Excellents (AUC > 0.9): {len(excellent)} modèles")
        for model in excellent[:3]:  # Top 3
            print(f"    - {model}")
        
        print(f"  Bons (0.8 ≤ AUC ≤ 0.9): {len(good)} modèles")
        print(f"  Modérés (AUC < 0.8): {len(moderate)} modèles")
        
        # Recommandations
        print(f"\n💡 RECOMMANDATIONS:")
        if len(excellent) >= 3:
            print("  ✅ Plusieurs modèles excellents - Utiliser un ensemble")
        elif len(excellent) >= 1:
            print("  👍 Au moins un modèle excellent identifié")
        else:
            print("  ⚠️ Performances limitées - Revoir les features ou la target")
        
        if best_model[1]['roc_auc_std'] > 0.05:
            print("  ⚠️ Variance élevée - Augmenter les données ou la régularisation")
        
        return {
            'n_models': n_models,
            'best_model': best_model[0],
            'best_score': best_model[1]['roc_auc_mean'],
            'auc_scores': auc_scores
        }

# Utilisation dans le pipeline principal
def run_advanced_modeling(X_train, y_train):
    """Lance l'analyse complète avec modèles avancés"""
    
    # Initialiser
    advanced_models = AdvancedCSGOModels()
    
    # Créer tous les modèles
    advanced_models.create_base_models()
    advanced_models.create_ensemble_models()
    
    # Évaluer
    results = advanced_models.evaluate_all_models(X_train, y_train)
    
    # Analyser
    ranking = advanced_models.get_model_ranking()
    stats_results = advanced_models.perform_statistical_tests(X_train, y_train)
    
    # Créer méta-ensemble
    meta_model = advanced_models.create_meta_ensemble(top_n=3)
    
    # Rapport final
    report = advanced_models.generate_model_report()
    
    return {
        'advanced_models': advanced_models,
        'results': results,
        'ranking': ranking,
        'meta_ensemble': meta_model,
        'report': report
    }

if __name__ == "__main__":
    print("🚀 Test des modèles avancés...")
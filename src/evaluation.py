"""
Module d'évaluation et visualisation des modèles CS:GO - VERSION CORRIGÉE
École89 - 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score
)
# CORRECTION: learning_curve est dans model_selection, pas metrics
from sklearn.model_selection import learning_curve, validation_curve, StratifiedKFold
import joblib
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.config import (
        MODELS_DIR, FEATURES_DATA_DIR, PROCESSED_DATA_DIR, COLORS, PLOT_STYLE, 
        FIGURE_SIZE, DPI, RANDOM_STATE, LOGGER
    )
except ImportError:
    # Configuration de base si config.py incomplet
    MODELS_DIR = Path("models")
    FEATURES_DATA_DIR = Path("data/features")
    PROCESSED_DATA_DIR = Path("data/processed")
    
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'accent': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff7f0e',
        'success': '#2ca02c'
    }
    
    PLOT_STYLE = 'default'
    FIGURE_SIZE = (10, 6)
    DPI = 100
    RANDOM_STATE = 42
    
    import logging
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)

# Configuration matplotlib
plt.style.use(PLOT_STYLE)
plt.rcParams['figure.figsize'] = FIGURE_SIZE
plt.rcParams['figure.dpi'] = DPI

class CSGOModelEvaluator:
    """Classe pour l'évaluation complète des modèles CS:GO"""
    
    def __init__(self):
        self.model = None
        self.model_name = None
        self.results = {}
        
    def load_model(self, model_path=None):
        """Charge un modèle sauvegardé"""
        if model_path is None:
            # Chercher le meilleur modèle
            model_files = list(MODELS_DIR.glob("best_model_*.pkl"))
            if not model_files:
                raise FileNotFoundError("Aucun modèle trouvé dans models/")
            model_path = model_files[0]
        
        self.model = joblib.load(model_path)
        self.model_name = model_path.stem
        LOGGER.info(f"📦 Modèle chargé: {model_path}")
        
    def load_test_data(self):
        """Charge les données de test"""
        try:
            # Essayer les données engineered d'abord
            X_test = pd.read_csv(FEATURES_DATA_DIR / "X_test_engineered.csv")
            y_test = pd.read_csv(FEATURES_DATA_DIR / "y_test_engineered.csv").iloc[:, 0]
            
            LOGGER.info(f"📁 Données engineered chargées: {X_test.shape}")
            return X_test, y_test
            
        except FileNotFoundError:
            # Fallback vers données processed
            X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
            y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").iloc[:, 0]
            
            LOGGER.info(f"📁 Données processed chargées: {X_test.shape}")
            return X_test, y_test
    
    def evaluate_model(self, X_test, y_test):
        """
        Évaluation complète du modèle
        
        Args:
            X_test, y_test: Données de test
            
        Returns:
            Dict avec toutes les métriques
        """
        if self.model is None:
            raise ValueError("Aucun modèle chargé")
        
        LOGGER.info("📊 Évaluation du modèle en cours...")
        
        # Prédictions
        y_pred = self.model.predict(X_test)
        
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X_test)[:, 1]
        else:
            y_proba = self.model.decision_function(X_test)
        
        # Métriques de base
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Rapport de classification
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Courbe ROC
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Courbe Precision-Recall
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        
        self.results = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'roc_data': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds, 'auc': roc_auc},
            'pr_data': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds, 'auc': pr_auc},
            'predictions': {'y_pred': y_pred, 'y_proba': y_proba, 'y_true': y_test}
        }
        
        LOGGER.info(f"✅ Évaluation terminée - Accuracy: {metrics['accuracy']:.4f}")
        return self.results
    
    def plot_confusion_matrix(self, figsize=(8, 6)):
        """Visualise la matrice de confusion"""
        if 'confusion_matrix' not in self.results:
            raise ValueError("Lancez d'abord evaluate_model()")
        
        plt.figure(figsize=figsize)
        
        conf_matrix = self.results['confusion_matrix']
        
        # Normalisation pour pourcentages
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # Création du heatmap
        sns.heatmap(
            conf_matrix_norm, 
            annot=True, 
            fmt='.2%',
            cmap='Blues',
            square=True,
            cbar_kws={'label': 'Pourcentage'},
            xticklabels=['Low Performer', 'High Performer'],
            yticklabels=['Low Performer', 'High Performer']
        )
        
        plt.title(f'Matrice de Confusion - {self.model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Vraie Classe', fontsize=12)
        plt.xlabel('Classe Prédite', fontsize=12)
        
        # Ajout des valeurs absolues
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'n={conf_matrix[i, j]}', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        plt.show()
        
        # Statistiques de la matrice
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        
        LOGGER.info(f"📊 Statistiques matrice de confusion:")
        LOGGER.info(f"  Sensibilité (Recall): {sensitivity:.3f}")
        LOGGER.info(f"  Spécificité: {specificity:.3f}")
        LOGGER.info(f"  Vrais Positifs: {tp}, Faux Positifs: {fp}")
        LOGGER.info(f"  Vrais Négatifs: {tn}, Faux Négatifs: {fn}")
    
    def plot_roc_curve(self, figsize=(8, 6)):
        """Visualise la courbe ROC"""
        if 'roc_data' not in self.results:
            raise ValueError("Lancez d'abord evaluate_model()")
        
        plt.figure(figsize=figsize)
        
        roc_data = self.results['roc_data']
        
        # Courbe ROC
        plt.plot(
            roc_data['fpr'], 
            roc_data['tpr'], 
            color=COLORS['primary'],
            linewidth=2,
            label=f'ROC Curve (AUC = {roc_data["auc"]:.3f})'
        )
        
        # Ligne de référence (classificateur aléatoire)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=12)
        plt.ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=12)
        plt.title(f'Courbe ROC - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        LOGGER.info(f"🎯 AUC-ROC: {roc_data['auc']:.4f}")
    
    def plot_precision_recall_curve(self, figsize=(8, 6)):
        """Visualise la courbe Precision-Recall"""
        if 'pr_data' not in self.results:
            raise ValueError("Lancez d'abord evaluate_model()")
        
        plt.figure(figsize=figsize)
        
        pr_data = self.results['pr_data']
        
        # Courbe Precision-Recall
        plt.plot(
            pr_data['recall'], 
            pr_data['precision'], 
            color=COLORS['secondary'],
            linewidth=2,
            label=f'PR Curve (AUC = {pr_data["auc"]:.3f})'
        )
        
        # Ligne de baseline (proportion de la classe positive)
        y_true = self.results['predictions']['y_true']
        baseline = y_true.mean()
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, 
                   label=f'Baseline (Prevalence = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Courbe Precision-Recall - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, X_test, top_n=15, figsize=(10, 8)):
        """Visualise l'importance des features"""
        if self.model is None:
            raise ValueError("Aucun modèle chargé")
        
        importance_data = None
        
        # Récupérer l'importance selon le type de modèle
        if hasattr(self.model, 'feature_importances_'):
            importance_data = pd.DataFrame({
                'feature': X_test.columns,
                'importance': self.model.feature_importances_
            })
            importance_type = "Feature Importance"
            
        elif hasattr(self.model, 'coef_'):
            importance_data = pd.DataFrame({
                'feature': X_test.columns,
                'importance': np.abs(self.model.coef_[0])
            })
            importance_type = "Coefficient Magnitude"
        
        else:
            LOGGER.warning("⚠️ Importance des features non disponible pour ce modèle")
            return
        
        # Trier et sélectionner top N
        importance_data = importance_data.sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=figsize)
        
        # Barplot horizontal
        bars = plt.barh(
            range(len(importance_data)), 
            importance_data['importance'].values,
            color=COLORS['primary'],
            alpha=0.8
        )
        
        # Customisation
        plt.yticks(range(len(importance_data)), importance_data['feature'].values)
        plt.xlabel(f'{importance_type}', fontsize=12)
        plt.title(f'Top {top_n} Features - {self.model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Plus important en haut
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontsize=9)
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        LOGGER.info(f"📊 Top 5 features importantes:")
        for i, row in importance_data.head().iterrows():
            LOGGER.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    def plot_prediction_distribution(self, figsize=(12, 5)):
        """Visualise la distribution des prédictions"""
        if 'predictions' not in self.results:
            raise ValueError("Lancez d'abord evaluate_model()")
        
        predictions = self.results['predictions']
        y_true = predictions['y_true']
        y_proba = predictions['y_proba']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Distribution des probabilités par classe vraie
        ax1.hist(y_proba[y_true == 0], bins=30, alpha=0.7, label='Low Performers', 
                color=COLORS['danger'], density=True)
        ax1.hist(y_proba[y_true == 1], bins=30, alpha=0.7, label='High Performers', 
                color=COLORS['success'], density=True)
        
        ax1.set_xlabel('Probabilité prédite')
        ax1.set_ylabel('Densité')
        ax1.set_title('Distribution des Probabilités par Classe')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Seuil de décision et métriques associées
        thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        f1_scores = []
        
        for thresh in thresholds:
            y_pred_thresh = (y_proba >= thresh).astype(int)
            if len(np.unique(y_pred_thresh)) == 2:  # Éviter division par zéro
                precisions.append(precision_score(y_true, y_pred_thresh, average='binary'))
                recalls.append(recall_score(y_true, y_pred_thresh, average='binary'))
                f1_scores.append(f1_score(y_true, y_pred_thresh, average='binary'))
            else:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
        
        ax2.plot(thresholds, precisions, label='Precision', color=COLORS['primary'])
        ax2.plot(thresholds, recalls, label='Recall', color=COLORS['secondary'])
        ax2.plot(thresholds, f1_scores, label='F1-Score', color=COLORS['accent'])
        
        # Seuil optimal (max F1)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        ax2.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7,
                   label=f'Seuil optimal: {optimal_threshold:.3f}')
        
        ax2.set_xlabel('Seuil de décision')
        ax2.set_ylabel('Score')
        ax2.set_title('Métriques vs Seuil de Décision')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Analyse des Prédictions - {self.model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        LOGGER.info(f"📊 Seuil optimal (max F1): {optimal_threshold:.3f}")
        LOGGER.info(f"📊 F1-Score optimal: {max(f1_scores):.3f}")
    
    def generate_evaluation_report(self):
        """Génère un rapport d'évaluation complet"""
        if not self.results:
            raise ValueError("Lancez d'abord evaluate_model()")
        
        LOGGER.info("📋 Génération du rapport d'évaluation...")
        
        metrics = self.results['metrics']
        class_report = self.results['classification_report']
        
        print("\n" + "="*60)
        print(f"RAPPORT D'ÉVALUATION - {self.model_name}")
        print("="*60)
        
        print(f"\n📊 MÉTRIQUES GÉNÉRALES:")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  F1-Score:     {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC:      {metrics['roc_auc']:.4f}")
        
        print(f"\n📋 RAPPORT PAR CLASSE:")
        for class_name, class_metrics in class_report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                class_label = "High Performer" if class_name == '1' else "Low Performer"
                print(f"  {class_label}:")
                print(f"    Precision: {class_metrics['precision']:.4f}")
                print(f"    Recall:    {class_metrics['recall']:.4f}")
                print(f"    F1-Score:  {class_metrics['f1-score']:.4f}")
                print(f"    Support:   {class_metrics['support']}")
        
        # Interprétation business
        print(f"\n💡 INTERPRÉTATION BUSINESS:")
        accuracy = metrics['accuracy']
        if accuracy > 0.85:
            print("  ✅ Excellente performance - Modèle prêt pour la production")
        elif accuracy > 0.75:
            print("  👍 Bonne performance - Quelques améliorations possibles")
        elif accuracy > 0.65:
            print("  ⚠️ Performance modérée - Optimisations nécessaires")
        else:
            print("  ❌ Performance faible - Révision complète requise")
        
        auc_roc = metrics['roc_auc']
        if auc_roc > 0.9:
            print("  🎯 Excellente capacité discriminante")
        elif auc_roc > 0.8:
            print("  👌 Bonne capacité discriminante")
        elif auc_roc > 0.7:
            print("  📈 Capacité discriminante acceptable")
        else:
            print("  📉 Capacité discriminante faible")
        
        print("\n" + "="*60)
        
        return {
            'model_name': self.model_name,
            'metrics': metrics,
            'classification_report': class_report,
            'evaluation_summary': f"Accuracy: {accuracy:.3f}, AUC-ROC: {auc_roc:.3f}"
        }

def main():
    """Fonction principale pour l'évaluation complète"""
    
    print("📊 " + "="*50)
    print("   ÉVALUATION DES MODÈLES CS:GO")
    print("   École89 - 2025")
    print("="*54)
    
    # Initialisation
    evaluator = CSGOModelEvaluator()
    
    try:
        # 1. Charger le modèle
        evaluator.load_model()
        
        # 2. Charger les données de test
        X_test, y_test = evaluator.load_test_data()
        
        # 3. Évaluation
        results = evaluator.evaluate_model(X_test, y_test)
        
        # 4. Visualisations
        print("🎨 Génération des visualisations...")
        
        # Matrice de confusion
        evaluator.plot_confusion_matrix()
        
        # Courbe ROC
        evaluator.plot_roc_curve()
        
        # Courbe Precision-Recall
        evaluator.plot_precision_recall_curve()
        
        # Importance des features
        evaluator.plot_feature_importance(X_test)
        
        # Distribution des prédictions
        evaluator.plot_prediction_distribution()
        
        # 5. Rapport final
        report = evaluator.generate_evaluation_report()
        
        print("\n✅ ÉVALUATION TERMINÉE!")
        print("📊 Toutes les visualisations ont été générées")
        print("💾 Consultez les graphiques pour l'analyse détaillée")
        
        return report
        
    except Exception as e:
        LOGGER.error(f"❌ Erreur pendant l'évaluation: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script principal pour exécuter le pipeline complet CS:GO ML
École89 - 2025

Usage:
    python main.py [--steps STEPS] [--data-size SIZE] [--quick]
    
Examples:
    python main.py                          # Pipeline complet
    python main.py --steps collect,model   # Seulement collecte et modélisation
    python main.py --quick                 # Version rapide (moins de données)
"""

import argparse
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Ajouter le dossier src au path
sys.path.append(str(Path(__file__).parent / "src"))

# Imports avec gestion d'erreur
try:
    from config.config import LOGGER, check_steam_api_key
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)
    
    def check_steam_api_key():
        LOGGER.info("⚠️ Configuration API Steam non disponible - Mode données d'exemple")
        return False

def create_project_structure():
    """Crée la structure de dossiers du projet"""
    directories = [
        Path("data"),
        Path("data/raw"),
        Path("data/processed"), 
        Path("data/features"),
        Path("models"),
        Path("notebooks")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info("📁 Structure de projet vérifiée/créée")
    return True

def main():
    """Point d'entrée principal"""
    
    print("🎮 " + "="*60)
    print("   PIPELINE ML CS:GO - PRÉDICTION DE PERFORMANCE")
    print("   École89 - 2025 - Projet Machine Learning")
    print("="*64)
    
    # Parse des arguments
    parser = argparse.ArgumentParser(
        description="Pipeline ML pour prédiction CS:GO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Étapes disponibles:
  collect     - Collecte des données (API Steam ou génération)
  preprocess  - Nettoyage et preprocessing
  features    - Feature engineering avancé
  model       - Entraînement des modèles ML
  evaluate    - Évaluation et visualisations

Exemples:
  python main.py                                    # Pipeline complet
  python main.py --steps collect,model             # Seulement collecte + modèles
  python main.py --data-size 1000 --quick         # Version rapide
        """
    )
    
    parser.add_argument(
        '--steps', 
        type=str, 
        default='collect,preprocess,model,evaluate',
        help='Étapes à exécuter (séparées par des virgules)'
    )
    
    parser.add_argument(
        '--data-size', 
        type=int, 
        default=800,
        help='Nombre de joueurs à générer (défaut: 800)'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Mode rapide (moins de données, moins d\'optimisation)'
    )
    
    args = parser.parse_args()
    
    # Configuration rapide
    if args.quick:
        args.data_size = min(args.data_size, 500)
        LOGGER.info("🚀 Mode rapide activé")
    
    # Parse des étapes
    steps = [step.strip() for step in args.steps.split(',')]
    valid_steps = ['collect', 'preprocess', 'features', 'model', 'evaluate']
    
    for step in steps:
        if step not in valid_steps:
            print(f"❌ Étape invalide: {step}")
            print(f"Étapes valides: {', '.join(valid_steps)}")
            sys.exit(1)
    
    print(f"📋 Étapes à exécuter: {' → '.join(steps)}")
    print(f"📊 Taille du dataset: {args.data_size} joueurs")
    
    # Vérifications initiales
    LOGGER.info("🔧 Vérifications initiales...")
    create_project_structure()
    check_steam_api_key()
    
    # Exécution du pipeline
    start_time = time.time()
    
    try:
        execute_pipeline(steps, args.data_size, args.quick)
        
        total_time = time.time() - start_time
        print(f"\n🎉 " + "="*60)
        print(f"   PIPELINE TERMINÉ AVEC SUCCÈS!")
        print(f"   Temps total: {total_time:.1f} secondes")
        print(f"   Étapes complétées: {len(steps)}")
        print("="*64)
        
    except Exception as e:
        print(f"\n❌ " + "="*60)
        print(f"   ERREUR DANS LE PIPELINE")
        print(f"   {str(e)}")
        print("="*64)
        sys.exit(1)

def execute_pipeline(steps, data_size, quick_mode):
    """Exécute le pipeline selon les étapes spécifiées"""
    
    results = {}
    
    # ========================================================================
    # ÉTAPE 1: COLLECTE DES DONNÉES
    # ========================================================================
    if 'collect' in steps:
        LOGGER.info("📡 Étape 1: Collecte des données...")
        
        try:
            from data_collection import CSGODataCollector
            
            collector = CSGODataCollector()
            df = collector.generate_sample_dataset(n_players=data_size)
            
            if df.empty:
                raise Exception("Échec de la collecte des données")
            
            filepath = collector.save_raw_data(df)
            
            # Correction automatique du data leakage
            LOGGER.info("🔧 Correction automatique du data leakage...")
            try:
                from data_leakage import create_independent_target
                df_fixed, forbidden = create_independent_target(df)
                
                # Sauvegarder la version corrigée
                fixed_path = Path("data/raw/csgo_raw_data_fixed.csv")
                df_fixed.to_csv(fixed_path, index=False)
                LOGGER.info(f"💾 Données corrigées sauvegardées: {fixed_path}")
            except ImportError:
                LOGGER.warning("⚠️ Module data_leakage non disponible - utilisation des données brutes")
            
            results['data_collection'] = {
                'status': 'success',
                'samples': len(df),
                'features': len(df.columns),
                'filepath': filepath
            }
            
            LOGGER.info(f"✅ Collecte terminée: {len(df)} joueurs, {len(df.columns)} features")
            
        except ImportError as e:
            LOGGER.error(f"❌ Erreur d'import pour la collecte: {e}")
            raise Exception("Module de collecte non disponible")
        except Exception as e:
            LOGGER.error(f"❌ Erreur lors de la collecte: {e}")
            raise
    
    # ========================================================================
    # ÉTAPE 2: PREPROCESSING
    # ========================================================================
    if 'preprocess' in steps:
        LOGGER.info("🧹 Étape 2: Preprocessing des données...")
        
        try:
            from data_preprocessing import CSGODataPreprocessor
            
            preprocessor = CSGODataPreprocessor()
            
            # Chargement et nettoyage
            df_raw = preprocessor.load_raw_data()
            df_clean = preprocessor.clean_data(df_raw)
            df_imputed = preprocessor.handle_missing_values(df_clean)
            df_features = preprocessor.create_derived_features(df_imputed)
            df_selected = preprocessor.select_features(df_features)
            
            # Division et normalisation
            X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_selected)
            X_train_scaled, X_val_scaled, X_test_scaled = preprocessor.scale_features(
                X_train, X_val, X_test
            )
            
            # Sauvegarde
            preprocessor.save_processed_data(
                X_train=X_train_scaled, X_val=X_val_scaled, X_test=X_test_scaled,
                y_train=y_train, y_val=y_val, y_test=y_test
            )
            
            results['preprocessing'] = {
                'status': 'success',
                'original_samples': len(df_raw),
                'cleaned_samples': len(df_clean),
                'final_features': len(X_train_scaled.columns)
            }
            
            LOGGER.info(f"✅ Preprocessing terminé: {len(df_clean)} échantillons, {len(X_train_scaled.columns)} features")
            
        except ImportError as e:
            LOGGER.error(f"❌ Erreur d'import pour le preprocessing: {e}")
            raise Exception("Module de preprocessing non disponible")
        except Exception as e:
            LOGGER.error(f"❌ Erreur lors du preprocessing: {e}")
            raise
    
    # ========================================================================
    # ÉTAPE 3: FEATURE ENGINEERING
    # ========================================================================
    if 'features' in steps:
        LOGGER.info("🔧 Étape 3: Feature Engineering...")
        
        try:
            from feature_engineering import CSGOFeatureEngineer
            
            engineer = CSGOFeatureEngineer()
            
            # Chargement des données préprocessées
            X_train, X_val, X_test, y_train, y_val, y_test = engineer.load_processed_data()
            
            # Features avancées
            X_train_adv, X_val_adv, X_test_adv = engineer.create_advanced_features(
                X_train, X_val, X_test
            )
            
            # Features polynomiales (si pas en mode rapide)
            if not quick_mode:
                X_train_poly, X_val_poly, X_test_poly = engineer.create_polynomial_features(
                    X_train_adv, X_val_adv, X_test_adv, degree=2
                )
            else:
                X_train_poly, X_val_poly, X_test_poly = X_train_adv, X_val_adv, X_test_adv
            
            # Sélection des meilleures features
            k_features = 30 if quick_mode else 40
            X_train_selected, X_val_selected, X_test_selected = engineer.select_best_features(
                X_train_poly, X_val_poly, X_test_poly, y_train, k=k_features
            )
            
            # Sauvegarde
            engineer.save_engineered_features(
                X_train_engineered=X_train_selected,
                X_val_engineered=X_val_selected,
                X_test_engineered=X_test_selected,
                y_train_engineered=y_train,
                y_val_engineered=y_val,
                y_test_engineered=y_test
            )
            
            results['feature_engineering'] = {
                'status': 'success',
                'initial_features': X_train.shape[1],
                'advanced_features': X_train_adv.shape[1],
                'final_features': X_train_selected.shape[1],
                'selected_features': engineer.selected_features[:10] if hasattr(engineer, 'selected_features') else []
            }
            
            LOGGER.info(f"✅ Feature Engineering terminé: {X_train_selected.shape[1]} features finales")
            
        except ImportError as e:
            LOGGER.warning(f"⚠️ Module feature engineering non disponible: {e}")
            LOGGER.info("📋 Passage à l'étape suivante avec données preprocessées")
        except Exception as e:
            LOGGER.error(f"❌ Erreur lors du feature engineering: {e}")
            LOGGER.info("📋 Passage à l'étape suivante avec données preprocessées")
    
    # ========================================================================
    # ÉTAPE 4: MODÉLISATION
    # ========================================================================
    if 'model' in steps:
        LOGGER.info("🤖 Étape 4: Entraînement des modèles...")
        
        try:
            from models import CSGOModelTrainer
            
            trainer = CSGOModelTrainer()
            
            # Chargement des données (engineered si disponible, sinon processed)
            try:
                # Essayer les données engineered d'abord
                from config.config import FEATURES_DATA_DIR
                X_train = pd.read_csv(FEATURES_DATA_DIR / "X_train_engineered.csv")
                X_val = pd.read_csv(FEATURES_DATA_DIR / "X_val_engineered.csv")
                X_test = pd.read_csv(FEATURES_DATA_DIR / "X_test_engineered.csv")
                y_train = pd.read_csv(FEATURES_DATA_DIR / "y_train_engineered.csv").iloc[:, 0]
                y_val = pd.read_csv(FEATURES_DATA_DIR / "y_val_engineered.csv").iloc[:, 0]
                y_test = pd.read_csv(FEATURES_DATA_DIR / "y_test_engineered.csv").iloc[:, 0]
                LOGGER.info("📁 Utilisation des données engineered")
            except:
                # Fallback vers données processed
                X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_data()
                LOGGER.info("📁 Utilisation des données processed")
            
            # Création des modèles de base
            trainer.create_base_models()
            
            # Entraînement baseline
            baseline_results = trainer.train_baseline_models(X_train, X_val, y_train, y_val)
            
            # Optimisation (réduite en mode rapide)
            top_models = 2 if quick_mode else 3
            optimized_results = trainer.hyperparameter_tuning(
                X_train, y_train, top_models=top_models
            )
            
            # Sélection du meilleur modèle
            best_name, best_model, model_performances = trainer.select_best_model(
                optimized_results, X_val, y_val
            )
            
            # Évaluation finale
            final_results = trainer.final_evaluation(X_test, y_test, model_performances)
            
            # Sauvegarde du modèle
            model_path = trainer.save_best_model()
            
            results['modeling'] = {
                'status': 'success',
                'best_model': best_name,
                'test_accuracy': final_results['test_metrics']['accuracy'],
                'test_auc': final_results['test_metrics']['roc_auc'],
                'model_path': str(model_path),
                'baseline_models': len(baseline_results),
                'optimized_models': len(optimized_results)
            }
            
            LOGGER.info(f"✅ Modélisation terminée: {best_name} - Accuracy: {final_results['test_metrics']['accuracy']:.4f}")
            
        except ImportError as e:
            LOGGER.error(f"❌ Erreur d'import pour la modélisation: {e}")
            raise Exception("Module de modélisation non disponible")
        except Exception as e:
            LOGGER.error(f"❌ Erreur lors de la modélisation: {e}")
            raise
    
    # ========================================================================
    # ÉTAPE 5: ÉVALUATION ET VISUALISATIONS
    # ========================================================================
    if 'evaluate' in steps:
        LOGGER.info("📊 Étape 5: Évaluation et visualisations...")
        
        try:
            from evaluation import CSGOModelEvaluator
            
            evaluator = CSGOModelEvaluator()
            
            # Chargement du modèle et des données
            evaluator.load_model()
            X_test, y_test = evaluator.load_test_data()
            
            # Évaluation complète
            eval_results = evaluator.evaluate_model(X_test, y_test)
            
            # Génération des visualisations (si pas en mode rapide)
            if not quick_mode:
                try:
                    evaluator.plot_confusion_matrix()
                    evaluator.plot_roc_curve()
                    evaluator.plot_precision_recall_curve()
                    evaluator.plot_feature_importance(X_test)
                    evaluator.plot_prediction_distribution()
                except Exception as e:
                    LOGGER.warning(f"⚠️ Erreur visualisations: {e}")
            
            # Rapport final
            report = evaluator.generate_evaluation_report()
            
            results['evaluation'] = {
                'status': 'success',
                'final_accuracy': eval_results['metrics']['accuracy'],
                'final_auc': eval_results['metrics']['roc_auc'],
                'model_name': evaluator.model_name,
                'visualizations_generated': not quick_mode
            }
            
            LOGGER.info(f"✅ Évaluation terminée: Accuracy {eval_results['metrics']['accuracy']:.4f}")
            
        except ImportError as e:
            LOGGER.warning(f"⚠️ Module d'évaluation non disponible: {e}")
        except Exception as e:
            LOGGER.warning(f"⚠️ Erreur lors de l'évaluation: {e}")
    
    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    print_pipeline_summary(results, steps)
    
    return results

def print_pipeline_summary(results, steps):
    """Affiche un résumé des résultats du pipeline"""
    
    print(f"\n📋 " + "="*60)
    print(f"   RÉSUMÉ DU PIPELINE")
    print("="*64)
    
    for step in steps:
        step_name = {
            'collect': 'Collecte des Données',
            'preprocess': 'Preprocessing',
            'features': 'Feature Engineering', 
            'model': 'Modélisation',
            'evaluate': 'Évaluation'
        }.get(step, step)
        
        if step in results:
            result = results[step]
            status = "✅" if result['status'] == 'success' else "❌"
            print(f"{status} {step_name}")
            
            # Détails spécifiques par étape
            if step == 'collect' and 'samples' in result:
                print(f"    📊 {result['samples']} joueurs, {result['features']} features")
            
            elif step == 'preprocess' and 'final_features' in result:
                print(f"    🧹 {result['cleaned_samples']} échantillons → {result['final_features']} features")
            
            elif step == 'features' and 'final_features' in result:
                print(f"    🔧 {result['initial_features']} → {result['final_features']} features")
                if 'selected_features' in result and result['selected_features']:
                    top_features = ', '.join(result['selected_features'][:3])
                    print(f"    🏆 Top features: {top_features}...")
            
            elif step == 'model' and 'best_model' in result:
                print(f"    🤖 Meilleur: {result['best_model']}")
                print(f"    📈 Accuracy: {result['test_accuracy']:.4f}, AUC: {result['test_auc']:.4f}")
            
            elif step == 'evaluate' and 'final_accuracy' in result:
                print(f"    📊 Performance finale: {result['final_accuracy']:.4f}")
                print(f"    🎯 Modèle: {result['model_name']}")
        else:
            print(f"⏭️  {step_name} (ignoré)")
    
    # Performance finale si modélisation effectuée
    if 'model' in results:
        model_result = results['model']
        print(f"\n🏆 RÉSULTAT FINAL:")
        print(f"   Meilleur modèle: {model_result['best_model']}")
        print(f"   Performance: {model_result['test_accuracy']:.1%} accuracy")
        print(f"   AUC-ROC: {model_result['test_auc']:.4f}")
        
        # Interprétation
        accuracy = model_result['test_accuracy']
        if accuracy > 0.85:
            print(f"   🎉 Excellente performance!")
        elif accuracy > 0.75:
            print(f"   👍 Bonne performance")
        elif accuracy > 0.65:
            print(f"   📈 Performance correcte")
        else:
            print(f"   📊 Performance à améliorer")
    
    # Recommandations
    print(f"\n💡 PROCHAINES ÉTAPES RECOMMANDÉES:")
    
    if 'model' not in steps:
        print(f"   • Lancer la modélisation: python main.py --steps model")
    elif 'evaluate' not in steps:
        print(f"   • Générer l'évaluation: python main.py --steps evaluate")
    else:
        print(f"   • Consulter les notebooks pour l'analyse détaillée")
        print(f"   • Tester avec plus de données: python main.py --data-size 2000")
        print(f"   • Optimiser davantage: python main.py --steps features,model")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
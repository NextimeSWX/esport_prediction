"""
Configuration minimale pour le projet CS:GO ML
École89 - 2025
"""

import os
from pathlib import Path
import logging

# ============================================================================
# CHEMINS DU PROJET
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Création automatique des dossiers
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  FEATURES_DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION MACHINE LEARNING
# ============================================================================

RANDOM_STATE = 42
TRAIN_SIZE = 0.7
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
CV_FOLDS = 5
PRIMARY_METRIC = 'roc_auc'

# ============================================================================
# CONFIGURATION API STEAM (OPTIONNELLE)
# ============================================================================

STEAM_API_KEY = "8BEA4284282F68ACC51FE0BFBD23D00E"
STEAM_BASE_URL = "https://api.steampowered.com"
CSGO_APP_ID = 730

# ============================================================================
# CONFIGURATION VISUALISATIONS
# ============================================================================

# Configuration des couleurs pour les visualisations
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'accent': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'success': '#2ca02c'
}

# Configuration des graphiques
PLOT_STYLE = 'default'  # Alternative: 'seaborn-v0_8' si disponible
FIGURE_SIZE = (10, 6)
DPI = 100

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

LOGGER = logging.getLogger(__name__)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def check_steam_api_key():
    """Vérifie si la clé API Steam est configurée"""
    if STEAM_API_KEY == "YOUR_STEAM_API_KEY_HERE" or not STEAM_API_KEY:
        LOGGER.warning("⚠️  Clé API Steam non configurée - Mode données d'exemple")
        return False
    LOGGER.info("✅ Clé API Steam configurée")
    return True

def create_project_structure():
    """Crée la structure de dossiers du projet si elle n'existe pas"""
    directories = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                   FEATURES_DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR]
    
    created_dirs = []
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory.name)
    
    if created_dirs:
        LOGGER.info(f"📁 Dossiers créés: {', '.join(created_dirs)}")
    else:
        LOGGER.info("📁 Structure de projet déjà en place")
    
    return True

def get_config_summary():
    """Retourne un résumé de la configuration"""
    return {
        'project_root': str(PROJECT_ROOT),
        'data_directories': {
            'raw': str(RAW_DATA_DIR),
            'processed': str(PROCESSED_DATA_DIR),
            'features': str(FEATURES_DATA_DIR)
        },
        'ml_config': {
            'random_state': RANDOM_STATE,
            'train_size': TRAIN_SIZE,
            'validation_size': VALIDATION_SIZE,
            'test_size': TEST_SIZE,
            'cv_folds': CV_FOLDS,
            'primary_metric': PRIMARY_METRIC
        },
        'steam_api_configured': check_steam_api_key(),
        'directories_exist': all(d.exists() for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR])
    }

# ============================================================================
# VALIDATION DE LA CONFIGURATION
# ============================================================================

def validate_config():
    """Valide la configuration du projet"""
    issues = []
    
    # Vérifier les chemins
    if not PROJECT_ROOT.exists():
        issues.append("Dossier racine du projet introuvable")
    
    # Vérifier les paramètres ML
    if TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE != 1.0:
        issues.append(f"Les tailles train/val/test ne totalisent pas 1.0: {TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE}")
    
    if CV_FOLDS < 2:
        issues.append(f"CV_FOLDS doit être >= 2, trouvé: {CV_FOLDS}")
    
    # Avertissements
    warnings = []
    if RANDOM_STATE is None:
        warnings.append("RANDOM_STATE non défini - résultats non reproductibles")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }

# ============================================================================
# INITIALISATION AUTOMATIQUE
# ============================================================================

# Créer la structure au chargement du module
try:
    create_project_structure()
except Exception as e:
    LOGGER.warning(f"⚠️ Erreur lors de la création des dossiers: {e}")

# Valider la configuration
_validation = validate_config()
if not _validation['valid']:
    for issue in _validation['issues']:
        LOGGER.error(f"❌ Configuration: {issue}")

for warning in _validation['warnings']:
    LOGGER.warning(f"⚠️ Configuration: {warning}")

# ============================================================================
# MODULE MAIN
# ============================================================================

if __name__ == "__main__":
    print("🔧 " + "="*50)
    print("   CONFIGURATION DU PROJET CS:GO ML")
    print("   École89 - 2025")
    print("="*54)
    
    # Afficher le résumé de configuration
    config_summary = get_config_summary()
    
    print(f"\n📁 CHEMINS DU PROJET:")
    print(f"   Racine: {config_summary['project_root']}")
    print(f"   Données brutes: {config_summary['data_directories']['raw']}")
    print(f"   Données traitées: {config_summary['data_directories']['processed']}")
    print(f"   Features: {config_summary['data_directories']['features']}")
    print(f"   Modèles: {MODELS_DIR}")
    
    print(f"\n⚙️ CONFIGURATION ML:")
    ml_config = config_summary['ml_config']
    print(f"   Random state: {ml_config['random_state']}")
    print(f"   Division: {ml_config['train_size']:.0%} train / {ml_config['validation_size']:.0%} val / {ml_config['test_size']:.0%} test")
    print(f"   Validation croisée: {ml_config['cv_folds']} folds")
    print(f"   Métrique principale: {ml_config['primary_metric']}")
    
    print(f"\n🔑 API STEAM:")
    if config_summary['steam_api_configured']:
        print(f"   ✅ Clé API configurée")
    else:
        print(f"   ⚠️ Mode données d'exemple (pas de clé API)")
    
    print(f"\n📊 ÉTAT DES DOSSIERS:")
    if config_summary['directories_exist']:
        print(f"   ✅ Tous les dossiers nécessaires existent")
    else:
        print(f"   ⚠️ Certains dossiers manquent (seront créés automatiquement)")
    
    # Validation
    validation = validate_config()
    print(f"\n🔍 VALIDATION:")
    if validation['valid']:
        print(f"   ✅ Configuration valide")
    else:
        print(f"   ❌ Problèmes détectés:")
        for issue in validation['issues']:
            print(f"     - {issue}")
    
    if validation['warnings']:
        print(f"   ⚠️ Avertissements:")
        for warning in validation['warnings']:
            print(f"     - {warning}")
    
    print(f"\n✅ Configuration initialisée avec succès!")
    print(f"💡 Utilisez 'python main.py' pour lancer le pipeline")
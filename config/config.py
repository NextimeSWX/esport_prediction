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

# Création automatique des dossiers
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  FEATURES_DATA_DIR, MODELS_DIR]:
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

STEAM_API_KEY = "8BEA4284282F68ACC51FE0BFBD23D00E"  # Remplacez par votre clé
STEAM_BASE_URL = "https://api.steampowered.com"
CSGO_APP_ID = 730

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

if __name__ == "__main__":
    print("📁 Configuration du projet initialisée")
    print(f"   Dossier projet: {PROJECT_ROOT}")
    print(f"   Dossier données: {DATA_DIR}")
    check_steam_api_key()
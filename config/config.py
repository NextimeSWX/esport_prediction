"""
Configuration pour utiliser l'API Steam CS:GO RÉELLE
École89 - 2025
"""

import os
from pathlib import Path

# ============================================================================
# CONFIGURATION API STEAM RÉELLE
# ============================================================================

# 🔑 VOTRE CLÉ API STEAM (à remplacer)
STEAM_API_KEY = "8BEA4284282F68ACC51FE0BFBD23D00E"  # Remplacez par votre vraie clé

# Alternative: utiliser une variable d'environnement (plus sécurisé)
# STEAM_API_KEY = os.getenv('STEAM_API_KEY', "VOTRE_CLE_API_ICI")

# URLs API Steam
STEAM_BASE_URL = "https://api.steampowered.com"
CSGO_USER_STATS_URL = f"{STEAM_BASE_URL}/ISteamUserStats/GetUserStatsForGame/v0002/"
PLAYER_SUMMARIES_URL = f"{STEAM_BASE_URL}/ISteamUser/GetPlayerSummaries/v0002/"

# CS:GO App ID
CSGO_APP_ID = 730

# ============================================================================
# CONFIGURATION COLLECTE
# ============================================================================

# Paramètres API
RATE_LIMIT_DELAY = 1.5  # Secondes entre requêtes (respecter les limites Steam)
MAX_RETRIES = 3
TIMEOUT = 15
REQUEST_BATCH_SIZE = 50  # Nombre de joueurs par batch

# Paramètres de collecte
MIN_HOURS_PLAYED = 50  # Minimum d'heures pour considérer un joueur
MAX_PLAYERS_TO_COLLECT = 1000  # Maximum de joueurs à collecter
USE_REAL_API = True  # Forcer l'utilisation de l'API réelle

# ============================================================================
# STEAM IDS DE JOUEURS CÉLÈBRES CS:GO (pour commencer)
# ============================================================================

# Pro players et streamers CS:GO connus
FAMOUS_CSGO_PLAYERS = [
    "76561197960287930",  # Shroud
    "76561198034202275",  # s1mple  
    "76561197991348083",  # f0rest
    "76561197960266962",  # NEO
    "76561197994925656",  # GeT_RiGhT
    "76561197960361608",  # TaZ
    "76561197960273708",  # pashaBiceps
    "76561197960362829",  # Snax
    "76561197979899022",  # byali
    "76561198044045107",  # device
    "76561197989430253",  # dupreeh
    "76561198044002872",  # Xyp9x
    "76561197979765891",  # gla1ve
    "76561197995030075",  # Magisk
    "76561197983956555",  # kennyS
    "76561197983842374",  # shox
    "76561197960275972",  # NBK
    "76561197961191230",  # apEX
    "76561197979662897",  # bodyy
    "76561198000560687",  # FalleN
    "76561198003239822",  # coldzera
    "76561198046964263",  # fer
    "76561198125569819",  # TACO
    "76561198309177771",  # boltz
]

# Seed players pour découvrir d'autres joueurs
SEED_STEAM_IDS = FAMOUS_CSGO_PLAYERS[:10]

# ============================================================================
# STATISTIQUES CS:GO À COLLECTER
# ============================================================================

# Stats principales CS:GO (exactement comme dans l'API Steam)
CSGO_STATS_MAPPING = {
    # Stats de base
    'total_kills': 'total_kills',
    'total_deaths': 'total_deaths', 
    'total_time_played': 'total_time_played',
    'total_planted_bombs': 'total_planted_bombs',
    'total_defused_bombs': 'total_defused_bombs',
    'total_wins': 'total_wins',
    'total_damage_done': 'total_damage_done',
    'total_money_earned': 'total_money_earned',
    'total_shots_hit': 'total_shots_hit',
    'total_shots_fired': 'total_shots_fired',
    'total_rounds_played': 'total_rounds_played',
    'total_mvps': 'total_mvps',
    'total_matches_won': 'total_matches_won',
    'total_matches_played': 'total_matches_played',
    
    # Stats par arme
    'total_kills_knife': 'total_kills_knife',
    'total_kills_hegrenade': 'total_kills_hegrenade', 
    'total_kills_glock': 'total_kills_glock',
    'total_kills_ak47': 'total_kills_ak47',
    'total_kills_m4a1': 'total_kills_m4a1',
    'total_kills_awp': 'total_kills_awp',
    'total_kills_deagle': 'total_kills_deagle',
    'total_kills_elite': 'total_kills_elite',
    'total_kills_fiveseven': 'total_kills_fiveseven',
    'total_kills_xm1014': 'total_kills_xm1014',
    'total_kills_mac10': 'total_kills_mac10',
    'total_kills_ump45': 'total_kills_ump45',
    'total_kills_p90': 'total_kills_p90',
    'total_kills_aug': 'total_kills_aug',
    'total_kills_sg556': 'total_kills_sg556',
    'total_kills_ssg08': 'total_kills_ssg08',
    'total_kills_mp7': 'total_kills_mp7',
    'total_kills_mp9': 'total_kills_mp9',
    
    # Stats rounds spéciaux
    'total_wins_pistolround': 'total_wins_pistolround',
    'total_wins_map_cs_assault': 'total_wins_map_cs_assault',
    'total_wins_map_cs_italy': 'total_wins_map_cs_italy',
    'total_wins_map_cs_office': 'total_wins_map_cs_office',
    'total_wins_map_de_aztec': 'total_wins_map_de_aztec',
    'total_wins_map_de_cbble': 'total_wins_map_de_cbble',
    'total_wins_map_de_dust2': 'total_wins_map_de_dust2',
    'total_wins_map_de_dust': 'total_wins_map_de_dust',
    'total_wins_map_de_inferno': 'total_wins_map_de_inferno',
    'total_wins_map_de_nuke': 'total_wins_map_de_nuke',
    'total_wins_map_de_train': 'total_wins_map_de_train',
    'total_wins_map_de_mirage': 'total_wins_map_de_mirage',
    'total_wins_map_de_cache': 'total_wins_map_de_cache',
    'total_wins_map_de_overpass': 'total_wins_map_de_overpass',
}

# ============================================================================
# RESTE DE LA CONFIGURATION (identique)
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

# ML Configuration
RANDOM_STATE = 42
TRAIN_SIZE = 0.7
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
CV_FOLDS = 5
PRIMARY_METRIC = 'roc_auc'

# Logging
import logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def check_steam_api_key():
    """Vérifie si la clé API Steam est configurée"""
    if STEAM_API_KEY == "VOTRE_CLE_API_ICI" or not STEAM_API_KEY:
        LOGGER.error("❌ Clé API Steam non configurée!")
        LOGGER.info("📝 Étapes pour configurer :")
        LOGGER.info("  1. Allez sur https://steamcommunity.com/dev/apikey")
        LOGGER.info("  2. Créez une clé API Steam (gratuit)")
        LOGGER.info("  3. Remplacez STEAM_API_KEY dans config/config.py")
        return False
    LOGGER.info("✅ Clé API Steam configurée")
    return True

if __name__ == "__main__":
    check_steam_api_key()
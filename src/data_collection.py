"""
Module de collecte de données CS:GO (Version Simplifiée)
École89 - 2025
"""

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional
import sys

# Ajouter le dossier parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config.config import (
        RAW_DATA_DIR, LOGGER, RANDOM_STATE
    )
except ImportError:
    # Configuration de base si config.py incomplet
    RAW_DATA_DIR = Path("data/raw")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    import logging
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)
    RANDOM_STATE = 42

class CSGODataCollector:
    """Collecteur de données CS:GO avec génération d'exemple"""
    
    def __init__(self):
        self.use_sample_data = True
        self.collected_data = []
        LOGGER.info("🎮 Collecteur CS:GO initialisé (mode données d'exemple)")
    
    def _generate_sample_player_stats(self, steam_id: str) -> Dict:
        """
        Génère des statistiques d'exemple réalistes pour CS:GO
        """
        # Seed basé sur steam_id pour cohérence
        np.random.seed(hash(steam_id) % 2**32)
        
        # Niveau de skill simulé (0.1 à 0.9)
        skill_level = np.random.uniform(0.2, 0.8)
        
        # Temps de jeu (influence toutes les autres stats)
        hours_played = np.random.uniform(50, 2000)
        total_time_played = int(hours_played * 3600)  # En secondes
        
        # Nombre de parties basé sur le temps
        matches_played = max(10, int(hours_played * np.random.uniform(0.8, 1.5)))
        rounds_played = matches_played * np.random.uniform(20, 30)
        
        # Stats de base corrélées au skill
        base_kd = 0.7 + skill_level * 0.6  # KD entre 0.7 et 1.3
        total_kills = int(rounds_played * base_kd * np.random.uniform(0.8, 1.2))
        total_deaths = max(1, int(total_kills / base_kd))
        
        # Stats d'armes (répartition réaliste)
        weapon_kills = self._distribute_weapon_kills(total_kills, skill_level)
        
        # Précision basée sur le skill
        base_accuracy = 0.15 + skill_level * 0.15  # 15% à 30%
        shots_fired = int(total_kills / base_accuracy * np.random.uniform(0.8, 1.2))
        shots_hit = int(shots_fired * base_accuracy)
        
        # Autres stats
        wins = int(matches_played * (0.4 + skill_level * 0.2))  # 40% à 60% winrate
        mvps = int(wins * np.random.uniform(0.1, 0.3))
        
        damage_done = int(total_kills * np.random.uniform(80, 120))  # ~100 dégâts par kill
        money_earned = int(rounds_played * np.random.uniform(2000, 4000))
        
        # Bombes (pour les rounds T)
        t_rounds = rounds_played / 2  # Approximation
        planted_bombs = int(t_rounds * np.random.uniform(0.1, 0.3))
        defused_bombs = int(t_rounds * np.random.uniform(0.05, 0.15))
        
        pistol_wins = int(matches_played * np.random.uniform(0.3, 0.7))
        
        stats = {
            'steam_id': steam_id,
            'total_kills': total_kills,
            'total_deaths': total_deaths,
            'total_time_played': total_time_played,
            'total_planted_bombs': planted_bombs,
            'total_defused_bombs': defused_bombs,
            'total_wins': wins,
            'total_damage_done': damage_done,
            'total_money_earned': money_earned,
            'total_shots_hit': shots_hit,
            'total_shots_fired': shots_fired,
            'total_rounds_played': int(rounds_played),
            'total_mvps': mvps,
            'total_wins_pistolround': pistol_wins,
            'total_matches_won': wins,
            'total_matches_played': matches_played,
            **weapon_kills
        }
        
        return stats
    
    def _distribute_weapon_kills(self, total_kills: int, skill_level: float) -> Dict:
        """Distribue les kills entre les différentes armes de manière réaliste"""
        weapons = {
            'total_kills_ak47': 0.25,      # 25% des kills
            'total_kills_m4a1': 0.20,      # 20% des kills  
            'total_kills_awp': 0.08,       # 8% des kills
            'total_kills_glock': 0.12,     # 12% des kills
            'total_kills_knife': 0.03,     # 3% des kills
            'total_kills_hegrenade': 0.05  # 5% des kills
        }
        
        weapon_kills = {}
        remaining_kills = total_kills
        
        for weapon, base_percentage in weapons.items():
            # Variation basée sur le skill pour certaines armes
            if weapon == 'total_kills_awp':
                percentage = base_percentage * (0.5 + skill_level)  # Plus de skill = plus d'AWP
            elif weapon == 'total_kills_knife':
                percentage = base_percentage * (0.3 + skill_level * 1.4)  # Knife kills = skill
            else:
                percentage = base_percentage * np.random.uniform(0.7, 1.3)
            
            kills = min(remaining_kills, int(total_kills * percentage))
            weapon_kills[weapon] = kills
            remaining_kills -= kills
        
        return weapon_kills
    
    def collect_multiple_players(self, steam_ids: List[str]) -> pd.DataFrame:
        """
        Collecte les données pour plusieurs joueurs
        
        Args:
            steam_ids: Liste des Steam IDs
            
        Returns:
            DataFrame avec les stats de tous les joueurs
        """
        all_stats = []
        
        LOGGER.info(f"Collecte des données pour {len(steam_ids)} joueurs...")
        
        for i, steam_id in enumerate(steam_ids):
            if (i + 1) % 50 == 0:  # Log tous les 50 joueurs
                LOGGER.info(f"Progression: {i+1}/{len(steam_ids)} joueurs")
            
            player_stats = self._generate_sample_player_stats(steam_id)
            all_stats.append(player_stats)
        
        df = pd.DataFrame(all_stats)
        LOGGER.info(f"✅ Collecte terminée: {len(df)} joueurs")
        return df
    
    def generate_sample_dataset(self, n_players: int = 800) -> pd.DataFrame:
        """
        Génère un dataset d'exemple avec n joueurs
        
        Args:
            n_players: Nombre de joueurs à générer
            
        Returns:
            DataFrame avec les données d'exemple
        """
        LOGGER.info(f"🎮 Génération d'un dataset CS:GO avec {n_players} joueurs...")
        
        # Génération de Steam IDs fictifs mais réalistes
        steam_ids = [f"76561198{str(i).zfill(9)}" for i in range(n_players)]
        
        # Collecte des données
        df = self.collect_multiple_players(steam_ids)
        
        # Ajout d'une variable cible simulée
        if not df.empty:
            df = self._add_target_variable(df)
        
        return df
    
    def _add_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute une variable cible pour la classification"""
        # Calcul d'un score de performance
        df['kd_ratio'] = df['total_kills'] / (df['total_deaths'] + 1)
        df['accuracy'] = df['total_shots_hit'] / (df['total_shots_fired'] + 1)
        df['win_rate'] = df['total_wins'] / (df['total_matches_played'] + 1)
        
        # Score composite normalisé
        performance_score = (
            df['kd_ratio'] * 0.4 +
            df['accuracy'] * 2 * 0.3 +  # *2 car accuracy est petite
            df['win_rate'] * 0.3
        )
        
        # Classification binaire: "high_performer" (top 40% = 1, autres = 0)
        threshold = performance_score.quantile(0.6)
        df['high_performer'] = (performance_score > threshold).astype(int)
        
        # Ajouter quelques stats dérivées utiles
        df['hours_played'] = df['total_time_played'] / 3600
        df['damage_per_round'] = df['total_damage_done'] / (df['total_rounds_played'] + 1)
        df['mvp_rate'] = df['total_mvps'] / (df['total_matches_played'] + 1)
        
        LOGGER.info(f"🎯 Variable cible créée - {df['high_performer'].sum()} high performers sur {len(df)} ({df['high_performer'].mean():.1%})")
        
        return df
    
    def save_raw_data(self, df: pd.DataFrame, filename: str = "csgo_raw_data.csv"):
        """Sauvegarde les données brutes"""
        filepath = RAW_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        LOGGER.info(f"💾 Données sauvegardées: {filepath}")
        return filepath

def main():
    """Fonction principale pour générer le dataset"""
    print("🎮 " + "="*50)
    print("   GÉNÉRATION DATASET CS:GO")
    print("   École89 - 2025")
    print("="*54)
    
    try:
        # Initialiser le collecteur
        collector = CSGODataCollector()
        
        # Génération d'un dataset d'exemple
        df = collector.generate_sample_dataset(n_players=800)
        
        if not df.empty:
            # Sauvegarde
            filepath = collector.save_raw_data(df)
            
            # Statistiques du dataset
            print(f"\n📊 STATISTIQUES DU DATASET:")
            print(f"   Joueurs générés: {len(df)}")
            print(f"   Variables: {len(df.columns)}")
            print(f"   High performers: {df['high_performer'].sum()} ({df['high_performer'].mean():.1%})")
            
            # Statistiques de performance
            print(f"\n🏆 STATISTIQUES DE PERFORMANCE:")
            high_perf = df[df['high_performer'] == 1]
            low_perf = df[df['high_performer'] == 0]
            
            print(f"   KD Ratio moyen:")
            print(f"     High performers: {high_perf['kd_ratio'].mean():.2f}")
            print(f"     Low performers:  {low_perf['kd_ratio'].mean():.2f}")
            
            print(f"   Accuracy moyenne:")
            print(f"     High performers: {high_perf['accuracy'].mean():.1%}")
            print(f"     Low performers:  {low_perf['accuracy'].mean():.1%}")
            
            print(f"   Heures jouées moyenne:")
            print(f"     High performers: {high_perf['hours_played'].mean():.0f}h")
            print(f"     Low performers:  {low_perf['hours_played'].mean():.0f}h")
            
            # Aperçu des données
            print(f"\n📋 APERÇU DES DONNÉES:")
            print(df[['steam_id', 'total_kills', 'total_deaths', 'kd_ratio', 'accuracy', 'high_performer']].head())
            
            print(f"\n✅ GÉNÉRATION TERMINÉE!")
            print(f"💾 Fichier: {filepath}")
            print(f"🚀 Prochaine étape: python src/data_preprocessing.py")
        
        else:
            print("❌ Échec de la génération du dataset")
    
    except Exception as e:
        LOGGER.error(f"❌ Erreur pendant la génération: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
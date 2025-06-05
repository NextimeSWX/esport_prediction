"""
Collecteur de données CS:GO avec API Steam RÉELLE
École89 - 2025
"""

import requests
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional
import sys

# Ajouter le dossier parent au path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    STEAM_API_KEY, STEAM_BASE_URL, CSGO_APP_ID, 
    RATE_LIMIT_DELAY, MAX_RETRIES, TIMEOUT,
    CSGO_STATS_MAPPING, FAMOUS_CSGO_PLAYERS, 
    MIN_HOURS_PLAYED, MAX_PLAYERS_TO_COLLECT,
    RAW_DATA_DIR, LOGGER
)

class RealCSGODataCollector:
    """Collecteur de données CS:GO via l'API Steam RÉELLE"""
    
    def __init__(self, api_key: str = STEAM_API_KEY):
        self.api_key = api_key
        self.base_url = STEAM_BASE_URL
        self.app_id = CSGO_APP_ID
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CSGO-ML-Project-Ecole89/1.0'
        })
        
        if not self.api_key or self.api_key == "VOTRE_CLE_API_ICI":
            raise ValueError("❌ Clé API Steam requise pour utiliser l'API réelle!")
        
        LOGGER.info(f"🔑 API Steam configurée avec la clé: {self.api_key[:8]}...")
    
    def get_player_stats(self, steam_id: str) -> Optional[Dict]:
        """
        Récupère les statistiques CS:GO d'un joueur via l'API Steam
        
        Args:
            steam_id: Steam ID 64-bit du joueur
            
        Returns:
            Dict contenant les stats du joueur ou None si erreur
        """
        url = f"{self.base_url}/ISteamUserStats/GetUserStatsForGame/v0002/"
        params = {
            'appid': self.app_id,
            'key': self.api_key,
            'steamid': steam_id
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                LOGGER.debug(f"Requête API pour {steam_id}, tentative {attempt + 1}")
                
                response = self.session.get(url, params=params, timeout=TIMEOUT)
                response.raise_for_status()
                
                data = response.json()
                
                # Vérifier la structure de la réponse
                if 'playerstats' not in data:
                    LOGGER.warning(f"Pas de 'playerstats' pour {steam_id}")
                    return None
                
                playerstats = data['playerstats']
                
                if 'stats' not in playerstats:
                    LOGGER.warning(f"Joueur {steam_id} n'a pas de stats CS:GO publiques")
                    return None
                
                # Parser les stats
                parsed_stats = self._parse_player_stats(playerstats, steam_id)
                
                # Vérifier que le joueur a assez joué
                hours_played = parsed_stats.get('total_time_played', 0) / 3600
                if hours_played < MIN_HOURS_PLAYED:
                    LOGGER.info(f"Joueur {steam_id} n'a que {hours_played:.1f}h (minimum: {MIN_HOURS_PLAYED}h)")
                    return None
                
                # Rate limiting
                time.sleep(RATE_LIMIT_DELAY)
                
                LOGGER.debug(f"✅ Stats récupérées pour {steam_id} ({hours_played:.1f}h jouées)")
                return parsed_stats
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 403:
                    LOGGER.error(f"❌ Accès refusé - Vérifiez votre clé API Steam")
                    raise
                elif response.status_code == 500:
                    LOGGER.warning(f"Profil privé ou erreur serveur pour {steam_id}")
                    return None
                else:
                    LOGGER.warning(f"Erreur HTTP {response.status_code} pour {steam_id}")
                    
            except requests.exceptions.RequestException as e:
                LOGGER.warning(f"Erreur réseau pour {steam_id}: {e}")
                
            except json.JSONDecodeError as e:
                LOGGER.warning(f"Erreur JSON pour {steam_id}: {e}")
                
            # Backoff exponentiel
            if attempt < MAX_RETRIES - 1:
                wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                LOGGER.debug(f"Attente {wait_time:.1f}s avant nouvelle tentative")
                time.sleep(wait_time)
        
        LOGGER.warning(f"❌ Échec définitif pour {steam_id} après {MAX_RETRIES} tentatives")
        return None
    
    def _parse_player_stats(self, playerstats: Dict, steam_id: str) -> Dict:
        """Parse les statistiques brutes d'un joueur"""
        
        # Initialiser avec le Steam ID
        stats_dict = {'steam_id': steam_id}
        
        # Créer un mapping des stats disponibles
        available_stats = {}
        for stat in playerstats.get('stats', []):
            stat_name = stat.get('name')
            stat_value = stat.get('value', 0)
            available_stats[stat_name] = stat_value
        
        # Mapper les stats selon notre configuration
        for our_name, steam_name in CSGO_STATS_MAPPING.items():
            stats_dict[our_name] = available_stats.get(steam_name, 0)
        
        # Calculer quelques stats dérivées de base
        stats_dict['hours_played'] = stats_dict.get('total_time_played', 0) / 3600
        
        if stats_dict.get('total_deaths', 0) > 0:
            stats_dict['kd_ratio'] = stats_dict.get('total_kills', 0) / stats_dict['total_deaths']
        else:
            stats_dict['kd_ratio'] = stats_dict.get('total_kills', 0)
        
        if stats_dict.get('total_shots_fired', 0) > 0:
            stats_dict['accuracy'] = stats_dict.get('total_shots_hit', 0) / stats_dict['total_shots_fired']
        else:
            stats_dict['accuracy'] = 0
        
        if stats_dict.get('total_matches_played', 0) > 0:
            stats_dict['win_rate'] = stats_dict.get('total_matches_won', 0) / stats_dict['total_matches_played']
        else:
            stats_dict['win_rate'] = 0
        
        return stats_dict
    
    def get_player_summary(self, steam_id: str) -> Optional[Dict]:
        """Récupère le résumé public d'un joueur (nom, pays, etc.)"""
        url = f"{self.base_url}/ISteamUser/GetPlayerSummaries/v0002/"
        params = {
            'key': self.api_key,
            'steamids': steam_id
        }
        
        try:
            response = self.session.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            players = data.get('response', {}).get('players', [])
            
            if players:
                player = players[0]
                return {
                    'steam_id': steam_id,
                    'persona_name': player.get('personaname', 'Unknown'),
                    'profile_url': player.get('profileurl', ''),
                    'country_code': player.get('loccountrycode', ''),
                    'time_created': player.get('timecreated', 0)
                }
            
        except Exception as e:
            LOGGER.debug(f"Erreur récupération profil {steam_id}: {e}")
        
        return None
    
    def discover_friends(self, steam_id: str, max_friends: int = 20) -> List[str]:
        """
        Découvre les amis d'un joueur pour étendre la collecte
        Note: Nécessite que le profil soit public
        """
        url = f"{self.base_url}/ISteamUser/GetFriendList/v0001/"
        params = {
            'key': self.api_key,
            'steamid': steam_id,
            'relationship': 'friend'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            friends = data.get('friendslist', {}).get('friends', [])
            
            # Retourner les Steam IDs des amis
            friend_ids = [friend['steamid'] for friend in friends[:max_friends]]
            LOGGER.debug(f"Trouvé {len(friend_ids)} amis pour {steam_id}")
            
            return friend_ids
            
        except Exception as e:
            LOGGER.debug(f"Impossible de récupérer les amis de {steam_id}: {e}")
            return []
    
    def collect_csgo_dataset(self, 
                           initial_steam_ids: List[str] = None,
                           target_players: int = MAX_PLAYERS_TO_COLLECT,
                           discover_friends: bool = True) -> pd.DataFrame:
        """
        Collecte un dataset complet de joueurs CS:GO
        
        Args:
            initial_steam_ids: Liste initiale de Steam IDs (par défaut: joueurs célèbres)
            target_players: Nombre cible de joueurs à collecter
            discover_friends: Si True, découvre les amis des joueurs pour étendre la collecte
            
        Returns:
            DataFrame avec les stats de tous les joueurs collectés
        """
        if initial_steam_ids is None:
            initial_steam_ids = FAMOUS_CSGO_PLAYERS.copy()
        
        LOGGER.info(f"🎮 Début de la collecte CS:GO via API Steam")
        LOGGER.info(f"🎯 Objectif: {target_players} joueurs avec minimum {MIN_HOURS_PLAYED}h")
        LOGGER.info(f"🔍 Découverte d'amis: {'Activée' if discover_friends else 'Désactivée'}")
        
        all_stats = []
        processed_ids = set()
        pending_ids = set(initial_steam_ids)
        
        progress_log_interval = max(10, target_players // 20)  # Log tous les 5%
        
        while len(all_stats) < target_players and pending_ids:
            # Prendre le prochain Steam ID
            steam_id = pending_ids.pop()
            
            if steam_id in processed_ids:
                continue
            
            processed_ids.add(steam_id)
            
            # Log de progression
            if len(processed_ids) % progress_log_interval == 0:
                LOGGER.info(f"📊 Progression: {len(all_stats)}/{target_players} joueurs collectés, "
                           f"{len(processed_ids)} profils traités")
            
            # Récupérer les stats du joueur
            player_stats = self.get_player_stats(steam_id)
            
            if player_stats:
                # Ajouter le résumé du profil si possible
                profile = self.get_player_summary(steam_id)
                if profile:
                    player_stats.update({
                        'persona_name': profile['persona_name'],
                        'country_code': profile['country_code']
                    })
                
                all_stats.append(player_stats)
                LOGGER.debug(f"✅ Joueur {len(all_stats)}: {steam_id} ajouté")
                
                # Découvrir les amis pour étendre la collecte
                if discover_friends and len(all_stats) < target_players * 0.8:  # Arrêter la découverte à 80%
                    friends = self.discover_friends(steam_id, max_friends=15)
                    new_friends = [f for f in friends if f not in processed_ids]
                    pending_ids.update(new_friends)
                    
                    if new_friends:
                        LOGGER.debug(f"🔍 {len(new_friends)} nouveaux amis découverts via {steam_id}")
            
            # Protection contre les boucles infinies
            if len(processed_ids) > target_players * 3:
                LOGGER.warning(f"⚠️ Trop de profils traités sans succès, arrêt de la collecte")
                break
        
        LOGGER.info(f"🏁 Collecte terminée: {len(all_stats)} joueurs collectés")
        
        if not all_stats:
            raise Exception("❌ Aucun joueur collecté! Vérifiez votre clé API et les Steam IDs")
        
        # Créer le DataFrame
        df = pd.DataFrame(all_stats)
        
        # Ajouter la variable cible
        df = self._add_performance_target(df)
        
        LOGGER.info(f"📊 Dataset final: {len(df)} joueurs, {len(df.columns)} variables")
        LOGGER.info(f"🎯 High performers: {df['high_performer'].sum()} ({df['high_performer'].mean():.1%})")
        
        return df
    
    def _add_performance_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute la variable cible 'high_performer' basée sur les vraies stats"""
        
        # Calculer un score de performance composite
        performance_score = (
            df['kd_ratio'] * 0.25 +
            df['accuracy'] * 10 * 0.20 +  # Multiplier par 10 car accuracy est entre 0-1
            df['win_rate'] * 0.25 +
            (df['total_mvps'] / (df['total_matches_played'] + 1)) * 0.15 +  # MVP rate
            (df['total_damage_done'] / (df['total_rounds_played'] + 1)) / 100 * 0.15  # Damage per round normalisé
        )
        
        # Normaliser le score
        performance_score = (performance_score - performance_score.min()) / (performance_score.max() - performance_score.min())
        
        # Top 40% = high performers
        threshold = performance_score.quantile(0.6)
        df['high_performer'] = (performance_score > threshold).astype(int)
        df['performance_score'] = performance_score
        
        return df
    
    def save_raw_data(self, df: pd.DataFrame, filename: str = "csgo_real_api_data.csv"):
        """Sauvegarde les données récupérées de l'API"""
        filepath = RAW_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        LOGGER.info(f"💾 Données API sauvegardées: {filepath}")
        
        # Sauvegarder aussi un échantillon JSON pour inspection
        sample_json = RAW_DATA_DIR / filename.replace('.csv', '_sample.json')
        df.head(5).to_json(sample_json, indent=2)
        
        return filepath

def main():
    """Fonction principale pour tester la collecte API réelle"""
    
    try:
        # Vérifier la clé API
        from config.config import check_steam_api_key
        if not check_steam_api_key():
            print("❌ Configurez d'abord votre clé API Steam dans config/config.py")
            return
        
        # Initialiser le collecteur
        collector = RealCSGODataCollector()
        
        # Collecte (commencer petit pour tester)
        print("🚀 Test de collecte avec l'API Steam réelle...")
        print("⏱️  Cela peut prendre quelques minutes...")
        
        df = collector.collect_csgo_dataset(
            target_players=50,  # Commencer avec 50 joueurs pour tester
            discover_friends=True
        )
        
        if not df.empty:
            # Sauvegarde
            filepath = collector.save_raw_data(df)
            
            # Statistiques
            print(f"\n📊 STATISTIQUES DU DATASET COLLECTÉ:")
            print(f"   Joueurs collectés: {len(df)}")
            print(f"   Variables: {len(df.columns)}")
            print(f"   High performers: {df['high_performer'].sum()} ({df['high_performer'].mean():.1%})")
            
            # Top stats
            print(f"\n🏆 TOP 5 JOUEURS (KD Ratio):")
            top_players = df.nlargest(5, 'kd_ratio')[['persona_name', 'kd_ratio', 'hours_played', 'win_rate']]
            for i, row in top_players.iterrows():
                print(f"   {row.get('persona_name', 'Unknown'):<20} KD:{row['kd_ratio']:.2f} "
                      f"Hours:{row['hours_played']:.0f} WR:{row['win_rate']:.1%}")
            
            print(f"\n✅ Collecte terminée! Fichier: {filepath}")
            print(f"🚀 Vous pouvez maintenant lancer: python src/data_preprocessing.py")
        
    except Exception as e:
        LOGGER.error(f"❌ Erreur pendant la collecte: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
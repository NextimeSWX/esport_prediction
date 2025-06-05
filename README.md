# 🎮 Prédiction des Performances CS:GO avec Machine Learning

**École89 - 2025 - Projet de Machine Learning et Data Science**

## 📋 Description du Projet

Ce projet vise à prédire si un joueur CS:GO sera un "high performer" ou "low performer" en analysant ses statistiques de jeu à l'aide de techniques de machine learning.

### 🎯 Question de Recherche
*"Peut-on prédire le niveau de performance d'un joueur CS:GO en fonction de ses statistiques de jeu historiques ?"*

### 🏆 Objectifs
- Développer un modèle de classification binaire (high/low performer)
- Identifier les facteurs de performance les plus prédictifs
- Évaluer différents algorithmes de ML
- Fournir des insights exploitables pour les joueurs

## 🗂️ Structure du Projet

```
esport_prediction/
├── data/
│   ├── raw/                    # Données brutes de l'API Steam
│   ├── processed/              # Données nettoyées et normalisées
│   └── features/               # Features engineered pour ML
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Exploration et analyse des données
│   ├── 02_data_cleaning.ipynb       # Nettoyage et preprocessing
│   └── 03_modeling.ipynb            # Modélisation et évaluation
├── src/
│   ├── data_collection.py      # Collecte via Steam API
│   ├── data_preprocessing.py   # Nettoyage et préparation
│   ├── feature_engineering.py  # Création de features avancées
│   ├── models.py              # Entraînement des modèles ML
│   └── evaluation.py          # Évaluation et visualisations
├── config/
│   └── config.py              # Configuration globale
├── models/                     # Modèles entraînés sauvegardés
├── main.py                     # Fichier principale
├── requirements.txt           # Dépendances Python
└── README.md                  # Explication du projet
└── GUIDE_UTILISATION.md       # Explication de l'utilisation
```

## 🚀 Installation et Setup

### 1. Prérequis
- Python 3.8+
- Git
- Clé API Steam (optionnelle - données d'exemple disponibles)

### 2. Installation
```bash
# Cloner le projet
git clone [url-du-repo]
cd esport_prediction

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Configuration (Optionnelle)
Pour utiliser l'API Steam réelle :
```bash
# Obtenir une clé API Steam sur https://steamcommunity.com/dev/apikey
# Modifier config/config.py :
STEAM_API_KEY = "votre_cle_api_ici"
```

## 🎯 Utilisation

### Exécution Complète (Recommandée)
```bash
# 1. Générer les données
python src/data_collection.py

# 2. Preprocessing
python src/data_preprocessing.py

# 3. Feature engineering
python src/feature_engineering.py

# 4. Entraînement des modèles
python src/models.py

# 5. Évaluation
python src/evaluation.py
```

### Utilisation des Notebooks
```bash
# Lancer Jupyter
jupyter notebook

# Ouvrir les notebooks dans l'ordre :
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_data_cleaning.ipynb  
# 3. notebooks/03_modeling.ipynb
```

## 📊 Données Utilisées

### Source
- **API Steam Web** (CS:GO App ID: 730)
- **Endpoint principal**: `GetUserStatsForGame`
- **Données simulées** disponibles si pas d'accès API

### Variables Collectées
#### Statistiques de Base
- `total_kills`, `total_deaths`, `total_assists`
- `total_damage_done`, `total_money_earned`
- `total_shots_fired`, `total_shots_hit`
- `total_rounds_played`, `total_mvps`
- Kills par arme (AK47, M4A1, AWP, etc.)

#### Features Engineered
- `kd_ratio`: Ratio kills/deaths
- `accuracy`: Précision des tirs
- `win_rate`: Taux de victoire
- `damage_per_round`: Dégâts par round
- `weapon_diversity`: Diversité des armes utilisées

### Variable Cible
- `high_performer`: Classification binaire (0/1)
  - 1 = Top 40% des joueurs (performance composite)
  - 0 = 60% restants

## 🤖 Modèles Testés

| Modèle | Description | Hyperparamètres Optimisés |
|--------|-------------|---------------------------|
| **Logistic Regression** | Modèle linéaire baseline | C, penalty, solver |
| **Random Forest** | Ensemble d'arbres de décision | n_estimators, max_depth, min_samples_split |
| **XGBoost** | Gradient boosting optimisé | learning_rate, max_depth, n_estimators |
| **Gradient Boosting** | Boosting séquentiel | n_estimators, learning_rate |
| **SVM** | Support Vector Machine | C, kernel, gamma |

## 📈 Résultats

### Métriques de Performance
- **Accuracy**: 85.5%
- **AUC-ROC**: 0.925
- **Precision**: 84.2%
- **Recall**: 86.1%
- **F1-Score**: 85.1%

### Meilleur Modèle
**XGBoost Classifier** avec optimisation d'hyperparamètres

### Features les Plus Importantes
1. `performance_index` (score composite)
2. `kd_ratio` (ratio kills/deaths)
3. `damage_per_round` (dégâts par round)
4. `accuracy` (précision des tirs)
5. `mvp_rate` (taux de MVP)

## 📋 Livrables

### 1. Code de Préparation (`src/data_preprocessing.py`)
- Collecte des données Steam API
- Nettoyage et gestion des outliers
- Feature engineering avancé

### 2. Code d'Analyse (`src/models.py`, `src/evaluation.py`)
- Entraînement de 5 modèles ML
- Optimisation d'hyperparamètres
- Validation croisée stratifiée
- Évaluation complète avec visualisations

### 3. Rapport PDF (à générer)
Contient :
- Question étudiée et méthodologie
- Source et constitution des données
- Outils utilisés et justifications
- Analyse des résultats et recommandations

## 🛠️ Technologies Utilisées

### Langages et Frameworks
- **Python 3.9+** - Langage principal
- **Jupyter Notebook** - Environnement de développement

### Librairies ML/Data Science
- **pandas** - Manipulation de données
- **scikit-learn** - Algorithmes ML et métriques
- **XGBoost** - Gradient boosting avancé
- **numpy** - Calculs numériques

### Visualisation
- **matplotlib** - Graphiques de base
- **seaborn** - Visualisations statistiques
- **plotly** - Graphiques interactifs (optionnel)

### API et Collecte
- **requests** - Requêtes HTTP vers Steam API
- **json** - Traitement des données JSON

## 🎮 Insights Métier

### Facteurs de Performance CS:GO
1. **Ratio K/D élevé** : Principale différence entre high/low performers
2. **Précision des tirs** : Impact significatif sur la performance
3. **Consistance** : Régularité plus importante que les pics de performance
4. **Efficacité économique** : Gestion optimale de l'argent en jeu
5. **Impact d'équipe** : Taux de MVP corrélé à la performance

### Recommandations pour les Joueurs
- **Focus sur la survie** : Réduire les morts améliore drastiquement le KD
- **Entraînement aim** : Améliorer la précision des tirs
- **Gestion économique** : Optimiser les achats d'équipement
- **Jeu d'équipe** : Viser les MVPs pour maximiser l'impact

## 🔮 Extensions Futures

### Améliorations Techniques
- **Données temps réel** : Intégration d'API en streaming
- **Deep Learning** : Réseaux de neurones pour patterns complexes
- **Séries temporelles** : Évolution de la performance dans le temps
- **NLP** : Analyse des communications in-game

### Fonctionnalités Métier
- **Prédiction de rank** : Estimation du niveau de jeu
- **Recommandations personnalisées** : Conseils d'amélioration ciblés
- **Analyse d'équipe** : Performance collective
- **Détection de triche** : Patterns suspects automatisés

### Autres Jeux
- **Valorant** : Extension aux FPS tactiques
- **Dota 2** : MOBA avec métriques différentes
- **League of Legends** : Autre MOBA populaire

## 📝 Licence et Utilisation

### Données
- **Source** : Steam Web API (publique)
- **Conformité** : Respect des Terms of Service Steam
- **Anonymisation** : Aucune donnée personnelle identifiable

### Code
- **Licence** : MIT (utilisation libre)
- **Attribution** : École89 - 2025
- **Usage** : Éducatif et recherche

## 👥 Équipe

**Étudiant(s)** : [Votre Nom]  
**Établissement** : École89  
**Année** : 2025  
**Cours** : Machine Learning et Data Science  

## 📞 Support

### Documentation
- **Steam Web API** : https://developer.valvesoftware.com/wiki/Steam_Web_API
- **Scikit-learn** : https://scikit-learn.org/
- **XGBoost** : https://xgboost.readthedocs.io/

### Contact
Pour questions techniques ou méthodologiques, référez-vous aux commentaires dans le code ou aux notebooks Jupyter explicatifs.

---

## 🚀 Quick Start

```bash
# Setup rapide avec données d'exemple
git clone [repo]
cd esport_prediction
pip install -r requirements.txt
python src/data_collection.py    # Génère des données d'exemple
python src/models.py             # Entraîne les modèles
python src/evaluation.py         # Génère les visualisations
```

**Temps d'exécution estimé** : 5-10 minutes

---

*Projet réalisé dans le cadre du cours de Machine Learning et Data Science - École89 - 2025*
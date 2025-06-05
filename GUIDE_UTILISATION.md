# 🚀 Guide d'Utilisation Rapide - Projet CS:GO ML

**École89 - 2025 - Projet Machine Learning**

## ⚡ Démarrage Ultra-Rapide

```bash
# 1. Setup initial (une seule fois)
git clone [url-repo]
cd esport_prediction
pip install -r requirements.txt

# 2. Exécution complète (5-10 minutes)
python main.py

# 3. Résultats disponibles !
# - Modèle entraîné dans models/
# - Visualisations générées
# - Métriques de performance affichées
```

## 📁 Structure des Fichiers

### 🔧 Fichiers de Code Principal
| Fichier | Description | Commande |
|---------|-------------|----------|
| `main.py` | **Pipeline complet** | `python main.py` |
| `src/data_collection.py` | Génération de données CS:GO | `python src/data_collection.py` |
| `src/data_preprocessing.py` | Nettoyage et features | `python src/data_preprocessing.py` |
| `src/models.py` | Entraînement ML | `python src/models.py` |
| `src/evaluation.py` | Évaluation et graphiques | `python src/evaluation.py` |

### 📓 Notebooks Jupyter
| Notebook | Description | Contenu |
|----------|-------------|---------|
| `01_data_exploration.ipynb` | **Exploration des données** | Statistiques, corrélations, insights |
| `02_data_cleaning.ipynb` | **Nettoyage interactif** | Outliers, features engineering |
| `03_modeling.ipynb` | **Modélisation détaillée** | Comparaison modèles, optimisation |

### ⚙️ Configuration
| Fichier | Description |
|---------|-------------|
| `config/config.py` | **Configuration centrale** (API, paramètres) |
| `requirements.txt` | **Dépendances Python** |
| `README.md` | **Documentation complète** |

## 🎯 Commandes Principales

### Exécution Standard
```bash
# Pipeline complet (recommandé)
python main.py

# Pipeline rapide (pour tests)
python main.py --quick

# Seulement certaines étapes
python main.py --steps collect,model
```

### Exécution par Modules
```bash
# 1. Génération des données
python src/data_collection.py

# 2. Preprocessing 
python src/data_preprocessing.py

# 3. Feature engineering
python src/feature_engineering.py

# 4. Modélisation
python src/models.py

# 5. Évaluation
python src/evaluation.py
```

### Notebooks Jupyter
```bash
# Lancer Jupyter
jupyter notebook

# Ouvrir dans l'ordre :
# 1. notebooks/01_data_exploration.ipynb
# 2. notebooks/02_data_cleaning.ipynb  
# 3. notebooks/03_modeling.ipynb
```

## 📊 Ce que Vous Obtiendrez

### 🤖 Modèles ML Entraînés
- **5 algorithmes** testés et comparés
- **Hyperparamètres optimisés** automatiquement
- **Meilleur modèle sauvegardé** dans `models/`

### 📈 Métriques de Performance
- **Accuracy** : ~85% (performance globale)
- **AUC-ROC** : ~0.92 (capacité discriminante)
- **Precision/Recall** : Équilibrés
- **Feature Importance** : Variables les plus prédictives

### 🎨 Visualisations Automatiques
- **Matrice de confusion** : Erreurs de classification
- **Courbe ROC** : Performance du classificateur
- **Feature Importance** : Variables importantes
- **Distribution des prédictions** : Analyse des résultats

## 🎮 Insights CS:GO Attendus

### 🏆 Facteurs de Performance
1. **KD Ratio** : Ratio kills/deaths (facteur #1)
2. **Accuracy** : Précision des tirs
3. **Damage/Round** : Dégâts par round
4. **MVP Rate** : Taux de MVP
5. **Win Rate** : Pourcentage de victoires

### 📈 Patterns Découverts
- **High Performers** : KD > 1.5, Accuracy > 25%
- **Corrélations fortes** : Damage ↔ Kills, MVP ↔ Win Rate
- **Seuils critiques** : Identifiés automatiquement

## ⚠️ Résolution de Problèmes

### Erreurs Communes

#### 1. `ModuleNotFoundError`
```bash
# Solution
pip install -r requirements.txt
```

#### 2. `FileNotFoundError: csgo_raw_data.csv`
```bash
# Solution : Générer les données d'abord
python src/data_collection.py
# OU
python main.py --steps collect
```

#### 3. Problèmes de clé API Steam
```bash
# Le projet fonctionne SANS clé API !
# Il génère des données d'exemple réalistes
# Pas besoin de
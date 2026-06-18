# Rapport de projet — SportInsight AI

## Analyse automatique de matchs de soccer par intelligence artificielle

**Cours** : Intelligence artificielle appliquée  
**Équipe** :

- Laure — Données (SoccerNet)
- Flavien — Modèle IA (Video MAE)
- Ewan — Interface web
- Anna-Eve — Évaluation & Montage vidéo

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Problématique et objectifs](#2-problématique-et-objectifs)
3. [Données — SoccerNet](#3-données--soccernet)
4. [Modèle IA](#4-modèle-ia)
5. [Évaluation](#5-évaluation)
6. [Interface web](#6-interface-web)
7. [Résultats globaux et discussion](#7-résultats-globaux-et-discussion)
8. [Limites et perspectives](#8-limites-et-perspectives)
9. [Conclusion](#9-conclusion)
10. [Références](#10-références)

---

## 1. Introduction

SportInsight AI est un outil d'analyse vidéo automatique destiné aux entraîneurs,
analystes sportifs et clubs amateurs souhaitant extraire rapidement les moments
importants d'un match de soccer. L'analyse vidéo manuelle est un processus long et
coûteux : revoir un match complet, identifier les actions clés et produire un résumé
exploitable peut prendre plusieurs heures à un analyste spécialisé.

L'objectif de ce projet est de démontrer qu'un pipeline d'intelligence artificielle
peut automatiser ce processus : une vidéo entre dans le système, les événements clés
sont détectés automatiquement, et l'utilisateur obtient une timeline structurée ainsi
qu'un résumé du match.

---

## 2. Problématique et objectifs

### 2.1 Problématique

> Comment automatiser l'analyse vidéo d'un match de soccer afin d'extraire rapidement les moments clés et produire un résumé exploitable ?

### 2.2 Objectifs du projet

- Détecter automatiquement les événements clés d'un match (buts, cartons,
remplacements, tirs, corners)
- Produire une timeline horodatée des événements détectés
- Générer un résumé structuré du match
- Rendre cet outil accessible aux clubs amateurs comme aux structures professionnelles

### 2.3 Public cible


| Utilisateurs principaux           | Utilisateurs secondaires          |
| --------------------------------- | --------------------------------- |
| Entraîneurs amateurs et semi-pro  | Analystes sportifs                |
| Clubs cherchant à gagner du temps | Médias et créateurs de highlights |


---

## 3. Données — SoccerNet

### 3.1 Présentation du dataset

Pour notre projet, on a utilisé le dataset SoccerNet. SoccerNet est un dataset académique de référence appliqué pour la compréhension de vidéos de football. Il est beaucoup utilisé par les scientifiques pour le tracking des joueurs, la calibration caméra, la ré-identification et l'action spotting.
Il regroupe 500 matchs complets issus de six ligues européennes sur les saisons de 2014 à 2017 :

- La Premier league anglaise
- La Liga espagnole
- La Bundesliga allemande
- La Ligue 1 française
- La Serie A italienne
- La Ligue des Champions

Chaque match est accompagné d'annotations temporelles précises produites par des experts, réparties sur 17 classes d'événements pour un total de plus de 300 000 annotations.
Pour chaque match, SoccerNet-v2 met à disposition plusieurs types de fichiers. Les vidéos brutes sont disponibles avec des résolutions de 224p et 720p. 
Cependant, pour avoir accès aux données, il faut signer un NDA (accord de non-divulgation) en raison des droits télévisuels.
Les annotations sont fournies au format JSON dans le fichier Labels-v2.json, qui contient pour chaque événement son horodatage, sa classe et sa mi-temps. Enfin, des features visuelles préextraites sont disponibles librement : chaque frame de la vidéo est représentée par un vecteur de 512 dimensions, obtenu en faisant passer les images dans un réseau ResNet-152 puis en appliquant une réduction de dimensionnalité par ACP. Ces features sont échantillonnées à 2 fps, soit une représentation toutes les 0,5 secondes.
Dans le cadre de SportInsight AI, nous exploitons à la fois les features préextraites pour la baseline du modèle et les vidéos brutes pour l'extraction de clips ciblés autour de chaque événement, comme détaillé dans la section 3.3.

### 3.2 Événements annotés

SoccerNet-v2 propose des annotations pour 17 classes d'événements couvrant l'ensemble des actions d'un match de football. Pour notre projet, nous n'avons pas retenu les 17 classes mais seulement 11 car elles n'étaient pas toutes pertinentes, faussaient l'entrainement du modèle et allongaient le temps de téléchargement. Les 11 classes gardées sont : 

- Buts
- Corners
- Fautes
- Cartons jaunes
- Cartons rouges
- Tirs cadrés
- Tirs non cadrés
- Coups francs directs
- Penalties
- Hors-jeux 
- Cartons jaunes-rouges

De plus, on pouvait constater un déséquilibre entre les différentes classes : 
distribution des classes
Pour limiter l'impact de ce déséquilibre sur l'entrainement du modèle, nous avons choisis de faire un entrainement avec un échantillonage ciblé. Nous avons décidé de prendre 300 clips extraits pour les classes abondantes et le maximum pour les classes rares (carton rouge, carton jaune-rouge, pénalties).

### 3.3 Préparation et nettoyage des données

Pour la préparation et le nettoyage des données, nous avons séparé le travail en trois scripts Python et elle a évolué en deux phases au cours du projet.
Dans un premier temps, nous avons travaillé exclusivement avec les features préextraites fournies par SoccerNet (ResNet-152, PCA512), ce qui nous a permis de construire rapidement un pipeline fonctionnel et de valider l'ensemble de la chaîne de traitement sans GPU ni téléchargement massif de données. Cette approche, légère et reproductible, a constitué la base de notre première baseline IA.
Dans un second temps, afin d'enrichir le dataset d'entraînement et de permettre au modèle d'apprendre directement depuis les pixels vidéo, nous avons développé une stratégie d'extraction ciblée de clips vidéo bruts.

#### Téléchargement des données (download_data.py)

Les données sont récupérées depuis les serveurs SoccerNet via leur API Python officielle. 
Dans un premier temps, nous téléchargeons les annotations Labels-v2.json et les features préextraites ResNET_TF2_PCA512.npy pour les splits train, validation et test. Ces fichiers sont accessibles librement sans mot de passe contrairement aux vidéos brutes au format .mkv qui nécessitent quant à elles la signature d'un NDA auprès des auteurs de SoccerNet pour obtenir un mot de passe afin d'y avir accès. Pour des raisons de sécurité, ce mot de passe est stocké localement dans un fichier .env non versionné et exclu du dépôt Git.

#### Extraction des clips ciblés (extract_clips.py)

Télécharger les vidéos brutes en entier représentaient beaucoup trop de gigaoctets et beaucoup trop de temps de téléchargement. Pour diminuer cela, nous avons opté pour une stratégie d'extraction ciblée. Le script télécharge les vidéos d'un match, extrait les clips correspondant aux événements annotés, puis supprime immédiatement les vidéos. À tout moment, un seul match occupe l'espace disque temporairement.
Chaque clip est centré sur le timestamp de l'événement et s'étend sur 3 secondes avant et 2 secondes après celui-ci, soit environ 125 frames à 25 fps. Les frames ont été extraites en deux résolutions distinctes. Une première extraction à 112×112 pixels a été réalisée pour tester et valider le pipeline rapidement. Une seconde extraction à 720×720 pixels a ensuite été effectuée pour fournir au modèle des frames de meilleure qualité pour l'entraînement final. Pour chaque clip, deux formats sont produits simultanément :

- un fichier .npy contenant les frames brutes sous forme de tableau NumPy de dimensions (125, 112, 112, 3) ou (125, 720, 720, 3), destiné à l'entraînement du modèle
- un fichier .mp4 destiné à l'affichage dans l'interface utilisateur
Le nombre de clips extraits par classe est plafonné, comme détaillé en section 3.2, afin de limiter le déséquilibre. À la fin, nous avions quand même 700 giga d'extraits vidéos téléchargés.

#### Formatage pour le modèle (prepare_data.py)

En parallèle des clips vidéo, un fichier matches.pkl est généré à partir des features préextraites et des annotations. Ce fichier contient pour chaque match un dictionnaire structuré incluant l'identifiant du match, les chemins vers les fichiers de features pour chaque mi-temps, et la liste des événements avec leur label, leur mi-temps et leur timestamp en secondes.

### 3.4 Format des données en entrée du modèle

Les données préparées sont mises à disposition du modèle sous deux formats complémentaires, selon l'architecture utilisée.

#### Format features préextraites (matches.pkl)

Pour la baseline et les modèles légers de type LSTM ou Transformer, le fichier matches.pkl constitue l'entrée principale. Il s'agit d'une liste de dictionnaires, un par match, structurés comme suit :

```Python
{
    "match_id":        "england_epl/2014-2015/Chelsea_Burnley",
    "features_path_1": "chemin/vers/1_ResNET_TF2_PCA512.npy",  # (N, 512)
    "features_path_2": "chemin/vers/2_ResNET_TF2_PCA512.npy",  # (M, 512)
    "events": [
        {
            "label":        "Goal",
            "half":         1,
            "time_seconds": 1342
        },
        ...
    ]
}
```

Les features sont des vecteurs de 512 dimensions extraits par ResNet-152 puis réduits par ACP, échantillonnés à 2 fps. Chaque valeur représente une frame de la vidéo, soit un point temporel toutes les 0,5 secondes. Les features ne sont pas chargées en mémoire dans le fichier matches.pkl, seuls les chemins sont stockés

#### Format clips vidéo (outputs/clips/)

Pour les architectures plus avancées travaillant directement sur les pixels vidéo, les clips extraits sont organisés par classe dans deux sous-dossiers :

```
outputs/clips/
├── npy/                        <- entraînement du modèle
│   ├── Goal/
│   │   ├── 0001.npy            # shape : (125, 720, 720, 3)
│   │   └── ...
│   └── ...
└── mp4/                        <- affichage dans l'interface
    ├── Goal/
    │   ├── 0001.mp4
    │   └── ...
    └── ...
```

Chaque fichier .npy contient une séquence de frames brutes de dimensions (125, 720, 720, 3). 
Les fichiers .mp4 correspondants sont destinés à l'interface utilisateur, permettant d'afficher les moments clés détectés.

---

## 4. Modèle IA

### 4.1 Choix de l'architecture

Le développement du modèle s'est déroulé en deux étapes complémentaires, en cohérence avec la préparation des données décrite en section 3.

**Baseline — Random Forest sur features préextraites.** Dans un premier temps, nous avons entraîné un classifieur Random Forest sur les vecteurs ResNet-152 + PCA512 fournis par SoccerNet (2 fps). Cette approche a permis de valider rapidement le pipeline de bout en bout — chargement des annotations, construction du dataset, entraînement et évaluation — sans GPU ni téléchargement massif de vidéos. Le modèle traite une fenêtre de 10 frames consécutives (soit 5 secondes à 2 fps) autour de chaque instant annoté. Cette baseline reste disponible dans l'interface web, mais elle ne capture pas la dynamique temporelle fine des séquences vidéo : chaque fenêtre est traitée indépendamment, sans modélisation explicite du mouvement.

**Modèle principal — VideoMAE.** Pour exploiter directement les pixels des clips extraits (section 3.3), nous avons retenu VideoMAE (*Video Masked Autoencoder*), un transformeur pré-entraîné sur Kinetics puis fine-tuné sur notre jeu de clips SoccerNet. Ce choix s'appuie sur plusieurs arguments : le modèle apprend des représentations spatio-temporelles à partir de courtes séquences vidéo ; le pré-entraînement sur Kinetics fournit une initialisation robuste malgré un volume de données limité ; l'architecture par patches (tubelet) est adaptée à la classification d'actions courtes comme les événements de soccer. VideoMAE constitue le modèle par défaut du projet et celui sur lequel portent les résultats d'évaluation de la section 5.

**Variante exploratoire — SlowFast.** Nous avons également implémenté une architecture SlowFast simplifiée, avec deux backbones 3D ResNet-18 (voie lente et voie rapide) dont les features sont concaténées avant classification. Cette variante permet de comparer une approche à double fréquence temporelle avec le transformeur VideoMAE, mais n'a pas été retenue comme modèle principal en raison de performances inférieures sur notre jeu de validation.

### 4.2 Architecture

#### VideoMAE — modèle principal

Le modèle utilisé est `MCG-NJU/videomae-base-finetuned-kinetics`, chargé via la bibliothèque Hugging Face Transformers et adapté aux classes d'événements retenues dans le projet (section 3.2). L'architecture repose sur un encodeur Vision Transformer de 12 couches (dimension cachée 768, 12 têtes d'attention) qui traite la vidéo sous forme de patches spatio-temporels :

- **Patches** : taille 16×16 pixels, regroupés en tubelets de 2 frames consécutives (`tubelet_size = 2`).
- **Entrée** : 16 frames RGB redimensionnées à 224×224 pixels (soit environ 1 seconde de vidéo à 16 fps lors de l'inférence).
- **Tête de classification** : couche linéaire remplaçant la tête Kinetics d'origine, produisant un logits par classe d'événement.

Lors du fine-tuning, chaque clip d'entraînement (125 frames à 25 fps, soit 5 secondes) est sous-échantillonné uniformément en 16 frames. Le processeur `VideoMAEImageProcessor` applique la normalisation attendue par le modèle pré-entraîné avant l'inférence.

#### Random Forest — baseline

Le classifieur Random Forest (100 arbres, `max_depth=15`) reçoit en entrée un vecteur aplati de 10 frames de features PCA512 (5 120 dimensions après normalisation par `StandardScaler`). Il prédit la classe d'événement frame par frame à 2 fps.

#### SlowFast — variante

L'architecture `SlowFastSimple` utilise deux réseaux R3D-18 en parallèle : la voie lente traite 8 frames espacées, la voie rapide 64 frames à haute fréquence (facteur α = 8). Les vecteurs de features (512 dimensions chacun) sont concaténés puis passés dans un classifieur fully-connected à deux couches.

### 4.3 Pipeline d'entraînement

Le fine-tuning de VideoMAE est orchestré par le module `videomae_finetune/train.py`. Le pipeline suit les étapes suivantes :

1. **Découverte des données** — Le script parcourt `outputs/clips/mp4/`, où les clips sont organisés par classe (un sous-dossier par type d'événement). Un plafond de 300 clips par classe est appliqué pour limiter le déséquilibre (section 3.2).
2. **Split train/validation** — Partition stratifiée 80/20 (`random_state=42`) afin de préserver la distribution des classes dans les deux jeux.
3. **Chargement des clips** — La classe `SoccerVideoDataset` lit chaque fichier vidéo avec OpenCV, convertit les frames en RGB, les redimensionne à la taille cible et sélectionne 16 frames par interpolation linéaire sur la durée du clip.
4. **Fine-tuning** — Le modèle pré-entraîné est instancié avec `num_labels` adapté au nombre de classes détectées. La configuration (`num_frames`, `image_size`, mappings `label2id`/`id2label`) est mise à jour avant l'entraînement.
5. **Évaluation et sauvegarde** — À chaque epoch, le F1-score macro est calculé sur le jeu de validation. Le checkpoint correspondant au meilleur score est sauvegardé dans `outputs/models/<nom_expérience>/best_model/`, accompagné du processeur d'images et du fichier `labels.json`.
6. **Historique** — Les métriques par epoch (loss d'entraînement, loss de validation, accuracy, F1 macro) sont exportées dans `metrics.json`.

Pour la baseline Random Forest, le pipeline `main.py` enchaîne le chargement des features (`data_loader.py`), la construction du dataset avec fenêtre temporelle (`dataset_builder.py`), l'entraînement (`train_model.py`) et la sauvegarde des artéfacts (`event_detection_model.pkl`, `scaler.pkl`, `idx_to_label.pkl`).

### 4.4 Stratégie d'entraînement et optimisation

#### Hyperparamètres VideoMAE


| Paramètre        | Valeur                                        | Rôle                                               |
| ---------------- | --------------------------------------------- | -------------------------------------------------- |
| `learning_rate`  | 5×10⁻⁵                                        | Taux d'apprentissage faible, adapté au fine-tuning |
| `weight_decay`   | 0,01                                          | Régularisation L2                                  |
| `batch_size`     | 4                                             | Compromis mémoire GPU / stabilité                  |
| `epochs`         | 50                                             | Convergence observée sans sur-apprentissage marqué |
| `num_frames`     | 16                                            | Fenêtre temporelle d'entrée                        |
| `image_size`     | 224 (inférence) / 112 (certaines expériences) | Résolution des frames                              |
| `warmup_ratio`   | 0,1                                           | Phase de montée en charge du scheduler             |
| `max_per_folder` | 300                                           | Plafond d'échantillons par classe                  |


#### Techniques d'optimisation

- **Optimiseur AdamW** avec décroissance de poids, appliqué uniquement aux paramètres entraînables (backbone gelé si l'option `--freeze-backbone` est activée).
- **Scheduler cosine avec warmup** : montée progressive du taux d'apprentissage sur les 10 % premiers steps, puis décroissance cosinusoïdale jusqu'à la fin de l'entraînement.
- **Pondération des classes** : la fonction de perte cross-entropy est pondérée par l'inverse de la fréquence de chaque classe dans le jeu d'entraînement, compensant le déséquilibre résiduel malgré le plafonnement à 300 clips.
- **Mixed precision (AMP)** : entraînement en précision mixte sur GPU pour accélérer les itérations et réduire la consommation mémoire.
- **Sélection du meilleur modèle** : sauvegarde basée sur le F1-score macro de validation, et non sur la loss, afin de privilégier l'équilibre entre classes.

#### Inférence sur vidéo complète

Lors de l'analyse d'une vidéo utilisateur (section 6), le modèle VideoMAE ne traite pas la vidéo en une seule passe. L'API applique une stratégie de **fenêtres glissantes** :

1. La vidéo est décodée à 16 fps et redimensionnée à 224×224.
2. Des fenêtres de 16 frames consécutives (~1 s) sont extraites avec un pas configurable (`stride_sec`, par défaut 1 seconde).
3. Chaque fenêtre est classifiée par batch (taille 8) ; les prédictions dont la confiance dépasse un seuil (par défaut 0,35) sont retenues.
4. Les détections proches dans le temps (écart ≤ 2 s, même classe) sont fusionnées en ne conservant que la prédiction la plus confiante.
5. Si aucune prédiction ne dépasse le seuil, un mode secours retourne les fenêtres les plus confiantes afin d'éviter un résultat vide.

Cette approche transforme un classifieur de clips courts en détecteur d'événements sur des séquences de durée variable (extraits de quelques minutes ou mi-temps complètes).

---

## 5. Évaluation et hyperparamètres

### 5.1 Métriques d'évaluation

Pour évaluer les performances du modèle VideoMAE sur la tâche de détection d'événements, quatre métriques principales ont été retenues, en cohérence avec les standards du benchmark SoccerNet.

#### 5.1.1 mAP — mean Average Precision
La mAP est la métrique de référence pour l'action spotting sur SoccerNet.
Elle mesure la capacité du modèle à détecter les bons événements au bon moment, en tenant compte à la fois de la précision de la classification et de la précision temporelle de la détection. Pour chaque classe d'événement, on calcule l'Average Precision (AP) en comparant les prédictions triées par score de confiance aux annotations de référence, puis la mAP est obtenue en faisant la moyenne des AP sur toutes les classes :

$$\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} AP_c$$

#### 5.1.2 Précision et Rappel

La précision mesure, parmi tous les événements détectés par le modèle, la proportion de vrais positifs :

$$\text{Précision} = \frac{TP}{TP + FP}$$

Le rappel mesure, parmi tous les événements réellement présents dans le match, la proportion que le modèle a effectivement détectée :

$$\text{Rappel} = \frac{TP}{TP + FN}$$

Ces deux métriques sont calculées en macro-average sur l'ensemble des classes afin de traiter équitablement les classes rares comme les cartons rouges et les classes fréquentes comme les tirs, ce qui est particulièrement important dans un dataset aussi déséquilibré que SoccerNet malgré un balancement.

#### 5.1.3 Erreur temporelle moyenne

L'erreur temporelle moyenne mesure l'écart en secondes entre le timestamp prédit par le modèle et le timestamp réel de l'événement. Un événement est considéré correctement localisé s'il se trouve dans une fenêtre de ±5 secondes autour de la vérité terrain, ce qui correspond à la tolérance standard définie par le benchmark SoccerNet.

#### 5.1.4 F1-score par classe

Le F1-score est la moyenne harmonique de la précision et du rappel, calculé individuellement pour chaque classe d'événement afin d'identifier précisément les catégories que le modèle détecte bien et celles qui posent problème :

$$F1 = 2 \times \frac{\text{Précision} \times \text{Rappel}}{\text{Précision} + \text{Rappel}}$$

---

### 5.2 Résultats d'évaluation

#### 5.2.1 Métriques globales

outputs\figures\videomae_soccernet_720_avec_balancement\fig3_metrics_table.png

| Métrique          | Valeur |
|-------------------|--------|
| Précision macro   | 43.3%  |
| Rappel macro      | 44.3%  |
| F1-score macro    | 43.1%  |

Le modèle atteint un F1-score macro de 43.1%, avec une précision et un rappel quasi symétriques, ce qui indique que le balancement des classes appliqué lors de l'entraînement a bien fonctionné : le modèle ne penche ni vers la sur-détection ni vers la sous-détection de manière systématique.

#### 5.2.2 Performances par classe d'événement

outputs\figures\videomae_soccernet_720_avec_balancement\fig1_f1_per_class.png

L'analyse par classe révèle des disparités très importantes, avec des F1-scores allant de 0% à 78.7% selon la catégorie. Yellow_card (78.7%), Corner (74.0%) et Direct_free-kick (67.5%) ressortent nettement au-dessus de la moyenne grâce à leurs indices visuels caractéristiques et récurrents, tandis que Shots_on_target (34.2%) et Shots_off_target (31.5%) souffrent d'une confusion mutuelle structurelle, la distinction entre tir cadré et non cadré reposant sur la trajectoire finale du ballon, une information difficilement capturable dans une fenêtre de 16 frames. Red_card, Penalty et Yellow_red_card restent à 0% malgré le balancement, leurs exemples étant trop rares pour être appris correctement.

#### 5.2.3 Matrice de confusion

outputs\figures\videomae_soccernet_720_avec_balancement\fig2_confusion_matrix.png

La matrice de confusion confirme ces observations et permet d'identifier trois patterns de confusion structurels : une confusion symétrique entre Shots_on_target et Shots_off_target (17 cas dans chaque sens), une absorption de Goal par Direct_free-kick et Offside (6 cas chacun) due à des séquences d'arrêt de jeu visuellement similaires, et une absence quasi totale de prédictions correctes pour les classes rares comme Penalty, dont les 7 instances sont entièrement absorbées par Direct_free-kick et Goal.

#### 5.2.4 Analyse du seuil de confiance

outputs\figures\videomae_soccernet_720_avec_balancement\fig5_threshold_precision_recall.png

La courbe précision/rappel en fonction du seuil montre une zone de stabilité entre 0.1 et 0.6, où les deux métriques oscillent autour de 0.45–0.52 sans écart marqué, ce qui confirme que le seuil par défaut de 0.5 est un choix raisonnable. Au-delà de 0.65 en revanche, les deux courbes chutent brutalement jusqu'à environ 0.17 à seuil 0.9, ce qui signifie que le modèle produit rarement des prédictions à haute confiance, comportement cohérent avec le nombre de classes similaires et la difficulté intrinsèque de la tâche.

---

### 5.3 Discussion

Les résultats obtenus confirment la faisabilité de l'approche VideoMAE fine-tuné pour la détection d'événements sur des vidéos de soccer en résolution 720p, tout en mettant en évidence les limites actuelles de la configuration expérimentale. Plusieurs facteurs expliquent les performances modestes en valeur absolue : le sous-ensemble de SoccerNet utilisé reste de taille réduite par rapport aux standards du benchmark, le traitement de vidéos brutes en 720p allonge significativement les temps d'inférence comparé à des features pré-extraites. Ces résultats constituent néanmoins une base de référence solide pour la comparaison avec les autres configurations expérimentées dans le cadre de ce projet.

---

## 6. Interface web

L'interface web constitue le point d'entrée de SportInsight AI : elle permet à un utilisateur non spécialiste de déposer une vidéo, lancer l'analyse et consulter les événements détectés sous forme de timeline interactive. Elle communique avec une API Python qui orchestre l'inférence et renvoie les résultats structurés.

### 6.1 Technologies utilisées

L'application repose sur une architecture **client-serveur** découplée, lancée en développement par le script `run-dev.sh` qui démarre simultanément l'API et le frontend.

#### Frontend


| Technologie                   | Rôle                                                            |
| ----------------------------- | --------------------------------------------------------------- |
| **React 19** + **TypeScript** | Interface utilisateur composants fonctionnels                   |
| **Vite 6**                    | Bundler et serveur de développement (port 5173)                 |
| **React Router 7**            | Navigation entre les pages (upload, résultats, figures)         |
| **CSS personnalisé**          | Mise en page et thème (`global.css`), sans framework UI externe |


Le frontend est volontairement léger : aucune bibliothèque de composants lourde n'est utilisée. Les graphiques (timeline, courbes) sont implémentés en SVG natif dans des composants dédiés (`TimelineChart`, `LineChart`, `SimpleBarChart`). Un mode démonstration (`VITE_USE_MOCK=true`) permet de tester l'interface sans API grâce à des données simulées.

#### Backend (API)


| Technologie                    | Rôle                                                  |
| ------------------------------ | ----------------------------------------------------- |
| **FastAPI**                    | Framework REST, documentation automatique sur `/docs` |
| **Uvicorn**                    | Serveur ASGI (port 8000)                              |
| **PyTorch** + **Transformers** | Chargement et inférence VideoMAE                      |
| **scikit-learn**               | Inférence Random Forest (baseline)                    |
| **OpenCV**                     | Décodage vidéo et extraction de frames                |


L'API expose des endpoints REST sous le préfixe `/api`. Le proxy Vite redirige les requêtes `/api` vers `localhost:8000` en développement, évitant les problèmes CORS. En production, les deux services peuvent être déployés séparément avec la variable `VITE_API_URL` pointant vers l'API.

### 6.2 Fonctionnalités de l'interface

L'interface est organisée autour de quatre écrans accessibles via une barre de navigation.

#### Page d'upload (`/`)

Point d'entrée de l'application. L'utilisateur peut :

- **Déposer une vidéo** par glisser-déposer ou sélection de fichier (formats MP4, MKV, WebM, AVI, MOV).
- **Choisir le modèle d'analyse** parmi ceux disponibles sur le serveur (VideoMAE par défaut, Random Forest si entraîné).
- **Ajuster les paramètres** :
  - *Seuil de confiance* (0,05–0,90) : filtre les prédictions peu fiables.
  - *Pas d'analyse* (1–5 s) : intervalle entre fenêtres glissantes ; un pas court améliore la précision temporelle au prix du temps de calcul.
  - *Type de vidéo* : mode auto (extrait ou match), 1ère ou 2ème mi-temps pour les vidéos longues (> 42 min).
- **Suivre la progression** via une barre et un indicateur de chargement pendant l'analyse asynchrone.

Une fois l'analyse terminée, l'utilisateur est redirigé automatiquement vers la timeline des résultats.

#### Page Timeline (`/resultats/:jobId/timeline`)

Écran principal de consultation. Il combine :

- Un **lecteur vidéo** synchronisé avec la vidéo analysée (servie par l'API, convertie en MP4 si nécessaire).
- Une **timeline interactive** affichant les événements détectés sous forme d'emojis positionnés sur l'axe temporel, avec possibilité de cliquer pour sauter à un instant précis.
- Une **barre latérale d'événements** listant les détections avec leur type, horodatage et score de confiance ; l'élément correspondant à la position de lecture est mis en surbrillance automatiquement.

#### Page Événements (`/resultats/:jobId/evenements`)

Vue tabulaire de toutes les détections, triées par ordre chronologique. Chaque ligne affiche le type d'événement (avec code couleur), le timestamp formaté et le pourcentage de confiance. Un message explicatif apparaît si le mode secours a été utilisé (aucune prédiction au-dessus du seuil).

#### Page Figures (`/figures`)

Écran de visualisation des graphiques d'évaluation générés par le script `Eval/evalution.py` et stockés dans `outputs/figures/`. L'utilisateur peut filtrer par modèle expérimental (par exemple `videomae_soccernet_720_avec_balancement`) et agrandir chaque figure (F1 par classe, matrice de confusion, courbes d'entraînement, analyse du seuil).

#### Gestion d'état

Un contexte React (`AnalysisContext`) centralise les résultats de l'analyse en cours : identifiant du job, données de prédiction et métadonnées. Le composant `SyncJobResults` recharge automatiquement les résultats lorsque l'utilisateur navigue directement vers une URL de résultats.

### 6.3 Intégration du modèle

L'intégration entre l'interface et les modèles d'IA repose sur un pipeline asynchrone orchestré par l'API.

#### Flux d'analyse

```
Utilisateur → POST /api/analyze → Job créé → Tâche en arrière-plan
                                                    ↓
                                          Décodage vidéo + inférence
                                                    ↓
                              Résultats JSON ← GET /api/jobs/{id}/results
                                                    ↓
                                         Affichage timeline + événements
```

1. **Upload** — Le frontend envoie la vidéo et les paramètres (modèle, seuil, pas, mi-temps) via `multipart/form-data` à `POST /api/analyze`.
2. **Job asynchrone** — L'API crée un identifiant de job, sauvegarde la vidéo dans `outputs/uploads/` et lance l'inférence en tâche de fond (`BackgroundTasks` FastAPI). Le frontend interroge `GET /api/jobs/{id}` toutes les secondes jusqu'à obtention du statut `completed`.
3. **Inférence** — Le pipeline (`api/services/pipeline.py`) charge le modèle sélectionné via un registre central (`api/services/models/registry.py`) et exécute la prédiction par fenêtres glissantes (section 4.4).
4. **Résultats** — Les prédictions sont enrichies de métadonnées (durée, mode d'analyse, modèle utilisé, seuil appliqué) et sérialisées en JSON dans `outputs/jobs/{id}_result.json`. Le frontend récupère ce fichier via `GET /api/jobs/{id}/results`.

#### Registre de modèles

L'API découvre automatiquement les modèles VideoMAE disponibles en parcourant `outputs/models/` : chaque sous-dossier contenant un répertoire `best_model/` avec des poids (`model.safetensors`) est enregistré comme modèle utilisable. Le Random Forest est ajouté s'il a été entraîné (`event_detection_model.pkl`). L'endpoint `GET /api/models` expose la liste des modèles avec leur disponibilité, permettant au frontend d'afficher uniquement les options prêtes à l'emploi.

#### Paramètres exposés à l'utilisateur

Les hyperparamètres d'inférence configurables depuis l'interface correspondent directement aux arguments du pipeline backend :


| Paramètre interface | Paramètre API | Effet                                                |
| ------------------- | ------------- | ---------------------------------------------------- |
| Seuil de confiance  | `threshold`   | Filtre les prédictions ; défaut 0,35 pour VideoMAE   |
| Pas d'analyse       | `stride_sec`  | Intervalle entre fenêtres glissantes (1–5 s)         |
| Type de vidéo       | `half`        | Limite l'analyse à une mi-temps sur les matchs longs |
| Modèle d'analyse    | `model`       | Sélection du modèle dans le registre                 |


#### Lecture vidéo

L'endpoint `GET /api/jobs/{id}/video` sert la vidéo analysée au lecteur HTML5. Si le fichier source est au format MKV ou WebM, l'API le convertit automatiquement en MP4 via FFmpeg (`api/services/video_playback.py`) pour garantir la compatibilité navigateur.

Cette architecture modulaire permet d'ajouter de nouveaux modèles (SlowFast, futurs transformeurs) en implémentant une fonction `predict` et en l'enregistrant dans le registre, sans modification du frontend.

---

## 7. Résultats globaux et discussion

### 7.1 Synthèse des performances

Les résultats détaillés lors de l'analyse technique (Section 5) permettent de valider les choix d'architecture et de prétraitement effectués pour la configuration de référence (VideoMAE, 720p avec balancement des classes). Cette section synthétise ces comportements afin d'en dégager les forces et les limites structurelles du pipeline actuel.

#### 7.1.1 Impact du balancement et comportement global

La symétrie presque parfaite obtenue entre la précision macro (43.3%) et le rappel macro (44.3%) valide empiriquement la stratégie de rééquilibrage du dataset appliquée en amont. Sans ce balancement, le modèle aurait naturellement convergé vers une sur-détection des classes ultra-majoritaires (comme les fautes) au détriment des autres. Ici, le modèle démontre une neutralité de prédiction essentielle : il ne souffre ni d'un conservatisme excessif (faible rappel) ni d'une propension aux fausses alertes (faible précision).

Cependant, le plafond de performance global (F1-score macro de 43.1%) rappelle la complexité d'opérer une classification d'actions directement de bout en bout (end-to-end) sur des flux vidéo bruts, sans passer par des descripteurs géométriques ou des données cinématiques (poses des joueurs, suivi du ballon).

#### 7.1.2 Typologie des comportements par classe

L'examen des performances par catégorie d'événement met en lumière une corrélation directe entre la nature de l'indice visuel et la capacité d'apprentissage de VideoMAE. On peut diviser les 17 classes du benchmark en trois grandes catégories comportementales :

- Les réussites visuelles structurales (F1 > 65%) :
Les excellents scores de Yellow_card (78.7%) ou Corner (74.0%) s'expliquent par des ruptures de plans ou des cadrages hautement stéréotypés dans la réalisation télévisuelle (gros plan sur l'arbitre, plan fixe sur le coin du terrain). Le modèle capte efficacement ces signatures contextuelles globales.

- Les zones de confusion sémantique (F1 entre 30% et 65%) :
Cette catégorie regroupe les actions partagées au sein d'un même espace temps ou d'une même dynamique. Le cas le plus flagrant reste la confusion stricte et symétrique entre Shots_on_target et Shots_off_target. La distinction entre un tir cadré et non cadré n'est pas une question d'attitude des joueurs (qui est identique au moment de la frappe), mais de trajectoire fine de la balle sur le long terme. Restreint à une fenêtre de 16 frames, le modèle est structurellement aveugle à cette issue. De la même manière, les phases d'arrêts de jeux (après un but ou un hors-jeu) partagent des signatures visuelles de transition similaires qui induisent le modèle en erreur.

- Le mur de la rareté extrême (F1 = 0%) :
Malgré les pénalités de poids ajustées lors de l'entraînement, les classes comme Penalty ou Red_card subissent un échec total. Cela démontre une limite inhérente aux approches par Deep Learning : lorsque le volume initial d'exemples est trop faible (inférieur au seuil critique d'apprentissage du modèle), le balancement mathématique des pertes ne suffit pas à compenser l'absence de diversité statistique des situations.

#### 7.1.3 Analyse de la calibration du modèle

L'étude de la distribution des scores de confiance montre que le modèle adopte un comportement prudent, générant très peu de prédictions avec une certitude absolue (effondrement des métriques au-delà d'un seuil de 0.65). Cette faible confiance généralisée est typique des jeux de données complexes et multi-classes où les frontières inter-classes sont poreuses.

Néanmoins, la plage de stabilité observée entre 0.1 et 0.6 confirme que la distribution des probabilités en sortie reste cohérente. Le choix empirique d'un seuil de décision standard à 0.5 s'avère optimal pour cette configuration, car toute tentative d'augmenter le seuil pour purifier la précision détruirait le rappel de manière disproportionnée.

#### 7.1.4 Analyse du seuil de confiance

En conclusion, si l'approche VideoMAE en résolution 720p est viable, ses performances absolues se heurtent à trois verrous techniques majeurs :

- La résolution spatiale effective : 
La réduction des images à 112 x 112 pixels requise pour le calcul limite la détection des micro-indices (la couleur du carton dans la main de l'arbitre, l'angle exact du pied).

- La résolution temporelle : 
La restriction à des clips courts (16 frames) empêche la modélisation des dépendances à long terme nécessaires pour caractériser l'issue d'une action (comme les tirs).

- Le volume de données : 
Le sous-ensemble SoccerNet exploité ici restreint la capacité de généralisation sur les événements rares.Ces conclusions posent les bases de comparaison nécessaires pour analyser, dans les sections suivantes, l'apport des autres configurations testées (variations de résolutions et impact de l'absence de balancement).

---

## 8. Limites et perspectives

### 8.1 Limites actuelles

- **Entrainement long** : L'entrainement du modèle prend plusieurs heure à réaliser 
- **Mauvaise représentation des classes** : Certaines classes dans le dataset sont mal représenter donc mal reconnu
- **Célébration des joueurs imcomprise** : La célébration des jouheur est mal comprise par le modèle qui se met à détécter tout et n'importe quoi
- **Cadrage des vidéo** : Les vidéos on un cadrage professionnel qui ne peut pas être retrouver par les clubs amateurs et ce qui nuirra donc au résultats obtenue


### 8.2 Perspectives d'amélioration

1. **Entrainement du modèle sur des matchs entier** : Entrainer le modèle sur des match entier et non sur des clips
2. **Reconnaissance des joueurs** : Nous pourrions mettre en place une reconnaissance des joueurs par leurs numéro pour permettre au model de dire quel joueur réalise quelle action
3. **Mise en place d'un couche de vérification** : Mettre en place une couche de vérification de détéction de la balle du terrain ou de l'arbitre permettrai au modèle d'avoir de meilleurs résultats

---

## 9. Conclusion

SportInsight AI démontre la viabilité d'un pipeline d'analyse vidéo automatique
pour le soccer. En combinant le dataset SoccerNet, l'architecture SlowFast et une
interface accessible, le prototype permet de détecter les événements clés d'un
match et de les restituer sous forme de timeline et de résumé structuré.

Les résultats obtenus, bien qu'inférieurs aux modèles state-of-the-art, valident
l'approche et ouvrent la voie à des améliorations concrètes. L'objectif principal
— rendre l'analyse vidéo sportive accessible à tous — reste au cœur de la vision
du projet.

---

## 10. Références

- Giancola, S. et al. (2022). *SoccerNet-v2: A Dataset and Benchmarks for Holistic
Understanding of Broadcast Soccer Videos*. CVPR Workshop.
- Feichtenhofer, C. et al. (2019). *SlowFast Networks for Video Recognition*. ICCV.
- SoccerNet official repository : [https://github.com/SoccerNet/soccernet](https://github.com/SoccerNet/soccernet)
- PyTorch documentation : [https://pytorch.org/docs/](https://pytorch.org/docs/)


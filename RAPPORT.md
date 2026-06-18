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

| Utilisateurs principaux | Utilisateurs secondaires |
|---|---|
| Entraîneurs amateurs et semi-pro | Analystes sportifs |
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
![distribution des classes](/outputs/exploration/distribution_classes.png)
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

### 4.2 Architecture 

### 4.3 Pipeline d'entraînement

### 4.4 Stratégie d'entraînement et optimisation

---

## 5. Évaluation et hyperparamètres

### 5.1 Métriques d'évaluation

Pour évaluer les performances du modèle Video MAE sur la tâche d'action spotting,
quatre métriques principales ont été retenues, en cohérence avec le benchmark
SoccerNet.

#### 5.1.1 mAP — mean Average Precision

La mAP (mean Average Precision) est la métrique de référence pour l'action spotting
sur SoccerNet. Elle mesure la capacité du modèle à détecter les bons événements au
bon moment, en tenant compte à la fois de la précision de la classification et de la
précision temporelle de la détection.

Pour chaque classe d'événement $c$, on calcule l'Average Precision (AP) en comparant
les prédictions triées par score de confiance aux annotations de référence. La mAP
est ensuite la moyenne des AP sur toutes les classes :

$$\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} AP_c$$

#### 5.1.2 Précision et Rappel

La **précision** mesure, parmi tous les événements détectés par le modèle, la
proportion de vrais positifs :

$$\text{Précision} = \frac{TP}{TP + FP}$$

Le **rappel** mesure, parmi tous les événements réellement présents dans le match,
la proportion que le modèle a effectivement détectée :

$$\text{Rappel} = \frac{TP}{TP + FN}$$

Ces deux métriques sont calculées en macro-average sur l'ensemble des classes
d'événements afin de traiter équitablement les classes rares (cartons rouges) et
les classes fréquentes (tirs).

#### 5.1.3 Erreur temporelle moyenne

L'erreur temporelle moyenne mesure l'écart en secondes entre le timestamp prédit
par le modèle et le timestamp réel de l'événement dans la vidéo. Un événement est
considéré correctement localisé s'il se trouve dans une fenêtre de ±5 secondes
autour de la vérité terrain.

#### 5.1.4 F1-score par classe

Le F1-score est la moyenne harmonique de la précision et du rappel. Il est calculé
individuellement pour chaque classe d'événement afin d'identifier les catégories
que le modèle détecte bien et celles qui posent problème.

$$F1 = 2 \times \frac{\text{Précision} \times \text{Rappel}}{\text{Précision} + \text{Rappel}}$$

### 5.2 Résultats d'évaluation

#### 5.2.1 Métriques globales

*(Insérer ici la Figure 5 — tableau des métriques)*

| Métrique | Valeur |
|---|---|
| Précision macro | *à compléter* |
| Rappel macro | *à compléter* |
| F1-score macro | *à compléter* |
| Erreur temporelle moyenne | *à compléter* |

#### 5.2.2 Performances par classe d'événement

*(Insérer ici la Figure 3 — F1-score par classe)*

L'analyse par classe révèle des disparités importantes entre les types d'événements.
Les buts obtiennent généralement les meilleurs scores car ils s'accompagnent de
réactions visuelles distinctives (célébrations, regroupements de joueurs). Les
corners et les tirs sont plus difficiles à distinguer visuellement, ce qui explique
leurs scores plus faibles.

#### 5.2.3 Matrice de confusion

*(Insérer ici la Figure 4 — matrice de confusion)*

La matrice de confusion permet d'identifier les confusions les plus fréquentes
entre classes. On observe notamment des confusions entre tirs et corners, deux
événements qui partagent des caractéristiques visuelles proches (ballon en jeu
dans la zone de surface).

#### 5.2.4 Courbes d'entraînement

*(Insérer ici la Figure 6 — courbes de loss et mAP)*

Les courbes de loss montrent une convergence stable du modèle sans signe
d'overfitting majeur. La mAP sur le jeu de validation progresse régulièrement
jusqu'à se stabiliser autour de l'epoch *X*, ce qui justifie le choix de
sauvegarder le checkpoint `best.pth` à cette epoch.

#### 5.2.5 Analyse du seuil de confiance

*(Insérer ici la Figure 2 — précision vs rappel selon le seuil)*

La courbe précision/rappel en fonction du seuil de confiance montre le compromis
classique entre ces deux métriques. Un seuil de 0.5 offre un bon équilibre pour
une utilisation générale. Dans un contexte où les fausses détections sont
pénalisantes (rapport officiel), un seuil plus élevé (0.7) est recommandé.
À l'inverse, pour une utilisation exploratoire où on ne veut manquer aucun
événement, un seuil plus bas (0.3) est préférable.

---

### 5.4 Discussion

Les résultats obtenus démontrent la faisabilité de l'approche SlowFast pour
l'action spotting sur des matchs de soccer. Les performances restent inférieures
aux modèles state-of-the-art du benchmark SoccerNet, ce qui s'explique par
plusieurs facteurs :

- **Volume de données limité** : le prototype a été entraîné sur un sous-ensemble
  réduit de SoccerNet en raison des contraintes de temps de calcul.
- **Vidéos brutes** : l'utilisation des vidéos brutes plutôt que des features
  pré-extraites augmente significativement le temps d'entraînement.
- **Déséquilibre des classes** : certains événements (cartons rouges) sont
  beaucoup plus rares que d'autres (tirs), ce qui pénalise les métriques macro.

---

## 6. Interface web

### 6.1 Technologies utilisées

### 6.2 Fonctionnalités de l'interface

### 6.3 Intégration du modèle

---

## 7. Résultats globaux et discussion

### 7.1 Synthèse des performances

### 7.2 Comparaison avec la baseline SoccerNet

### 7.3 Analyse du pipeline complet

---

## 8. Limites et perspectives

### 8.1 Limites actuelles

- **Features incomplètes** : tous les matchs du dépôt SoccerNet ne disposent pas
  de features complètes, ce qui limite le volume de données utilisables.
- **Modèle baseline** : le Random Forest utilisé comme baseline ne capture pas la
  dimension séquentielle et temporelle des événements.
- **Absence de contexte temporel** : la version actuelle traite chaque frame
  indépendamment, sans fenêtre contextuelle pour enrichir la décision.

### 8.2 Perspectives d'amélioration

1. **Contexte temporel** : intégrer des fenêtres glissantes autour de chaque
   événement pour enrichir le signal d'entrée du modèle.
2. **Modèles séquentiels** : tester des architectures LSTM ou Transformers pour
   mieux capturer la dynamique temporelle du match.
3. **Résumé automatique** : passer de la détection d'événements à un vrai résumé
   narratif générant des highlights et des rapports automatiques.
4. **Analyse en temps réel** : optimiser le pipeline pour une détection en direct.

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
- SoccerNet official repository : https://github.com/SoccerNet/soccernet
- PyTorch documentation : https://pytorch.org/docs/
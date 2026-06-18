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

Pour évaluer les performances du modèle VideoMAE sur la tâche de détection
d'événements, quatre métriques principales ont été retenues, en cohérence
avec les standards du benchmark SoccerNet.

#### 5.1.1 mAP — mean Average Precision

La mAP est la métrique de référence pour l'action spotting sur SoccerNet.
Elle mesure la capacité du modèle à détecter les bons événements au bon
moment, en tenant compte à la fois de la précision de la classification et
de la précision temporelle de la détection. Pour chaque classe d'événement
$c$, on calcule l'Average Precision (AP) en comparant les prédictions triées
par score de confiance aux annotations de référence, puis la mAP est obtenue
en faisant la moyenne des AP sur toutes les classes :

$$\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} AP_c$$

#### 5.1.2 Précision et Rappel

La **précision** mesure, parmi tous les événements détectés par le modèle,
la proportion de vrais positifs :

$$\text{Précision} = \frac{TP}{TP + FP}$$

Le **rappel** mesure, parmi tous les événements réellement présents dans
le match, la proportion que le modèle a effectivement détectée :

$$\text{Rappel} = \frac{TP}{TP + FN}$$

Ces deux métriques sont calculées en macro-average sur l'ensemble des
classes afin de traiter équitablement les classes rares comme les cartons
rouges et les classes fréquentes comme les tirs — ce qui est particulièrement
important dans un dataset aussi déséquilibré que SoccerNet.

#### 5.1.3 Erreur temporelle moyenne

L'erreur temporelle moyenne mesure l'écart en secondes entre le timestamp
prédit par le modèle et le timestamp réel de l'événement. Un événement est
considéré correctement localisé s'il se trouve dans une fenêtre de ±5
secondes autour de la vérité terrain, ce qui correspond à la tolérance
standard définie par le benchmark SoccerNet.

#### 5.1.4 F1-score par classe

Le F1-score est la moyenne harmonique de la précision et du rappel, calculé
individuellement pour chaque classe d'événement afin d'identifier précisément
les catégories que le modèle détecte bien et celles qui posent problème :

$$F1 = 2 \times \frac{\text{Précision} \times \text{Rappel}}{\text{Précision} + \text{Rappel}}$$

---

### 5.2 Résultats d'évaluation

#### 5.2.1 Métriques globales

*(Figure — tableau des métriques)*

| Métrique          | Valeur |
|-------------------|--------|
| Précision macro   | 43.3%  |
| Rappel macro      | 44.3%  |
| F1-score macro    | 43.1%  |

Le modèle atteint un F1-score macro de 43.1%, avec une précision et un
rappel quasi symétriques, ce qui indique que le balancement des classes
appliqué lors de l'entraînement a bien fonctionné : le modèle ne penche
ni vers la sur-détection ni vers la sous-détection de manière systématique.

#### 5.2.2 Performances par classe d'événement

*(Figure — F1-score par classe)*

L'analyse par classe révèle des disparités très importantes, avec des
F1-scores allant de 0% à 78.7% selon la catégorie. Yellow_card (78.7%),
Corner (74.0%) et Direct_free-kick (67.5%) ressortent nettement au-dessus
de la moyenne grâce à leurs indices visuels caractéristiques et récurrents,
tandis que Shots_on_target (34.2%) et Shots_off_target (31.5%) souffrent
d'une confusion mutuelle structurelle, la distinction entre tir cadré et
non cadré reposant sur la trajectoire finale du ballon, une information
difficilement capturable dans une fenêtre de 16 frames. Red_card, Penalty
et Yellow_red_card restent à 0% malgré le balancement, leurs exemples
étant trop rares pour être appris correctement.

#### 5.2.3 Matrice de confusion

*(Figure — matrice de confusion)*

La matrice de confusion confirme ces observations et permet d'identifier
trois patterns de confusion structurels : une confusion symétrique entre
Shots_on_target et Shots_off_target (17 cas dans chaque sens), une
absorption de Goal par Direct_free-kick et Offside (6 cas chacun) due
à des séquences d'arrêt de jeu visuellement similaires, et une absence
quasi totale de prédictions correctes pour les classes rares comme Penalty,
dont les 7 instances sont entièrement absorbées par Direct_free-kick et Goal.

#### 5.2.4 Courbes d'entraînement

*(Figure — courbes de loss et F1)*

Les courbes de loss montrent une convergence stable sans signe d'overfitting
majeur, et le F1-score macro sur la validation progresse régulièrement
avant de se stabiliser, ce qui justifie le choix de sauvegarder le
checkpoint `best_model/` à l'epoch correspondant au meilleur score de
validation.

#### 5.2.5 Analyse du seuil de confiance

*(Figure — précision vs rappel selon le seuil)*

La courbe précision/rappel en fonction du seuil montre une zone de stabilité
entre 0.1 et 0.6, où les deux métriques oscillent autour de 0.45–0.52 sans
écart marqué, ce qui confirme que le seuil par défaut de 0.5 est un choix
raisonnable. Au-delà de 0.65 en revanche, les deux courbes chutent
brutalement jusqu'à environ 0.17 à seuil 0.9, ce qui signifie que le
modèle produit rarement des prédictions à haute confiance — comportement
cohérent avec le nombre de classes similaires et la difficulté intrinsèque
de la tâche.

---

### 5.3 Discussion

Les résultats obtenus confirment la faisabilité de l'approche VideoMAE
fine-tuné pour la détection d'événements sur des vidéos de soccer en
résolution 720p, tout en mettant en évidence les limites actuelles de
la configuration expérimentale. Plusieurs facteurs expliquent les
performances modestes en valeur absolue : le sous-ensemble de SoccerNet
utilisé reste de taille réduite par rapport aux standards du benchmark,
le traitement de vidéos brutes en 720p allonge significativement les temps
d'inférence comparé à des features pré-extraites, et la résolution d'entrée
réduite à 112px lors du fine-tuning entraîne une perte d'information pour
les événements nécessitant la détection de détails fins comme le geste de
l'arbitre pour un carton rouge. Ces résultats constituent néanmoins une
base de référence solide pour la comparaison avec les autres configurations
expérimentées dans le cadre de ce projet.

---

## 6. Interface web

### 6.1 Technologies utilisées

### 6.2 Fonctionnalités de l'interface

### 6.3 Intégration du modèle

---

## 7. Résultats globaux et discussion

### 7.1 Synthèse des performances

Les résultats présentés dans cette section portent sur la configuration
VideoMAE fine-tuné à 112px d'entrée, entraîné sur des clips extraits de
vidéos SoccerNet en résolution 720p avec balancement des classes. Les
métriques ont été calculées sur le jeu de validation (20% des données,
split stratifié, `random_state=42`) via le script d'évaluation dédié.

#### 7.1.1 Métriques globales

*(Figure — tableau des métriques)*

| Métrique          | Valeur |
|-------------------|--------|
| Précision macro   | 43.3%  |
| Rappel macro      | 44.3%  |
| F1-score macro    | 43.1%  |

Le modèle atteint un F1-score macro de **43.1%**, avec une précision et
un rappel équilibrés (43.3% et 44.3% respectivement). Cette symétrie
indique que le balancement des classes a permis d'éviter un biais systématique
vers la sur- ou sous-détection : le modèle ne favorise ni les fausses alertes
ni les omissions de manière générale. Les performances restent cependant
modestes, ce qui s'explique par la difficulté intrinsèque de la tâche et
les contraintes de la configuration expérimentale décrites en section 5.4.

#### 7.1.2 Performances par classe d'événement

*(Figure — F1-score par classe)*

L'analyse par classe révèle des disparités importantes, avec des F1-scores
allant de 0% à 78.7% selon la catégorie d'événement.

Les classes les mieux détectées sont **Yellow_card (78.7%)**, **Corner (74.0%)**
et **Direct_free-kick (67.5%)**. Ces événements bénéficient d'indices visuels
caractéristiques et récurrents : geste de l'arbitre brandissant un carton,
regroupement des joueurs en bord de terrain pour les corners, ou position
spécifique du ballon pour les coups francs directs. Leur représentation
relativement équilibrée dans le dataset après balancement contribue également
à leurs bons résultats.

Les classes intermédiaires — **Foul (62.2%)**, **Offside (59.3%)**,
**Goal (56.7%)** et **Indirect_free-kick (53.1%)** — affichent des scores
au-dessus de la moyenne mais révèlent des ambiguïtés visuelles confirmées
par la matrice de confusion. En particulier, 13 cas d'Indirect_free-kick
sont classifiés comme Yellow_card, ce qui pourrait s'expliquer par une
co-occurrence fréquente de ces événements dans les séquences d'entraînement.

Les classes faibles — **Shots_on_target (34.2%)** et **Shots_off_target
(31.5%)** — souffrent d'une confusion mutuelle structurelle : 17 exemples
de chaque classe sont confondus l'un avec l'autre. La distinction entre
tir cadré et non cadré repose sur la trajectoire finale du ballon, information
difficilement capturable dans une fenêtre temporelle courte de 16 frames.

Enfin, trois classes demeurent en **échec total (F1 = 0%)** : **Red_card**,
**Penalty** et **Yellow_red_card**. Malgré le balancement, ces événements
restent très rares dans SoccerNet et leurs exemples sont absorbés par des
classes visuellement proches : les penaltys sont confondus avec
Direct_free-kick (5 cas) et Goal (2 cas), tandis que les cartons rouges
sont quasi absents des prédictions.

#### 7.1.3 Matrice de confusion

*(Figure — matrice de confusion)*

La matrice de confusion confirme et précise les observations précédentes.
Les éléments diagonaux les plus élevés correspondent aux classes les mieux
apprises : Direct_free-kick (51), Yellow_card (50), Corner (47) et
Goal (38). On observe trois patterns de confusion structurels :

- **Tirs** : les classes Shots_on_target et Shots_off_target sont confondues
  mutuellement (17 cas dans chaque sens), formant un bloc de confusion
  symétrique qui traduit l'incapacité du modèle à distinguer ces deux
  catégories sans information sur la destination finale du ballon.
- **Événements d'arrêt de jeu** : Goal est confondu avec Direct_free-kick
  (6 cas) et Offside (6 cas), deux situations qui génèrent des séquences
  d'arrêt de jeu visuellement similaires.
- **Classes rares** : Penalty (7 instances au total) n'est jamais prédit
  correctement, ses exemples étant absorbés par Direct_free-kick et Goal.

#### 7.1.4 Analyse du seuil de confiance

*(Figure — précision vs rappel selon le seuil)*

La courbe précision/rappel en fonction du seuil de confiance montre une
zone de stabilité entre 0.1 et 0.6, où les deux métriques oscillent autour
de 0.45–0.52 sans écart marqué. Cela indique que le modèle calibre
raisonnablement ses scores de confiance dans cette plage, et que le **seuil
par défaut de 0.5 constitue un choix justifié**.

Au-delà de 0.65, les deux courbes chutent brutalement pour atteindre environ
0.17 à seuil 0.9. Cette dégradation rapide signifie que le modèle produit
rarement des prédictions à haute confiance, comportement cohérent avec la
difficulté du problème et le nombre de classes similaires. Dans ce contexte,
augmenter le seuil pour filtrer les prédictions peu confiantes dégraderait
massivement le rappel sans gain suffisant en précision.

### 7.1.5 Discussion

Les résultats obtenus valident la faisabilité de l'approche VideoMAE pour
la détection d'événements sur vidéos de soccer en résolution 720p. Le
balancement des classes a permis d'obtenir un comportement global équilibré
entre précision et rappel, ce qui constitue un prérequis pour une comparaison
fiable entre configurations.

Plusieurs facteurs limitent néanmoins les performances absolues. Le
sous-ensemble de SoccerNet utilisé reste de taille réduite au regard des
standards du benchmark, et le traitement de vidéos brutes en 720p allonge
significativement les temps d'inférence par rapport à des features
pré-extraites. Par ailleurs, la résolution d'entrée réduite à 112px lors
du fine-tuning peut entraîner une perte d'information pour les événements
nécessitant la détection de détails fins, comme le geste de l'arbitre pour
un carton rouge ou la trajectoire précise d'un tir.

Ces résultats serviront de référence pour la comparaison avec les autres
configurations expérimentées dans le cadre de ce projet, notamment les
variantes entraînées sur des clips de résolution différente ou sans
balancement des classes.

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
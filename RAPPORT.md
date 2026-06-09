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

### 3.2 Événements annotés

### 3.3 Préparation et nettoyage des données

### 3.4 Format des données en entrée du modèle

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
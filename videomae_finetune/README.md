# VideoMAE fine-tuning on SOCCER

Ce dossier contient un point d'entrée simple pour fine-tuner VideoMAE sur les clips déjà rangés par classe dans `SOCCER/outputs/clips/mp4`.

## Ce que fait le script

- découvre automatiquement les classes à partir des sous-dossiers
- fait un split train/validation stratifié
- charge un VideoMAE pré-entraîné sur Kinetics
- sauvegarde le meilleur modèle dans `outputs/models/videomae_soccernet`

## Prérequis

Installer les dépendances du projet, puis ajouter celles de VideoMAE :

```bash
pip install -r requirement.txt
pip install -r videomae_finetune/requirements.txt
```

## Lancement

```bash
python -m videomae_finetune.train --data-root SOCCER/outputs/clips/mp4
```

Options utiles :

- `--batch-size 4` pour réduire la mémoire
- `--epochs 8` pour un premier essai
- `--freeze-backbone` si tu veux n'entraîner que la tête de classification au départ

## Sorties

- `outputs/models/videomae_soccernet/` : modèle, processor et métriques
- `outputs/models/videomae_soccernet/labels.json` : mapping des labels
- `outputs/models/videomae_soccernet/metrics.json` : historique et meilleur score

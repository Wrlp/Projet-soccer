# API SportInsight

Lancer avec le front : **`./run-dev.sh`**

**Modèles d'analyse :**
- **VideoMAE** (défaut) — `outputs/models/videomae_soccernet/best_model/`
- **SlowFast** — `SlowFast/checkpoints/best.pth`

Le front envoie `model=videomae|slowfast` sur `POST /api/analyze`. Liste : `GET /api/models`.

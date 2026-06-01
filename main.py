"""
Main - Pipeline complet d'entraînement du modèle IA
"""

from data_loader import load_matches, explore_data
from dataset_builder import build_dataset, prepare_train_test_split
from train_model import train_random_forest, evaluate_model, save_model


def main():
    print("\n" + "=" * 60)
    print("SPORTINSIGHT AI - ENTRAINEMENT DU MODELE")
    print("=" * 60 + "\n")
    
    # 1. Charger les données
    matches = load_matches()
    explore_data(matches)
    
    # 2. Construire le dataset avec une fenêtre temporelle fixe de 10 frames
    X, y, label_to_idx, idx_to_label = build_dataset(matches, window_size_frames=10)
    
    # 3. Préparer train/test
    X_train, X_test, y_train, y_test, scaler = prepare_train_test_split(X, y)
    
    # 4. Entraîner le modèle OPTIMISE
    model = train_random_forest(X_train, y_train)
    
    # 5. Évaluer
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # 6. Sauvegarder
    save_model(model, scaler, label_to_idx, idx_to_label)
    
    print("=" * 60)
    print("ENTRAINEMENT TERMINE")
    print("=" * 60 + "\n")
    
    # 7. Évaluation des hyperparamètres avec le modèle entraîné
    print("\n" + "=" * 60)
    print("LANCEMENT DE L'EVALUATION DES HYPERPARAMÈTRES")
    print("=" * 60 + "\n")
    
    try:
        from evalution_hyperparams import run_evaluation
        run_evaluation(
            model=model,
            X_test=X_test,
            y_test=y_test,
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
            verbose=True
        )
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'évaluation des hyperparamètres : {e}")
        import traceback
        traceback.print_exc()
        print("[WARNING] L'entraînement s'est terminé correctement, mais l'évaluation a échoué.")


if __name__ == "__main__":
    main()

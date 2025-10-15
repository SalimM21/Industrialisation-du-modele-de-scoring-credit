import os

# --- Définir la structure complète du projet ---
structure = {
    "credit_scoring_mlops": {
        "data": {
            "raw": {},
            "processed": {},
            "reference": {},
            "new_data": {}
        },
        "notebooks": {
            "exploration.ipynb": "",
            "model_comparison.ipynb": "",
            "monitoring_drift.ipynb": ""
        },
        "src": {
            "__init__.py": "",
            "data_preprocessing": {
                "preprocessing.py": "",
                "feature_selection.py": "",
                "split_data.py": ""
            },
            "training": {
                "train_lr.py": "",
                "train_xgb.py": "",
                "train_nn.py": "",
                "evaluate_models.py": ""
            },
            "export": {
                "export_model.py": "",
                "register_mlflow.py": ""
            },
            "deployment": {
                "app": {
                    "main.py": "",
                    "schemas.py": "",
                    "utils.py": ""
                },
                "Dockerfile": "",
                "requirements.txt": "",
                "entrypoint.sh": "",
                "test_api.py": ""
            },
            "monitoring": {
                "drift_detection.py": "",
                "retrain_trigger.py": "",
                "generate_report.py": ""
            },
            "utils": {
                "spark_session.py": "",
                "config.yaml": "",
                "logger.py": ""
            }
        },
        "ci_cd": {
            "github_actions.yml": "",
            "docker-compose.yml": "",
            "Jenkinsfile": ""
        },
        "models": {
            "xgb_credit_model.pkl": "",
            "xgb_credit_model.onnx": "",
            "model_metadata.json": "",
            "mlruns": {}
        },
        "tests": {
            "test_preprocessing.py": "",
            "test_training.py": "",
            "test_export.py": "",
            "test_api.py": "",
            "test_monitoring.py": ""
        },
        "dashboards": {
            "credit_model_monitoring.html": "",
            "drift_report.html": ""
        },
        "docker-compose.yml": "",
        "README.md": "",
        "requirements.txt": "",
        "Makefile": ""
    }
}

# --- Fonction récursive pour créer les dossiers et fichiers ---
def create_structure(base_path, structure_dict):
    for name, content in structure_dict.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, 'a').close()

# --- Exécution ---
if __name__ == "__main__":
    base_dir = os.getcwd()
    create_structure(base_dir, structure)
    print("✅ Arborescence du projet 'credit_scoring_mlops' créée avec succès !")

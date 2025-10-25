"""
Tests unitaires pour vérifier que les modèles exportés
(Pickle, ONNX, PMML) sont corrects et fonctionnels.
"""

import pytest
import os
import pickle
import onnx
import onnxruntime as ort
from sklearn.datasets import make_classification
from src.export_model import export_model

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def synthetic_data():
    X, y = make_classification(
        n_samples=20,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    return X, y

@pytest.fixture
def model_path(tmp_path):
    return tmp_path / "model_test.pkl"

# -----------------------------
# Test Pickle
# -----------------------------
def test_pickle_export(synthetic_data, model_path):
    X, y = synthetic_data
    
    # Exporter un modèle simple (Logistic Regression)
    model_file = model_path
    model = export_model(model_type="lr", X=X, y=y, output_path=str(model_file))
    
    # Vérifier que le fichier Pickle existe
    assert os.path.exists(model_file)
    
    # Charger le modèle et tester prédiction
    with open(model_file, "rb") as f:
        loaded_model = pickle.load(f)
    preds = loaded_model.predict(X)
    assert len(preds) == X.shape[0]

# -----------------------------
# Test ONNX
# -----------------------------
def test_onnx_export(synthetic_data, tmp_path):
    X, y = synthetic_data
    onnx_path = tmp_path / "model_test.onnx"
    
    model = export_model(model_type="lr", X=X, y=y, output_path=str(onnx_path), export_format="onnx")
    
    # Vérifier que le fichier ONNX existe
    assert os.path.exists(onnx_path)
    
    # Charger ONNX et vérifier session
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    
    ort_session = ort.InferenceSession(str(onnx_path))
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: X.astype("float32")})
    assert outputs[0].shape[0] == X.shape[0]

# -----------------------------
# Test PMML (optionnel)
# -----------------------------
def test_pmml_export(synthetic_data, tmp_path):
    X, y = synthetic_data
    pmml_path = tmp_path / "model_test.pmml"
    
    model = export_model(model_type="lr", X=X, y=y, output_path=str(pmml_path), export_format="pmml")
    
    # Vérifier que le fichier PMML existe
    assert os.path.exists(pmml_path)
    # PMML peut être vérifié en parsant le XML
    with open(pmml_path, "r") as f:
        content = f.read()
        assert "<PMML" in content

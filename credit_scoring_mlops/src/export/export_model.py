"""
Export du meilleur modèle de scoring crédit en formats ONNX et PMML.

Fonctionnalités :
- Chargement du meilleur modèle (Logistic Regression, XGBoost ou Neural Network)
- Conversion en ONNX
- Conversion en PMML (si compatible)
- Sauvegarde des fichiers exportés pour déploiement
"""

import joblib
import os
import numpy as np
from sklearn2pmml import sklearn2pmml, PMMLPipeline
from sklearn.pipeline import Pipeline
import onnxmltools
import xgboost as xgb
from tensorflow.keras.models import load_model
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Paths modèles
lr_model_path = '../models/logistic_regression_model.pkl'
xgb_model_path = '../models/xgboost_model.pkl'
nn_model_path = '../models/nn_model'

# Dataset dummy pour définir les input shapes
X_test = np.load('../data/processed/X_test.npy')
input_shape = X_test.shape[1]

# Dossier export
export_dir = '../models/exported'
os.makedirs(export_dir, exist_ok=True)

def export_sklearn_model(model, model_name):
    """Export sklearn model en ONNX et PMML"""
    # ONNX
    initial_type = [('float_input', FloatTensorType([None, input_shape]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    onnx_path = os.path.join(export_dir, f"{model_name}.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"{model_name} exporté en ONNX : {onnx_path}")

    # PMML
    pipeline = PMMLPipeline([("classifier", model)])
    pmml_path = os.path.join(export_dir, f"{model_name}.pmml")
    sklearn2pmml(pipeline, pmml_path, with_repr=True)
    print(f"{model_name} exporté en PMML : {pmml_path}")

def export_xgboost_model(model, model_name):
    """Export XGBoost model en ONNX"""
    import onnxmltools
    initial_type = [('float_input', FloatTensorType([None, input_shape]))]
    onnx_model = onnxmltools.convert.convert_xgboost(model, initial_types=initial_type)
    onnx_path = os.path.join(export_dir, f"{model_name}.onnx")
    onnxmltools.utils.save_model(onnx_model, onnx_path)
    print(f"{model_name} exporté en ONNX : {onnx_path}")

def export_keras_model(model, model_name):
    """Export Keras model en ONNX"""
    import tf2onnx
    onnx_path = os.path.join(export_dir, f"{model_name}.onnx")
    spec = (np.random.randn(1, input_shape).astype(np.float32),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"{model_name} exporté en ONNX : {onnx_path}")

# Export des modèles si existants
if os.path.exists(lr_model_path):
    lr_model = joblib.load(lr_model_path)
    export_sklearn_model(lr_model, "LogisticRegression")

if os.path.exists(xgb_model_path):
    xgb_model = joblib.load(xgb_model_path)
    export_xgboost_model(xgb_model, "XGBoost")

if os.path.exists(nn_model_path):
    nn_model = load_model(nn_model_path)
    export_keras_model(nn_model, "NeuralNetwork")

# Dùng dataset breast cancer để luyện model
from pathlib import Path
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

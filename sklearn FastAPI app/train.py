# Dùng dataset breast cancer để luyện model
from pathlib import Path
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def main():
    data = load_breast_cancer()
    X = data.data
    Y = data.target

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)

    artifact = {
        "model": model,
        "target_names": data.target_names.tolist(),
        "feature_names": data.feature_names,
    }

    output_path = Path("artifacts/breast_cancer_model.joblib")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    print(f"Model saved to: {output_path}")
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
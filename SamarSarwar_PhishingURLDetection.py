"""
Phishing Website Classification using Machine Learning
Dataset: UCI Machine Learning Repository - Phishing Websites (ID 327)

Important:
- This project classifies labelled dataset records; it does not verify a live website.
- A prediction is not a guarantee that a website is safe or malicious.
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset():
    """Download the UCI dataset and return features and target."""
    print("Downloading the UCI Phishing Websites dataset...")
    dataset = fetch_ucirepo(id=327)

    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()

    # Convert feature columns to numeric where possible.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([float("inf"), float("-inf")], pd.NA)

    # Remove rows with missing values in features or target.
    combined = X.copy()
    combined["_target_"] = y
    combined = combined.dropna()
    y = combined.pop("_target_")
    X = combined

    if X.empty:
        raise ValueError("No usable rows were found after cleaning the dataset.")

    # UCI dataset target convention: -1 = phishing, 1 = legitimate.
    # Keep original labels for training and create readable names for reports.
    return X, y


def readable_label(value):
    """Convert known UCI target values to readable names."""
    value_text = str(value).strip().lower()
    if value_text in {"-1", "-1.0", "phishing", "phishing website"}:
        return "Phishing"
    if value_text in {"1", "1.0", "legitimate", "legitimate website"}:
        return "Legitimate"
    return str(value)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    X, y = load_dataset()
    print(f"\nDataset rows: {len(X):,}")
    print(f"Number of features: {X.shape[1]}")
    print("\nTarget class counts:")
    print(y.value_counts())

    # Save a quick dataset overview.
    overview = pd.DataFrame({
        "metric": ["Rows after cleaning", "Number of features", "Number of classes"],
        "value": [len(X), X.shape[1], y.nunique()],
    })
    overview.to_csv(OUTPUT_DIR / "dataset_overview.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    print("\nTraining the Random Forest model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    class_order = list(model.classes_)
    target_names = [readable_label(label) for label in class_order]
    report = classification_report(
        y_test,
        predictions,
        labels=class_order,
        target_names=target_names,
        zero_division=0,
    )
    print("\nClassification report:")
    print(report)

    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        file.write(report)

    # Confusion matrix figure.
    cm = confusion_matrix(y_test, predictions, labels=class_order)
    display_labels = [readable_label(label) for label in class_order]
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
    )
    plt.title("Confusion Matrix - Phishing Website Classification")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.show()

    # Feature importance plot.
    importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    importance.head(15).sort_values().plot(kind="barh", figsize=(9, 6))
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.show()

    importance.rename("importance").to_csv(OUTPUT_DIR / "feature_importance.csv")

    print("\nFiles saved in the 'outputs' folder:")
    print("- dataset_overview.csv")
    print("- classification_report.txt")
    print("- confusion_matrix.png")
    print("- feature_importance.png")
    print("- feature_importance.csv")
    print("\nProject finished. Use the measured results in your report.")
    print("Reminder: this is a dataset-based educational classifier, not a live URL scanner.")


if __name__ == "__main__":
    main()

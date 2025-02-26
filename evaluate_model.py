from sklearn.metrics import accuracy_score, confusion_matrix
from Train_model import train_model

def evaluate_model():
    model, X_test, y_test, _ = train_model()
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:\n", conf_matrix)

if __name__ == "__main__":
    evaluate_model()

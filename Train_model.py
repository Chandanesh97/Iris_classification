# Imports the logistic regression model for classification
from sklearn.linear_model import LogisticRegression
from Loading_Preprocessing_iris_Data import load_and_preprocess_data

def train_model():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    model = LogisticRegression(multi_class='ovr', max_iter=200) #One-vs-Rest strategy (best for multi-class classification).
    model.fit(X_train, y_train)
    return model, X_test, y_test, scaler

if __name__ == "__main__":
    model, X_test, y_test, scaler = train_model()
    print("Model training completed")

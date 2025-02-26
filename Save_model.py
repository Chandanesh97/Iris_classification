import pickle
from Train_model import train_model

def save_model():
    model, _, _, scaler = train_model()

    with open('logistic_regression_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    save_model()

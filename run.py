import os
import tqdm
import joblib
import train_distributions

MODELS_PATH = "models"


def run():
    try:
        os.mkdir(MODELS_PATH)
    except:
        pass

    dist_model = train_distributions.get_trained_model()
    path = os.path.join(MODELS_PATH, "model.pkl")
    joblib.dump(dist_model, path)
    print("Model saved at ", path)


if __name__ == "__main__":
    run()

import os
import tqdm
import joblib
import train_distributions
import preprocess
import cv2
import train_contour
import warnings

MODELS_PATH = "models"


def run():
    print("Preprocessing train images...")
    try:
        os.mkdir(preprocess.PREPROCESSED_PATH)
    except:
        pass

    for ind in tqdm.tqdm(range(1, 791)):
        im = preprocess.read_im(ind, preprocessed=False)
        proc = preprocess.full_pipeline(im)
        try:
            cv2.imwrite(os.path.join(preprocess.PREPROCESSED_PATH, f"{ind}.jpg"), proc)
        except Exception as e:
            print(e)
            import sys

            sys.exit(-1)

    try:
        os.mkdir(MODELS_PATH)
    except:
        pass

    dist_model = train_distributions.get_trained_model()
    path = os.path.join(MODELS_PATH, "model.pkl")
    joblib.dump(dist_model, path)
    print("Model saved at ", path)
    print("Training counter model...")
    counter_model = train_contour.get_model()
    path = os.path.join(MODELS_PATH, "counter_model.pkl")
    joblib.dump(counter_model, path)
    print("Model saved at ", path)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run()

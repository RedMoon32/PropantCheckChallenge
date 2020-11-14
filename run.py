import os
import tqdm
import joblib
import train_distributions
import preprocess
import cv2
import train_contour
import warnings

MODELS_PATH = "models"

PREPROCESSED_PATH = "preprocessed_imgs"
IM_PATH = './data/train'

def read_im(index, base = IM_PATH):
    return cv2.imread(os.path.join(base, str(int(index))+'.jpg'))

def run():
    print('Preprocessing train images...')
    try:
        os.mkdir(PREPROCESSED_PATH)
    except:
        pass

    for ind in tqdm.tqdm(range(1, 791)):
        im = read_im(ind)
        proc = preprocess.full_pipeline(im)
        try:
            cv2.imwrite(os.path.join(PREPROCESSED_PATH, f"{ind}.jpg"), proc)
        except Exception as e:
            print(e)

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
    joblib.dump(counter_model, path)
    print("Model saved at ", path)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run()

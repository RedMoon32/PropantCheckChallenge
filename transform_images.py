import preprocess
import cv2
import os
import tqdm

TRAIN = "./data/train"
PREPROCESSED_PATH = "preprocessed_imgs"

def read_im(im_id):
    return cv2.imread(os.path.join(TRAIN, str(im_id)+".jpg"))

def run():
    try:
        os.mkdir(PREPROCESSED_PATH)
    except:
        pass

    for ind in tqdm.tqdm(range(1, 791)):
        im = read_im(ind)
        cv2.imwrite(os.path.join(PREPROCESSED_PATH, f"{ind}.jpg"), preprocess.full_pipeline(im))
            


if __name__ == '__main__':
    run()
    
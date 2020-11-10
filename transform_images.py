import preprocess
import cv2
import os

TRAIN = "./data/train"
PREPROCESSED_PATH = "preprocessed_imgs"

def read_im(im_id):
    return cv2.imread(os.path.join(TRAIN, str(im_id)+".jpg"))

def run():
    try:
        os.mkdir(PREPROCESSED_PATH)
    except:
        pass

    for ind in range(1, 791):
        im = read_im(ind)
        try:
            cv2.imwrite(os.path.join(PREPROCESSED_PATH, f"{ind}.jpg"), preprocess.full_pipeline(im))
            print(f"Image #{ind} processed")
        except Exception as e:
            print(e)


if __name__ == '__main__':
    run()
    
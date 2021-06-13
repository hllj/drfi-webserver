import logging
import click
import os
import cv2
import time

from config import ConfigLoader, Config

from utils.img_data import Img_Data
from utils.get_transparent_image import get_transparent_image

from pipeline.inference.fusion import Fusion
from pipeline.inference.similarities import Similarities
from pipeline.inference.saliency import Saliency

from region_detect import Super_Region
from feature_process import Features

from grabcut import grabcut_inference

C_LIST = [20, 80, 350, 900]
ROOT_FOLDER = os.getcwd()
logging.basicConfig(level=logging.DEBUG, filename="inference.txt", filemode="w")


def predict(config_path, inference_img):
    t0 = time.clock()
    config_loader = ConfigLoader(config_path)
    args = config_loader.load()
    config = Config(args)

    threshold = config.get_inference_threshold()

    output = os.path.join(ROOT_FOLDER, config.get_inference_output())

    similarities = Similarities(config.get_same_region_path())
    saliency = Saliency(config.get_salience_path())
    fusion = Fusion(config.get_fusion_path(), config.get_fusion_model_type())

    similarities_model = similarities.get_model()

    img_data = Img_Data(inference_img)
    img = cv2.imread(inference_img)

    X = saliency.inference(img_data, similarities_model)
    Y = fusion.inference(X, img_data)
    cv2.imwrite("mask.png", Y)
    Y = grabcut_inference(img, Y)
    result_img = get_transparent_image(img, Y, threshold)
    if os.path.isdir(output) is False:
        os.mkdir(output)
    img_name = inference_img.split("/")[-1].split(".")[0] + ".png"
    result_path = os.path.join(output, img_name)
    cv2.imwrite(result_path, result_img)
    print("finished~( •̀ ω •́ )y")
    t1 = time.clock() - t0
    print("Time elapsed: ", t1)
    return Y, result_img, img_name, result_path


@click.command()
@click.option("--config_path", default="./config/config.yaml")
def main(config_path):
    t0 = time.clock()
    config_loader = ConfigLoader(config_path)
    args = config_loader.load()
    config = Config(args)

    inference_img = config.get_inference_img_path()
    threshold = config.get_inference_threshold()

    output = os.path.join(ROOT_FOLDER, config.get_inference_output())

    similarities = Similarities(config.get_same_region_path())
    saliency = Saliency(config.get_salience_path())
    fusion = Fusion(config.get_fusion_path(), config.get_fusion_model_type())

    similarities_model = similarities.get_model()

    img_data = Img_Data(inference_img)

    X = saliency.inference(img_data, similarities_model)
    Y = fusion.inference(X, img_data)
    cv2.imwrite("mask.png", Y)
    Y = grabcut_inference(inference_img, Y)
    # Y = inference(cv2.imread(inference_img), Y)
    cv2.imwrite("mask_postprocess.png", Y)
    result_img = get_transparent_image(inference_img, Y, threshold)
    if os.path.isdir(output) is False:
        os.mkdir(output)
    img_name = inference_img.split("/")[-1].split(".")[0]
    result_path = output + img_name + ".png"
    cv2.imwrite(result_path, result_img)
    print("finished~( •̀ ω •́ )y")
    t1 = time.clock() - t0
    print("Time elapsed: ", t1)


if __name__ == "__main__":
    main()

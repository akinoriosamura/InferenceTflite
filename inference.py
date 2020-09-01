from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import io

from tensorflow.lite.python.interpreter import Interpreter


# メインの実行
if __name__ == '__main__':
    # 引数のパース
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default='/tmp/grace_hopper.bmp',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default='/tmp/mobilenet_v1_1.0_224_quant.tflite',
        help='.tflite model to be executed')
    # parser.add_argument(
    #     '--input_mean',
    #     default=127.5, type=float,
    #     help='input_mean')
    # parser.add_argument(
    #     '--input_std',
    #     default=127.5, type=float,
    #     help='input standard deviation')
    args = parser.parse_args()


    # インタプリタの生成
    interpreter = Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()

    # 入力情報と出力情報の取得
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 入力テンソル種別の取得(Floatingモデルかどうか)
    floating_model = input_details[0]['dtype'] == np.float32

    # 幅と高さの取得(NxHxWxC, H:1, W:2)
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # 入力画像のリサイズ
    if (".png" in args.image) or (".PNG" in args.image):
        img = np.array(cv2.imread(args.image))
    else:
        img = np.array(cv2.imread(args.image))
    img = cv2.resize(img, (width, height))
    img_tmp = img.copy()

    # 入力データの生成
    input_data = img.astype(np.float32)
    input_data = (input_data) / 256
    input_data = np.expand_dims(input_data, axis=0)

    # Floatingモデルのデータ変換
    # if floating_model:
    #     input_data = (np.float32(input_data) - args.input_mean) / args.input_std
    # 入力をインタプリタに指定
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 推論の実行
    import time
    st = time.time()
    interpreter.invoke()
    print("el: ", time.time() - st)

    # 出力の取得
    import pdb; pdb.set_trace()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results_land = np.squeeze(output_data).reshape(-1, 3)
    iris = interpreter.get_tensor(output_details[1]['index'])
    results_iris = np.squeeze(iris).reshape(-1, 3)
    for land in results_land:
        cv2.circle(img, (int(land[0]), int(land[1])), 1, (0, 0, 255))
    for re in results_iris:
        cv2.circle(img, (int(re[0]), int(re[1])), 1, (0, 255, 0))

    save_img_path = args.image[:-4] + "_labeled.png"
    cv2.imwrite(save_img_path, img)

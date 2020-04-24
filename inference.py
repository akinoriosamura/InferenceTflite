from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import cv2
import os

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
        img = np.array(cv2.imread(args.image, -1))
    else:
        img = np.array(cv2.imread(args.image))
    img = cv2.resize(img, (width, height))

    # 入力データの生成
    input_data = np.expand_dims(img, axis=0)

    # Floatingモデルのデータ変換
    # if floating_model:
    #     input_data = (np.float32(input_data) - args.input_mean) / args.input_std
    input_data = input_data / 256
 
    # 入力をインタプリタに指定
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 推論の実行
    interpreter.invoke()

    # 出力の取得
    import pdb; pdb.set_trace()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    pred_path = os.path.join("./preds", args.image[:-4] + "_pred.png") 
    cv2.imwrite(pred_path, results)

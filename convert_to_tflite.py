import tensorflow as tf
import argparse
import sys
import os


def convert_keras_model(model_path):
    base_path, model_name = os.path.split(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
    tflite_model = converter.convert()

    tflite_name = "converted_{}.tflite".format(model_name[:-3])
    with open(os.path.join(base_path, tflite_name), "wb") as f:
        f.write(tflite_model)

    print("The model {} was successfully converted to {}".format(
        model_name, tflite_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', '--model', help='trained model path',
                        required=True)

    args = parser.parse_args()

    convert_keras_model(args.model)

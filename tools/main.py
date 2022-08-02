import argparse
from faulthandler import disable
import sys
import pycls.core.config as config
from pycls.core.config import cfg
from pycls.ir.constructor.tensorflow.anynet import anynet
import pycls.core.builders as builders
import tensorflow as tf
import torch
from tensorflow.python.framework.ops import disable_eager_execution


def parse_args():
    """Parse command line options (mode and config)."""
    parser = argparse.ArgumentParser(description="Run a model.")
    help_s = "Config file location"
    parser.add_argument("--cfg", help=help_s, required=True, type=str)
    help_s = "See pycls/core/config.py for all options"
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    """Execute operation (train, test, time, etc.)."""
    args = parse_args()
    config.load_cfg(args.cfg)
    config.assert_cfg()
    cfg.freeze()

    # net = builders.get_model()()
    # torch.onnx.export(net, torch.randn(1, 3, 224, 224), 'shufflenetv2_origin.onnx')

    net = anynet()
    net.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(net)
    tflite_model = converter.convert()

    with open('shufflenetv2_0.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    disable_eager_execution()
    main()

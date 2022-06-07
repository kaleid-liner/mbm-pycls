import argparse
import sys
import pycls.core.config as config
from pycls.core.config import cfg
from pycls.ir.constructor.tensorflow.blocks import anynet
import tensorflow as tf


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

    net = anynet()
    net.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(net)
    tflite_model = converter.convert()

    with open('resnet56.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    main()

import argparse
from faulthandler import disable
import sys
import pycls.core.config as config
from pycls.core.config import cfg
from pycls.ir.constructor.tensorflow.anynet import anynet
import pycls.core.builders as builders
import tensorflow as tf
import numpy as np
import torch
from pycls.core.net import complexity
from tensorflow.python.framework.ops import disable_eager_execution


def parse_args():
    """Parse command line options (mode and config)."""
    parser = argparse.ArgumentParser(description="Run a model.")
    help_s = "Config file location"
    parser.add_argument("--cfg", help=help_s, required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--quant", required=False, action="store_true", dest="quant")
    parser.set_defaults(quant=False)
    help_s = "See pycls/core/config.py for all options"
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 224, 224, 3)
      yield [data.astype(np.float32)]


def main():
    """Execute operation (train, test, time, etc.)."""
    args = parse_args()
    config.load_cfg(args.cfg)
    config.assert_cfg()
    cfg.freeze()

    net = builders.get_model()()
    print(complexity(net))
    net.eval()
    torch.onnx.export(net, torch.randn(1, 3, 224, 224), args.output + (".quant" if args.quant else "") + ".onnx")

    # net = anynet()
    # # net.summary()

    # converter = tf.lite.TFLiteConverter.from_keras_model(net)
    # if args.quant:
    #     converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #     converter.representative_dataset = representative_dataset
    #     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #     converter.inference_input_type = tf.uint8
    #     converter.inference_output_type = tf.uint8
    # tflite_model = converter.convert()

    # with open(args.output + (".quant" if args.quant else "") + ".tflite", 'wb') as f:
    #     f.write(tflite_model)


if __name__ == "__main__":
    disable_eager_execution()
    main()

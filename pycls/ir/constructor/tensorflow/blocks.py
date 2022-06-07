from tkinter import W
import keras.layers as layers
import keras
from pycls.core.config import cfg


def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "res_stem_cifar": res_stem_cifar,
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "res_basic_block": res_basic_block,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


def res_stem_cifar(x, w_in, w_out, name):
    x = layers.Conv2D(w_out, 3, padding="same", name=name + "_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_bn")(x)
    x = layers.ReLU(name=name + "_relu")(x)
    return x


def basic_transform(x, w_in, w_out, stride, name):
    x = layers.Conv2D(w_out, 3, stride, padding="same", name=name + "_0_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_0_bn")(x)
    x = layers.ReLU(name=name + "_0_relu")(x)
    x = layers.Conv2D(w_out, 3, padding="same", name=name + "_1_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_1_bn")(x)
    x = layers.ReLU(name=name + "_1_relu")(x)
    return x


def res_basic_block(x, w_in, w_out, stride, name, params=None):
    if (w_in != w_out) or (stride != 1):
        x_proj = layers.Conv2D(w_out, 1, stride, padding="same", name=name + "_proj_conv")(x)
        x_proj = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_proj_bn")(x_proj)
    else:
        x_proj = x
    
    x = basic_transform(x, w_in, w_out, stride, name)
    x = x_proj + x
    x = layers.ReLU(name=name + "_relu")(x)
    return x


def anyhead(x, w_in, head_width, num_classes, name):
    if head_width > 0:
        x = layers.Conv2D(head_width, 1, padding="same", name=name + "_conv")(x)
        x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_bn")(x)
        x = layers.ReLU(name=name + "_relu")(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, name=name + "_dense")(x)
    return x


def anystage(x, w_in, w_out, stride, d, block_fun, name, params):
    for i in range(d):
        x = block_fun(x, w_in, w_out, stride, name + "_b{}".format(i), params)
        stride, w_in = 1, w_out

    return x


def meeting_point(inputs, w_in, devices, w_outs=None, name=""):
    if w_outs is None:
        w_outs = [w_in // len(inputs)] * len(inputs)
    return layers.Concatenate(name=name + "_concat")([
        layers.Conv2D(w_out, 1, padding="same", name=name + "_{}_conv".format(device))(x)
        for x, device, w_out in zip(inputs, devices, w_outs)
    ])


def anynet(input_shape=(224, 224, 3)):
    nones = [None for _ in cfg.ANYNET.DEPTHS]
    p = {
        "stem_type": cfg.ANYNET.STEM_TYPE,
        "stem_w": cfg.ANYNET.STEM_W,
        "block_type": cfg.ANYNET.BLOCK_TYPE,
        "depths": cfg.ANYNET.DEPTHS,
        "widths": cfg.ANYNET.WIDTHS,
        "strides": cfg.ANYNET.STRIDES,
        "bot_muls": cfg.ANYNET.BOT_MULS if cfg.ANYNET.BOT_MULS else nones,
        "group_ws": cfg.ANYNET.GROUP_WS if cfg.ANYNET.GROUP_WS else nones,
        "head_w": cfg.ANYNET.HEAD_W,
        "se_r": cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else 0,
        "num_classes": cfg.MODEL.NUM_CLASSES,
        "devices": cfg.ANYNET.DEVICES,
        "original_widths": cfg.ANYNET.ORIGINAL_WIDTHS
    }

    img_input = layers.Input(shape=input_shape)

    stem_fun = get_stem_fun(p["stem_type"])
    block_fun = get_block_fun(p["block_type"])

    x = stem_fun(img_input, 3, p["stem_w"], "stem")
    prev_w = p["stem_w"]
    keys = ["depths", "widths", "strides", "bot_muls", "group_ws", "original_widths"]
    devices = p["devices"]
    for i, (ds, ws, ss, b, g, o) in enumerate(zip(*[p[k] for k in keys])):
        x_outs = []
        for device, d, w, s in zip(devices, ds, ws, ss):
            params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"]}
            x_outs.append(anystage(x, prev_w, w, s, d, block_fun, "s{}_{}".format(i, device), params))
        x = meeting_point(x_outs, o, devices, name="s{}_mp".format(i))
        prev_w = o

    x = anyhead(x, prev_w, p["head_w"], p["num_classes"], name="head")

    return keras.Model(img_input, x, name="anynet")

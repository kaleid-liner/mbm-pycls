from tkinter import W
import keras.layers as layers
import keras
from pycls.core.config import cfg


MP_END_FLAG = "stem"


def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "res_stem_cifar": res_stem_cifar,
        'res_stem_in': res_stem,
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "res_basic_block": res_basic_block,
        'res_bottleneck_block': res_bottleneck_block,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


def res_stem_cifar(x, w_in, w_out, name):
    x = layers.Conv2D(w_out, 3, padding="same", name=name + "_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_bn")(x)
    x = layers.ReLU(name=name + "_relu")(x)
    return x


def res_stem(x, w_in, w_out, name):
    x = layers.Conv2D(w_out, 7, padding="same", name=name + "_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name+"_bn")(x)
    x = layers.ReLU(name=name + "_relu")(x)
    x = layers.MaxPool2D(3, 2, padding="same", name=name + "_pool")(x)
    return x


def res_bottleneck_block(x, w_in, w_out, stride, name, params):
    if (w_in != w_out) or (stride != 1):
        x_proj = layers.Conv2D(w_out, 1, stride, padding="same", name=name + "_proj_conv")(x)
        x_proj = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_proj_bn")(x_proj)
    else:
        x_proj = x

    x = bottleneck_transform(x, w_in, w_out, stride, name, params)
    x = layers.Add(name=name + "_add")([x_proj, x])
    x = layers.ReLU(name=name + "_relu")(x)
    return x
    

def bottleneck_transform(x, w_in, w_out, stride, name, params):
    w_b = int(round(w_out * params["bot_mul"]))
    w_se = int(round(w_in * params["se_r"]))
    groups = w_b // params["group_w"]
    x = layers.Conv2D(w_b, 1, padding="same", name=name + "_a_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_a_bn")(x)
    x = layers.ReLU(name=name + "_a_relu")(x)
    x = layers.Conv2D(w_b, 3, strides=stride, groups=groups, padding="same", name=name + "_b_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_b_bn")(x)
    x = layers.ReLU(name=name + "_b_relu")(x)
    x = layers.Conv2D(w_out, 1, padding="same", name=name + "_c_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_c_bn")(x)
    return x


def basic_transform(x, w_in, w_out, stride, name):
    x = layers.Conv2D(w_out, 3, stride, padding="same", name=name + "_0_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_0_bn")(x)
    x = layers.ReLU(name=name + "_0_relu")(x)
    x = layers.Conv2D(w_out, 3, padding="same", name=name + "_1_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_1_bn")(x)
    return x


def res_basic_block(x, w_in, w_out, stride, name, params=None):
    if (w_in != w_out) or (stride != 1):
        x_proj = layers.Conv2D(w_out, 1, stride, padding="same", name=name + "_proj_conv")(x)
        x_proj = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_proj_bn")(x_proj)
    else:
        x_proj = x
    
    x = basic_transform(x, w_in, w_out, stride, name)
    x = layers.Add(name + "_add")([x_proj, x])
    x = layers.ReLU(name + "_relu")(x)
    return x


def anyhead(x, w_in, head_width, num_classes, name):
    if head_width > 0:
        x = layers.Conv2D(head_width, 1, padding="same", name=name + "_conv")(x)
        x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_bn")(x)
        x = layers.ReLU(name=name + "_relu")(x)
    
    x = layers.GlobalAveragePooling2D(name=name + "_gap")(x)
    x = layers.Dense(num_classes, name=name + "_dense")(x)
    return x


def anystage(x, w_in, w_out, stride, d, block_fun, name, params):
    for i in range(d):
        x = block_fun(x, w_in, w_out, stride, name + "_b{}".format(i + 1), params)
        stride, w_in = 1, w_out

    return x


def meeting_point(inputs, w_in, devices, w_outs=None, name=""):
    if len(inputs) == 1:
        return inputs[0]
    if w_outs is None:
        w_outs = [w_in // len(inputs)] * len(inputs)
    devices = [
        "stem_" + d if d == devices[0] else d
        for d in devices
    ]
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

    prev_w = p["stem_w"]
    keys = ["depths", "widths", "strides", "bot_muls", "group_ws", "original_widths"]
    devices = p["devices"]
    x = stem_fun(img_input, 3, p["stem_w"], "st_mp_" + devices[0])

    for i, (ds, ws, ss, b, gs, o) in enumerate(zip(*[p[k] for k in keys])):
        x_outs = []
        for j, (device, d, w, s, g) in enumerate(zip(devices, ds, ws, ss, gs)):
            params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"]}
            if j == 0:
                x = block_fun(x, prev_w, o, s, "s{}_mp_{}_b0".format(i + 1, device), params)
            if w != o:
                x_out = layers.Conv2D(w, 1, padding="same", name="s{}_mp_{}_convfirst".format(i + 1, devices[0]))(x)
            else:
                x_out = x
            d, s = d - 1, 1
            x_outs.append(anystage(x_out, w, w, s, d, block_fun, "s{}_{}".format(i + 1, device), params))
        x = meeting_point(x_outs, o, devices, name="s{}_mp".format(i + 1))
        prev_w = o

    x = anyhead(x, prev_w, p["head_w"], p["num_classes"], name="head_" + devices[0])

    return keras.Model(img_input, x, name="anynet")

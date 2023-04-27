import keras.layers as layers
import keras
import tensorflow as tf
from pycls.core.config import cfg
from pycls.ir.common import *
from pycls.ir.utils import make_divisible
from .op_patch import group_conv


def activation(x, name, mp_start=False):
    activation_fun = cfg.MODEL.ACTIVATION_FUN
    if "cpu" in name and not mp_start:
        activation_fun = "relu"
    if activation_fun == "relu":
        return layers.ReLU(name=name)(x)
    elif activation_fun == "swish" or activation_fun == "silu":
        mp_start_name = "{}_{}".format(name, MP_START) if mp_start else name
        return layers.Multiply(name=mp_start_name + "_mul")([x, tf.nn.sigmoid(x, name=name + "_sigmoid")])


def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "res_stem_cifar": res_stem_cifar,
        "res_stem_in": res_stem,
        "simple_stem_in": simple_stem,
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "res_basic_block": res_basic_block,
        "res_bottleneck_block": res_bottleneck_block,
        "inverted_residual": inverted_residual,
        "mbconv": mbconv,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


def res_stem_cifar(x, w_in, w_out, k=3, name="", mp_start=False):
    name = "{}_{}".format(name, MP_START) if mp_start else name
    x = layers.Conv2D(w_out, k, padding="same", name=name + "_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_bn")(x)
    x = layers.ReLU(name=name + "_relu")(x)
    return x


def res_stem(x, w_in, w_out, k=7, name="", mp_start=False):
    x = layers.Conv2D(w_out, k, strides=2, padding="same", name=name + "_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name+"_bn")(x)
    x = layers.ReLU(name=name + "_relu")(x)
    name = "{}_{}".format(name, MP_START) if mp_start else name
    x = layers.MaxPool2D(3, 2, padding="same", name=name + "_pool")(x)
    return x


def simple_stem(x, w_in, w_out, k=3, name="", mp_start=False):
    if cfg.MODEL.ACTIVATION_FUN == "relu":
        name = "{}_{}".format(name, MP_START) if mp_start else name
        mp_start = False
    x = layers.Conv2D(w_out, k, strides=2, padding="same", name=name + "_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_bn")(x)
    x = activation(x, name + "_act", mp_start=mp_start)
    return x


def res_bottleneck_block(x, w_in, w_out, stride, name, params, mp_start=False):
    if (w_in != w_out) or (stride != 1):
        x_proj = layers.Conv2D(w_out, 1, stride, padding="same", name=name + "_proj_conv")(x)
        x_proj = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_proj_bn")(x_proj)
    else:
        x_proj = x

    x = bottleneck_transform(x, w_in, w_out, stride, name, params)
    name = "{}_{}".format(name, MP_START) if mp_start else name
    x = layers.Add(name=name + "_add")([x_proj, x])
    x = layers.ReLU(name=name + "_relu")(x)
    return x
    

def bottleneck_transform(x, w_in, w_out, stride, name, params, mp_start=False):
    w_b = int(round(w_out * params["bot_mul"]))
    w_se = int(round(w_in * params["se_r"]))
    groups = w_b // params["group_w"]
    x = layers.Conv2D(w_b, 1, padding="same", name=name + "_a_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_a_bn")(x)
    x = layers.ReLU(name=name + "_a_relu")(x)
    x = group_conv(x, w_b, 3, strides=stride, padding="same", groups=groups, name=name + "_b_conv")
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_b_bn")(x)
    x = layers.ReLU(name=name + "_b_relu")(x)
    name = "{}_{}".format(name, MP_START) if mp_start else name
    x = layers.Conv2D(w_out, 1, padding="same", name=name + "_c_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_c_bn")(x)
    return x


def basic_transform(x, w_in, w_out, stride, name, mp_start=False):
    x = layers.Conv2D(w_out, 3, stride, padding="same", name=name + "_0_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_0_bn")(x)
    x = layers.ReLU(name=name + "_0_relu")(x)
    name = "{}_{}".format(name, MP_START) if mp_start else name
    x = layers.Conv2D(w_out, 3, padding="same", name=name + "_1_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_1_bn")(x)
    return x


def res_basic_block(x, w_in, w_out, stride, name, params=None, mp_start=False):
    if (w_in != w_out) or (stride != 1):
        x_proj = layers.Conv2D(w_out, 1, stride, padding="same", name=name + "_proj_conv")(x)
        x_proj = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_proj_bn")(x_proj)
    else:
        x_proj = x
    
    x = basic_transform(x, w_in, w_out, stride, name)
    name = "{}_{}".format(name, MP_START) if mp_start else name
    x = layers.Add(name=name + "_add")([x_proj, x])
    x = layers.ReLU(name=name + "_relu")(x)
    return x


def channel_shuffle(x, num_groups, name):
    n, h, w, c = x.shape
    x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups], name=name + "_reshape1")
    # x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3], name=name + "_transpose")
    output = tf.reshape(x_reshaped, [-1, h, w, c], name=name + "_reshape2")
    
    return output


def inverted_residual(x, w_in, w_out, stride, name, params=None):
    def branch1(x, w_in, w_out, stride):
        x = layers.DepthwiseConv2D(3, strides=stride, padding="same", name=name + "_branch1_conv1")(x)
        x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_branch1_bn1")(x)
        x = layers.Conv2D(w_out, 1, padding="same", name=name + "_branch1_conv2")(x)
        x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_branch1_bn2")(x)
        x = layers.ReLU(name=name + "_branch1_relu")(x)
        return x

    def branch2(x, w_in, w_out, stride):
        x = layers.Conv2D(w_out, 1, padding="same", name=name + "_branch2_conv1")(x)
        x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_branch2_bn1")(x)
        x = layers.ReLU(name=name + "_branch2_relu1")(x)
        x = layers.DepthwiseConv2D(3, strides=stride, padding="same", name=name + "_branch2_conv2")(x)
        x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_branch2_bn2")(x)
        x = layers.Conv2D(w_out, 1, padding="same", name=name + "_branch2_conv3")(x)
        x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_branch2_bn3")(x)
        x = layers.ReLU(name=name + "_branch2_relu2")(x)
        return x

    branch_features = w_out // 2
    if stride == 1:
        x1, x2 = tf.split(x, 2, axis=-1, name=name + "_split")
        x = tf.concat([x1, branch2(x2, w_in, branch_features, stride)], axis=-1, name=name + "_concat")
    else:
        x = tf.concat([
            branch1(x, w_in, branch_features, stride),
            branch2(x, w_in, branch_features, stride)
        ], axis=-1, name=name + "_concat")

    x = channel_shuffle(x, 2, name=name + "_shuffle")
    return x


def mbconv(x, w_in, w_out, stride, name, params, mp_start=False):
    w_exp = int(w_in * params["bot_mul"])
    k = params["k"]
    f_x = x
    if w_exp != w_in:
        f_x = layers.Conv2D(w_exp, 1, padding="same", name=name + "_exp_conv")(f_x)
        f_x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_exp_bn")(f_x)
        f_x = activation(f_x, name + "_exp_act")
    f_x = layers.DepthwiseConv2D(k, stride, padding="same", name=name + "_dw")(f_x)
    f_x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_dw_bn")(f_x)
    f_x = activation(f_x, name + "_dw_act")

    has_skip = stride == 1 and w_in == w_out
    mp_start_name = "{}_{}".format(name, MP_START)
    if not has_skip:
        name = mp_start_name if mp_start else name
    f_x = layers.Conv2D(w_out, 1, padding="same", name=name + "_proj_conv")(f_x)
    f_x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_proj_bn")(f_x)
    if has_skip:
        name = mp_start_name if mp_start else name
        f_x = layers.Add(name=name + "_add")([x, f_x])
    return f_x


def anyhead(x, w_in, head_width, num_classes, name):
    if head_width > 0:
        x = layers.Conv2D(head_width, 1, padding="same", name=name + "_conv")(x)
        x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_bn")(x)
        x = activation(x, name + "_act")
    
    x = layers.GlobalAveragePooling2D(name=name + "_gap")(x)
    x = layers.Dense(num_classes, name=name + "_dense")(x)
    return x


def anystage(x, w_in, w_out, stride, d, block_fun, name, params):
    for i in range(d):
        x = block_fun(x, w_in, w_out, stride, name + "_b{}".format(i + 1), params)
        stride, w_in = 1, w_out

    return x


def meeting_point(inputs, w_ins, o_w, w_outs=None, names=None):
    if len(inputs) == 1:
        return layers.Concatenate(name=names[-1] + "_concat")(inputs)
    if w_outs is None:
        w_outs = [o_w // len(inputs)] * len(inputs)
    return layers.Concatenate(name=names[-1] + "_concat")([
        layers.Conv2D(w_out, 1, padding="same", name=name + "_convlast")(x)
        if w_in != w_out else x
        for x, w_in, w_out, name in zip(inputs, w_ins, w_outs, names)
    ])


def anynet(input_shape=(224, 224, 3), include_head=True, include_stem=True):
    nones = [None for _ in cfg.ANYNET.DEPTHS]
    p = {
        "stem_type": cfg.ANYNET.STEM_TYPE,
        "stem_w": cfg.ANYNET.STEM_W,
        "stem_k": cfg.ANYNET.STEM_K,
        "block_type": cfg.ANYNET.BLOCK_TYPE,
        "depths": cfg.ANYNET.DEPTHS,
        "widths": cfg.ANYNET.WIDTHS,
        "strides": cfg.ANYNET.STRIDES,
        "kernels": cfg.ANYNET.KERNELS if cfg.ANYNET.KERNELS else nones,
        "bot_muls": cfg.ANYNET.BOT_MULS if cfg.ANYNET.BOT_MULS else nones,
        "group_ws": cfg.ANYNET.GROUP_WS if cfg.ANYNET.GROUP_WS else nones,
        "head_w": cfg.ANYNET.HEAD_W,
        "se_r": cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else 0,
        "num_classes": cfg.MODEL.NUM_CLASSES,
        "devices": cfg.ANYNET.DEVICES,
        "original_widths": cfg.ANYNET.ORIGINAL_WIDTHS,
        "mb_downsample": cfg.ANYNET.MB_DOWNSAMPLE,
        "stem_device": cfg.ANYNET.STEM_DEVICE,
        "head_device": cfg.ANYNET.HEAD_DEVICE,
        "merge_device": cfg.ANYNET.MERGE_DEVICE,
        "mb_ver": cfg.ANYNET.MB_VER
    }

    img_input = layers.Input(shape=input_shape)

    stem_fun = get_stem_fun(p["stem_type"])
    block_fun = get_block_fun(p["block_type"])

    prev_w = p["stem_w"]
    keys = ["depths", "widths", "strides", "bot_muls", "group_ws", "original_widths", "kernels"]
    devices = p["devices"]

    mp_start = p["mb_downsample"] and len(p["depths"][0]) > 1
    x = stem_fun(img_input, 3, p["stem_w"], p["stem_k"], 
        name="st_{}".format(p["stem_device"]),
        mp_start=mp_start
    )

    for i, (ds, ws, ss, bs, gs, o, k) in enumerate(zip(*[p[k] for k in keys])):
        x_outs = []
        if gs is None:
            gs = [None for _ in ds]
        if not isinstance(bs, list):
            bs = [bs for _ in ds]
        if len(ds) < 2:
            branching = False
        else:
            branching = True
        for j, (device, d, w, s, b, g) in enumerate(zip(devices, ds, ws, ss, bs, gs)):
            params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"], "k": k}
            if p["mb_downsample"]:
                x_out = x
            else:
                if j == 0:
                    x = block_fun(x, prev_w, o, s, 
                        "s{}_{}_b0".format(i + 1, device), params, True)
                if w != o:
                    x_out = layers.Conv2D(w, 1, padding="same", 
                        name="s{}_{}_convfirst".format(i + 1, device))(x)
                else:
                    x_out = x
                d, s, prev_w = d - 1, 1, w
            if p["mb_ver"] == 1 and branching:
                w_in = make_divisible(w / o * prev_w, 8)
                x_out = layers.Conv2D(w_in, 1, padding="same",
                    name="s{}_{}_convfirst".format(i + 1, device))(x_out)
                x_out = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS,
                    name="s{}_{}_convfirst_bn".format(i + 1, device))(x_out)
                x_out = anystage(x_out, w_in, w, s, d, block_fun, 
                    "s{}_{}".format(i + 1, device), params)
            else:
                x_out = anystage(x_out, prev_w, w, s, d, block_fun, 
                    "s{}_{}".format(i + 1, device), params)
            x_outs.append(x_out)
        mp_start = p["mb_downsample"] and i != len(p["depths"]) - 1 and (len(p["depths"][i + 1]) > 1)
        names = ["s{}_{}".format(i + 1, d) for d in devices]
        names.append("s{}_{}{}_{}".format(
            i + 1,
            p["merge_device"],
            "_" + MP_START if mp_start else "",
            MP_END
        ))
        if p["mb_ver"] == 1:
            w_outs = None
        elif p["mb_ver"] == 2:
            w_outs = ws
        x = meeting_point(x_outs, ws, o, w_outs=w_outs, names=names)
        prev_w = o

    if include_head:
        x = anyhead(x, prev_w, p["head_w"], p["num_classes"], 
            name="head_{}".format(p["head_device"]))
    else:
        x = layers.Add(name="placeholder_add_{}".format(p["head_device"]))([x, x])

    return keras.Model(img_input, x, name="anynet")

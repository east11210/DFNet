import os
from pathlib import Path
import argparse
import cv2
import numpy as np

import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

from torch.backends._coreml.preprocess import (
    CompileSpec,
    TensorSpec,
    CoreMLComputeUnit,
)

import coremltools as ct

from model import DFNet

def module_spec():
    return {
        "forward": CompileSpec(
            inputs=(
                TensorSpec(
                    shape=[1, 3, 512, 512],
                ),
                TensorSpec(
                    shape=[1, 1, 512, 512],
                ),
            ),
            outputs=(
                TensorSpec(
                    shape=[1, 3, 512, 512],
                ),
            ),
            backend=CoreMLComputeUnit.ALL,
            allow_low_precision=True,
        ),
    }

device = torch.device("cpu")
model = DFNet().to(device)
checkpoint = torch.load("./model/model_places2.pth", map_location=device)
model.load_state_dict(checkpoint)

# model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.eval()

img = cv2.imread("/Users/lijf/git/yingfeng/test-data/beach.jpg", cv2.IMREAD_COLOR)
msk = cv2.imread("/Volumes/ramdisk/beach_smi0.png", cv2.IMREAD_GRAYSCALE)
inputSize = (512, 512)
img = cv2.resize(img, inputSize)
msk = cv2.resize(msk, inputSize)

img = np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.uint8)
msk = np.ascontiguousarray(np.expand_dims(msk, 0)).astype(np.uint8)

msk = np.expand_dims(msk, 0)
img = np.expand_dims(img, 0)

img = torch.from_numpy(img).to(device)
msk = torch.from_numpy(msk).to(device)

img = img.float().div(255)
msk = msk.float().div(255)

imgMiss = img * msk

traced_script_module = torch.jit.trace(model, (imgMiss, msk))

mlmodel = ct.convert(
    traced_script_module,
    inputs=[ct.TensorType(name="imgMiss", shape=imgMiss.shape), ct.TensorType(name="mask", shape=msk.shape)],
    outputs=[ct.TensorType(name="result")]
)
mlmodel.save("/Volumes/ramdisk/place2.mlmodel")

# mlmodel = torch._C._jit_to_backend("coreml", model, module_spec())
# mlmodel._save_for_lite_interpreter("/Volumes/ramdisk/dfnet_place2_coreml.pt")

# optimized_traced_model = optimize_for_mobile(traced_script_module, optimization_blocklist={torch.utils.mobile_optimizer.MobileOptimizerType.CONV_BN_FUSION})
# optimized_traced_model._save_for_lite_interpreter("/Volumes/ramdisk/dfnet_place2.pt")

# traced_script_module._save_for_lite_interpreter("/Volumes/ramdisk/dfnet_place2.pt")
"""
Train and test E1_RGB_NIR  (4 input channels: ["rgb", "nir"]).

Pipeline (all logic lives in py_files/multispectral_experiment.py so the four
experiments cannot drift apart):
    1. verify identical splits / class mapping / no test leakage
    2. train the pretrained YOLO11s with the existing MultispectralDetectionTrainer
       using Ultralytics 8.4.184 default hyperparameters
    3. evaluate weights/best.pt on the held-out TEST split
    4. write test metrics + graphs to test_results/E1_RGB_NIR/

Usage:
    python training_files/train_E1_RGB_NIR.py              # train, then test
    python training_files/train_E1_RGB_NIR.py --test-only  # test an existing best.pt
    python training_files/train_E1_RGB_NIR.py --preflight  # quick evaluator check, no training
"""

import sys

MODULE_DIR = "/home/netali/projects/Multispectral/py_files"
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import multispectral_moudule as mm
import multispectral_experiment as me

EXPERIMENT_NAME = "E1_RGB_NIR"
EXPECTED_INPUT_CHANNELS = 4
EXPECTED_CHANNELS = ["rgb", "nir"]

# Fail immediately if the project definitions do not match this experiment.
assert EXPERIMENT_NAME in mm.EXPERIMENTS, EXPERIMENT_NAME
assert mm.EXPERIMENT_CHANNELS[EXPERIMENT_NAME] == EXPECTED_CHANNELS, (
    mm.EXPERIMENT_CHANNELS[EXPERIMENT_NAME]
)
assert me.input_channels_for(EXPERIMENT_NAME) == EXPECTED_INPUT_CHANNELS


if __name__ == "__main__":
    sys.exit(me.main(EXPERIMENT_NAME))
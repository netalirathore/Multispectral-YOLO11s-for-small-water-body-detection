import torch
import torch.nn as nn
from ultralytics import YOLO


def initialize_multispectral_yolo(
    weights="yolo11s.pt",
    input_channels=3,
    num_classes=4,
    device=None
):
    """
    Initialize pretrained YOLO11s for multispectral detection.

    Supports 3, 4, and 5 input channels.
    Preserves pretrained RGB, backbone, neck, and
    compatible regression-head weights.

    Replaces classification output convolutions
    for the requested number of classes.
    """

    if input_channels not in (3, 4, 5):
        raise ValueError("input_channels must be 3, 4, or 5")

    if num_classes < 1:
        raise ValueError("num_classes must be positive")

    yolo = YOLO(weights)
    model = yolo.model

    # -------------------------------------------------
    # 1. Expand first convolution
    # -------------------------------------------------

    first_conv = next(
        m for m in model.modules()
        if isinstance(m, nn.Conv2d)
    )

    old_weights = first_conv.weight.detach().clone()

    if first_conv.in_channels != 3:
        raise RuntimeError("Expected pretrained RGB model.")

    if input_channels > 3:

        new_conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None,
            padding_mode=first_conv.padding_mode
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = old_weights

            mean_kernel = old_weights.mean(
                dim=1,
                keepdim=True
            )

            for channel in range(3, input_channels):
                new_conv.weight[:, channel:channel+1] = mean_kernel

            if first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

        # Replace first convolution
        for name, module in model.named_modules():
            if module is first_conv:
                parent = model
                parts = name.split(".")

                for part in parts[:-1]:
                    parent = getattr(parent, part)

                setattr(parent, parts[-1], new_conv)
                break

    # -------------------------------------------------
    # 2. Replace classification output convolutions
    # -------------------------------------------------

    detect_head = model.model[-1]

    if not hasattr(detect_head, "cv3"):
        raise RuntimeError("YOLO detection head not found.")

    old_num_classes = detect_head.nc

    for branch in detect_head.cv3:

        old_classifier = branch[-1]

        if not isinstance(old_classifier, nn.Conv2d):
            raise RuntimeError("Unexpected classification head structure.")

        new_classifier = nn.Conv2d(
            in_channels=old_classifier.in_channels,
            out_channels=num_classes,
            kernel_size=old_classifier.kernel_size,
            stride=old_classifier.stride,
            padding=old_classifier.padding,
            dilation=old_classifier.dilation,
            groups=old_classifier.groups,
            bias=old_classifier.bias is not None
        )

        # Keep the classifier trainable and use fresh initialization.
        branch[-1] = new_classifier

    detect_head.nc = num_classes

    # Detect.no (= nc + 4 * reg_max) is used when reshaping outputs at
    # inference/validation time; it must follow the new class count.
    detect_head.no = num_classes + detect_head.reg_max * 4

    # Fresh classifier convs need the standard YOLO bias prior
    # (log(5 / nc / (640 / stride) ** 2)); without it the initial
    # cls_loss is in the hundreds.
    detect_head.bias_init()

    # Update model class configuration
    model.names = {
        i: str(i) for i in range(num_classes)
    }

    # Update number of classes in model arguments
    if isinstance(model.args, dict):
        model.args["nc"] = num_classes
    else:
        model.args.nc = num_classes

    # Enable training
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    if device is not None:
        model = model.to(device)

    print(f"Pretrained classes: {old_num_classes}")
    print(f"New classes: {num_classes}")
    print(f"Input channels: {input_channels}")

    return model
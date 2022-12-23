import torchmetrics
import torch
import shutil
import os

from utils import csvlogger_start, dotdict

from models.models_builder import build_segmentor, build_backbone

from configs.config_v0 import DataConfig, SegFormerConfig


def get_segmentation_metrics(num_classes, ignore=False):
    """>>> https://torchmetrics.readthedocs.io/en/stable/references/modules.html#"""
    if ignore:
        train_iou = torchmetrics.JaccardIndex(
            num_classes=num_classes + 1, average="none", ignore_index=num_classes
        )
        val_iou = torchmetrics.JaccardIndex(
            num_classes=num_classes + 1, average="none", ignore_index=num_classes
        )
        test_iou = torchmetrics.JaccardIndex(
            num_classes=num_classes + 1, average="none", ignore_index=num_classes
        )

        train_precision = torchmetrics.Precision(
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )

        val_precision = torchmetrics.Precision(
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )

        test_precision = torchmetrics.Precision(
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )

        val_recall = torchmetrics.Recall(
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )

        test_recall = torchmetrics.Recall(
            num_classes=num_classes + 1,
            average="none",
            ignore_index=num_classes,
            mdmc_average="global",
        )
    else:
        train_iou = torchmetrics.JaccardIndex(num_classes=num_classes, average="none")
        val_iou = torchmetrics.JaccardIndex(num_classes=num_classes, average="none")

        test_iou = torchmetrics.JaccardIndex(num_classes=num_classes, average="none")

        train_precision = torchmetrics.Precision(
            num_classes=num_classes, average="none", mdmc_average="global"
        )

        val_precision = torchmetrics.Precision(
            num_classes=num_classes, average="none", mdmc_average="global"
        )

        test_precision = torchmetrics.Precision(
            num_classes=num_classes, average="none", mdmc_average="global"
        )

        val_recall = torchmetrics.Recall(
            num_classes=num_classes, average="none", mdmc_average="global"
        )

        test_recall = torchmetrics.Recall(
            num_classes=num_classes, average="none", mdmc_average="global"
        )
    return (
        train_iou,
        val_iou,
        test_iou,
        train_precision,
        val_precision,
        test_precision,
        val_recall,
        test_recall,
    )


class ExtraModelConfig:
    # model settings
    norm_cfg = dict(type="SyncBN", requires_grad=True)
    find_unused_parameters = True

    if SegFormerConfig.model_cls_backbone == "mit_b5":
        model = dict(
            type="EncoderDecoder",
            pretrained="/data/SSD1/data/weights/"
            + SegFormerConfig.model_cls_backbone
            + ".pth",
            backbone=dict(type="mit_b5", style="pytorch"),
            decode_head=dict(
                type="SegFormerHead",
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                dropout_ratio=0.1,
                num_classes=19,
                norm_cfg=norm_cfg,
                align_corners=False,
                decoder_params=dict(embed_dim=768),
                loss_decode=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
            ),
        )
    else:

        model = dict(
            type="EncoderDecoder",
            pretrained="/data/SSD1/data/weights/"
            + SegFormerConfig.model_cls_backbone
            + ".pth",
            backbone=dict(type=SegFormerConfig.model_cls_backbone, style="pytorch"),
            decode_head=dict(
                type="SegFormerHead",
                in_channels=[32, 64, 160, 256],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                dropout_ratio=0.1,
                num_classes=DataConfig.num_classes,
                norm_cfg=norm_cfg,
                align_corners=False,
                decoder_params=dict(embed_dim=256),
                loss_decode=dict(
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
                ),
            ),
            # model training and testing settings
        )
    train_cfg = dotdict(dict())
    # print("==>> train_cfg: ", train_cfg)
    test_cfg = dotdict(dict(mode="whole"))
    # print("==>> test_cfg: ", test_cfg)


def get_segformer_model():

    # model_backbone = build_backbone(ExtraModelConfig.model)
    # print("==>> model_backbone: ", model_backbone)

    model = build_segmentor(
        ExtraModelConfig.model,
        train_cfg=ExtraModelConfig.train_cfg,
        test_cfg=ExtraModelConfig.test_cfg,
    )

    return model


if __name__ == "__main__":
    model = get_segformer_model()
    print("==>> model: ", model)

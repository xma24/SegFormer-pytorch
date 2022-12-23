class DataConfig:
    data_root = "/data/SSD1/data/cityscapes-xin/"
    logger_root = "/data/SSD1/results/xma/main_csv_logs/"
    work_dirs = "/data/SSD1/results/xma/work_dirs/"
    fix_train_data = False
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # resize_size = [512, 1024]
    # crop_size = [512, 1024]

    workers = 16
    pin_memory = True
    random_seed = 42

    class_ignore = True

    num_classes = 19
    cls_names = "none"
    classes = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "ambiguous",
    ]

    class_idx = {
        "road": 0,
        "sidewalk": 1,
        "building": 2,
        "wall": 3,
        "fence": 4,
        "pole": 5,
        "traffic light": 6,
        "traffic sign": 7,
        "vegetation": 8,
        "terrain": 9,
        "sky": 10,
        "person": 11,
        "rider": 12,
        "car": 13,
        "truck": 14,
        "bus": 15,
        "train-plant": 16,
        "motorcycle": 17,
        "bicycle": 18,
        "ambiguous": 255,
    }
    dataset_name = "cityscapes"
    dataloader_name = "dataloader_v0"
    palette = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
    extra_args = {}


class NetConfig:
    model_name = "model_v0"
    model_real_name = "SegFormer"
    lr = 0.008
    backbone_lr = 0.008
    opt = "SGD"  # AdamW, Adam, SGD
    # WEIGHT_DECAY = 0.0005
    # BETA = 0.5
    # MOMENTUM = 0.9
    # EPS = 0.00000001 # 1e-8
    # AMSGRAD = False0
    dropout = 0.3

    extra_args = {}


class TrainingConfig:
    interpolation: False
    logger_name = "wandb"  # "neptune", "csv", "wandb"
    cpus = False
    num_gpus = "autocount"
    num_nodes = 1
    max_epochs = 100
    wandb_name = "pt-" + NetConfig.model_real_name + "-" + DataConfig.dataset_name
    ckpt_path = "none"
    onnx_model = "./work_dirs/default.onnx"
    resume = "none"
    strategy = "ddp"
    accelerator = "gpu"
    progress_bar_refresh_rate = 1

    batch_size = 16

    subtrain = False
    subtrain_ratio = 1
    precision = 16

    use_torchhub = False
    use_timm = False

    single_lr = False
    lr_find = False

    pl_resume = False
    pl_resume_lr = 0.008
    pl_resume_backbone_lr = 0.008
    pl_resume_max_epoch = 10
    pl_resume_path = ""

    pretrained_weights = True
    pre_lr = 0.08
    pre_backbone_lr = 0.008
    pretrained_weights_max_epoch = 100
    pretrained_weights_path = ""
    pretrained_cls_bakcbone_weights = "/data/SSD1/data/weights/mit_b5.pth"

    scheduler = "cosineAnn"  # "step", "cosineAnnWarm", "poly", "cosineAnn"
    T_max = 100  # for cosineAnn; The same with max epoch
    eta_min = 0  # for cosineAnn
    T_0 = 5  # for cosineAnnWarm
    T_mult = 1  # for cosineAnnWarm
    lr_gamma = 0.5
    poly_lr: False
    extra_args = {}


class ValidationConfig:
    batch_size = 1
    val_interval = 1

    sub_val = False
    subval_ratio = 1
    extra_args = {}


class TestingConfig:
    batch_size = 1
    ckpt_path = "none"
    multiscale = False
    imageration = [1.0]
    slidingscale = False
    extra_args = {}


class SegFormerConfig:
    work_dir = DataConfig.work_dirs
    load_from = ""
    resume_from = ""
    no_validate = True
    model_cls_backbone = "mit_b5"

import argparse

def parser():
    parser = argparse.ArgumentParser(description="Random Training")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        required=True,
        help="training directory (default: None)",
    )

    parser.add_argument(
        "--dataset", type=str, default="CIFAR10", help="dataset name (default: CIFAR10)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        required=True,
        metavar="PATH",
        help="path to datasets location (default: None)",
    )
    parser.add_argument(
        "--use_test",
        dest="use_test",
        action="store_true",
        help="use test dataset instead of validation (default: False)",
    )
    parser.add_argument("--split_classes", type=int, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 128)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        metavar="N",
        help="number of workers (default: 4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
        metavar="MODEL",
        help="model name (default: None)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="checkpoint to resume training from (default: None)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=25,
        metavar="N",
        help="save frequency (default: 25)",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=5,
        metavar="N",
        help="evaluation frequency (default: 5)",
    )
    parser.add_argument(
        "--lr_init",
        type=float,
        default=0.01,
        metavar="LR",
        help="initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
    )

    parser.add_argument("--swa", action="store_true", help="swa usage flag (default: off)")
    parser.add_argument(
        "--swa_start",
        type=float,
        default=161,
        metavar="N",
        help="SWA start epoch number (default: 161)",
    )
    parser.add_argument(
        "--swa_lr", type=float, default=0.02, metavar="LR", help="SWA LR (default: 0.02)"
    )
    parser.add_argument(
        "--swa_c_epochs",
        type=int,
        default=1,
        metavar="N",
        help="SWA model collection frequency/cycle length in epochs (default: 1)",
    )
    parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")
    parser.add_argument(
        "--max_num_models",
        type=int,
        default=20,
        help="maximum number of SWAG models to save",
    )

    parser.add_argument(
        "--swa_resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="checkpoint to restor SWA from (default: None)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="CE",
        help="loss to use for training model (default: Cross-entropy)",
    )

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--num_channels", type=int, default=64, help="number of channels for resnet"
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="depth of convnet"
    )
    parser.add_argument("--no_schedule", action="store_true", help="store schedule")

    args = parser.parse_args()
    return args
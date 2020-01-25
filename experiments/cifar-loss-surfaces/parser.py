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
        "--use_test",
        dest="use_test",
        action="store_true",
        help="use test dataset instead of validation (default: False)",
    )
    parser.add_argument(
        "--max_depth",
        dest="max_depth",
        action="max_depth",
        help="depth of conv net",
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
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 200)",
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


    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--num_channels", type=int, default=64, help="number of channels for resnet"
    )
    parser.add_argument("--no_schedule", action="store_true", help="store schedule")

    args = parser.parse_args()
    return args

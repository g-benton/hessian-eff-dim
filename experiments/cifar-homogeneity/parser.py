import argparse

def parser():
    parser = argparse.ArgumentParser(description="Random Training")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 200)",
    )

    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )

    args = parser.parse_args()
    return args

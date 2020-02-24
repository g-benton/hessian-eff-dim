import argparse

def parser():
    parser = argparse.ArgumentParser(description="Loss Surfaces")
    parser.add_argument(
        "--range", type=float, default=10.0, help="parameter distance around MAP"
    )
    parser.add_argument(
        "--n_pts", type=int, default=25, help="number of points to compute the loss surface at"
    )

    args = parser.parse_args()
    return args

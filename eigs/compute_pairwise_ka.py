import numpy as np

import argparse

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--file1", type=str, required=True, help="file 1"
)
parser.add_argument(
    "--file2", type=str, required=True, help="file 2"
)
args = parser.parse_args()

npz1 = np.load(args.file1)
npz2 = np.load(args.file2)

trkkt = np.power(npz1['pos_evals'], 2).sum()**0.5
            
trkp_kpt = np.power(npz2['pos_evals', 2).sum()**0.5

q1_prod_q2 = npz1['pos_bases'].T @ npz2['pos_bases']

t1_q12 = np.diag(npz1['pos_evals']) @ q1_prod_q2

t2_q21 = np.diag(npz2['pos_evals']) @ q1_prod_q2.T

cross_term = t1_q12 @ t2_q21
trace_cross_term = np.diag(cross_term).sum()

kern_align = trace_cross_term / (trkkt * trkp_kpt)
print('Kernel Alignment:', kern_align)
print('Log Kernel Alignment:', np.log(kern_align))
import re
import math
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import ReadFasta


class DDE:
    def __init__(self):
        self.AA = 'ACDEFGHIKLMNPQRSTVWY'
        self.Codons = {
            **{'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2},
            **{'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6},
            **{'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6},
            **{'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2},
        }
        self.DPs = [aa1 + aa2 for aa1 in self.AA for aa2 in self.AA]
        self.encodings = []
        self.marks = []
        self.headers = [aa + '.DDE' for aa in self.DPs]

        self.TM = {
            pair: self.Codons[pair[0]] / 61 * self.Codons[pair[1]] / 61
            for pair in self.DPs
        }

    # parallel functions
    def dde_para(self, seq):
        DP_count = dict(zip(self.DPs, [0] * len(self.DPs)))
        for i in range(len(seq) - 1):
            DP_count[seq[i] + seq[i + 1]] += 1
        DC = {key: DP_count[key] / (len(seq) - 1) for key in DP_count}

        TV = {
            key: self.TM[key] * (1 - self.TM[key]) / (len(seq) - 1)
            for key in self.TM
        }

        DDE = [(DC[key] - self.TM[key]) / math.sqrt(TV[key]) for key in DC]
        return DDE

    def fit(self, fastas):
        fastas = ReadFasta.ReadFasta(fastas)
        self.marks = [i[0] for i in fastas]
        DDE = Parallel(n_jobs=-1)(
            delayed(self.dde_para)(re.sub('-', '', seq[1])) for seq in fastas
        )
        self.encodings = pd.DataFrame(
            np.array(DDE).reshape(len(fastas), 400),
            index=self.marks,
            columns=self.headers,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage='USAGE:', description='Generating DDE vectors'
    )
    parser.add_argument(
        '--file', required=True, help='input protein sequence fasta file'
    )
    parser.add_argument(
        '--out', default='DDE.csv', help='the generated DDE vectors file'
    )
    args = parser.parse_args()
    output = args.out if args.out is not None else 'DDE.csv'
    dde = DDE()
    dde.fit(fastas=args.file)
    dde.encodings.to_csv(output)

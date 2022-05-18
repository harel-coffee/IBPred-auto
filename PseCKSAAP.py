import os
import re
import time
import argparse
import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
import CheckFasta
import ReadFasta


data_path = os.path.join(os.path.dirname(__file__), 'datas')


class PseCKSAAP:
    def __init__(self, gap=9, delta=10):
        self.gap = gap
        self.delta = delta
        if self.gap < 0:
            print('Error: the gap should be equal or greater than 0' + '\n\n')
            exit()
        if self.delta < 1:
            print(
                'Error: the delta should be equal or greater than 1' + '\n\n'
            )
            exit()
        self.AA = 'ACDEFGHIKLMNPQRSTVWY'
        self.DPs = [aa1 + aa2 for aa1 in self.AA for aa2 in self.AA]

        self.N = 14

        self.PROPERTY = pd.read_csv(
            os.path.join(data_path, 'property.csv'), index_col=0
        )

        # normalization of properties
        self.PROPERTY_N = (self.PROPERTY - self.PROPERTY.mean()) / np.std(
            self.PROPERTY
        )

        self.THETA = pd.DataFrame(
            dict(
                zip(
                    [aa + bb for aa in self.AA for bb in self.AA],
                    [
                        self.PROPERTY_N.loc[aa] * self.PROPERTY_N.loc[bb]
                        for aa in self.AA
                        for bb in self.AA
                    ],
                )
            )
        )

        self.encodings = []
        self.marks = []  # index of outfile, the names/marks of sequences

        PSE_NAME = ['Pse_' + i for i in self.PROPERTY.columns]
        self.headers = [
            rs + '.' + str(g) for g in range(self.gap + 1) for rs in self.DPs
        ] + [
            x + '_delta' + str(i)
            for i in range(1, self.delta + 1)
            for x in PSE_NAME
        ]

    # parallel functions
    def dp_para(self, seq, g):
        DP_count = dict(zip(self.DPs, [0] * 400))
        for i in range(len(seq) - g - 1):
            DP_count[seq[i] + seq[i + g + 1]] += 1
        V = [DP_count[key] / (len(seq) - g - 1) for key in DP_count]
        return V

    def pse_para(self, seq, d):
        theta = np.zeros((len(seq) - d, self.N))
        for i in range(len(seq) - d):
            theta[i] = self.THETA[seq[i] + seq[i + d]]
        S = (1 / (len(seq) - d)) * sum(theta)
        return S

    def fit(self, fastas):
        start_t = time.time()
        fastas = ReadFasta.ReadFasta(fastas)
        if CheckFasta.minSequenceLength(fastas) < self.gap + 2:
            print(
                'Error: all the sequence length should be larger than '
                + 'the gap + 2: '
                + str(self.gap + 2)
                + '\n\n'
            )
            exit()
        if CheckFasta.minSequenceLengthWithNormalAA(fastas) < self.delta + 1:
            print(
                'Error: all the sequence length should be larger than '
                + 'the delta + 1: '
                + str(self.delta + 1)
                + '\n\n'
            )
            exit()

        # # serial
        # for i in fastas:
        #     mark, sequence = i[0], re.sub('-', '', i[1])
        #     self.marks.append(mark)
        #     code = []
        #     # DP (CKSAAP)
        #     for g in range(self.gap + 1):
        #         DP_count = dict(zip(self.DPs, [0] * 400))
        #         for i in range(len(sequence) - g - 1):
        #             DP_count[sequence[i] + sequence[i + g + 1]] += 1
        #         V = [
        #             DP_count[key] / (len(sequence) - g - 1)
        #             for key in DP_count
        #         ]
        #         code += V
        #     # Pse
        #     for d in range(1, self.delta + 1):
        #         theta = np.zeros((len(sequence) - d, self.N))
        #         for i in range(len(sequence) - d):
        #             theta[i] = self.THETA[sequence[i] + sequence[i + d]]
        #         code = np.append(
        #             code,
        #             (1 / (len(sequence) - d)) * sum(theta)
        #         )
        #     self.encodings.append(code)

        # parallel
        self.marks = [i[0] for i in fastas]
        count = Parallel(n_jobs=-1)(
            delayed(self.dp_para)(re.sub('-', '', i[1]), g)
            for i, g in product(fastas, range(self.gap + 1))
        )
        theta = Parallel(n_jobs=-1)(
            delayed(self.pse_para)(re.sub('-', '', i[1]), d)
            for i, d in product(fastas, range(1, self.delta + 1))
        )
        self.encodings = np.c_[
            np.array(count).reshape(
                len(fastas), len(self.DPs) * (self.gap + 1)
            ),
            np.array(theta).reshape(len(fastas), self.N * self.delta),
        ]
        self.encodings = pd.DataFrame(
            self.encodings, index=self.marks, columns=self.headers
        )
        print('psecksaap features %.2f seconds' % (time.time() - start_t))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage='USAGE:', description='Generating PseCKSAAP vectors'
    )
    parser.add_argument(
        '--file', required=True, help='input protein sequence fasta file'
    )
    parser.add_argument(
        '--gap',
        type=int,
        default=9,
        help='the k-space value: the gap of two amino acids',
    )
    parser.add_argument(
        '--delta', type=int, default=10, help='the delta value'
    )
    parser.add_argument(
        '--out',
        default='PseCKSAAP.csv',
        help='the generated PseCKSAAP vectors file',
    )
    args = parser.parse_args()
    output = args.out if args.out is not None else 'PseCKSAAP.csv'
    psecksaap = PseCKSAAP(gap=args.gap, delta=args.delta)
    psecksaap.fit(fastas=args.file)
    psecksaap.encodings.to_csv(output)

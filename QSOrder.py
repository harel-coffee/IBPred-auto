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


class QSOrder:
    def __init__(self, nlag=5, w=0.5):
        self.nlag = nlag
        self.w = w
        self.AA1 = 'ARNDCQEGHILKMFPSTWYV'

        self.Schneider_Wrede = pd.read_csv(
            os.path.join(data_path, 'Schneider_Wrede.csv'), index_col=0
        )
        self.Grantham = pd.read_csv(
            os.path.join(data_path, 'Grantham.csv'), index_col=0
        )

        self.DictAA1 = dict(
            zip([i for i in self.AA1], [i for i in range(len(self.AA1))])
        )

        self.encodings = []
        # index of outfile, the names/marks of sequences
        self.marks = []
        self.headers = (
            ['Schneider.Xr.' + aa for aa in self.AA1]
            + ['Grantham.Xr.' + aa for aa in self.AA1]
            + ['Schneider.Xd.' + str(n) for n in range(1, self.nlag + 1)]
            + ['Grantham.Xd.' + str(n) for n in range(1, self.nlag + 1)]
        )

    def AA_count(self, seq, AA):
        return seq.count(AA)

    def dSW_para(self, seq, nlag):
        return sum(
            [
                self.Schneider_Wrede.loc[seq[j], seq[j + nlag]] ** 2
                for j in range(len(seq) - nlag)
            ]
        )

    def dGM_para(self, seq, nlag):
        return sum(
            [
                self.Grantham.loc[seq[j], seq[j + nlag]] ** 2
                for j in range(len(seq) - nlag)
            ]
        )

    def fit(self, fastas):
        start_t = time.time()
        fastas = ReadFasta.ReadFasta(fastas)
        if CheckFasta.minSequenceLengthWithNormalAA(fastas) < self.nlag + 1:
            print(
                'Error: all the sequence length should be larger than '
                + 'the nlag+1: '
                + str(self.nlag + 1)
                + '\n\n'
            )
            return 0
        # # serial --form iFeature
        # for i in fastas:
        #     mark, sequence = i[0], re.sub('-', '', i[1])
        #     self.marks.append(mark)
        #     Dict = dict(
        #         zip(
        #             [aa for aa in self.AA1],
        #             [sequence.count(aa) for aa in self.AA1],
        #         )
        #     )
        #     code = []
        #     arraySW = []
        #     arrayGM = []
        #     for n in range(1, self.nlag + 1):
        #         arraySW.append(
        #             sum(
        #                 [
        #                     self.Schneider_Wrede.loc[
        #                         sequence[j], sequence[j + n]
        #                     ]
        #                     ** 2
        #                     for j in range(len(sequence) - n)
        #                 ]
        #             )
        #         )
        #         arrayGM.append(
        #             sum(
        #                 [
        #                     self.Grantham.loc[sequence[j], sequence[j + n]]
        #                     ** 2
        #                     for j in range(len(sequence) - n)
        #                 ]
        #             )
        #         )
        #     for aa in self.AA1:
        #         code.append(Dict[aa] / (1 + self.w * sum(arraySW)))
        #     for aa in self.AA1:
        #         code.append(Dict[aa] / (1 + self.w * sum(arrayGM)))
        #     for d in arraySW:
        #         code.append((self.w * d) / (1 + self.w * sum(arraySW)))
        #     for d in arrayGM:
        #         code.append((self.w * d) / (1 + self.w * sum(arrayGM)))
        #     self.encodings.append(code)

        # parallel
        self.marks = [i[0] for i in fastas]
        AA_counts = np.array(
            Parallel(n_jobs=-1)(
                delayed(self.AA_count)(re.sub('-', '', i[1]), AA)
                for i, AA in product(fastas, self.AA1)
            )
        ).reshape(len(fastas), len(self.AA1))
        distanceSW = np.array(
            Parallel(n_jobs=-1)(
                delayed(self.dSW_para)(re.sub('-', '', i[1]), n)
                for i, n in product(fastas, range(1, self.nlag + 1))
            )
        ).reshape(len(fastas), self.nlag)
        distanceGM = np.array(
            Parallel(n_jobs=-1)(
                delayed(self.dGM_para)(re.sub('-', '', i[1]), n)
                for i, n in product(fastas, range(1, self.nlag + 1))
            )
        ).reshape(len(fastas), self.nlag)
        sum_distSW = distanceSW.sum(axis=1).reshape(len(fastas), 1)
        sum_distGW = distanceGM.sum(axis=1).reshape(len(fastas), 1)
        self.encodings = np.c_[
            np.multiply(AA_counts, 1 / (1 + self.w * sum_distSW)),
            np.multiply(AA_counts, 1 / (1 + self.w * sum_distGW)),
            np.multiply(self.w * distanceSW, 1 / (1 + self.w * sum_distSW)),
            np.multiply(self.w * distanceGM, 1 / (1 + self.w * sum_distGW)),
        ]

        self.encodings = pd.DataFrame(
            self.encodings, index=self.marks, columns=self.headers
        )
        print(
            'qsorder features comsumed %.2f seconds' % (time.time() - start_t)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage="it's usage tip.",
        description="Generating QSOrder feature vectors",
    )
    parser.add_argument(
        "--file", required=True, help="input protein sequence fasta file"
    )
    parser.add_argument(
        "--nlag",
        type=int,
        default=30,
        help="the nlag value for QSOrderr descriptor",
    )
    parser.add_argument(
        "--w",
        type=float,
        default=0.1,
        help="the weight factor for QSOrder descriptor",
    )
    parser.add_argument(
        "--out",
        default='QSOrder.csv',
        help="the generated QSOrder vectors file",
    )
    args = parser.parse_args()
    output = args.out if args.out is not None else 'QSOrder.csv'
    qsorder = QSOrder(nlag=args.nlag, w=args.w)
    qsorder.fit(fastas=args.file)
    qsorder.encodings.to_csv(output)

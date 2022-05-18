import re
import math
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import ReadFasta


class CTD:
    def __init__(self):
        self.properties = (
            ['hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101']
            + ['hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101']
            + ['hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101']
            + ['hydrophobicity_FASG890101', 'normwaalsvolume', 'polarity']
            + ['polarizability', 'charge', 'secondarystruct', 'solventaccess']
        )
        self.polar = (
            ['RKEDQN', 'QSTNGDE', 'QNGSWTDERA', 'KPDESNQT']
            + ['KDEQPSRNTG', 'RDKENQHYP', 'KERSQD', 'GASTPDC']
            + ['LIFWCMVY', 'GASDT', 'KR', 'EALMQKRH', 'ALFCGIVW']
        )
        self.neutral = (
            ['GASTPHY', 'RAHCKMV', 'HMCKV', 'GRHA', 'AHYMLV']
            + ['SGTAW', 'NTPG', 'NVEQIL', 'PATGS', 'CPNVEQIL']
            + ['ANCQGHILMFPSTWYV', 'VIYCWFT', 'RKQEND']
        )
        self.hydrophobicity = (
            ['CLVIMFW', 'LYPFIW', 'LPFYI', 'YMFWLCVI']
            + ['FIWC', 'CVLIMF', 'AYHWVMFLIC', 'MHKFRYW']
            + ['HQRKNED', 'KMHFRYW', 'DE', 'GNPSD', 'MSPTHY']
        )

        self.group1 = dict(zip(self.properties, self.polar))
        self.group2 = dict(zip(self.properties, self.neutral))
        self.group3 = dict(zip(self.properties, self.hydrophobicity))

        self.encodings = []
        self.marks = []
        self.headers = [
            p + '.' + i
            for i in ['G1', 'G2', 'G3', 'Tr1221', 'Tr1331', 'Tr2332']
            + [
                g + '.residue' + d
                for g in ('G1', 'G2', 'G3')
                for d in ['0', '25', '50', '75', '100']
            ]
            for p in self.properties
        ]

    def code_para(self, seq):
        code = []
        aaPair = [seq[j : j + 2] for j in range(len(seq) - 1)]
        for p in self.properties:  # C
            c1 = self.Count(self.group1[p], seq) / len(seq)
            c2 = self.Count(self.group2[p], seq) / len(seq)
            c3 = 1 - c1 - c2
            code += [c1, c2, c3]
        for p in self.properties:  # T
            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (
                    pair[0] in self.group1[p] and pair[1] in self.group2[p]
                ) or (pair[0] in self.group2[p] and pair[1] in self.group1[p]):
                    c1221 += 1
                    continue
                if (
                    pair[0] in self.group1[p] and pair[1] in self.group3[p]
                ) or (pair[0] in self.group3[p] and pair[1] in self.group1[p]):
                    c1331 += 1
                    continue
                if (
                    pair[0] in self.group2[p] and pair[1] in self.group3[p]
                ) or (pair[0] in self.group3[p] and pair[1] in self.group2[p]):
                    c2332 += 1
            code += [
                c1221 / len(aaPair),
                c1331 / len(aaPair),
                c2332 / len(aaPair),
            ]
        for p in self.properties:  # D
            code += (
                self.Count1(self.group1[p], seq)
                + self.Count1(self.group2[p], seq)
                + self.Count1(self.group3[p], seq)
            )
        return code

    # modified from iFeature
    def fit(self, fastas):
        fastas = ReadFasta.ReadFasta(fastas)
        self.marks = [i[0] for i in fastas]
        ctd = Parallel(n_jobs=-1)(
            delayed(self.code_para)(re.sub('-', '', seq[1])) for seq in fastas
        )
        self.encodings = pd.DataFrame(
            np.array(ctd).reshape(
                len(fastas), int(len(self.properties) * 3 * (2 + 5))
            ),
            index=self.marks,
            columns=self.headers,
        )

    def Count(self, seq1, seq2):
        sum = 0
        for aa in seq1:
            sum += seq2.count(aa)
        return sum

    def Count1(self, aaSet, sequence):
        number = 0
        for aa in sequence:
            if aa in aaSet:
                number = number + 1
        cutoffNums = [
            1,
            math.floor(0.25 * number),
            math.floor(0.50 * number),
            math.floor(0.75 * number),
            number,
        ]
        cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]

        code = []
        for cutoff in cutoffNums:
            myCount = 0
            for i in range(len(sequence)):
                if sequence[i] in aaSet:
                    myCount += 1
                    if myCount == cutoff:
                        code.append((i + 1) / len(sequence) * 100)
                        break
            if myCount == 0:
                code.append(0)
        return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage='USAGE:', description='Generating CTD vectors'
    )
    parser.add_argument(
        '--file', required=True, help='input protein sequence fasta file'
    )
    parser.add_argument(
        '--out', default='CTD.csv', help='the generated CTD vectors file'
    )
    args = parser.parse_args()
    output = args.out if args.out is not None else 'CTD.csv'

    ctd = CTD()
    ctd.fit(fastas=args.file)
    ctd.encodings.to_csv(output)

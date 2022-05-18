import re
import os
import sys


# from iFeature
def ReadFasta(file):
    if os.path.exists(file) is False:
        print('Error: "' + file + '" does not exist.')
        sys.exit(1)

    with open(file) as f:
        records = f.read()

    if re.search('>', records) is None:
        print('The input file seems not in fasta format.')
        sys.exit(1)

    records = records.split('\n>')
    records[0] = records[0].split('>')[1]

    myFasta = []
    for fasta in records:
        array = fasta.split('\n')

        name = array[0]

        sequence = re.sub(
            '[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper()
        )
        myFasta.append([name, sequence])
    return myFasta

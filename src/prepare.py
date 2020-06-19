import io
import sys
import xml.etree.ElementTree
import random
import re
import os
import errno
import zipfile


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython prepare.py data\n')
        sys.exit(1)

    # Maybe implement other ratios if necessary, Test data set split ratio
    split = 0.20
    random.seed(20170426)

    input = sys.argv[1]
    output = sys.argv[2]

    input_train = os.path.join(input, 'trainset.zip')
    input_test = os.path.join(input, 'testset.zip')
    print(input_train, input_test)
    
    zf = zipfile.ZipFile(input_train, 'r')
    zf.extractall(output)
    zf.close()
    zf = zipfile.ZipFile(input_test, 'r')
    zf.extractall(output)
    zf.close()





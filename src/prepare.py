import io
import sys
import xml.etree.ElementTree
import random
import re
import os
import errno


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
    # Test data set split ratio
    split = 0.20
    random.seed(20170426)

    input = sys.argv[0]
    output_train = os.path.join('data', 'prepared', 'train.tsv')
    output_test = os.path.join('data', 'prepared', 'test.tsv')

    mkdir_p(os.path.join('data', 'prepared'))
    writepath = os.path.join('data', 'prepared', 'train.txt')
    mode = 'a' if os.path.exists(writepath) else 'w'
    with open(writepath, mode) as f:
        f.write('Training')
    writepath = os.path.join('data', 'prepared', 'test.txt')
    mode = 'a' if os.path.exists(writepath) else 'w'
    with open(writepath, mode) as f:
        f.write('Testing')




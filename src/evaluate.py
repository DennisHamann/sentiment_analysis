import sys
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

    if len(sys.argv) != 4:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython evaluate.py model features output\n')
        sys.exit(1)

    model_file = sys.argv[1]
    matrix_file = os.path.join(sys.argv[2], 'test.pkl')
    metrics_file = sys.argv[3]
    mkdir_p(sys.argv[3])
    writepath = os.path.join(sys.argv[3], 'auc.json')
    mode = 'a' if os.path.exists(writepath) else 'w'
    with open(writepath, mode) as fd:
        fd.write('0 - 100')

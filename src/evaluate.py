import sys
import os

if __name__ == '__main__':


    if len(sys.argv) != 4:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython evaluate.py model features output\n')
        sys.exit(1)

    model_file = sys.argv[1]
    matrix_file = os.path.join(sys.argv[2], 'test.pkl')
    metrics_file = sys.argv[3]

    with open(metrics_file, 'w') as fd:
        fd.write('0 - 100')

import sys
import os







def train_model(input, output):
    # todo choose model, (remove shadowing?)
    print('train model....')

    with open(output, 'w') as f:
        f.write('This is a trained model')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython train.py features model\n')
        sys.exit(1)
    input = sys.argv[1]
    output = sys.argv[2]
    print('data path:', input)
    print('output path:', output)
    writepath = output
    mode = 'a' if os.path.exists(writepath) else 'w'
    with open(writepath, mode) as f:
        f.write('Output')

import os
import sys
import errno


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_features(data_prepated_path: str, output_path: str):
    # todo implement
    print('e.g. do principal component analysis in this step')
    print('split test train set')

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for mode in ['test', 'train']:
        with open(os.path.join(output_path, f'features_{mode}.txt'), 'w') as f:
            f.write('this is a feature array of shape '
                    '(num_distinct_images_in_triplets, '
                    'len(feature_vector))')

        with open(os.path.join(output_path, f'triplet_{mode}.txt'), 'w') as f:
            f.write('this is a dict containing image relations')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython featurization.py data-dir-path features-dir-path\n')
        sys.exit(1)
    train_input = os.path.join(sys.argv[1], 'train.tsv')
    test_input = os.path.join(sys.argv[1], 'test.tsv')
    train_output = os.path.join(sys.argv[2], 'train.pkl')
    test_output = os.path.join(sys.argv[2], 'test.pkl')
    mkdir_p(sys.argv[2])

from joeynmt.metrics import chrf, bleu, token_accuracy, sequence_accuracy
import scripts.optimizers as optimizers
import pickle

EVAL_HELPER = {'bleu':bleu, 'chrf':chrf, 'token_accuracy':token_accuracy, 'sequence_accuracy':sequence_accuracy}
STEPSIZE_HELPER = {'adam': optimizers.Adam, 'momentum': optimizers.Momentum, 'sgd': optimizers.SGD}

def serialize(x, path):
    """
    Pickles a given object to a file.

    :param x: an object, could be anything
    :param path: a filename string
    """
    with open(path, 'wb') as f:
        pickle.dump(x, f)


def subarray_generator(arr, subarray_size):
    """
    Yields blocks of size subarray_size from a given array. The blocks are split off from 
    the first dimension of the array, e.g. array 100 x 20, subarray_size 10 -> block size: 10 x 20
    """
    i = 0
    while i < len(arr):
        yield arr[i : i + subarray_size], i
        i += subarray_size

                                                   
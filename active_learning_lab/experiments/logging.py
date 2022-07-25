import logging

from active_learning_lab.utils.numpy import get_class_histogram
from small_text.utils.labels import csr_to_list
from scipy.sparse import csr_matrix


def log_class_distribution(y, num_classes):
    logging.info('Class Distribution:')
    if isinstance(y, csr_matrix):
        multilabel = True
        y = csr_to_list(y)
    else:
        multilabel = False
    logging.info(get_class_histogram(y, num_classes, normalize=False, multilabel=multilabel))


def log_run_info(run_id, run_max, len_train, len_test):
    logging.info('#--------------------------------')
    logging.info('## Split: %d of %d', run_id, run_max)
    logging.info('##   Train %d / Test %d', len_train, len_test)
    logging.info('#--------------------------------')

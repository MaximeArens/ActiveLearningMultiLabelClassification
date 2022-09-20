"""Example of a transformer-based active learning multi-label text classification.

Note:
This examples requires the datasets library.
"""
import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from small_text.active_learner import PoolBasedActiveLearner
from small_text.exceptions import ActiveLearnerException
from small_text.initialization import random_initialization_stratified
from small_text.integrations.transformers import TransformerModelArguments
from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory
from small_text.query_strategies import PoolExhaustedException, EmptyPoolException
from small_text.query_strategies import RandomSampling

#from examplecode.data.example_data_multilabel import (
#    get_train_test
#)
#from examplecode.data.example_data_transformers import preprocess_data
#from examplecode.shared import evaluate_multi_label


TRANSFORMER_MODEL = TransformerModelArguments('distilroberta-base')


try:
    import datasets  # noqa: F401
except ImportError:
    raise ActiveLearnerException('This example requires the "datasets" library. '
                                 'Please install datasets to run this example.')


def get_train_test():
    return get_go_emotions_dataset()


def main(num_iterations=10):
    # Active learning parameters
    num_classes = 28
    clf_factory = TransformerBasedClassificationFactory(TRANSFORMER_MODEL,
                                                        num_classes,
                                                        kwargs=dict({
                                                            'device': 'cpu',
                                                            'multi_label': True
                                                        }))
    query_strategy = RandomSampling()

    # Prepare some data
    train, test = get_train_test()

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL.model, cache_dir='.cache/')
    train = preprocess_data(tokenizer, train['text'][:5000], train['labels'][:5000], multi_label=True)

    test = preprocess_data(tokenizer, test['text'][:5000], test['labels'][:5000], multi_label=True)

    # Active learner
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)
    indices_labeled = initialize_active_learner(active_learner, train.y)

    try:
        perform_active_learning(active_learner, train, indices_labeled, test, num_iterations)
    except PoolExhaustedException:
        print('Error! Not enough samples left to handle the query.')
    except EmptyPoolException:
        print('Error! No more samples left. (Unlabeled pool is empty)')


def perform_active_learning(active_learner, train, indices_labeled, test, num_iterations):
    # Perform 10 iterations of active learning...
    for i in range(num_iterations):
        # ...where each iteration consists of labelling 20 samples
        indices_queried = active_learner.query(num_samples=1000)

        # Simulate user interaction here. Replace this for real-world usage.
        y = train.y[indices_queried]

        # Return the labels for the current query to the active learner.
        active_learner.update(y)

        indices_labeled = np.concatenate([indices_queried, indices_labeled])

        print('Iteration #{:d} ({} samples)'.format(i, len(indices_labeled)))
        evaluate_multi_label(active_learner, train[indices_labeled], test)


def initialize_active_learner(active_learner, y_train):

    indices_initial = random_initialization_stratified(y_train, n_samples=2000)
    y_initial = y_train[indices_initial]

    active_learner.initialize_data(indices_initial, y_initial)

    return indices_initial


import numpy as np

from small_text.integrations.transformers.datasets import TransformersDataset


def preprocess_data(tokenizer, data, labels, max_length=500, multi_label=False):

    data_out = []

    for i, doc in enumerate(data):
        encoded_dict = tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            return_attention_mask=True,
            return_tensors='pt',
            truncation='longest_first'
        )

        if multi_label:
            data_out.append((encoded_dict['input_ids'],
                             encoded_dict['attention_mask'],
                             np.sort(labels[i])))
        else:
            data_out.append((encoded_dict['input_ids'],
                             encoded_dict['attention_mask'],
                             labels[i]))

    return TransformersDataset(data_out, multi_label=multi_label)


def evaluate_multi_label(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    # https://github.com/scikit-learn/scikit-learn/issues/18611
    print('Train accuracy: {:.2f}'.format(
        f1_score(y_pred.toarray(), train.y.toarray(), average='micro')))
    print('Test accuracy: {:.2f}'.format(f1_score(y_pred_test.toarray(),
                                                  test.y.toarray(), average='micro')))
    print('---')


def get_go_emotions_dataset():
    import datasets
    from datasets import concatenate_datasets
    go_emotions = datasets.load_dataset('go_emotions')

    return concatenate_datasets([go_emotions['train'], go_emotions['validation']]), go_emotions['test']

if __name__ == '__main__':
    import argparse
    import logging
    logging.getLogger('small_text').setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='An example that shows active learning '
                                                 'for multi-class multi-label text classification '
                                                 'using transformers.')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='number of active learning iterations')

    args = parser.parse_args()

    main(num_iterations=args.num_iterations)

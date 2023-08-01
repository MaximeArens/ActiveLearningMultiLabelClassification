import numpy as np

from abc import ABC, abstractmethod
from pathlib import Path

from scipy.sparse import csr_matrix
from scipy.special import softmax

from small_text.base import LABEL_IGNORED
from small_text.exceptions import LearnerNotInitializedException
from small_text.utils.data import list_length
from small_text.utils.labels import concatenate, get_ignored_labels_mask, remove_by_index
from small_text.version import __version__ as version
from small_text.active_learner import AbstractPoolBasedActiveLearner
from active_learning_lab.experiments.query_strategies import query_strategy_from_str
from small_text.utils.labels import csr_to_list
from active_learning_lab.utils.numpy import get_class_histogram



import math



class AlternatePoolBasedActiveLearner(AbstractPoolBasedActiveLearner):
    """A pool-based active learner in which a pool holds all available unlabeled data.

    It uses a classifier, a query strategy and manages the mutually exclusive partition over the
    whole training data into labeled and unlabeled.

    Parameters
    ----------
    clf_factory : small_text.classifiers.factories.AbstractClassifierFactory
        A factory responsible for creating new classifier instances.
    query_strategy : small_text.query_strategies.QueryStrategy
        Query strategy which is responsible for selecting instances during a `query()` call.
    dataset : ~small_text.data.datasets.Dataset
        A training dataset that is supported by the underlying classifier.
    reuse_model : bool, default=False
        Reuses the previous model during retraining (if a previous model exists),
        otherwise creates a new model for each retraining.

    Attributes
    ----------
    indices_labeled : numpy.ndarray
        Indices of instances (relative to `self.x_train`) constituting the labeled pool.
    indices_ignored : numpy.ndarray or scipy.sparse.csr_matrix
        Indices of instances (relative to `self.x_train`) which have been ignored,
        i.e. which will never be returned by a query.
    y : numpy.ndarray or scipy.sparse.csr_matrix
        Labels for the the current labeled pool. Each tuple `(x_indices_labeled[i], y[i])`
        represents one labeled sample.
    indices_queried : numpy.ndarray or None
        Queried indices returned by the last `query()` call, or `None` if no query has been
        executed yet.
    """

    def __init__(self, clf_factory, query_strategy, dataset, wait_strategy, reuse_model=False):
        self._clf = None
        self._clf_factory = clf_factory
        self._query_strategy = query_strategy

        self._index_to_position = None

        self.dataset = dataset
        self.wait_strategy = wait_strategy
        self.reuse_model = reuse_model

        self.indices_labeled = np.empty(shape=0, dtype=int)
        self.indices_ignored = np.empty(shape=0, dtype=int)

        self.y = None
        self.indices_queried = None

    def initialize_data(self, indices_initial, y_initial, indices_ignored=None,
                        indices_validation=None, retrain=True):
        """(Re-)Initializes the current labeled pool.

        This is required once before the first `query()` call, and whenever the labeled pool
        is changed from the outside, i.e. when `self.x_train` changes.

        Parameters
        ----------
        indices_initial : numpy.ndarray
            A list of indices (relative to `self.x_train`) of initially labeled samples.
        y_initial : numpy.ndarray or or scipy.sparse.csr_matrix
            Label matrix. One row corresponds to an index in `x_indices_initial`. If the
            passed type is numpy.ndarray (dense) all further label-based operations assume dense
            labels, otherwise sparse labels for scipy.sparse.csr_matrix.
        indices_ignored : numpy.ndarray
            List of ignored samples which will be invisible to the query strategy.
        indices_validation : numpy.ndarray, default=None
            The given indices (relative to `self.x_indices_labeled`) define a custom validation set
            if provided. Otherwise each classifier that uses a validation set will be responsible
            for creating a validation set. Only used if `retrain=True`.
        retrain : bool, default=True
            Retrains the model after the update if True.
        """
        self.indices_labeled = indices_initial
        self._index_to_position = self._build_index_to_position_dict()
        self.y = y_initial

        if isinstance(self.y, csr_matrix):
            self.multi_label = True
        else:
            self.multi_label = False

        if indices_ignored is not None:
            self.indices_ignored = indices_ignored
        else:
            self.indices_ignored = np.empty(shape=(0), dtype=int)

        if retrain:
            self._retrain(indices_validation=indices_validation)

    def query(self, num_samples=10, representation=None, query_strategy_kwargs=dict()):
        """Performs a query step, which selects a number of samples from the unlabeled pool.
        A query step must be followed by an update step.

        Parameters
        ----------
        num_samples : int, default=10
            Number of samples to query.
        representation : numpy.ndarray, default=None
            Alternative representation for the samples in the unlabeled pool. his can be used
            if you want to rely pre-computed fixed representations instead of embeddings that
            change during each active learning iteration.
        query_strategy_kwargs : dict, default=dict()

        Returns
        -------
        queried_indices : numpy.ndarray[int]
            List of queried indices (relative to the current unlabeled pool).

        Raises
        ------
        LearnerNotInitializedException
            Thrown when the active learner was not initialized via `initialize_data(...)`.
        ValueError
            Raised when args or kwargs are not used and consumed.
        """
        if self._index_to_position is None:
            raise LearnerNotInitializedException()

        size = list_length(self.dataset)
        if representation is not None and size != list_length(representation):
            raise ValueError('Number of rows of alternative representation x must match the train '
                             'set (dim 0).')

        self.mask = np.ones(size, bool)
        self.mask[np.concatenate([self.indices_labeled, self.indices_ignored])] = False
        indices = np.arange(size)

        representation = self.dataset if representation is None else representation
        # calculate num_samples for alternative query and active learning query
        num_samples_alternate = math.floor(num_samples / 2)
        num_samples_active_learning = math.ceil(num_samples / 2)
        #get round of active learning
        round_al = int(self.y.shape[0]/num_samples)
        # instances to query during wait time
        wait_indices_queried = self.alternative_query(representation,
                                                 indices[self.mask],
                                                 num_samples_alternate,
                                                 round_al,
                                                 self.wait_strategy)
        # update mask with instances queried during wait time
        self.mask[np.concatenate([self.indices_labeled, self.indices_ignored, wait_indices_queried])] = False
        # instances to query with active learning
        active_learning_indices_queried = self.query_strategy.query(self._clf,
                                                         representation,
                                                         indices[self.mask],
                                                         self.indices_labeled,
                                                         self.y,
                                                         n=num_samples_active_learning,
                                                         **query_strategy_kwargs)
        self.indices_queried = np.concatenate([wait_indices_queried, active_learning_indices_queried])
        return self.indices_queried


    def alternative_query(self, dataset, indices_unlabeled, n, round_al, wait_strategy):
        if round_al == 1:
            query_strategy_kwargs = dict()
            strategy = query_strategy_from_str('rd', query_strategy_kwargs)
            indices_to_query = strategy.query(self._clf,
                           dataset,
                           indices_unlabeled,
                           self.indices_labeled,
                           self.y,
                           n=n,
                           **query_strategy_kwargs)
        else:
            if wait_strategy == "random":
                query_strategy_kwargs = dict()
                strategy = query_strategy_from_str('rd', query_strategy_kwargs)
                indices_to_query = strategy.query(self._clf,
                                         dataset,
                                         indices_unlabeled,
                                         self.indices_labeled,
                                         self.y,
                                         n=n,
                                         **query_strategy_kwargs)
            if wait_strategy == "stale":
                confidence = self.query_strategy.scores_
                if hasattr(self.query_strategy, 'lower_is_better'):
                    if not self.query_strategy.lower_is_better:
                        confidence = -confidence
                if self.query_strategy.scores_.size == self.mask.size:
                    indices_partitioned = np.argpartition(confidence[indices_unlabeled], range(n))[:n]
                    indices_to_query = np.array([indices_unlabeled[i] for i in indices_partitioned])
                else:
                    indices_partitioned = np.argpartition(confidence, range(n))[:n]
                    indices_to_query = np.array([indices_unlabeled[i] for i in indices_partitioned])

            if wait_strategy == "equilibrate_label":

                num_labels = self.y.shape[1]
                label_histogram = get_class_histogram(csr_to_list(self.y), num_labels, normalize=False, multilabel=self.multi_label)

                if self.query_strategy.scores_.size == self.mask.size:
                    confidence_score = self.query_strategy.scores_[indices_unlabeled]
                    proba = self.query_strategy.proba_[indices_unlabeled]
                else:
                    confidence_score = self.query_strategy.scores_
                    proba = self.query_strategy.proba_
                    
                if hasattr(self.query_strategy, 'lower_is_better'):
                    if not self.query_strategy.lower_is_better:
                        confidence_score = -confidence_score

                # Calculate intermediate weighting vector
                inter_weight = [0]*num_labels
                for i, _ in enumerate(inter_weight):
                    if label_histogram[i] != 0:
                        inter_weight[i] = (np.amax(label_histogram) + label_histogram[i])/(2*label_histogram[i])
                for i, _ in enumerate(inter_weight):
                    max_weight = max(inter_weight)
                    if label_histogram[i] == 0:
                        inter_weight[i] = 2*max_weight
                normalized_weight = softmax(inter_weight)
                equilibrated_score = np.full_like(confidence_score, 0)
                for i, _ in enumerate(equilibrated_score):
                    proba_label = proba[i, :]
                    label_score = np.sum(np.multiply(proba_label, normalized_weight))
                    equilibrated_score = label_score*confidence_score

                # Get the indices with highest scores
                indices_partitioned = np.argpartition(equilibrated_score, range(n))[-n:]
                indices_to_query = np.array([indices_unlabeled[i] for i in indices_partitioned])
        return indices_to_query

    def update(self, y, indices_validation=None):
        """Performs an update step, which passes the label for each of the
        previously queried indices.

        An update step must be preceded by a query step. At the end of the update step the
        current model is retrained using all available labels.

        Parameters
        ----------
        y : numpy.ndarray or scipy.sparse.csr_matrix
            Labels provided in response to the previous query. Each label at index i corresponds
            to the sample x[i] for single-label data (ndarray) and each row of labels at index i
            corresponds to the sample x[i] for multi-label data (csr_matrix). Setting the label /
            row of labels to ` small_text.base import LABEL_IGNORED` will ignore the respective
            sample.
        indices_validation : numpy.ndarray, default=None
            The given indices (relative to `self.x_indices_labeled`) define a custom validation set
            if provided. Otherwise each classifier that uses a validation set will be responsible
            for creating a validation set.
        """
        if self.indices_queried.shape[0] != y.shape[0]:
            raise ValueError('Query-update mismatch: indices queried - {} / labels provided - {}'
                             .format(self.indices_queried.shape[0], y.shape[0])
                             )

        ignored = get_ignored_labels_mask(y, LABEL_IGNORED)
        if ignored.any():
            y = remove_by_index(y, np.arange(y.shape[0])[ignored])
            self.indices_ignored = np.concatenate([self.indices_ignored, self.indices_queried[ignored]])

        self.indices_labeled = np.concatenate([self.indices_labeled, self.indices_queried[~ignored]])
        self._index_to_position = self._build_index_to_position_dict()

        if self.indices_labeled.shape[0] != np.unique(self.indices_labeled).shape[0]:
            raise ValueError('Duplicate indices detected in the labeled pool! '
                             'Please re-examine your query strategy.')

        if not ignored.all():
            self.y = concatenate(self.y, y)
            self._retrain(indices_validation=indices_validation)

        self.indices_queried = None
        self.mask = None

    def update_label_at(self, index, y, retrain=False, indices_validation=None):
        """Updates the label for the given x_index (with regard to `self.x_train`).

        Notes
        -----
        After adding labels the current model might not reflect the labeled data anymore.
        You should consider if a retraining is necessary when using this operation.
        Since retraining is often time-consuming, `retrain` is set to `False` by default.

        Parameters
        ----------
        index : int
            Data index (relative to `self.x_train`) for which the label should be updated.
        y : int or numpy.ndarray
            The new label(s) to be assigned for the sample at `self.x_indices_labeled[x_index]`.
        retrain : bool, default=False
            Retrains the model after the update if True.
        indices_validation : numpy.ndarray, default=None
            The given indices (relative to `self.x_indices_labeled`) define a custom validation set
            if provided. This is only used if `retrain` is `True`.
        """
        position = self._index_to_position[index]
        self.y[position] = y

        if retrain:
            self._retrain(indices_validation=indices_validation)

    def remove_label_at(self, x_index, retrain=False, x_indices_validation=None):
        """Removes the labeling for the given x_index (with regard to `self.x_train`).

        Notes
        -----
        After removing labels the current model might not reflect the labeled data anymore.
        You should consider if a retraining is necessary when using this operation.
        Since retraining is often time-consuming, `retrain` is set to `False` by default.

        Parameters
        ----------
        x_index : int
            Data index (relative to `self.x_train`) for which the label should be removed.
        retrain : bool, default=None
            Retrains the model after removal if True.
        x_indices_validation : numpy.ndarray, default=None
            The given indices (relative to `self.x_indices_labeled`) define a custom validation set
            if provided. This is only used if `retrain` is `True`.
        """
        position = self._index_to_position[x_index]
        self.y = remove_by_index(self.y, position)
        self.indices_labeled = np.delete(self.indices_labeled, position)

        if retrain:
            self._retrain(indices_validation=x_indices_validation)

    def ignore_sample_at(self, index, retrain=False, indices_validation=None):
        """Ignores the sample at the given `x_index.

        Any labels which had previously been assigned to this sample will be removed.

        Notes
        -----
        If ignoring a sample incurs the removal of a label label, the current model might not
        reflect the labeled data anymore. You should consider if a retraining is necessary when
        using this operation. Since retraining is often time-consuming, `retrain` is set to
        `False` by default.

        Parameters
        ----------
        index : int
           Data index (relative to `self.x_train`) for which the label should be ignored.
        retrain : bool, default=False
            Retrains the model after the removal if True.
        indices_validation : numpy.ndarray, default=None
            The given indices (relative to `self.x_indices_labeled`) define a custom validation set
            if provided. This is only used if `retrain` is `True`.
        """

        labeling_exists = index in self._index_to_position
        if labeling_exists:
            position = self._index_to_position[index]
            self.y = remove_by_index(self.y, position)
            self.indices_labeled = np.delete(self.indices_labeled, position)

        self.indices_ignored = np.concatenate([self.indices_ignored, [index]])

        if labeling_exists and retrain:
            self._retrain(indices_validation=indices_validation)

    def save(self, file):
        """Serializes the current active learner object into a single file for later re-use.

        Parameters
        ----------
        file : str or path or file
            Serialized output file to be written.
        """
        if isinstance(file, (str, Path)):
            with open(file, 'wb+') as f:
                self._save(f)
        else:
            self._save(file)

    def _save(self, file_handle):
        import dill as pickle
        pickle.dump(version, file_handle)
        pickle.dump(self, file_handle)

    @classmethod
    def load(cls, file):
        """Deserializes a serialized active learner.

        Parameters
        ----------
        file : str or path or file
            File to be loaded.
        """
        if isinstance(file, (str, Path)):
            with open(file, 'rb') as f:
                return cls._load(f)
        else:
            return cls._load(file)

    @classmethod
    def _load(cls, file_handle):
        import dill as pickle
        _ = pickle.load(file_handle)  # version, will be used in the future
        return pickle.load(file_handle)

    @property
    def classifier(self):
        return self._clf

    @property
    def query_strategy(self):
        return self._query_strategy

    def _retrain(self, indices_validation=None):
        if self._clf is None or not self.reuse_model:
            if hasattr(self, '_clf'):
                del self._clf
            self._clf = self._clf_factory.new()

        dataset = self.dataset[self.indices_labeled].clone()
        dataset.y = self.y

        if indices_validation is None:
            self._clf.fit(dataset)
        else:
            indices = np.arange(self.indices_labeled.shape[0])
            mask = np.isin(indices, indices_validation)

            train = dataset[indices[~mask]]
            valid = dataset[indices[mask]]

            self._clf.fit(train, validation_set=valid)

    def _build_index_to_position_dict(self):
        return dict({
            x_index: position for position, x_index in enumerate(self.indices_labeled)
        })

import numpy as np
from scipy.special import rel_entr
from scipy.spatial.distance import cdist

from small_text.utils.context import build_pbar_context
from small_text.query_strategies import (
    BreakingTies,
    EmbeddingBasedQueryStrategy,
    LeastConfidence,
    RandomSampling,
    PredictionEntropy,
    SubsamplingQueryStrategy,
    ConfidenceBasedQueryStrategy,
    QueryStrategy)


def query_strategy_from_str(query_strategy_name, kwargs):

    if query_strategy_name == 'bt':
        strategy = BreakingTies()
    elif query_strategy_name == 'lc':
        strategy = LeastConfidence()
    elif query_strategy_name == 'pe':
        strategy = PredictionEntropy()
    elif query_strategy_name == 'ca':
        strategy = ContrastiveActiveLearning()
    elif query_strategy_name == 'rd':
        strategy = RandomSampling()
    elif query_strategy_name == 'lct':
        strategy = LeastConfidenceThreshold()
    elif query_strategy_name == 'lmc':
        strategy = LeastMeanConfidence()
    elif query_strategy_name == 'ml':
        strategy = MaxLoss()
    elif query_strategy_name == 'mml':
        strategy = MeanMaxLoss()
    elif query_strategy_name == 'binmin':
        strategy = BinaryMinimum()
    elif query_strategy_name == 'cmn':
        strategy = MinConfidenceNoWeighting()
    elif query_strategy_name == 'mmc':
        strategy = MaximumLossReductionWithMaximalConfidence()
    elif query_strategy_name == 'mmu':
        strategy = MaxMarginUncertainty()
    elif query_strategy_name == 'lci':
        strategy = LabelCardinalityInconsistency()
    elif query_strategy_name == 'cvirs':
        strategy = CategoryVectorInconsistencyAndRanking()
    elif query_strategy_name == 'hrf':
        strategy = HighReadabilityFirst()
    elif query_strategy_name == 'pr':
        strategy = ProgressiveReadability()

    else:
        raise ValueError(f'Unknown query strategy string: {query_strategy_name}')

    if kwargs is not None and 'subsample' in kwargs:
        subsample_size = int(kwargs['subsample'])
        strategy = SubsamplingQueryStrategy(strategy, subsample_size)

    return strategy


class HighReadabilityFirst(QueryStrategy):
    """Selects instances with better readability first."""

    def query(self, clf, _dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        readability = _dataset.readability
        if len(indices_unlabeled) <= n:
            return np.array(indices_unlabeled)

        indices_partitioned = np.argpartition(readability[indices_unlabeled], n)[:n]
        return np.array([indices_unlabeled[i] for i in indices_partitioned])

    def __str__(self):
        return 'HighReadabilityFirst()'


class ProgressiveReadability(QueryStrategy):
    """Following curriculum learning principles selects instances from descending readability. Easiest examples to start
    followed by more difficult ones and so on. """

    def query(self, clf, _dataset, indices_unlabeled, indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        readability = _dataset.readability

        if (len(indices_labeled) % 100) == 0:
            if _dataset.readability[indices_labeled[len(indices_labeled)-1]] == _dataset.readability[indices_labeled[len(indices_labeled)-50]]:
                for i in range(len(_dataset.readability)):
                    if _dataset.readability[i] == _dataset.readability[indices_labeled[len(indices_labeled)-1]]:
                        _dataset.readability[i] = _dataset.readability[i] + 1000

        if len(indices_unlabeled) <= n:
            return np.array(indices_unlabeled)

        indices_partitioned = np.argpartition(_dataset.readability[indices_unlabeled], n)[:n]
        return np.array([indices_unlabeled[i] for i in indices_partitioned])

    def __str__(self):
        return 'ProgressiveReadability()'


class MaxLoss(ConfidenceBasedQueryStrategy):
    """
    Max Loss query strategy for multilabel classification.

    For more details on this query strategy, see Li et al., Multilabel SVM active learning for image classification
    (http://dx.doi.org/10.1109/ICIP.2004.1421535)

    """
    def __init__(self):
        super().__init__(lower_is_better=False)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        proba = clf.predict_proba(dataset)
        predictions = 2 * proba - 1
        num_class = np.shape(_y)[1]
        cls_mtx = -np.ones(shape=(len(dataset), num_class))
        for inst_id, most_certain_class in enumerate(proba.argmax(axis=1)):
            cls_mtx[inst_id, most_certain_class] = 1
        cls_loss = np.maximum(1 - np.multiply(cls_mtx, predictions), 0).sum(axis=1)
        return cls_loss

    def __str__(self):
        return 'MaxLoss()'


class MeanMaxLoss(ConfidenceBasedQueryStrategy):
    """
    Mean Max Loss query strategy for multilabel classification.

    For more details on this query strategy, see Li et al., Multilabel SVM active learning for image classification
    (http://dx.doi.org/10.1109/ICIP.2004.1421535)

    """
    def __init__(self):
        super().__init__(lower_is_better=False)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        proba = clf.predict_proba(dataset)
        predictions = 2 * proba - 1
        num_class = np.shape(_y)[1]
        cls_mtx = 2*np.eye(num_class, num_class)-1
        loss_mtx = np.maximum(1-np.dot(predictions, cls_mtx), 0)
        return loss_mtx.mean(axis=1)

    def __str__(self):
        return 'MeanMaxLoss()'


class BinaryMinimum(ConfidenceBasedQueryStrategy):
    """
    Binary minimum for query strategy for multilabel classification.

    For more details on this query strategy, see Klaus Brinker, On Active Learning in Multi-label Classification
    (https://link.springer.com/chapter/10.1007%2F3-540-31314-1_24)

    """
    def __init__(self):
        super().__init__(lower_is_better=True)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        proba = clf.predict_proba(dataset)
        num_class = np.shape(_y)[1]
        threshold = 0.5 * np.ones(shape=(len(dataset), num_class))
        distance = np.abs(proba-threshold)
        distance_min = distance.min(axis=1)
        return distance_min

    def __str__(self):
        return 'BinaryMinimum()'


class MinConfidenceNoWeighting(ConfidenceBasedQueryStrategy):
    """
    MinConfidence query strategy for multilabel classification.
    For more details on this query strategy, see
    Esuli and Sebastiani., Active Learning Strategies for Multi-Label Text Classification
    (http://dx.doi.org/10.1007/978-3-642-00958-7_12)

    """
    def __init__(self):
        super().__init__(lower_is_better=True)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        proba = clf.predict_proba(dataset)
        # range from [0,1] to [-1,1]
        proba = 2 * proba - 1
        return np.abs(proba).min(axis=1)

    def __str__(self):
        return 'MinConfidenceNoWeighting()'


class MaximumLossReductionWithMaximalConfidence(ConfidenceBasedQueryStrategy):
    """


    """
    def __init__(self):
        super().__init__(lower_is_better=bool)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):

        return None

    def __str__(self):
        return 'MaximumLossReductionWithMaximalConfidence()'


class MaxMarginUncertainty(ConfidenceBasedQueryStrategy):
    """


    """

    def __init__(self):
        super().__init__(lower_is_better=False)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        proba = clf.predict_proba(dataset)
        # range from [0,1] to [-1,1]
        proba = 2 * proba - 1
        positive = np.copy(proba)
        positive[proba <= 0] = 100
        negative = np.copy(proba)
        positive[proba >= 0] = -100
        separation_margin = positive.min(axis=1) - negative.max(axis=1)
        return 1. / separation_margin

    def __str__(self):
        return 'MaxMarginUncertainty()'


class LabelCardinalityInconsistency(ConfidenceBasedQueryStrategy):
    """


    """

    def __init__(self):
        super().__init__(lower_is_better=False)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        proba = clf.predict_proba(dataset)
        average_positive_label = _y.mean(axis=0).sum()
        label_cardinality = np.sqrt((proba.sum(axis=1) - average_positive_label)**2)
        return label_cardinality

    def __str__(self):
        return 'LabelCardinalityInconsistency()'


class LeastConfidenceThreshold(ConfidenceBasedQueryStrategy):
    """Selects instances where the prediction score is closest to the decision threshold"""

    def __init__(self):
        super().__init__(lower_is_better=False)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        proba = clf.predict_proba(dataset)
        uncertainty_array = (1/(np.abs(proba-(1-proba))+0.01))/100
        confidence = []
        for uncertainty in uncertainty_array:
            score = sum(uncertainty)
            confidence.append(score / clf.num_classes)
        return np.array(confidence)

    def __str__(self):
        return 'LeastConfidenceThreshold()'


class LeastMeanConfidence(ConfidenceBasedQueryStrategy):
    """Selects instances with the least mean prediction confidence (on all classes)"""

    def __init__(self):
        super().__init__(lower_is_better=True)

    def get_confidence(self, clf, dataset, _indices_unlabeled, _indices_labeled, _y):
        proba = clf.predict_proba(dataset)
        uncertainty_array = np.abs(proba - (1-proba))
        confidence = []
        for uncertainty in uncertainty_array:
            score = sum(uncertainty)
            confidence.append(score / clf.num_classes)
        return np.array(confidence)

    def __str__(self):
        return 'LeastMeanConfidence()'


class ContrastiveActiveLearning(EmbeddingBasedQueryStrategy):
    """Contrastive Active Learning [MVB+21]_ selects instances whose k-nearest neighbours
    exhibit the largest mean Kullback-Leibler divergence.

    .. [MVB+21] Katerina Margatina, Giorgos Vernikos, Loïc Barrault, and Nikolaos Aletras. 2021.
       Active Learning by Acquiring Contrastive Examples.
       In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing,
          pages 650–663.
    """

    def __init__(self, k=10, embed_kwargs=dict(), normalize=True, batch_size=100, pbar='tqdm'):
        """
        Parameters
        ----------
        k : int
            Number of nearest neighbours whose KL divergence is considered.
        embed_kwargs : dict
            Embedding keyword args which are passed to `clf.embed()`.
        normalize : bool, default=True
            Embeddings will be L2 normalized if `True`, otherwise they remain unchanged.
        batch_size : int, default=100
            Batch size which is used to process the embeddings.
        """
        self.embed_kwargs = embed_kwargs
        self.normalize = normalize
        self.k = k
        self.batch_size = batch_size
        self.pbar = pbar

    def query(self, clf, dataset, indices_unlabeled, indices_labeled, y, n=10, pbar='tqdm',
              embeddings=None, embed_kwargs=dict()):

        return super().query(clf, dataset, indices_unlabeled, indices_labeled, y, n=n,
                             embed_kwargs=self.embed_kwargs, pbar=self.pbar)

    def sample(self, _clf, dataset, indices_unlabeled, _indices_labeled, _y, n, embeddings,
               embeddings_proba=None):
        from sklearn.neighbors import NearestNeighbors

        if embeddings_proba is None:
            raise ValueError('Error: embeddings_proba is None. '
                             'This strategy requires a classifier whose embed() method '
                             'supports the return_proba kwarg.')

        if self.normalize:
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, axis=1)

        nn = NearestNeighbors(n_neighbors=n)
        nn.fit(embeddings)

        return self._contrastive_active_learning(dataset, embeddings, embeddings_proba,
                                                 indices_unlabeled, nn, n)

    def _contrastive_active_learning(self, dataset, embeddings, embeddings_proba,
                                     indices_unlabeled, nn, n):
        scores = []

        embeddings_unlabelled_proba = embeddings_proba[indices_unlabeled]
        embeddings_unlabeled = embeddings[indices_unlabeled]

        num_batches = int(np.ceil(len(dataset) / self.batch_size))
        offset = 0
        for batch_idx in np.array_split(np.arange(indices_unlabeled.shape[0]), num_batches,
                                        axis=0):

            nn_indices = nn.kneighbors(embeddings_unlabeled[batch_idx],
                                       n_neighbors=self.k,
                                       return_distance=False)

            kl_divs = np.apply_along_axis(lambda v: np.mean([
                rel_entr(embeddings_proba[i], embeddings_unlabelled_proba[v])
                for i in nn_indices[v - offset]]),
                0,
                batch_idx[None, :])

            scores.extend(kl_divs.tolist())
            offset += batch_idx.shape[0]

        scores = np.array(scores)
        indices = np.argpartition(-scores, n)[:n]

        return indices

    def __str__(self):
        return f'ContrastiveActiveLearning(k={self.k}, ' \
               f'embed_kwargs={str(self.embed_kwargs)}, ' \
               f'normalize={self.normalize})'


class CategoryVectorInconsistencyAndRanking(QueryStrategy):
    """Uncertainty Sampling based on Category Vector Inconsistency and Ranking of Scores [RCV18]_
    selects instances based on the inconsistency of predicted labels and per-class label rankings.
    """

    def __init__(self, batch_size=2048, prediction_threshold=0.5, epsilon=1e-8, pbar='tqdm'):
        """
        Parameters
        ----------
        batch_size : int
            Batch size in which the computations are performed. Increasing the size increases
            the amount of memory used.
        prediction_threshold : float
            Confidence value above which a prediction counts as positive.
        epsilon : float
            A small value that is added to the argument of the logarithm to avoid taking the
            logarithm of zero.
        pbar : 'tqdm' or None, default='tqdm'
            Displays a progress bar if 'tqdm' is passed.
        """
        self.batch_size = batch_size
        self.prediction_threshold = prediction_threshold
        self.epsilon = epsilon
        self.pbar = pbar

    def query(self, clf, dataset, indices_unlabeled, _indices_labeled, y, n=10):
        self._validate_query_input(indices_unlabeled, n)

        y_proba = clf.predict_proba(dataset[indices_unlabeled])
        scores = self._compute_scores(indices_unlabeled, y, y_proba)

        if len(indices_unlabeled) == n:
            return np.array(indices_unlabeled)

        indices_queried = np.argpartition(-scores, n)[:n]
        return np.array([indices_unlabeled[i] for i in indices_queried])

    def _compute_scores(self, indices_unlabeled, y, proba):
        y_pred_unlabeled = (proba > self.prediction_threshold).astype(int)
        vector_inconsistency_scores = self._compute_vector_inconsistency(y,
                                                                         y_pred_unlabeled,
                                                                         proba.shape[1])
        ranking_scores = self._compute_ranking(indices_unlabeled, proba)
        return vector_inconsistency_scores * ranking_scores

    def _compute_vector_inconsistency(self, y, y_pred_unlabeled, num_classes):
        y_arr = y.toarray()

        num_batches = int(np.ceil(len(y_pred_unlabeled) / self.batch_size))

        vector_inconsistency = np.array([], dtype=np.float32)
        num_unlabeled = y_pred_unlabeled.shape[0]

        with build_pbar_context(self.pbar, tqdm_kwargs={'total': num_unlabeled}) as pbar:
            for batch_idx in np.array_split(np.arange(num_unlabeled), num_batches, axis=0):
                y_pred_unlabeled_sub = y_pred_unlabeled[batch_idx]
                # as an exception the variables a,b,c,d of the contingency table are adopted
                a = y_pred_unlabeled_sub.dot(y_arr.T)
                b = np.logical_not(y_pred_unlabeled_sub).dot(y_arr.T)
                c = y_pred_unlabeled_sub.dot(np.logical_not(y_arr).T)
                d = np.logical_not(y_pred_unlabeled_sub).dot(np.logical_not(y_arr).T).astype(int)

                hamming_distance = (b + c) / num_classes

                distance = self._distance(y_pred_unlabeled_sub, y_arr, num_classes,
                                          a, b, c, d, hamming_distance)
                distance = distance.sum(axis=1) / y_pred_unlabeled_sub.shape[0]
                vector_inconsistency = np.append(vector_inconsistency, distance)

                pbar.update(batch_idx.shape[0])

        return vector_inconsistency

    def _distance(self, y_pred_unlabeled_sub, y_arr, num_classes, a, b, c, d,
                  hamming_distance):

        distance = hamming_distance

        y_arr_ones = y_arr.sum(axis=1)
        y_arr_zeros = y_arr.shape[1] - y_arr_ones
        entropy_labeled = self._entropy(y_arr_ones, num_classes) \
            + self._entropy(y_arr_zeros, num_classes)
        entropy_labeled = np.tile(entropy_labeled[np.newaxis, :],
                                  (y_pred_unlabeled_sub.shape[0], 1))

        y_pred_unlabeled_sub_ones = y_pred_unlabeled_sub.sum(axis=1)
        y_pred_unlabeled_sub_zeros = y_pred_unlabeled_sub.shape[1] - y_pred_unlabeled_sub_ones
        entropy_unlabeled = self._entropy(y_pred_unlabeled_sub_ones, num_classes) \
            + self._entropy(y_pred_unlabeled_sub_zeros, num_classes)
        entropy_unlabeled = np.tile(entropy_unlabeled[:, np.newaxis], (1, y_arr.shape[0]))

        joint_entropy = self._entropy(b + c, num_classes) + self._entropy(a + d, num_classes)
        joint_entropy += (b + c) / num_classes \
            * (self._entropy(b, b + c)
               + self._entropy(c, b + c))
        joint_entropy += (a + d) / num_classes \
            * (self._entropy(a, a + d) + self._entropy(d, a + d))

        entropy_distance = 2 * joint_entropy - entropy_unlabeled - entropy_labeled
        entropy_distance /= (joint_entropy + self.epsilon)

        entropy_distance[hamming_distance == 1] = 1

        return entropy_distance

    def _entropy(self, numerator, denominator):
        ratio = numerator / (denominator + self.epsilon)
        result = -ratio * np.log2(ratio + self.epsilon)
        return result

    def _compute_ranking(self, indices_unlabeled, proba_unlabeled):
        num_unlabeled, num_classes = proba_unlabeled.shape[0], proba_unlabeled.shape[1]
        ranks = self._rank_by_margin(proba_unlabeled)

        ranking_denom = num_classes * (num_unlabeled - 1)

        ranking_scores = [
            sum([num_unlabeled - ranks[j, i]
                 for j in range(num_classes)]) / ranking_denom
            for i in range(indices_unlabeled.shape[0])
        ]
        return np.array(ranking_scores)

    def _rank_by_margin(self, proba):
        num_classes = proba.shape[1]

        #proba_sum = proba.sum(axis=1)
        #margin = proba - np.tile(proba_sum[:, np.newaxis], (1, num_classes))
        #margin = np.absolute(margin)
        margin = abs(2*proba - 1)
        ranks = np.array([
            np.argsort(margin[:, j])
            for j in range(num_classes)
        ])
        return ranks

    def __str__(self):
        return f'CategoryVectorInconsistencyAndRanking(batch_size={self.batch_size}, ' \
               f'prediction_threshold={self.prediction_threshold}, ' \
               f'epsilon={self.epsilon})'

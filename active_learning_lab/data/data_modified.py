import math
import torch
import datasets
import numpy as np

from enum import Enum

from sklearn.preprocessing import normalize
from small_text.data.datasets import SklearnDataset
from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
from small_text.integrations.transformers.datasets import TransformersDataset
from transformers import AutoTokenizer
from small_text.utils.labels import list_to_csr


TEST_SET_RATIO_DEFAULT = 0.1


class DataSets(Enum):
    JIGSAW = 'jigsaw'

    @staticmethod
    def from_str(enum_str):
        if enum_str == 'jigsaw':
            return DataSets.JIGSAW

        raise ValueError('Enum DataSets does not contain the given element: '
                         '\'{}\''.format(enum_str))


class DataSetType(Enum):
    TENSOR_PADDED_SEQ = 'tps'
    BOW = 'bow'
    RAW = 'raw'
    TRANSFORMERS = 'transformers'

    @staticmethod
    def from_str(enum_str):
        if enum_str == 'tps':
            return DataSetType.TENSOR_PADDED_SEQ
        elif enum_str == 'bow':
            return DataSetType.BOW
        elif enum_str == 'raw':
            return DataSetType.RAW
        elif enum_str == 'transformers':
            return DataSetType.TRANSFORMERS

        raise ValueError('Enum DataSetType does not contain the given element: '
                         '\'{}\''.format(enum_str))


class RawDataset(SklearnDataset):

    def __init__(self, x, y, target_labels=None):
        super().__init__(x, y, target_labels=target_labels)
        self.x = np.array(x)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y
        if self.track_target_labels:
            self._infer_target_labels(self._y)

    @property
    def target_labels(self):
        return self._target_labels

    @target_labels.setter
    def target_labels(self, target_labels):
        self._target_labels = target_labels

    def __getitem__(self, item):
        if isinstance(item, list) or isinstance(item, np.ndarray) or isinstance(item, slice):
            return RawDataset(np.array(self._x[item]), np.array(self._y[item]))

        ds = RawDataset(self._x[item], self._y[item])
        if len(ds._x.shape) <= 1:
            ds._x = np.expand_dims(ds._x, axis=0)
            ds._y = np.expand_dims(ds._y, axis=0)

        return ds

    def __iter__(self):
        for i in range(self._x.shape[0]):
            yield self[i]

    def __len__(self):
        return self._x.shape[0]


class UnknownDataSetException(ValueError):
    pass


def load_dataset(dataset, dataset_kwargs, classifier_name, classifier_kwargs, dataset_type=None):
    """

    Parameters
    ----------
    dataset : str
        Dataset name which can be consumed by `DataSets.from_str()`.
    dataset_kwargs : dict
        Additional arguments for the dataset.
    classifier_name : str
        Name of the classifier. Can be used to infer the dataset type.
    classifier_kwargs : dict
        Additional arguments for the classifier. Some arguments from this may be used for
        constructing a transformers tokenizer.
    dataset_type : DataSetType, default=None
        Data set type or None.

    Returns
    -------
    train : Dataset
        Training set for the dataset identified by `dataset_name`.
    test : Dataset
        Test set for the dataset identified by `dataset_name`.
    """
    dataset_type_expected = get_dataset_type(classifier_name, dataset_kwargs, dataset_type)
    dataset = DataSets.from_str(dataset)

    train, test = _load_dataset(dataset, dataset_type_expected, dataset_kwargs, classifier_kwargs)

    return train, test


def get_dataset_type(classifier_name, dataset_kwargs, dataset_type):

    if 'dataset_type' in dataset_kwargs:
        dataset_type_expected = DataSetType.from_str(dataset_kwargs['dataset_type'])
    elif dataset_type is not None:
        if isinstance(dataset_type, DataSetType):
            dataset_type_expected = dataset_type
        else:
            dataset_type_expected = DataSetType.from_str(dataset_type)
    else:
        dataset_type_expected = get_dataset_type_for_classifier(classifier_name)

    return dataset_type_expected


def get_dataset_type_for_classifier(classifier_name):

    if classifier_name == 'svm':
        return DataSetType.BOW
    elif classifier_name == 'kimcnn':
        return DataSetType.TENSOR_PADDED_SEQ
    elif classifier_name == 'transformer':
        return DataSetType.TRANSFORMERS

    raise ValueError(f'No dataset type defined for classifier_name {classifier_name}')


def _load_dataset(dataset, dataset_type, dataset_kwargs, classifier_kwargs,
                  test_set_ratio=TEST_SET_RATIO_DEFAULT):

    if dataset == DataSets.JIGSAW:
        return _load_jigsaw(dataset, dataset_type, dataset_kwargs, classifier_kwargs)
    # elif dataset == DataSets.MR:
    #     return _load_mr(dataset, dataset_type, dataset_kwargs, classifier_kwargs)
    # elif dataset == DataSets.CR:
    #     return _load_cr(dataset, dataset_type, dataset_kwargs, classifier_kwargs,
    #                     test_set_ratio=test_set_ratio)

    raise UnknownDataSetException(f'Unknown dataset / type combination: '
                                  f'{dataset} - {str(dataset_type)}')


def _load_jigsaw(dataset, dataset_type, dataset_kwargs, classifier_kwargs):
    # TODO : verifier que ça marche avec des datasets multi-label
    import datasets
    jigsaw_dataset = datasets.load_dataset('jigsaw_toxicity_pred')
    if dataset_type == DataSetType.TRANSFORMERS:
        tokenizer = _get_huggingface_tokenizer(classifier_kwargs)
        return _text_to_transformers_dataset(tokenizer,
                                             jigsaw_dataset['train']['text'],
                                             jigsaw_dataset['train']['label'],
                                             jigsaw_dataset['test']['text'],
                                             jigsaw_dataset['test']['label'],
                                             dataset_kwargs['max_length'],
                                             multi_label=True)
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_text_classification_dataset(
            jigsaw_dataset['train']['text'],
            jigsaw_dataset['train']['label'],
            jigsaw_dataset['test']['text'],
            jigsaw_dataset['test']['label'])
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow(jigsaw_dataset['train']['text'],
                            jigsaw_dataset['train']['label'],
                            jigsaw_dataset['test']['text'],
                            jigsaw_dataset['test']['label'])
    elif dataset_type == DataSetType.RAW:
        return RawDataset(jigsaw_dataset['train']['text'],
                          jigsaw_dataset['train']['label']), \
               RawDataset(jigsaw_dataset['test']['text'],
                          jigsaw_dataset['test']['label'])
    else:
        raise ValueError(f'Unsupported dataset type for dataset {str(dataset)}')

    
def _get_huggingface_tokenizer(classifier_kwargs):
    tokenizer_name = classifier_kwargs['transformer_tokenizer'] \
        if 'transformer_tokenizer' in classifier_kwargs else classifier_kwargs['transformer_model']
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir='.transformers_cache/',
    )
    return tokenizer


def _create_bow_preprocessor(preprocessor):
    result = preprocessor.vectorize((1, 2), preprocessor.get_train_docs(),
                                    preprocessor.get_test_docs())

    return (SklearnDataset(normalize(result['x']), result['y']),
            SklearnDataset(normalize(result['x_test']), result['y_test']))


def _text_to_bow(x, y, x_test, y_test, max_features=50000, ngram_range=(1, 2)):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    #
    # x = vectorizer.fit_transform(x)
    # x_test = vectorizer.transform(x_test)
    #
    # return (SklearnDataset(normalize(x), y),
    #         SklearnDataset(normalize(x_test), y_test))
    x_train = normalize(vectorizer.fit_transform(x))
    x_test = normalize(vectorizer.transform(x_test))
    # TODO : get equivalent of len(train) and NUM_LABELS with shape of x et y
    y_train = list_to_csr(y, shape=(len(train), NUM_LABELS))
    y_test = list_to_csr(y_test, shape=(len(test), NUM_LABELS))

    return SklearnDataset(x_train, y_train), SklearnDataset(x_test, y_test)


def _text_to_transformers_dataset(tokenizer, train_text, train_labels, test_text,
                                  test_labels, max_length, multi_label):

    return _transformers_prepare(tokenizer, train_text, train_labels, multi_label, max_length=max_length), \
           _transformers_prepare(tokenizer, test_text, test_labels, multi_label, max_length=max_length)


def _transformers_prepare(tokenizer, data, labels, multi_label, max_length=60):

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

    return TransformersDataset(data_out)


def _text_to_text_classification_dataset(x_text, y, x_test_text, y_test):
    # TODO : adapter multi-label data to classifieur kimcnn
    try:
        from torchtext.legacy import data
    except AttributeError:
        from torchtext import data

    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False, unk_token=None, pad_token=None)

    fields = [('text', text_field), ('label', label_field)]

    train = data.Dataset([data.Example.fromlist([text, labels], fields)
                          for text, labels in zip(x_text, y)],
                         fields)
    test = data.Dataset([data.Example.fromlist([text, labels], fields)
                         for text, labels in zip(x_test_text, y_test)],
                        fields)

    text_field.build_vocab(train, min_freq=1)
    label_field.build_vocab(train)

    train_tc = _tt_dataset_to_text_classification_dataset(train)
    test_tc = _tt_dataset_to_text_classification_dataset(test)

    return train_tc, test_tc


def _tt_dataset_to_text_classification_dataset(dataset):
    assert dataset.fields['text'].vocab.itos[0] == '<unk>'
    assert dataset.fields['text'].vocab.itos[1] == '<pad>'
    unk_token_idx = 1

    vocab = dataset.fields['text'].vocab

    data = [
        (torch.LongTensor([vocab.stoi[token] if token in vocab.stoi else unk_token_idx
                           for token in example.text]),
         dataset.fields['label'].vocab.stoi[example.label])
        for example in dataset.examples
    ]

    return PytorchTextClassificationDataset(data, vocab)
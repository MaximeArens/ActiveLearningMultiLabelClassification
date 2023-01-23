import math
import torch
import datasets
import numpy as np

from enum import Enum
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from small_text.data.datasets import SklearnDataset
from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
from small_text.integrations.transformers.datasets import TransformersDataset
from transformers import AutoTokenizer
from small_text.utils.labels import list_to_csr
from datasets import DatasetDict
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import ast
import textstat as ts


TEST_SET_RATIO_DEFAULT = 0.1


class DataSets(Enum):
    JIGSAW = 'jigsaw'
    JIGSAWMC = 'jigsaw_mc'
    GO_EMOTIONS = 'go_emotions'
    ECTHRA = 'ecthr_a'
    ECTHRB = 'ecthr_b'
    EURLEX = 'eur_lex'
    UNFAIRTOS = 'unfair_tos'
    NEWSGROUPS = '20_news'

    @staticmethod
    def from_str(enum_str):
        if enum_str == 'jigsaw':
            return DataSets.JIGSAW
        if enum_str == 'jigsaw_mc':
            return DataSets.JIGSAWMC
        if enum_str == 'go_emotions':
            return DataSets.GO_EMOTIONS
        if enum_str == 'ecthr_a':
            return DataSets.ECTHRA
        if enum_str == 'ecthr_b':
            return DataSets.ECTHRB
        if enum_str == 'eur_lex':
            return DataSets.EURLEX
        if enum_str == 'unfair_tos':
            return DataSets.UNFAIRTOS
        if enum_str == '20_news':
            return DataSets.NEWSGROUPS

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


def create_labels_column(dataset):
    dataset_train = pd.DataFrame(dataset['train'])
    dataset_test = pd.DataFrame(dataset['test'])

    labels_train = dataset_train[dataset_train.columns[1]].astype(str)
    labels_test = dataset_test[dataset_test.columns[1]].astype(str)
    for label in dataset_train.columns[2:]:
        labels_train = labels_train + ',' + dataset_train[label].astype(str)
        labels_test = labels_test + ',' + dataset_test[label].astype(str)
    labels_train = list(labels_train.array)
    labels_test = list(labels_test.array)

    y_test = list_to_csr(labels_train, shape=len(labels_train))
    # From different label column such as (label1, 1) ; (label2, 0) ; (label3,1)
    # to one label column such as ['label1', 'label3']
    # labels_train = list(dataset_train.iloc[:, 1:].dot(dataset_train.columns[1:] + ',').str[:-1].str.split(',').array)
    # labels_test = list(dataset_test.iloc[:, 1:].dot(dataset_test.columns[1:] + ',').str[:-1].str.split(',').array)

    # From one label column such as ['label1', 'label3']
    # to one label column such as [1, 0, 1]
    # mlb = MultiLabelBinarizer(classes=dataset_train.columns[1:], sparse_output=True)
    # labels_train = mlb.fit_transform(labels_train)

    dataset['train'] = dataset['train'].add_column('labels', labels_train)
    dataset['test'] = dataset['test'].add_column('labels', labels_test)

    return dataset


def _load_dataset(dataset, dataset_type, dataset_kwargs, classifier_kwargs,
                  test_set_ratio=TEST_SET_RATIO_DEFAULT):
    if dataset == DataSets.JIGSAW:
        return _load_jigsaw(dataset, dataset_type, dataset_kwargs, classifier_kwargs)
    if dataset == DataSets.JIGSAWMC:
        return _load_jigsaw_mc(dataset, dataset_type, dataset_kwargs, classifier_kwargs)
    elif dataset == DataSets.GO_EMOTIONS:
        return _load_go_emotions(dataset, dataset_type, dataset_kwargs, classifier_kwargs)
    elif dataset == DataSets.EURLEX:
        return _load_eurlex(dataset, dataset_type, dataset_kwargs, classifier_kwargs)
    elif dataset == DataSets.ECTHRA:
        return _load_ecthr_a(dataset, dataset_type, dataset_kwargs, classifier_kwargs)
    elif dataset == DataSets.ECTHRB:
        return _load_ecthr_b(dataset, dataset_type, dataset_kwargs, classifier_kwargs)
    elif dataset == DataSets.UNFAIRTOS:
        return _load_unfair_tos(dataset, dataset_type, dataset_kwargs, classifier_kwargs)
    elif dataset == DataSets.NEWSGROUPS:
        return _load_newsgroups(dataset, dataset_type, dataset_kwargs, classifier_kwargs)
    # elif dataset == DataSets.CR:
    #     return _load_cr(dataset, dataset_type, dataset_kwargs, classifier_kwargs,
    #                     test_set_ratio=test_set_ratio)

    raise UnknownDataSetException(f'Unknown dataset / type combination: '
                                  f'{dataset} - {str(dataset_type)}')


def _load_jigsaw(dataset, dataset_type, dataset_kwargs, classifier_kwargs):
    jigsaw_dataset = DatasetDict.from_csv({'train': './data/jigsaw_datasets/train.csv',
                                           'test': './data/jigsaw_datasets/test.csv'})
    y_train = [ast.literal_eval(x) for x in jigsaw_dataset['train']['labels']]
    y_test = [ast.literal_eval(x) for x in jigsaw_dataset['test']['labels']]
    if dataset_type == DataSetType.TRANSFORMERS:
        tokenizer = _get_huggingface_tokenizer(classifier_kwargs)
        return _text_to_transformers_dataset(tokenizer,
                                             jigsaw_dataset['train']['text'],
                                             y_train,
                                             jigsaw_dataset['test']['text'],
                                             y_test,
                                             dataset_kwargs['max_length'],
                                             multi_label=True)
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_text_classification_dataset(
            jigsaw_dataset['train']['text'],
            y_train,
            jigsaw_dataset['test']['text'],
            y_test,
            multilabel=True
        )
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow(jigsaw_dataset['train']['text'],
                            y_train,
                            jigsaw_dataset['test']['text'],
                            y_test,
                            num_labels=6)
    elif dataset_type == DataSetType.RAW:
        return RawDataset(jigsaw_dataset['train']['text'],
                          y_train), \
               RawDataset(jigsaw_dataset['test']['text'],
                          y_test)
    else:
        raise ValueError(f'Unsupported dataset type for dataset {str(dataset)}')


def _load_jigsaw_mc(dataset, dataset_type, dataset_kwargs, classifier_kwargs):
    jigsaw_dataset = DatasetDict.from_csv({'train': './data/jigsaw_datasets/train.csv',
                                           'test': './data/jigsaw_datasets/test.csv'})
    y_train = [ast.literal_eval(x) for x in jigsaw_dataset['train']['labels']]
    y_test = [ast.literal_eval(x) for x in jigsaw_dataset['test']['labels']]

    Y = y_train + y_test
    le = preprocessing.LabelEncoder()
    le.fit([str(tuple(label)) for label in Y])

    y_train_mc = np.ndarray.tolist(le.transform([str(tuple(label)) for label in y_train]))
    y_test_mc = np.ndarray.tolist(le.transform([str(tuple(label)) for label in y_test]))

    if dataset_type == DataSetType.TRANSFORMERS:
        tokenizer = _get_huggingface_tokenizer(classifier_kwargs)
        return _text_to_transformers_dataset(tokenizer,
                                             jigsaw_dataset['train']['text'],
                                             y_train_mc,
                                             jigsaw_dataset['test']['text'],
                                             y_test_mc,
                                             dataset_kwargs['max_length'],
                                             multi_label=False)
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_text_classification_dataset(
            jigsaw_dataset['train']['text'],
            y_train_mc,
            jigsaw_dataset['test']['text'],
            y_test_mc,
            multilabel=False
        )
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow_mc(jigsaw_dataset['train']['text'],
                            y_train_mc,
                            jigsaw_dataset['test']['text'],
                            y_test_mc)
    elif dataset_type == DataSetType.RAW:
        return RawDataset(jigsaw_dataset['train']['text'],
                          y_train_mc), \
               RawDataset(jigsaw_dataset['test']['text'],
                          y_test_mc)
    else:
        raise ValueError(f'Unsupported dataset type for dataset {str(dataset)}')


def _load_go_emotions(dataset, dataset_type, dataset_kwargs, classifier_kwargs):
    import datasets
    go_emotions_dataset = datasets.load_dataset('go_emotions')
    if dataset_type == DataSetType.TRANSFORMERS:
        tokenizer = _get_huggingface_tokenizer(classifier_kwargs)
        return _text_to_transformers_dataset(tokenizer,
                                             go_emotions_dataset['train']['text'],
                                             remove_neutral(go_emotions_dataset['train']['labels']),
                                             go_emotions_dataset['test']['text'],
                                             remove_neutral(go_emotions_dataset['test']['labels']),
                                             dataset_kwargs['max_length'],
                                             multi_label=True)
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_text_classification_dataset(
            go_emotions_dataset['train']['text'],
            remove_neutral(go_emotions_dataset['train']['labels']),
            go_emotions_dataset['test']['text'],
            remove_neutral(go_emotions_dataset['test']['labels']),
            multilabel=True)
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow(go_emotions_dataset['train']['text'],
                            remove_neutral(go_emotions_dataset['train']['labels']),
                            go_emotions_dataset['test']['text'],
                            remove_neutral(go_emotions_dataset['test']['labels']),
                            num_labels=27)
    elif dataset_type == DataSetType.RAW:
        return RawDataset(go_emotions_dataset['train']['text'],
                          remove_neutral(go_emotions_dataset['train']['labels']), \
               RawDataset(go_emotions_dataset['test']['text'],
                          remove_neutral(go_emotions_dataset['test']['labels'])))
    else:
        raise ValueError(f'Unsupported dataset type for dataset {str(dataset)}')


def _load_eurlex(dataset, dataset_type, dataset_kwargs, classifier_kwargs):
    import datasets
    eurlex_dataset = datasets.load_dataset('lex_glue', 'eurlex')

    if dataset_type == DataSetType.TRANSFORMERS:
        tokenizer = _get_huggingface_tokenizer(classifier_kwargs)
        return _text_to_transformers_dataset(tokenizer,
                                             eurlex_dataset['train']['text'],
                                             eurlex_dataset['train']['labels'],
                                             eurlex_dataset['test']['text'],
                                             eurlex_dataset['test']['labels'],
                                             dataset_kwargs['max_length'],
                                             multi_label=True)
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_text_classification_dataset(
            eurlex_dataset['train']['text'],
            eurlex_dataset['train']['labels'],
            eurlex_dataset['test']['text'],
            eurlex_dataset['test']['labels'],
            multilabel=True)
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow(eurlex_dataset['train']['text'],
                            eurlex_dataset['train']['labels'],
                            eurlex_dataset['test']['text'],
                            eurlex_dataset['test']['labels'],
                            num_labels=127)
    elif dataset_type == DataSetType.RAW:
        return RawDataset(eurlex_dataset['train']['text'],
                          eurlex_dataset['train']['labels'], \
               RawDataset(eurlex_dataset['test']['text'],
                          eurlex_dataset['test']['labels']))
    else:
        raise ValueError(f'Unsupported dataset type for dataset {str(dataset)}')


def _load_ecthr_a(dataset, dataset_type, dataset_kwargs, classifier_kwargs):
    import datasets
    ecthr_a_dataset = datasets.load_dataset('lex_glue', 'ecthr_a')
    train_text = []
    test_text = []
    for seq in ecthr_a_dataset['train']['text']:
        train_text.append(' '.join(str(e) for e in seq))
    for seq in ecthr_a_dataset['test']['labels']:
        test_text.append(' '.join(str(e) for e in seq))

    if dataset_type == DataSetType.TRANSFORMERS:
        tokenizer = _get_huggingface_tokenizer(classifier_kwargs)
        return _text_to_transformers_dataset(tokenizer,
                                             train_text,
                                             ecthr_a_dataset['train']['labels'],
                                             test_text,
                                             ecthr_a_dataset['test']['labels'],
                                             dataset_kwargs['max_length'],
                                             multi_label=True)
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_text_classification_dataset(
            train_text,
            ecthr_a_dataset['train']['labels'],
            test_text,
            ecthr_a_dataset['test']['labels'],
            multilabel=True)
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow(train_text,
                            ecthr_a_dataset['train']['labels'],
                            test_text,
                            ecthr_a_dataset['test']['labels'],
                            num_labels=10)
    elif dataset_type == DataSetType.RAW:
        return RawDataset(train_text,
                          ecthr_a_dataset['train']['labels'], \
               RawDataset(test_text,
                          ecthr_a_dataset['test']['labels']))
    else:
        raise ValueError(f'Unsupported dataset type for dataset {str(dataset)}')


def _load_ecthr_b(dataset, dataset_type, dataset_kwargs, classifier_kwargs):
    import datasets
    ecthr_b_dataset = datasets.load_dataset('lex_glue', 'ecthr_b')
    train_text = []
    test_text = []
    for seq in ecthr_b_dataset['train']['text']:
        train_text.append(' '.join(str(e) for e in seq))
    for seq in ecthr_b_dataset['test']['labels']:
        test_text.append(' '.join(str(e) for e in seq))

    if dataset_type == DataSetType.TRANSFORMERS:
        tokenizer = _get_huggingface_tokenizer(classifier_kwargs)
        return _text_to_transformers_dataset(tokenizer,
                                             train_text,
                                             ecthr_b_dataset['train']['labels'],
                                             test_text,
                                             ecthr_b_dataset['test']['labels'],
                                             dataset_kwargs['max_length'],
                                             multi_label=True)
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_text_classification_dataset(
            train_text,
            ecthr_b_dataset['train']['labels'],
            test_text,
            ecthr_b_dataset['test']['labels'],
            multilabel=True)
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow(train_text,
                            ecthr_b_dataset['train']['labels'],
                            test_text,
                            ecthr_b_dataset['test']['labels'],
                            num_labels=10)
    elif dataset_type == DataSetType.RAW:
        return RawDataset(train_text,
                          ecthr_b_dataset['train']['labels'], \
               RawDataset(test_text,
                          ecthr_b_dataset['test']['labels']))
    else:
        raise ValueError(f'Unsupported dataset type for dataset {str(dataset)}')


def _load_unfair_tos(dataset, dataset_type, dataset_kwargs, classifier_kwargs):
    import datasets
    unfair_tos_dataset = datasets.load_dataset('lex_glue', 'unfair_tos')

    if dataset_type == DataSetType.TRANSFORMERS:
        tokenizer = _get_huggingface_tokenizer(classifier_kwargs)
        return _text_to_transformers_dataset(tokenizer,
                                             unfair_tos_dataset['train']['text'],
                                             unfair_tos_dataset['train']['labels'],
                                             unfair_tos_dataset['test']['text'],
                                             unfair_tos_dataset['test']['labels'],
                                             dataset_kwargs['max_length'],
                                             multi_label=True)
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_text_classification_dataset(
            unfair_tos_dataset['train']['text'],
            unfair_tos_dataset['train']['labels'],
            unfair_tos_dataset['test']['text'],
            unfair_tos_dataset['test']['labels'],
            multilabel=True)
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow(unfair_tos_dataset['train']['text'],
                            unfair_tos_dataset['train']['labels'],
                            unfair_tos_dataset['test']['text'],
                            unfair_tos_dataset['test']['labels'],
                            num_labels=8)
    elif dataset_type == DataSetType.RAW:
        return RawDataset(unfair_tos_dataset['train']['text'],
                          unfair_tos_dataset['train']['labels'], \
               RawDataset(unfair_tos_dataset['test']['text'],
                          unfair_tos_dataset['test']['labels']))
    else:
        raise ValueError(f'Unsupported dataset type for dataset {str(dataset)}')


def _load_newsgroups(dataset, dataset_type, dataset_kwargs, classifier_kwargs):
    import datasets
    news_dataset = datasets.load_dataset('SetFit/20_newsgroups')
    test = news_dataset['train']['label']
    labels_train, labels_test = get_numerical_labels_from_string(news_dataset['train']['label'],
                                                                 news_dataset['test']['label'])
    if dataset_type == DataSetType.TRANSFORMERS:
        tokenizer = _get_huggingface_tokenizer(classifier_kwargs)
        return _text_to_transformers_dataset(tokenizer,
                                             news_dataset['train']['text'],
                                             labels_train,
                                             news_dataset['test']['text'],
                                             labels_test,
                                             dataset_kwargs['max_length'],
                                             multi_label=True)
    elif dataset_type == DataSetType.TENSOR_PADDED_SEQ:
        return _text_to_text_classification_dataset(
            news_dataset['train']['text'],
            labels_train,
            news_dataset['test']['text'],
            labels_test,
            multilabel=True)
    elif dataset_type == DataSetType.BOW:
        return _text_to_bow(news_dataset['train']['text'],
                            labels_train,
                            news_dataset['test']['text'],
                            labels_test,
                            num_labels=27)
    elif dataset_type == DataSetType.RAW:
        return RawDataset(news_dataset['train']['text'],
                          labels_train, \
               RawDataset(news_dataset['test']['text'],
                          labels_test))
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


def _text_to_bow(x, y, x_test, y_test, num_labels, max_features=50000, ngram_range=(1, 2)):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    #
    # x = vectorizer.fit_transform(x)
    # x_test = vectorizer.transform(x_test)
    #
    # return (SklearnDataset(normalize(x), np.array(y)),
    #         SklearnDataset(normalize(x_test), np.array(y_test)))
    x_train = normalize(vectorizer.fit_transform(x))
    x_test = normalize(vectorizer.transform(x_test))

    y_train = list_to_csr(y, shape=(len(y), num_labels))
    y_test = list_to_csr(y_test, shape=(len(y_test), num_labels))

    train_dataset = SklearnDataset(x_train, y_train)
    test_dataset = SklearnDataset(x_test, y_test)
    train_dataset.readability = np.asarray([ts.text_standard(y, float_output=True) for y in x])
    return train_dataset, test_dataset


def _text_to_bow_mc(x, y, x_test, y_test, max_features=50000, ngram_range=(1, 2)):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    x = vectorizer.fit_transform(x)
    x_test = vectorizer.transform(x_test)

    return (SklearnDataset(normalize(x), np.array(y)),
            SklearnDataset(normalize(x_test), np.array(y_test)))

def _text_to_transformers_dataset(tokenizer, train_text, train_labels, test_text,
                                  test_labels, max_length, multi_label):
    train_dataset = _transformers_prepare(tokenizer, train_text, train_labels, multi_label, max_length=max_length)
    test_dataset = _transformers_prepare(tokenizer, test_text, test_labels, multi_label, max_length=max_length)
    train_dataset.readability = np.asarray([ts.text_standard(x, float_output=True) for x in train_text])
    return train_dataset, test_dataset


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

    return TransformersDataset(data_out, multi_label=multi_label)


def _text_to_text_classification_dataset(x_text, y, x_test_text, y_test, multilabel):
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

    train_tc = _tt_dataset_to_text_classification_dataset(train, multilabel)
    test_tc = _tt_dataset_to_text_classification_dataset(test, multilabel)

    return train_tc, test_tc


def _tt_dataset_to_text_classification_dataset(dataset, multilabel):
    assert dataset.fields['text'].vocab.itos[0] == '<unk>'
    assert dataset.fields['text'].vocab.itos[1] == '<pad>'
    unk_token_idx = 1

    vocab = dataset.fields['text'].vocab
    labels = list(set(dataset.fields['label'].vocab.itos))
    labels = np.array(labels)

    data = []
    for example in dataset.examples:
        label_list = []
        for label in example.label:
            label_list.append(dataset.fields['label'].vocab.stoi[label])
        data.append((torch.LongTensor([vocab.stoi[token] if token in vocab.stoi else unk_token_idx
                                       for token in example.text]), label_list))

    return PytorchTextClassificationDataset(data, vocab, multi_label=multilabel, target_labels=labels)


def remove_neutral(labels_list):

    for labels in labels_list:
        try:
            labels.remove(27)
        except:
            None
    return labels_list


def get_numerical_labels_from_string(string_labels_list_train, string_labels_list_test):
    string_labels_list = string_labels_list_train.copy()
    string_labels_list.extend(string_labels_list_test)
    flat_string_labels_list = []
    for sublist in string_labels_list:
        for item in sublist:
            flat_string_labels_list.append(item)
    unique_flat_string_labels_list = list(set(flat_string_labels_list))
    le = LabelEncoder()
    le.fit(unique_flat_string_labels_list)
    train = []
    test = []
    for sublist in string_labels_list_train:
        train.append(list(le.transform(sublist)))
    for sublist in string_labels_list_test:
        test.append(list(le.transform(sublist)))
    return train, test







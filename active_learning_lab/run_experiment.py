import sys
sys.path.append("C:\\Users\\Maxime\\Documents\\Synapse\\RetD\\Scripts\\ActiveLearningMultiLabelClassification")

import re
import os
import inspect
import torch
import json
import ast
import mlflow
import datasets
from sentence_transformers import SentenceTransformer

from contextlib import closing
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory

from IPython.utils.io import Tee

from active_learning_lab.args import get_parser
from active_learning_lab.experiments.experiment_builder import (
    ActiveLearningExperimentBuilder
)
from active_learning_lab.utils.experiment import (
    get_tmp_path,
    set_random_seed,
    suppress_known_thirdparty_warnings,
    write_experiment_info
)
from active_learning_lab.utils.logging import setup_logger


def main(args, experiment_name, train_embeddings):
    sys.stderr = sys.stdout
    active_run = mlflow.active_run()

    tmp_path = get_tmp_path(active_run)
    log_file_path = tmp_path.joinpath('out.log').resolve()

    try:
        with TemporaryDirectory(dir=str(tmp_path.resolve())) as tmp_dir:
            with closing(Tee(str(log_file_path), 'w', channel='stdout')):
                logger = setup_logger()
                logger.info("Args: " + str(args))
                write_experiment_info(logger, mlflow, experiment_name)

                builder = get_active_learner_builder(args, tmp_dir)
                exp = builder.build(experiment_name)

                # [!] This is the entry point to the actual experiment
                results = exp.run(builder.train, builder.test, train_embeddings)

                process_results(exp, results)

                write_experiment_info(logger, mlflow, experiment_name)
    finally:
        mlflow.log_artifact(str(log_file_path))
        rmtree(tmp_path, 'out.log')


def get_active_learner_builder(args, tmp_dir):
    builder = ActiveLearningExperimentBuilder(args['active_learner']['num_queries'],
                                              args['active_learner']['query_size'],
                                              args['active_learner']['query_strategy'],
                                              args['active_learner']['query_strategy_kwargs'],
                                              args['general']['runs'],
                                              str(tmp_dir)) \
        .with_classifier(args['classifier']['classifier_name'],
                         args['classifier']['validation_set_size'],
                         args['classifier']['classifier_kwargs']) \
        .with_initialization(args['active_learner']['initialization_strategy'],
                             args['active_learner']['initialization_strategy_kwargs']) \
        .with_dataset(args['dataset']['dataset_name'], args['dataset']['dataset_kwargs'])

    return builder


def process_results(al_exp, results):
    for name, file in results.artifacts:
        if '/' in name:
            artifact_dir = re.sub('^file:', '', mlflow.get_artifact_uri())
            basedir = Path(artifact_dir).joinpath(name).parents[0]
            if not basedir.exists():
                basedir.mkdir()
            file_rel = str(file.relative_to(al_exp.tmp_dir))
            mlflow.log_artifact(file, file_rel[:file_rel.rindex('/')])
        else:
            mlflow.log_artifact(file)


def get_dataset_text(dataset, sentence_transformer):
    if dataset == 'jigsaw':
        jigsaw_dataset = datasets.DatasetDict.from_csv({'train': './data/jigsaw_datasets/train.csv',
                                               'test': './data/jigsaw_datasets/test.csv'})
        x_train = jigsaw_dataset['train']['text']
    if dataset == 'go_emotions':
        go_emotions_dataset = datasets.load_dataset('go_emotions')
        x_train = go_emotions_dataset['train']['text']
    if dataset == 'unfair_tos':
        unfair_tos_dataset = datasets.load_dataset('lex_glue', 'unfair_tos')
        x_train = unfair_tos_dataset['train']['text']
    if dataset == 'eur_lex':
        eurlex_dataset = datasets.load_dataset('lex_glue', 'eurlex')
        x_train = eurlex_dataset['train']['text']
    model = SentenceTransformer(sentence_transformer)
    x_train_embeddings = model.encode(x_train)
    return x_train_embeddings


if __name__ == '__main__':
    arg_list = sys.argv

    del arg_list[0]
    # parser = get_parser()
    #args = parser.parse_args(arg_list)
    classifier_config = arg_list[0]

    strategies = ['rd', 'ml', 'mml', 'cmn', 'cvirs', 'mmu', 'lci']

    with open('../config/' + classifier_config) as json_file:
        args = json.load(json_file)
        args['dataset']['dataset_name'] = arg_list[1]
    train_embeddings = get_dataset_text(args['dataset']['dataset_name'],
                                        "sentence-transformers/nli-distilbert-base")
    for strategy in strategies:
        args['active_learner']['query_strategy'] = strategy

        suppress_known_thirdparty_warnings()
        set_random_seed(args['general']['seed'], args['general']['max_reproducibility'])

        client = mlflow.tracking.MlflowClient()
        if 'cpu' in arg_list:
            experiment_name = args['experiment_name'] + "_" + args['dataset']['dataset_name'] + "_" +\
                              args['active_learner']['query_strategy'] + '_cpu'
        else:
            experiment_name = args['experiment_name'] + "_" + args['dataset']['dataset_name'] + "_" + \
                              args['active_learner']['query_strategy']
            #experiment = mlflow.get_experiment_by_name(experiment_name) \
        experiment = mlflow.set_experiment(experiment_name)
        if experiment is None:
            raise ValueError('No mlflow experiments with name \'{}\' exists. '
                                 'Please create the experiment first.'.format(experiment_name))

        if torch.cuda.is_available():
            # Find the location of the torch package
            package_path = os.path.dirname(inspect.getfile(torch))
            full_path = os.path.join(package_path, 'utils/data/sampler.py')
            # Read in the file
            with open(full_path, 'r') as file:
                filedata = file.read()

            # Replace the target string
            filedata = filedata.replace('generator = torch.Generator()', 'generator = torch.Generator(device=\'cuda\')')
            filedata = filedata.replace('yield from torch.randperm(n, generator=generator).tolist()',
                                        'yield from torch.randperm(n, generator=generator, device=\'cuda\').tolist()')

            # Write the file out again
            with open(full_path, 'w') as file:
                file.write(filedata)

        with mlflow.start_run(experiment_id=experiment.experiment_id):
            mlflow.log_param('experiment_name', experiment_name)
            mlflow.log_param('classifier_name', args['classifier']['classifier_name'])
            classifier_pretrained_model = args['classifier']['classifier_kwargs']['transformer_model'] \
            if args['classifier']['classifier_name'] == 'transformer' else ''
            mlflow.log_param('classifier_pretrained_model', classifier_pretrained_model)
            mlflow.log_param('dataset_name', args['dataset']['dataset_name'])
            mlflow.log_param('query_strategy', args['active_learner']['query_strategy'])
            mlflow.log_param('description', args['general']['description'])

            main(args, experiment_name, train_embeddings)

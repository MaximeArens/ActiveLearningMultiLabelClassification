import mlflow

if __name__ == '__main__':
    client = mlflow.tracking.MlflowClient()
    datasets = ['jigsaw', 'go_emotions', 'eur_lex', '20_news']
    models = ['svm', 'bert', 'distilRoberta']
    strategies = ['rd', 'ml', 'mml', 'cmn', 'cvirs', 'mmu', 'lci']
    for d in datasets:
        for m in models:
            for s in strategies:
                experiment = mlflow.set_experiment(m + '_' + d + '_' + s)
    for d in datasets:
        for m in models:
            for s in strategies:
                experiment = mlflow.set_experiment(m + '_' + d + '_' + s + '_cpu')

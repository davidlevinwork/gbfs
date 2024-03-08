from sklearn.manifold import TSNE

from gbfs.algorithms.gbafs import GBAFS

if __name__ == '__main__':
    tsne = TSNE(n_components=2, perplexity=10)

    x = GBAFS(
        dataset_path='cardiotocography.csv',
        separability_metric='jm',
        dim_reducer_model=tsne,
    )
    y = x.select_features()
    x.plot_feature_space()

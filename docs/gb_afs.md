# GB-AFS

GB-AFS (Graph-Based Automatic Feature Selection) is an approach designed to identify the optimal subset of features necessary for maintaining predictive performance, without necessitating user-specified parameters, such as the desired number of features to include. This self-sufficiency is what attributes the 'Automatic' aspect to its name. Operating as a filter-based methodology, GB-AFS is model-agnostic, allowing for the integration of feature selection seamlessly into the preprocessing phase, regardless of the predictive model being used.

The primary innovation and strength of GB-AFS lie in its unique capability to autonomously determine the smallest set of features required, circumventing the common limitation among filter-based methods that typically rely on user input for configuration.

## Using GB-AFS: Code Examples and Visualization

GBFS offers a versatile and user-friendly Python library for feature selection in multi-class classification tasks. This section guides you through initializing the GB-AFS object with your dataset, selecting features, and visualizing the feature space.

### Initialization and Parameters

To start using GB-AFS, you first need to initialize the GB-AFS object with your dataset and selection criteria:


``` py title="main.py" linenums="1"
from gbfs import GBAFS

gbafs = GBAFS(
    dataset_path="path/to/your/dataset.csv",
    separability_metric="your_separability_metric",
    dim_reducer_model="your_dimensionality_reduction_method",
    label_column="class",
    verbose=1
)
```

#### Parameters Explained

- `dataset_path`: Path to your dataset file. Ensure your dataset is in a CSV format or another compatible format.
- `separability_metric`: Metric for evaluating feature separability. 
- `dim_reducer_model`: Dimensionality reduction model applying to your dataset. Must implement a `fit_transform` method for compatibility.
- `label_column`: Name of the column with labels in your dataset. Defaults to `'class'`.
- `verbose`: Verbosity level of the process; `0` for no logging, `1` for logging.

Current supported metrics for `separability_metric` are `jm`, `bhattacharyya`, and `wasserstein`. To request support for additional metrics, please open an issue in the repository.


### Feature Selection
Once the GB-AFS object is initialized, you can proceed with the feature selection process:

``` py title="main.py" linenums="1"
selected_features = gbafs.select_features()

print("Selected Feature Indices:", selected_features)
```
This method returns a list of indices for the features deemed most relevant by the GB-AFS algorithm.

### Visualizing the Feature Space
GB-AFS also includes a method to visualize the selected features within the feature space, providing insights into their distribution and separability:

``` py title="main.py" linenums="1"
gbafs.plot_feature_space()
```

This method generates a scatter plot highlighting the selected features. Features are displayed with their separability power indicated by color intensity, and selected features are marked distinctly.

## References and Further Reading

For a deeper understanding of the GB-AFS method and its background, consider exploring the [official paper](./gb_afs.md).
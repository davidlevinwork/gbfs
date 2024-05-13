# GB-BC-FS

§GB-BC-FS (Graph-Based Budget-Constraint Feature Selection) is an approach designed to efficiently handle large datasets with numerous features under budget constraints. Unlike traditional methods that generate multiple candidate solutions, GB-BC-FS starts with a single solution and refines it to meet budgetary limits, significantly reducing computational time. The process utilizes the GB-AFS method, which selects a minimal set of features necessary for accuracy in multi-class classification by assessing the discriminative power of features across class pairs.

To ensure feature diversity and accommodate budget constraints, the method includes a heuristic refinement step that adjusts the initial feature set. This adjustment is based on a scoring function that favors lower-cost features, simplifying the feature selection process and enhancing robustness. The approach also evaluates the interrelationships among features, moving beyond traditional isolated assessments.

## Using GB-BC-FS: Code Examples and Visualization

GBFS offers a versatile and user-friendly Python library for feature selection in multi-class classification tasks. This section guides you through initializing the GB-BC-FS object with your dataset, selecting features, and visualizing the feature space.

### Initialization and Parameters

To start using GB-AFS, you first need to initialize the GB-AFS object with your dataset and selection criteria:


``` py title="main.py" linenums="1"
from gbfs import GBBCFS

gbbcfs = GBBCFS(
    dataset_path="path/to/your/dataset.csv",
    separability_metric="your_separability_metric",
    dim_reducer_model="your_dimensionality_reduction_method",
    label_column="class",
    budget=20,  # Maximum budget for feature selection
    alpha=0.5,
    epochs=100,
)
```

#### Parameters Explained

- `dataset_path`: Path to your dataset file. Ensure your dataset is in a CSV format or another compatible format.
- `separability_metric`: Metric for evaluating feature separability. 
- `dim_reducer_model`: Dimensionality reduction model applying to your dataset. Must implement a `fit_transform` method for compatibility.
- `label_column`: Name of the column with labels in your dataset. Defaults to `'class'`.
- `budget`: Numeric limit for the total allowable cost of selected features.
- `alpha`: A parameter defines the cost function's scoring method related to the heuristic. See the paper for further details.
- `epochs`: The number of iterations the heuristic uses to solve for each potential k value. See the paper for further details.

Current supported metrics for `separability_metric` are `jm`, `bhattacharyya`, and `wasserstein`. To request support for additional metrics, please open an issue in the repository.

___
> <span style="background-color: yellow;">**_NOTE:_**</span> This method requires that the cost associated with each feature be specified in the dataset file, specifically on the second line. Please verify this arrangement to ensure proper functionality.
___

### Feature Selection
Once the GB-BC-FS object is initialized, you can proceed with the feature selection process:

``` py title="main.py" linenums="1"
selected_features = gbbcfs.select_features()

print("Selected Feature Indices:", selected_features)
```
This method returns the list of features found by the GB-BC-FS algorithm, such that it meets the budget constraints.
### Visualizing the Feature Space
GB-BC-FS includes a visualization method for presenting selected features within the feature space, offering insights into their distribution and separation. This visualization features two adjacent graphs: the **left** graph depicts the initial outcomes of the GB-AFS method, while the **right** graph shows the final results of the GB-BC-FS algorithm after heuristic adjustments have been applied.

The feature spaces displayed in both graphs will be similar, yet there may be notable differences in how the features are clustered and selected. This assumes that the initial solution provided by the GB-AFS algorithm was deemed inadequate, prompting the activation of heuristics.
``` py title="main.py" linenums="1"
gbbcfs.plot_feature_space()
```

This method generates a scatter plot highlighting the selected features. Features are displayed with their separability power indicated by color intensity, and selected features are marked distinctly.

### Get the Selected Features
You can get the selected features in a dictionary format using the following command:
``` py title="main.py" linenums="1"
print(gbbcfs.selected_features_to_cost)
```

## References and Further Reading

For a deeper understanding of the GB-AFS method and its background, consider exploring the [official paper](./gb_afs.md).
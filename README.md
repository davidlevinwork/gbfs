## GB-FS

**gbfs** is a comprehensive repository dedicated to advancing Graph-Based Feature Selection methodologies in machine learning. Our project houses two significant contributions to the field: GB-AFS and GB-BC-FS, each developed to address the intricate challenges of feature selection with graph-based solutions.

[![Downloads](https://static.pepy.tech/badge/gbfs)](https://pepy.tech/project/gbfs) [![Downloads](https://static.pepy.tech/badge/gbfs/month)](https://pepy.tech/project/gbfs)
[![ci Status](https://github.com/davidlevinwork/gbfs/actions/workflows/ci.yml/badge.svg)](https://github.com/davidlevinwork/gbfs/actions/workflows/ci.yml)
[![Tests Status](https://github.com/davidlevinwork/gbfs/actions/workflows/tests.yml/badge.svg)](https://github.com/davidlevinwork/gbfs/actions/workflows/tests.yml)


## Table of contents
- [Our Contributions](#our-contributions)
- [Installation](#installation)
- [Usage](#usage)
  - [GB-AFS](#GB-AFS)
    - [Initialization](#Initialization)
    - [Feature-Selection](#Feature-Selection)
    - [Visualization](#visualization)
  - [GB-BC-FS](#GB-BC-FS)
    - [Status](#status)
- [Documentation](#documentation)
- [Contribution](#contribution)
- [Citation](#citation)

## Our Contributions

- **GB-AFS (Graph-Based Automatic Feature Selection)**: A method that automates the process of feature selection for multi-class classification tasks, ensuring the minimal yet most effective set of features is utilized for model training.
  
- **GB-BC-FS (Graph-Based Budget-Constrained Feature Selection)**: Currently in development, this method seeks to enhance feature selection by integrating budget constraints, ensuring the cost of each feature is considered.

## Installation

`gbfs` has been tested with Python 3.10.

**pip**
```bash
$ pip install gbfs 
```

**Clone from GitHub**
```bash
$ git clone https://github.com/davidlevinwork/gbfs.git && cd gbfs
$ poetry install
$ poetry shell
```

## Usage

### GB-AFS

#### Initialization

To begin working with GB-AFS, the first step is to initialize the GB-AFS object:

``` py bash
from gbfs import GBAFS

gbafs = GBAFS(
    dataset_path="path/to/your/dataset.csv",
    separability_metric="your_separability_metric",
    dim_reducer_model="your_dimensionality_reduction_method",
    label_column="class",
)
```

#### Feature-Selection
After initializing the GB-AFS object, you can move forward with the process of selecting features:

``` py bash
selected_features = gbafs.select_features()

print("Selected Feature Indices:", selected_features)
```

#### Visualization
GB-AFS also incorporates a technique for visualizing the chosen features within the feature space, offering insights into their distribution and how distinct they are:

``` py bash
gbafs.plot_feature_space()
```

### GB-BC-FS

#### Status
Currently in development.

## Documentation
For more information on available commands and usage, refer to the [documentation](https://davidlevinwork.github.io/gbfs/).

## Contribution
Contributions to `gbfs` are welcome! If you encounter any issues or have suggestions for improvements, please open an issue.

## Citation
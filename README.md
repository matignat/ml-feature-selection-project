# Feature Selection Project

## Project Overview
The main objective is to build, evaluate, and optimize predictive models for both classification and regression tasks on a highly dimensional dataset. 
The project demonstrates the critical role of feature selection, regularization, and hyperparameter tuning in preventing overfitting and improving model generalization.

## Dataset
The dataset is artificially generated and consists of:
*   **2000 samples**
*   **400 input variables** (`Input1` to `Input400`), which are standard-scaled real numbers.
*   **Two target variables**:
    *   `Class`: A discrete binary variable (0 or 1) for the classification task.
    *   `Output`: A continuous real number for the regression task.
    
Both output variables have a non-trivial dependency on a specific subset of the input variables.

_Note: Due to GitHub file size limits, the data.csv file is not included. Please run the generate_data.py script first to create the synthetic dataset before running the notebook._

## Methodology and What It Shows
The project is divided into several stages to show the progression from naive models to optimized pipelines:

### 1. Baseline Models
*   **Classification**: A basic `LogisticRegression` model using all 400 features.
*   **Regression**: A basic `LinearRegression` model using all 400 features.
*   *Findings*: Evaluated using 5-fold cross-validation, the baseline models show signs of overfitting and struggle with the high dimensionality of the data, as many input features act as noise.

### 2. Advanced Classification
*   A machine learning pipeline combining `SelectKBest` (for feature selection) and a `RandomForestClassifier`.
*   *Findings*: Through `GridSearchCV`, the model identifies that reducing the feature space from 400 to just 40 informative features drastically improves performance. The Random Forest classifier effectively captures non-linear relationships and interactions between these features.

### 3. Advanced Regression
*   A pipeline integrating `SelectKBest`, `StandardScaler`, and an `ElasticNet` model (which combines L1 Lasso and L2 Ridge regularization).
*   *Findings*: The ElasticNet model pushes irrelevant coefficients exactly to zero (L1) while stabilizing correlated predictors (L2). Hyperparameter tuning reveals that using a subset of 80 features yields the best predictive capability.

### 4. External Validation
*   The project includes a custom function `evaluate_on_validation()` designed to test both baseline and advanced models on an unseen external dataset.

## Results Summary
*   **Classification**: Accuracy improved significantly from **~51.7%** (baseline) to **~72.7%** (advanced pipeline).
*   **Regression**: Explained variance ($R^2$ score) improved from **~0.375** (baseline) to **~0.459** (advanced pipeline).

These results underline that the target variables heavily depend on a specific subset of features rather than the entire 400-dimensional space.

## How to Run the Project

### Prerequisites
Make sure you have Python installed along with the following libraries:
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Execution
1. Clone this repository to your local machine.
2. Ensure the training dataset `data.csv` is placed in the same directory as the Jupyter Notebook.
3. Open the Jupyter Notebook:
   ```bash
   jupyter ml_predictive_modeling-optimization.ipynb
   ```
4. Run all cells sequentially to load the data, train the baseline models, perform hyperparameter tuning via GridSearch, and view the visualization plots.
5. **To validate on new data**: Place a file named `validation_data.csv` (matching the original data structure) in the root directory. The final cell in the notebook will automatically load it, process the data, and output the evaluation metrics (Accuracy and R²) for all four models.

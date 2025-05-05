# Support Vector Machine for Ad Purchase Prediction

## Project Overview
This notebook implements Support Vector Machine (SVM) models to predict whether users will purchase a product based on demographic data from social network advertisements.

## Dataset
- **Source**: Social Network Ads dataset from Kaggle
- **Features**: Age and Estimated Salary
- **Target**: Purchase decision (binary: 0/1)
- **Access**: Data retrieved via KaggleHub API

## Implementation

### Data Processing Pipeline
- Data loading from Kaggle repository
- Null value checking (dataset is complete)
- Feature selection (Age, EstimatedSalary)
- Train-test split (75/25)
- Feature standardization using StandardScaler

### SVM Implementation
- **Linear Kernel**: Basic implementation with C=1
- **RBF Kernel**: Non-linear implementation with gamma=0.1
- **Hyperparameter Tuning**: GridSearchCV to optimize:
  - C values: [0.1, 1, 10, 100]
  - Gamma values: [1, 0.5, 0.1, 0.01]

### Visualization and Analysis
- Decision boundary visualization for both kernels
- Comparative analysis of linear vs non-linear boundaries
- Cross-validation (5-fold) to assess model stability

## Key Results
- Visualization shows different decision boundaries between linear and RBF kernels
- GridSearchCV identifies optimal hyperparameters for the RBF kernel
- Cross-validation provides insight into model performance consistency

## Dependencies
- pandas
- numpy 
- matplotlib
- scikit-learn
- kagglehub

## Usage
1. Ensure kagglehub is configured for dataset access
2. Run cells sequentially to reproduce analysis
3. Examine decision boundary plots to understand model behavior
4. Review cross-validation results for performance assessment

## Conclusions
The notebook demonstrates SVM's ability to create effective decision boundaries for purchase prediction, with the RBF kernel capturing non-linear relationships in the data that the linear kernel might miss.

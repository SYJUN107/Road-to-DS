- Why Feature Selection: Reduce overfitting, improve interpretability, speed up training.
- Methods: Filter (correlation, chi-squared, ANOVA, info gain), Wrapper (forward/backward selection, RFE), Embedded (Lasso, tree-based importance).
- Dimensionality Reduction: PCA (math, conceptual interpretation, limitations), t-SNE, UMAP (conceptual, use cases).
---
These techniques aim to reduce the number of input variables, addressing the "curse of dimensionality" and improving model performance.
# Why Feature Selection?

The process of choosing a subset of relevant features from the original feature set.

1. **Reduce Overfitting:** Fewer features mean less complexity, making the model generalize better and less likely to memorize noise in the training data.
2. **Improve Interpretability:** Fewer features simplify the model, making it easier to understand which factors are most important.
3. **Speed Up Training & Prediction:** Models with fewer features train faster, predict faster, and require less memory.

# Feature Selection Methods

These methods explicitly select a _subset of the original features_.

- **1. Filter Methods:**
    - **Concept:** Evaluate features based on their individual scores (e.g., correlation with the target, statistical tests), independent of the learning algorithm. Features are selected or removed based on these scores.
    - **Examples:**
        - **Correlation:** Removing highly correlated features (reduces redundancy).
        - **Chi-squared test:** For categorical features vs. categorical target (measures independence).
	        - Filters features that has no dependence with the target feature.
        - **ANOVA F-test:** For continuous features vs. categorical target (measures variance between groups).
	        - The logic is to see if the mean of a feature varies for the different categorical target values, if they do vary, it may mean that they have some predictive signal about the target.
        - **Information Gain / Mutual Information:** Measures the dependency between features and the target variable.
	        - Information gain is a specific application of mutual information in decision trees used to decide on which features to split in each node.
	        - In feature engineering it computes the IG or MI between a feature and a target, ranks it by the score and then keep the top-k features.
    - **Pros:** Computationally inexpensive, fast.
    - **Cons:** Don't consider feature interactions, may select suboptimal subsets as features that relies on interactions to give predictive signal may get discarded.

- **2. Wrapper Methods:**
    - **Concept:** Use a specific machine learning model as a "black box" to evaluate feature subsets. They train a model on various subsets and choose the subset that yields the best model performance.
    - **Examples:**
        - **Forward Selection:** Start with no features, iteratively add the feature that most improves model performance.
        - **Backward Elimination:** Start with all features, iteratively remove the feature whose removal least harms model performance.
        - **Recursive Feature Elimination (RFE):** Trains a model, ranks features by importance, removes the least important ones, and repeats. 
	        - It could capture interaction between features indirectly, depending on the model.
	        - Even with models that can capture interactions, RFE can still drop a feature that’s important only through its interaction with another if, at that stage, its individual importance is slightly lower than that of an independent feature. Once removed, it’s never reconsidered.
    - **Pros:** Find feature subsets optimized for a specific model, consider feature interactions.
    - **Cons:** Computationally very expensive (requires training many models), prone to overfitting.
	    - It tends to be *greedy*, meaning it only takes into account local importance, if a feature A that relies on a feature B to be meaningful and feature B is not in the subset that is chosen, it may not choose A and therefore not get features that may be important.
	    - It's also *not exhaustive* as it do not try all possible combinations.

- **3. Embedded Methods:**
    - **Concept:** Feature selection is integrated into the model training process itself. The model learns which features are important during training.
    - **Examples:**
        - **Lasso Regression (L1 Regularization):** Adds a penalty to the sum of the absolute values of coefficients, which can shrink some coefficients exactly to zero, effectively performing feature selection.
        - **Tree-based Models (e.g., Random Forest, Gradient Boosting):** Can provide "feature importance" scores based on how much each feature contributes to reducing impurity or error in the trees. Features can then be ranked and selected.
	        - IG and IM are embedded method here as they are used inside the tree model and the model are not created and used for filtering.
    - **Pros:** Less computationally expensive than wrappers, better than filters at considering interactions.
    - **Cons:** Feature selection is specific to the model being used.

# Dimensionality Reduction

Transforms the original features into a new, lower-dimensional set of features (components) while retaining as much relevant information as possible. These new features are often abstract and less interpretable.

- **1. Principal Component Analysis (PCA)**
    - **Concept:** A linear dimensionality reduction technique. It transforms correlated features into a smaller set of uncorrelated features called Principal Components (PCs). Each PC captures the maximum possible variance of the data, with the first PC capturing the most variance, the second PC the second most, and so on.
    - **How it works (Conceptual):** Finds orthogonal directions (eigenvectors of the covariance matrix) in the data where data varies most. Projects the data onto these chosen principal components.
    - **Limitations:**
        - **Linearity:** Assumes linear relationships between features and components (can't capture non-linear structure).
	        - To capture non linearity you have to engineer interactions between features first, such as using ratios, powers ands so.
        - **Mean-based:** Sensitive to outliers.
        - **Sensitive to scale:** Large values can dominate the variance.
        - **Interpretability:** Principal components are often linear combinations of original features, making them hard to interpret directly.
        - **Retains Variance, Not Necessarily Discriminatory Power:** Maximizes variance, not necessarily class separation; sometimes, low-variance components might be crucial for discrimination.

- **2. t-Distributed Stochastic Neighbor Embedding (t-SNE)**
    - **Concept:** A non-linear dimensionality reduction technique primarily used for **visualization**. It attempts to maintain the local structure of the data, meaning points that are close together in the high-dimensional space remain close in the low-dimensional map. It's good at revealing clusters but doesn't preserve global distances well.
    - **Use Cases:** Highly effective for visualizing high-dimensional datasets, identifying clusters, and exploring data structure, especially in bioinformatics, NLP, and image processing.

- **3. Uniform Manifold Approximation and Projection (UMAP)**
    - **Concept:** Another non-linear dimensionality reduction technique, also excellent for **visualization** and can be used for general dimensionality reduction. It builds a graph representation of the data and then optimizes a low-dimensional layout to match the high-dimensional graph as closely as possible. It generally preserves both local and a more meaningful global structure better than t-SNE, and is often faster.
    - **Use Cases:** Very similar to t-SNE but often preferred for its speed, scalability, and ability to preserve more global structure, making it suitable for larger datasets and as a preprocessing step for ML models.
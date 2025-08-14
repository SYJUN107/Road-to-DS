- Bias-Variance Tradeoff: Definition, high/low bias/variance examples, how to diagnose and mitigate.
- Overfitting vs. Underfitting: Symptoms, causes, and solutions (regularization, cross-validation, feature selection).
- Inductive Bias.
- Curse of Dimensionality: Causes, effects, counter-measures.
- Parametric vs. Non-Parametric Models.
- Supervised, Unsupervised, Semi-supervised, Reinforcement Learning (brief definitions, examples).
- Generalization, Cross-validation strategies (K-Fold, Stratified, Leave-One-Out, Time-Series CV).
---
# Bias-Variance Tradeoff
The bias-variance tradeoff is the relationship between the complexity of the model, its accuracy and how well it responds to unseen data. A model will have high bias when it oversimplifies the problem, not adjusting enough to the data or not being able to capture the underlying information, therefore having bad accuracy. The variance comes when the model adjusts too well to the existing data, therefore a small variation in the input of some variable can lead to big variations in the output, leading to high variance.
The Bias-Variance dilemma is the conflict in reducing both types of errors at the same time.
![[Pasted image 20250807022204.png]]
### Symptoms and Mitigation
- High Bias (Underfitting)
	- Symptoms
		- Poor performance in training and test.
		- Predictions are consistenly off.
	- Mitigation
		- Use more complex model (e.g. deeper trees, high complexity models, ...)
		- Reduce Regularization (e.g. lower L1/L2 penalties)
		- Add more relevant features.
		- Use ensemble methods like **boosting**.
			- Boosting uses several weak models that corrects or tries to minimize error of previous model.
- High Variance (Overfitting)
	- Symptoms
		- Excellent performance on training data but poor performance on test or unseen data.
	- Mitigation
		- Simplify model (e.g. prune trees, try simpler models, reduce layers in ANNs)
		- Increase Regularization
		- Apply techniques such as early stopping or dropout for NNs.
		- Use ensemble methods like **bagging**.
			- Uses several weak models with subsets of the original dataset, achieving smoothing basing on the average of outputs.
# Overfitting vs Underfitting 
The core of ML is to get a model that **generalizes** well. That means the model performs good both on training data and unseen data. However, that is often not the case, which lead to overfitting and underfitting.
Overfitting is when model adjusts itself too much to the training data and not being able to generalize properly, leading to poor performance when receiving unseen data. Underfitting is when the model is unable to capture the underlying patterns or information of the data leading to poor performance even in training. 
Overfitting is equivalent to High Variance and Underfitting to High Bias. The diagnosis and mitigation can refer to the one in the section of Bias-Variance Tradeoff.

# Inductive Bias
Inductive bias is caused by assumptions or suppositions that models make in order to generalize well from training data to unseen data. 
This happens as ML models needs to make predictions with data it has not seen. No matter how big the dataset used for training was, it still covers only a fraction of all possible data. Because of this, in order to generalize, the model needs to make assumptions or have preferences when making predictions. This assumptions is the inductive bias we are talking about.
According to the [No Free Lunch Theorem](https://en.wikipedia.org/wiki/No_free_lunch_theorem), no ML model is universally better than all others across all aspects in all problems. An algorithm performs better than other on a certain problem only because its inductive bias (its assumptions) aligns well with the problems characteristics. If a model do not have inductive bias, it will perform equally poor on all problems as there is no guiding line that tells the model which hypothesis from the infinite mathematical functions that exists to choose from that fits the training data and performs well in unseen data.
### Causes
#### Algorithms
The inductive bias is inherent in algorithms:
- **Linear Regression**: Assumes linear relation between input and output. Its inductive bias prefers simple, linear solutions.
- **Decision Trees**: Prefers splitting data based on a hierarchical manner. Its inductive bias makes it prefer smaller trees, either because of greedy splitting algorithms or pruning.
- **SVMs**: Prefers finding hyperplanes that maximizes the margin between classes. Its inductive bias prefer widely separated boundaries.
- **NNs**: With many layers and non linear activations, they have bias towards hierarchical (because of layered abstractions), distributed (instead of one neuron representing one concept, multiple neurons encode this information) and smooth non-linear functions (non linear functions that changes gradually and are differentiable).
#### Objective Function/Loss Function
The choice of the objective/loss function implicitly guides the model towards a certain type of solution.
- Squared Root Error: Continuous predictions
- Cross-Entropy: Probabilistic predictions
#### Regularization Techniques
Regularization is a set of techniques that prevents overfitting to training data.
L1 and L2 are the most common type of regularization types:
- **L1 Regularization (Lasso)**: Adds a penalty proportional to the absolute value of the weights. 
	- Used when there is suspicion that only a small subset of features are relevant.
- **L2 Regularization (Ridge)**: Adds a penalty proportional to the square of weights.
	- Used where all features are potentially relevant.
In this case, the Inductive bias of these regularization techniques leads to the output of simpler models, avoiding overly complex functions (that leads to overfit).
# Curse of Dimensionality
The "Curse of Dimensionality" refers to various phenomenas that happens when working with high dimensionality data (many features).
### Causes
The fundamental cause is the exponential growth of volume with the increase in the number of dimensions (feature).
- **Data Sparsity**: with the increase in dimensions, it requires more data to fill out the dataset with enough information ("filling the empty spaces").
	- Any fixed-sized dataset will become sparse in a high-dimensional space, making it harder for algorithms to find meaningful patterns from the sparse data.
	- There is less overlap of points between existing ones because of the sparsity.
- **Distortion in Distances**: In a high-dimensional space, the distance between different points tends to be fuzzier, as with so many dimensions, points tends to appear equidistant from each other.
	- This makes distance-based metrics lose their discriminatory capability as concepts like *near* and *far* becomes fuzzier.
- **Increase Complexity of Feature Interactions**: The number of potential feature interactions increase proportionally to the number of dimensions, so higher dimensions means higher number of possible interactions to check and verify.
### Effects
#### Increased Risk of Overfitting
With high sparsity in the distribution of points, there is high chance of finding patterns that exists purely by chance because of the data sample, leading to overfitting. 
#### Worsening of Model Performance
As previously mentioned, algorithm based on distance such as SVMs, K-Means Clustering or K-Nearest Neighbors cant perform well as distance loses its meaning.
#### Increased Computational Cost
As there are more features, models needs more calculations and memory to train and learn from the parameters, feature interactions and so. This means that the model will generally need more time and computational resources to train the model.
#### Data Visualization and Interpretation
Humans can visualize 1D, 2D and 3D. However, it becomes increasingly difficult to visualize higher dimensions as there is no clear way of representing the parameters.
#### Increase Need for Data
In order to maintain data density (if data points are close together, high density, it is easier to find patterns and so compared to being all dispersed around) as we increase the number of dimensions, we will need more data. This can be difficult in real world case scenarios as data may not be something easy to obtain.
### Counter-Measures
#### Dimensionality Reduction Techniques
##### Feature Selection
The goal is to identify and keep only the most relevant parameters/features.
Methods:
- **Filter methods**: Evaluating each parameter independent of the model.
	- Its fast and scalable but does not consider feature interaction.
	- E.g., correlation, chi-squared, ...
- **Wrapper methods**: Uses learning models to evaluate subsets of characteristics and find the best performing subset.
	- It does customizable iterative loops evaluating subsets of features. As it is iterative, it can be computationally costly. It can also lead to overfitting if not validated correctly.
	- E.g., recursive feature elimination, forward/backward selection, ...
- **Embedded methods**: The model internally makes prioritization or filtering of features.
	- More efficient than wrappers as it uses the model capabilities for feature selection, penalizing and prioritizing different features. However, it depends entirely on the model chosen, so it can be not interpretable if the model is very complex.
	- E.g., Lasso regression built-in selection, tree based feature importance, ...
##### Feature Extraction
Transform high dimensional data into lower dimensional space. This can be achieved by combining or projecting original features.
###### Dimension 
- **PCA** (Principal Component Analysis): Linear transformations that finds orthogonal components (eigenvectors) that capture most variance in data.
- **t-SNE** (t-distributed Stochastic Neighbor Embedding): Non-linear technique primarily used for visualization, aiming to preserve local neighborhoods (points that are close in the high dimensional space remains close in the representation).
- **UMAP** (Uniform Manifold Approximation and Projection): Another non-linear technique often faster than t-SNE and is able to capture global structures.
	- *Note*: Local structures refers to close points remains close in lower dimension spaces while global structures refers to the distribution of groups of points maintains the same. Note that this means that a representation can maintain local structure, meaning that points close to each other remains the same, however, the distribution of group of points may not represent the same way how it was in the higher dimension.
- **Autoencoders**: Learns a compressed version (encoding) of the input data.
#### Regularization
Prevents overfitting in high dimensions by adding penalty to the loss function when going for overly complex solutions.
The main regularization techniques are L1 (Lasso) and L2 (Ridge) but there are others such as Elastic Net that is a combination of both.

| Feature                            | L1 (Lasso)                                                                | L2 (Ridge)                                                  |
| ---------------------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Penalty Term                       | Sum of *absolute values* of weights                                       | Sum of *squared values* of weights                          |
| Math → Derived form (Optimization) | $$\lambda \sum_{i=1}^{N} \lvert w_{i}\lvert \rightarrow 2 \lambda w_{i}$$ | $$\lambda \sum_{i=1}^{N} w_{i}^{2} \rightarrow \pm\lambda$$ |
| Effect on Weights                  | Reduces many weights to 0                                                 | Shrinks weights evenly                                      |
| Sparsity                           | Produces sparse models (feature selection)                                | Keeps all features but with small weights                   |
| Best Use Case                      | High-dimensional data, feature selection                                  | When features may be correlated and noisy                   |
#### Tree Based Algorithms
Given that many models perform greedy splits based on the most relevant features, ignoring the less relevant, so it performs feature selection implicitly. Because of this, it do not degrade the model significantly, but they may increase the computational cost.
#### Collect more data
The issue arises because models struggle to learn from sparse datasets, as it may not be able to find enough points to discover meaningful patterns. To solve this, a straight forward solution is to get more data. Even thought this may not be always a feasible solution as the amount of data required increases exponentially with the number of dimensions. However it can still help alleviate this issue to some extent.
#### Domain Knowledge and Feature Engineering
Basically, know the domain of the problem and handpick and create new relevant features that simplifies and reduces the number of features, maintaining only relevant ones.
# Parametric vs Non-Parametric
**Parametric models** assumes fixed, known functional form for the relationship between input and output. Given that it makes assumptions that depends strongly on the function chosen, if chosen the one that aligns correctly with the problem, its performance will be better and may require less data to perform well. 
- It is *more efficient* if assumptions are correct but can lead to *underfitting* if assumptions are wrong.
- It may be *prone to underfitting* if the function do not align with the problem correctly. 
- E.g., Linear regression, logistic regression, ...

On the other hand, **non-parametric models** refers to those models that makes minimal or no assumptions regarding the form of the function. Their complexity can grow with the data as they learn from the data itself. 
- *Highly flexible* and can capture complex non linear relationships.
- Requires more data, are *slower*, *less interpretable* and prone to *overfitting*.
- E.g., K-NN, Tree based models, Deep Neural Networks, ...

So in summary:
- In Parametric you decide the model's function and then fit the data.
- In Non-Parametric the data decides the model's form.
# Main ML Paradigms
### Supervised learning
Trains model with a dataset that has *each input paired with the correct output label*. 
- The **goal** is to learn a mapping function that maps a input to a corresponding output.
- **Key idea**: Learn from labeled examples.
- E.g.
	- Image Classification: Classification of images based on a label.
	- Spam detection: Detect spams with labeled mails.
### Unsupervised learning
Trains model with a dataset that has *no corresponding output label*. 
- The **goal** is to find hidden patterns, structures or relationships within the data itself.
- **Key ideas**: Discovering tables within unlabeled data.
- E.g.
	- Clustering: Grouping similar data together.
	- Dimensionality Reduction: Simplifying complex data by finding its main components.
### Semi-supervised learning
Trains the model on a dataset with a *small portion of labeled data* and a *large portion of unlabeled data*. It addresses the situation where labeling data is expensive or time-consuming.
- **Key idea**: Leverage both labeled and unlabeled data.
- E.g.
	- Medical Image Analysis: Using few labeled MRI scans and many unlabeled ones to detect diseases.
	- Text classification: Train on small set of labeled reviews and then using unlabeled reviews to refine the model.
### Reinforcement Learning (RL)
Trains an agent to make a sequence of decisions in an environment to maximize a cumulative reward. The agent learns through a trial-error process by interacting with the environment and receiving feedback (rewards/penalties) from it.
- **Key idea**: Learning from interacting with the environment and obtaining feedback.
- E.g.
	- Game playing: An agent learn to play chess by making moves and receiving feedback from it.
	- Robotics: A robot learning to walk or grab objects by trying different actions and optimizing efficiency.
# Generalization & Cross-validation strategies
### Generalization
**Definition**: A model's ability to predict on new, unseen data. This demonstrates that the model have learned the patterns correctly, not just memorized the data.
**Importance**: Prevents overfitting. A model that generalizes poorly is useless in real world scenarios.
**Measurements**: Performance metrics (accuracy, RMSE, F1, ...) evaluated from a different dataset than that used for training or through cross-validation.
### Cross-Validation Strategies
**Purpose**: A resampling strategy designed to be more robust for testing model's performance than a simple train-test split. It helps in hyper-parameter tuning and model selection providing a reliable estimate on how it will perform on unseen data while making full use of the data set.
#### K-Fold CV
**How it works**: The dataset is shuffled and divided into K equally sized folds. The model is trained K times using one of the K folds to validate and the rest (K-1 folds) to train at a time. The measurement then is done averaging the performance on all iterations.
**Use Case**: General-purpose, widely used for performance estimation and hyperparameter tuning.
#### Stratified K-Fold CV
**How it works**: Same as K-Fold CV but it keeps the same distribution of samples for each target class as the complete dataset.
**Use case**: In classification problems, mainly for *imbalanced datasets*, to ensure a representative distribution of the problem.
#### Leave-One-Out CV
**How it works**: An extreme case of K-Fold CV by having K the same number as the number of samples (N). It uses only one sample to validate and the rest (N-1) to train.
**Use case**: On very small datasets (rarely). Provides a extremely robust estimate but is extremely computationally expensive.
#### Time-Series CV (Walk-Forward Validation)
**How it works**: Data is split chronologically, respecting the temporal order. The model is trained on an initial segment of historical data and validated using the following split. Then, the training window can expand or slide forward in time, validated using the subsequent periods.
**Use case**: Any time-dependent data (e.g., stock trading, sensor info, sales forecasting, ...) where future information cannot be used to predict the past. Crucial for realistic evaluation and avoiding data leakage.

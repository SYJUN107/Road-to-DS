- Common Techniques: One-hot encoding, label encoding, target encoding, frequency encoding, numerical scaling (standardization, normalization).
- Feature Interaction: Polynomial features, interaction terms.
- Time-based Features: Lag features, rolling averages, cyclical features, Fourier transforms for periodicity.
- Domain-Specific Feature Creation (brief examples for NLP, CV, RecSys).
- Embedding Techniques (brief intro, deep dive in DL section).
- Feature Scaling considerations for different models.
---
This involves creating new features or transforming existing ones to make the learning algorithm perform better.

# Common Feature Encoding & Transformation Techniques
### One Hot Encoding
- **Concept:** Converts categorical variables into a binary (0 or 1) vector. Each category becomes a new column, indicating presence (1) or absence (0).
- **Use Case:** When categorical features have no intrinsic ordinal relationship (e.g., colors: Red, Blue, Green). Prevents models from assuming an ordering.
### Label Encoding
- **Concept:** Assigns a unique integer to each category of a categorical variable (e.g., Red=0, Blue=1, Green=2).
- **Use Case:** When categorical features _do_ have an ordinal relationship (e.g., small=0, medium=1, large=2). Can be used for tree-based models where ordinality doesn't imply magnitude interpretation to the model.
	- By *magnitude interpretation* we refer that the differences between the values are meaningful in magnitude.
### Target Encoding
- **Concept:** Target encoding is essentially about replacing a categorical value with a numeric feature that captures its relationship to the target variable. (e.g., 'City A' replaced by average house price in City A).
- **Use Case:** High cardinality categorical features. Captures correlation with target but prone to overfitting; often requires smoothing or cross-validation.
### Frequency Encoding
- **Concept:** Replaces a categorical value with its frequency (count or proportion) in the dataset.
- **Use Case:** High cardinality categorical features where the frequency itself might be a useful predictor.

# Numerical Scaling
Transforms numerical features to a similar scale to prevent features with larger ranges from dominating models.
### Standardization (Z-score Normalization)
- **Concept:** Rescales data to have a mean of 0 and a standard deviation of 1. 
	- Formula: $X_{scaled} = (X - \mu) / \sigma$.
- **Use Case:** Models sensitive to feature scales (e.g., SVM, K-NN, Logistic Regression, Neural Networks). Assumes data is normally distributed (or close enough).
### Normalization (Min-Max Scaling)
- **Concept:** Rescales data to a fixed range, typically [0, 1]. Formula: $X_{scaled} = (X - X_{min}) / (X_{max} - X_{min})$.
- **Use Case:** When a specific bounded range is desired (e.g., for image processing, or some neural network activation functions). Sensitive to outliers.

# Feature Interaction

Creating new features that capture the combined effect of two or more existing features.
### Polynomial Features
- **Concept:** Creates new features by raising existing features to a power (e.g., $X^2, X^3$).
- **Use Case:** To model non-linear relationships. Allows linear models to fit curved data.
### Interaction Terms
- **Concept:** Creates new features by multiplying two or more existing features (e.g., $X_1 \times X_2$).
- **Use Case:** To capture scenarios where the effect of one feature on the target depends on the value of another feature.

# Time-based Features

Essential for time-series data, extracting knowledge from the temporal dimension.
### Lag Features
- **Concept:** The value of a variable at a previous time step (e.g., "sales yesterday", "temperature 3 hours ago").
- **Use Case:** Capturing temporal dependencies and auto-correlation.
### Rolling Averages (and other statistics)
- **Concept:** Calculates a statistic (mean, sum, min, max, std dev) over a _sliding window_ of past time points for a feature (e.g., "average sales over the last 7 days").
- **Use Case:** Smoothing out noise, identifying trends, or volatility over a period.
### Cyclical Features
- **Concept:** Encoding cyclical patterns (e.g., day of week, month, hour of day) using sine and cosine transformations to preserve continuity and relationship.
- **Use Case:** When patterns repeat periodically (e.g., higher traffic on weekdays). `sin(2*pi*hour/24), cos(2*pi*hour/24)`.
	- It is used so that the model can perceive notions such that 23h and 0h are close instead of far apart if we encode it purely numerically.
### Fourier Transforms for Periodicity
- **Concept:** Decomposes a time series into its constituent frequencies and amplitudes.
- **Use Case:** Identifying dominant periodic components within complex time series, which can then be used as new features.
# Domain-Specific Feature Creation
Leveraging expert knowledge of the problem domain to create highly informative features.
### NLP (Natural Language Processing)
- **Examples:** TF-IDF (Term Frequency-Inverse Document Frequency), N-grams (sequences of words), sentiment scores, readability scores, part-of-speech tags, topic models.
### CV (Computer Vision)
- **Examples:** Edge detection features, color histograms, texture patterns (e.g., Haralick features), SIFT/SURF descriptors (local features for object recognition).
### RecSys (Recommender Systems)
- **Examples:** User's average rating, item's average rating, number of interactions for user/item, time since last viewed/purchased.

# Embedding Techniques (Brief Intro)

- **Concept:** Representing high-dimensional, sparse or discrete data (like words, users, items, categories) as dense, continuous, low-dimensional vectors. These embeddings learn to capture semantic or functional relationships between entities.
- **Role:** Often learned via neural networks (e.g., Word2Vec, BERT) or matrix factorization (e.g., for users/items in RecSys). They translate discrete IDs into a meaningful vector space where similar items are closer.
- **Note:** Can be used for dimensionality reduction. However, it has to be trained or use an pre-trained model. Usually used for texts.
- _(Deep dive into specific embedding methods belongs in the Deep Learning section)._

# Feature Scaling Considerations for Different Models

Whether to scale features depends on the underlying mechanism of the ML algorithm:
### Models Requiring Scaling (Sensitive to Feature Magnitudes/Distances):
- **Distance-based:** K-Nearest Neighbors (K-NN), K-Means, Support Vector Machines (SVMs) – features with larger ranges would disproportionately influence distance calculations.
- **Gradient Descent-based:** Linear Regression, Logistic Regression, Neural Networks – unscaled features can lead to vastly different gradient magnitudes, making optimization difficult, slow, or unstable.
- **Regularized Models:** Ridge, Lasso – regularizers penalize coefficients. Unscaled features mean the penalty applies differently to coefficients for features with larger ranges, potentially penalizing important features too much or too little.
### Models NOT Highly Sensitive to Scaling (Tree-based)
- Decision Trees, Random Forests, Gradient Boosting Machines (XGBoost, LightGBM) – these models make decisions based on thresholds (e.g., "Is Age > 30?"), so the absolute scale of features doesn't directly impact split points; only their relative order matters.
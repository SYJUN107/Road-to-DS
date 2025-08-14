- Linear Algebra: Vectors, Matrices, Eigenvalues/Eigenvectors, SVD (conceptual understanding and applications in PCA, Recommender Systems).
- Calculus & Optimization: Derivatives, Gradients, Gradient Descent (variants: SGD, Adam, RMSProp - conceptual differences), Jensen's Inequality, Convexity (brief).
- Probability & Statistics (focus on ML applications): Bayes' Theorem, PDF/PMF, Likelihood, MLE, MAP, Central Limit Theorem.
---
# Linear Algebra
Linear algebra is the math of vectors, matrices and linear transformations. It's the base of many ML algorithms.
### Vectors
An ordered list of numbers that can represent:
- **Coordinates**: Points in space: $[x, y, z]$
- **Directions and magnitudes**: An arrow from the origin.
- **Data points**: A list of features for a single observation (an image's pixel's values).
**Role in ML**: The primary way individual data points are represented and worked upon in ML models. Operations are calculated to obtain distances, similarities and so between different data points. 
### Matrices
A rectangular array of numbers, arranged in rows and columns that can represent:
- **Datasets**: Collection of vectors. Each row is usually a data point and each column a feature.
- **Linear transformations**: Operations that scale, rotate or shear vectors.
- **Systems of equations**.
**Role in ML**: Most data in ML is stored and manipulated as matrices. Matrix multiplication is central to many algorithms such as neural network forward passes.
### Eigenvalues and Eigenvectors
**Eigenvector**: A special non-zero vector that when a linear transformation is applied upon it, it only changes in magnitude (it scales) and does not change it's direction. It represents a "*principal direction*" or axis along which variance/change is maximized.
**Eigenvalues**: The scalar factor by which the eigenvector is scaled. It tells how much a eigenvector stretches or compresses.
**Role in ML**:  Essential for understanding the key direction in variance in data, fundamental to dimension reduction (because we can get the direction that best explains the variance).
	- *Note*: It only works with square matrices (because it works upon the covariance matrix which is always square). If it is not square other metrics can be used such as SVD.
### Singular Value Decomposition (SVD)
A powerful technique that decomposes any matrix, be it square or not, into three simpler matrices:
$A = U \Sigma V^T$
- $U$ (Left singular vector): Orthogonal matrix representing the basis of the column space (e.g., data points/features).
- $\Sigma$ (Singular values): A diagonal matrix containing non-negative values (singular values) in descending order that represents the "importance" of each corresponding singular vector pair. They are analogous to eigenvalues.
- $V^T$ (Right singular vector): Orthogonal matrix representing the basis for the row space (e.g., features/latent components).
It essentially breaks a complex transformation into a rotation, a scaling and another rotation, corresponding to each element of the formula above.
**Role in ML**: Highly versatile for dimensionality reduction, noise reduction and uncovering latent structures. It is a generalization of the Eigenvalue Decomposition.
### Applications in ML
#### Principal Component Analysis (PCA)
A dimensionality reduction system which transforms the data into a new coordinates systems, where the first few dimensions (principal components) captures the maximum variance in data.
**How it works**: 
- It finds the eigenvectors of the data's covariance matrix (eigendecomposition). These eigenvectors are the principal components
- The eigenvalues indicate the amount of variance captured by each principal components.
- Data is then projected onto the top K eigenvectors (those with the largest eigenvalues) to reduce the number of dimensions to K while preserving the most variance.
*Note*: If data not square just use SVD to do PCA. In fact, most of the time PCA is done using SVD, including many libraries. PCA is mainly used if you work specifically with covariance matrix for some reason.
#### Recommender System (Latent Factor Models/SVD-based)
The objective is to create a predictive system that can predict user preferences by uncovering "latent factors" that influence ratings (user ratings of certain products). 
**How it uses SVD**:
- A user-item interaction matrix where users are rows, items are columns and ratings are entries is often sparse (one user could not interact with all items) and high dimensional (many items);
- SVD is applied to this matrix (or a modified version of it, preprocessed maybe) to decompose it to:
	- User features matrix ($U$)
	- Item features matrix ($V^T$)
	- A Singular values matrix ($\Sigma$) indicating the importance of this latent features.
- By keeping only the top K singular values and their corresponding vectors we can obtain a low-dimensional representation (latent factor) for both users and items.
- These representations might represent their underlying tastes (user that likes X genres).
- 
# Calculus & Optimization
# Probability & statistics

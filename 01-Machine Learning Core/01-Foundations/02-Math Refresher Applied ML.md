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
- The predictions (recommendations) can be obtained by multiplying the top K singular values and their corresponding user and item vectors such that: $R_{aprox}=U_k \Sigma_k V_k^T$ . By reconstructing it this way, $R_{aprox}$ will have all entries filled, including those that were originally missing, with predicted score representing a estimation on how the user will rate certain item. These scores can then be used to make recommendations.
# Calculus & Optimization
At its core, ML is about optimization, either minimizing a loss function or maximizing a reward. Calculus provides tools for this.
### Derivatives & Gradients
**Derivative (univariate)**: For a single variable function, the derivative measures the instantaneous rate of change of the function at a specific point. Geometrically, it shows the slope of the tangent to the function's curve at a specific point. It tells you which way the function increase/decrease its value.
**Gradient (Multivariate)**: For a function of multiple variables, the gradient is a vector of all partial derivatives of the function. 
**Role in ML**: We need to find the direction of the steepest descent (negative gradient) which minimizes the loss function. 
### Gradient Descent
It is an iterative optimization algorithm that finds the minimum of a function. To achieve this, it takes steps in the opposite direction of the gradient of the loss function with respect to the model's parameters.
**Process**:
1. Initialize model parameters randomly.
2. Calculate the gradient of the loss function according to the model's parameters.
3. Update parameters by moving a step in the direction opposite to the gradient.
	- The size of the step is controlled by the learning rate.
4. Repeat until convergence (e.g., loss stops decreasing significantly).
Even thought the goal is to reach a global minimum, it could get stuck inside a local minimum, especially in complex, non-convex loss landscapes.
### Gradient Descent Variants
These variants addresses the trade-off of different aspects of the optimization algorithm, such as computational cost, speed of convergence or scalability.
#### Stochastic Gradient Descent (SGD)
- **Conceptual Difference:** Instead of calculating the gradient over the entire dataset, it calculates it using *only one random chosen training example at a time*.
- **Pros:** Much faster for larger datasets plus it can also escape shallow local minima due to noisy updates.
- **Cons:** Noisy updates can lead to oscillatory convergence which requires careful tuning of the learning rate.
- **Note:** Often SGD in ML literature actually refers to Mini-Batch Gradient Descent.
#### Adam (Adaptative Moment Estimation)
- **Conceptual Difference:** An adaptive learning rate optimization algorithm which combines ideas of RMSProp and momentum:
	- It stores exponentially [decaying averages](https://www.studocu.com/en-us/messages/question/10613568/what-is-a-decaying-average#:~:text=A%20decaying%20average%2C%20also%20known%20as%20an%20exponential,average%20more%20responsive%20to%20changes%20in%20the%20data.) (AKA EMA) of past gradients (*momentum concept*).
	- It stores exponentially decaying averages of past **squared gradients** (*RMSProp concept*).
	- it then uses these moments to adaptively decide on the learning rate of **each parameter**. It essentially provides a personalized learning rate for each parameter based on its past behavior.
- **Pros:** Generally very effective, fast convergence, good performance across many problem types, often do not require extensive learning rate tuning.
- **Cons:** Can sometimes generalize poorly (overfitting) compared to SGD with momentum (thought this is often fine-tuned away).
#### RMSProp (Root Mean Square Propagation)
- **Conceptual Difference:** It's also an adaptive learning rate algorithm which maintains an exponentially decaying average of the **squared gradients** for each parameter. It then divides the learning rate by the square root of this EMA, effectively scaling down learning rates for parameters with frequent occurrence and/or large gradients.
- **Pros:** Addresses the problem of [vanishing/exploding gradients](https://deepgram.com/ai-glossary/vanishing-and-exploding-gradients) in RNNs. Helps with controlling the learning rate effectively.
- **Cons:** Doesn't use the concept of momentum like how Adam does making it prone to slow convergence and less stable updates compared to Adam.
### Jensen's Inequality
- **Super useful videos:** 
	- [video1 (quite visual and high level)](https://www.youtube.com/watch?v=u0_X2hX6DWE)
	- [video2 (more technical)](https://www.youtube.com/watch?v=GDJFLfmyb20)
- **Concept:** Imagine you have a line function $f(X)=X$ . If you first get the average of $X \{x1, x2, ...\}$, be it $\bar{X}$, and then introduce it to the function $f$ such that $f(\bar{X})$, it will be the same as if you were to first get all outputs and then average them ($\bar{f(X)}$), resulting in: $f(\bar{X})=\bar{f(X)}$.
  What Jensen's inequality tells us is that if the function curves upwards, lets call it $g$ (convex function), $g(\bar{X})\le \bar{g(X)}$. This can be visualized as curving $f$, the middle point ($\bar{X}$) will curve below $f$ and the extremes will curve above $f$. For concave function ($h$) it is the other way around such that $h(\bar{X})\ge \bar{h(X)}$.
	- Convex function: $$ f\left( \text{average of inputs} \right) \leq \text{average of outputs} $$![[Screenshot 2025-08-21 at 17.02.05.png]]
	- Concave function $$ f\left( \text{average of inputs} \right) \geq \text{average of outputs} $$
- **Role in ML:** Used in various proofs and derivations in convex optimization, information theory (e.g., proof of non-negativity of KL divergence) and the theoretical foundation for algorithms like Expectation Maximization.
### Convexity
- **Concept:** A property of a function where any line between 2 points lies above or below the graph. 
	- *Convex Function*: Has a "bowl" shape, meaning any local minimum is also the global minimum.
	- *Convex Set*: A set of points is called convex if, for any two points within the set, the line segment connecting them lies entirely within the set. Geometrically, this means the region includes all line segments between any pair of points in the set—it doesn’t “cave in” or exclude any part of those segments.
- **Role in ML:**
	- **Optimization:** For convex loss functions, gradient descent is guaranteed to find global minimum (if the learning rate is chosen correctly). This makes optimization much straight forward.
	- **Guarantees:** Many algorithms optimizes convex loss function providing theoretical guarantees on convergence and optimality.
	- **Non-convexity:** DNNs typically have non convex loss landscapes, meaning they can get stuck on local minima, making optimization a bigger challenge.
# Probability & statistics
These concepts provides the framework for you to understand uncertainty, modeling data and making informed decisions in ML.
### Baye's Theorem
**Concept:** A fundamental theory that describes how to update the probability of a hypothesis $H$ based on new evidence $E$. It provides a way of getting posterior probability using prior knowledge and likelihood.
**Formula**: $P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}$
- $P(H|E)$: Posterior probability of the hypothesis based on given evidence.
- $P(E|H)$: Likelihood of evidence given a probability.
- $P(H)$: Marginal (total) probability of the hypothesis.
- $P(E)$: Marginal probability of the evidence.
**ML Application:** Core to Bayesian models such as Naive Bayes Classifier or Bayesian inference. It is also the foundation for many probabilistic graphical models such as Hidden Markov Models. This concept is useful for incorporating beliefs into models such that some evidence (prior condition) can give us more information of the likelihood of something happening.
### Probability Density Function (PDF) & Probability Mass Function (PMF)
**Concept:** Functions that describes the probability distribution of a random variable.
- **PMF (for discrete random variables):** Gives the probability for a discrete random variable to take on a certain value such as $P(X=x)$. The sum of all PMF values for all possible outcomes is 1.
- **PDF (for continuous random variables):** The probability that a continuous random variable falls within a range of values $f(x)$ is the probability density at $x$. The probability of any single point is 0 (as it is extremely unlikely because continuous values have infinite values between them). The total area under the curve is 1.
**ML Application:** It is used to model the distribution of input features or output variables (e.g., Gaussian distribution for continuous features, Bernoulli for binary outcomes). It is essential for probabilistic methods.
### Likelihood
**Concept:** The probability or the probability density of observing a given data when model parameters are fixed. It's written as $P(\text{Data}|\text{Parameters})$.
**Crucial distinction:** Likelihood is not the same as probability. The key difference lies in the fact that probability refers to the probability of an outcome based on given parameters, $P(\text{Outcome}|\text{Input Parameters}$); whereas likelihood refers to how plausible different parameters are when you know the outcome, $P(\text{Input Parameters} | \text{Outcome})$.
**ML Application:** The core function to be maximized in the Maximum Likelihood Estimation. It quantifies how well a set of parameters can explain over the observed data.
- In this case by explain it means how probable is to get the observed data with certain parameters.
### Maximum Likelihood Estimation (LME)
**Concept:** A method for estimating the parameters of a statistical model. It finds the set of parameters that maximizes the likelihood function, meaning the parameters that makes the observed data most probable.
**Method:** Often involves taking the logarithm of the likelihood function and then taking derivatives and setting to zero to find the maximum.
**ML Application:** Widely used to "fit" parameters for many models such as Linear Regression (under Gaussian noise assumption), Logistic Regression, NB, Gaussian Mixture Models, and in Neural Network training objectives. It generally yields consistent and asymptotically (when sample size grows to infinity) efficient estimators. 
### Maximum A Posteriori (MAP)
**Concept:** An extension of MLE that incorporates prior knowledge or beliefs about the parameters. It finds the parameter values that **maximize the posterior probability** ($P(\text{Parameters}|\text{Data})$) of the parameters given the data, which is proportional to (Likelihood * Prior).
**Formula (Proportional):** $\text{MAP Estimate} \propto P(\text{Data} | \text{Parameters}) \cdot P(\text{Parameters})$
**ML Application:** Used when there's reliable prior information (e.g., regularization in linear models can be viewed as MAP estimation with a Gaussian or Laplace prior on parameters). It helps to prevent overfitting when data is sparse, by nudging parameters towards values suggested by the prior.
### Central Limit Theorem (CLT)
**Concept:** States that the distribution of sample means (or sums) from almost any population distribution will be approximately **normally distributed** (aka Gaussian; it do not include skewed versions), provided the sample size is sufficiently large, regardless of the original population's distribution.
**Key Conditions:** Samples must be independent and identically distributed (i.i.d.), and the population must have a finite mean and variance (these values exists, is a real number and not infinite).
**ML Application:**
- **Statistical Inference:** Justifies the use of normal distribution-based statistical tests and confidence intervals for sample means, even if the underlying data isn't normal.
- **Neural Networks:** Helps explain why aggregate activations or weights might tend towards normal distributions.
- **Bootstrapping:** Underpins the validity of using resampling methods to estimate properties of sample distributions.

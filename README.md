# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Minh Hoang's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/maxie.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| AdaGrad Optimizer | Implement a vectorized AdaGrad update in NumPy with accumulated squared gradients and adaptive per-parameter learning rates. | https://www.tensortonic.com/problems/adagrad-optimizer |
| Implement Adam Optimizer Step | Implement one vectorized Adam optimizer step in NumPy with first and second moments, bias correction, and elementwise parameter updates. | https://www.tensortonic.com/problems/adam-optimizer |
| Implement AdamW (Decoupled Weight Decay) | Implement one AdamW optimizer step in NumPy with first and second moments plus decoupled weight decay. | https://www.tensortonic.com/problems/adamw-optimizer |
| Compute AUC (Area Under ROC) | Calculate binary-classification ROC AUC from false-positive and true-positive rates using trapezoidal integration. | https://www.tensortonic.com/problems/auc |
| Bag-of-Words Vector | Build a NumPy bag-of-words count vector from an ordered vocabulary while ignoring out-of-vocabulary tokens. | https://www.tensortonic.com/problems/bag-of-words |
| Bernoulli Probability Mass Function & Moments | Compute the Bernoulli probability mass function, expected value, and variance for a valid success probability. | https://www.tensortonic.com/problems/bernoulli-pmf |
| Bigram Probabilities (Add-1 Smoothing) | Estimate bigram probabilities from token sequences using add-one smoothing over a fixed vocabulary. | https://www.tensortonic.com/problems/bigram-probabilities |
| Binary Focal Loss | Compute binary focal loss from predicted probabilities with class balancing, focusing strength, and stable logarithms. | https://www.tensortonic.com/problems/binary-focal-loss |
| Binning | Assign numeric values to ordered bins using supplied boundaries while handling values at interval edges. | https://www.tensortonic.com/problems/binning |
| Binomial Probability Mass Function | Compute binomial probability mass and cumulative probabilities from trial count, success probability, and outcome. | https://www.tensortonic.com/problems/binomial-pmf-cdf |
| BLEU Score | Calculate a BLEU translation score from candidate and reference tokens using clipped n-gram precision and brevity penalty. | https://www.tensortonic.com/problems/bleu-score |
| Implement BM25 Ranking Score | Implement BM25 document ranking with term frequency saturation, inverse document frequency, and length normalization. | https://www.tensortonic.com/problems/bm25 |
| Catalog Coverage | Measure recommendation catalog coverage as the fraction of available items appearing across user recommendation lists. | https://www.tensortonic.com/problems/catalog-coverage |
| Implement Causal Masking for Attention | Create a causal attention mask that blocks each token from attending to future positions in a sequence. | https://www.tensortonic.com/problems/causal-masking |
| Compute Accuracy, Precision, Recall, F1 | Compute binary accuracy, precision, recall, and F1 score from predicted and true class labels. | https://www.tensortonic.com/problems/classification-metrics |
| Cohen's Kappa | Calculate Cohen's kappa from two label sequences by comparing observed agreement with chance agreement. | https://www.tensortonic.com/problems/cohens-kappa |
| Color to Grayscale | Convert an RGB image to grayscale using weighted color channels while preserving its spatial dimensions. | https://www.tensortonic.com/problems/color-to-grayscale |
| Implement Contrastive Loss (Siamese) | Implement Siamese-network contrastive loss using pair labels, embedding distances, and a separation margin. | https://www.tensortonic.com/problems/contrastive-loss |
| Cosine Annealing LR Scheduler | Compute a cosine-annealed learning rate between configured maximum and minimum values across training steps. | https://www.tensortonic.com/problems/cosine-annealing-lr |
| Cosine Embedding Loss | Compute cosine embedding loss for similar and dissimilar vector pairs using labels and a configurable margin. | https://www.tensortonic.com/problems/cosine-embedding-loss |
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with dot products, Euclidean norms, and zero-vector handling. | https://www.tensortonic.com/problems/cosine-similarity |
| Compute Covariance Matrix | Compute a sample covariance matrix from centered observations, preserving feature-to-feature relationships. | https://www.tensortonic.com/problems/covariance-matrix |
| Implement Cross-Entropy Loss | Compute multiclass cross-entropy loss from class probabilities and integer labels with stable logarithms. | https://www.tensortonic.com/problems/cross-entropy-loss |
| Data Drift Detection | Detect feature drift by computing total variation distance between reference and production histograms. | https://www.tensortonic.com/problems/data-drift-detection |
| Implement Dice Loss | Compute Dice loss for segmentation predictions using overlap, total mass, and a numerical smoothing term. | https://www.tensortonic.com/problems/dice-loss |
| Implement Dot Product | Implement the dot product of equal-length numeric vectors by summing element-wise products without library shortcuts. | https://www.tensortonic.com/problems/dot-product |
| Edit Distance | Compute Levenshtein edit distance between two strings using dynamic programming over insertions, deletions, and substitutions. | https://www.tensortonic.com/problems/edit-distance |
| Calculate Eigenvalues of a Matrix | Calculate the eigenvalues of a square matrix and return them in the format required by the numerical contract. | https://www.tensortonic.com/problems/eigenvalues |
| ELU Activation | Apply the ELU activation element-wise, retaining positive inputs and exponentially transforming negative values. | https://www.tensortonic.com/problems/elu-activation |
| Compute Entropy for a Node | Compute decision-tree node entropy from class labels using empirical class probabilities and base-two logarithms. | https://www.tensortonic.com/problems/entropy-node |
| ETL Deduplication | Deduplicate ETL records by configured key fields while applying the required policy for repeated entries. | https://www.tensortonic.com/problems/etl-deduplication |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors as the square root of summed squared differences. | https://www.tensortonic.com/problems/euclidean-distance |
| Expected Calibration Error | Calculate expected calibration error by binning prediction confidence and weighting accuracy-confidence gaps. | https://www.tensortonic.com/problems/expected-calibration-error |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Feature Store Lookup | Combine stored offline and request-time features in input order, using defaults for unknown user IDs. | https://www.tensortonic.com/problems/feature-store-lookup |
| Implement Focal Loss | Compute mean binary focal loss from predicted probabilities using a configurable focusing parameter. | https://www.tensortonic.com/problems/focal-loss |
| Implement GELU Activation (Gaussian Error Linear Unit) | Implement the Gaussian Error Linear Unit activation element-wise using the required GELU approximation. | https://www.tensortonic.com/problems/gelu |
| Geometric Probability Mass Function & Mean | Compute the geometric distribution probability mass and mean from a valid success probability. | https://www.tensortonic.com/problems/geometric-pmf-mean |
| Implement Gradient Descent for a 1D Quadratic | Optimize a one-dimensional quadratic with iterative gradient descent and return the parameter trajectory. | https://www.tensortonic.com/problems/gradient-descent-quadratic |
| Build a Mini GRU Cell (Forward Pass) | Implement a GRU cell forward pass with reset, update, and candidate gates for one sequence timestep. | https://www.tensortonic.com/problems/gru-cell-forward |
| He Initialization | Scale raw weights into the He uniform range using a bound derived from the layer fan-in. | https://www.tensortonic.com/problems/he-initialization |
| Implement Hinge Loss (Binary SVM) | Compute binary SVM hinge loss from signed labels and prediction scores using the required margin. | https://www.tensortonic.com/problems/hinge-loss |
| Hit Rate at K | Calculate recommendation hit rate at K by checking whether each user's relevant items appear in top-ranked results. | https://www.tensortonic.com/problems/hit-rate-at-k |
| Implement Huber Loss | Compute Huber loss with quadratic errors near zero and linear penalties beyond a configurable threshold. | https://www.tensortonic.com/problems/huber-loss |
| Image Histogram | Count grayscale image pixels into intensity bins and return the histogram in ascending intensity order. | https://www.tensortonic.com/problems/image-histogram |
| Implement InfoNCE Loss | Compute InfoNCE contrastive loss from query and key embeddings using temperature-scaled similarities. | https://www.tensortonic.com/problems/info-nce-loss |
| Intersection over Union (IoU) | Compute intersection over union for two axis-aligned bounding boxes from overlap and combined area. | https://www.tensortonic.com/problems/iou-bounding-box |
| K-Means Assignment Step | Assign each sample to its nearest K-means centroid using Euclidean distance and deterministic tie handling. | https://www.tensortonic.com/problems/k-means-assignment |
| Implement KL Divergence | Compute Kullback-Leibler divergence between discrete probability distributions with safe zero-probability handling. | https://www.tensortonic.com/problems/kl-divergence |
| Label Smoothing Loss | Compute multiclass cross-entropy with label smoothing by distributing target mass across all classes. | https://www.tensortonic.com/problems/label-smoothing-loss |
| Implement Leaky ReLU (with α) | Apply Leaky ReLU element-wise with a configurable negative slope while retaining positive inputs. | https://www.tensortonic.com/problems/leaky-relu |
| Linear Layer Forward | Implement a dense linear layer forward pass by multiplying inputs by weights and adding a bias vector. | https://www.tensortonic.com/problems/linear-layer-forward |
| Linear Regression Closed Form | Fit linear regression with the closed-form normal equation and return coefficients for the supplied design matrix. | https://www.tensortonic.com/problems/linear-regression-closed-form |
| Log Loss (Per-Sample) | Compute binary log loss for each prediction with clipped probabilities to prevent undefined logarithms. | https://www.tensortonic.com/problems/log-loss-per-sample |
| Log Transform | Apply a numerically safe logarithmic transform to numeric features using the required offset or base. | https://www.tensortonic.com/problems/log-transform |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Implement Majority Class Classifier | Fit a majority-class baseline and predict the most frequent training label for every requested sample. | https://www.tensortonic.com/problems/majority-classifier |
| Make Diagonal Matrix | Construct a square diagonal matrix from a one-dimensional vector while setting every off-diagonal entry to zero. | https://www.tensortonic.com/problems/make-diagonal |
| Implement Manhattan Distance | Compute Manhattan distance between equal-length vectors by summing absolute coordinate differences. | https://www.tensortonic.com/problems/manhattan-distance |
| Matrix Inverse | Compute a square matrix inverse in NumPy while returning no result for invalid, non-square, or singular inputs. | https://www.tensortonic.com/problems/matrix-inverse |
| Implement Matrix Normalization | Normalize a NumPy matrix using the specified axis and norm while safely handling zero-magnitude slices. | https://www.tensortonic.com/problems/matrix-normalization |
| Matrix Trace | Compute the trace of a square matrix by summing its main diagonal entries without changing the input. | https://www.tensortonic.com/problems/matrix-trace |
| Matrix Transpose | Implement matrix transpose in NumPy without built-in transpose helpers, preserving rectangular shapes and the original input. | https://www.tensortonic.com/problems/matrix-transpose |
| Mean, Median, Mode | Calculate the mean, median, and deterministic mode of a numeric collection, including tied frequencies. | https://www.tensortonic.com/problems/mean-median-mode |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| Implement Micro-F1 | Compute multiclass micro-F1 by aggregating true positives, false positives, and false negatives across labels. | https://www.tensortonic.com/problems/metrics-f1-micro |
| Implement Min-Max Normalization | Normalize each NumPy feature to the zero-to-one range with explicit handling for constant columns. | https://www.tensortonic.com/problems/minmax-normalization |
| Model Versioning | Select a production model by highest accuracy, then lower latency, then the most recent timestamp. | https://www.tensortonic.com/problems/model-versioning-basics |
| NDCG (Normalized Discounted Cumulative Gain) | Calculate normalized discounted cumulative gain at K from ranked relevance scores and their ideal ordering. | https://www.tensortonic.com/problems/ndcg |
| Implement Nesterov Momentum (NAG) | Implement a Nesterov accelerated-gradient update using lookahead momentum and the current gradient. | https://www.tensortonic.com/problems/nesterov-momentum |
| Normalize 3D Vectors | Normalize a 3D vector to unit length in NumPy while returning the required result for a zero vector. | https://www.tensortonic.com/problems/normalize-3d |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| PCA Projection | Project centered observations onto supplied principal components to produce lower-dimensional features. | https://www.tensortonic.com/problems/pca-projection |
| Compute Pearson Correlation Matrix | Compute the Pearson correlation matrix between numeric features using centered covariance and standard deviations. | https://www.tensortonic.com/problems/pearson-correlation |
| Percentiles / Quantiles | Calculate requested percentiles from numeric data using the interpolation rule specified by the problem. | https://www.tensortonic.com/problems/percentiles |
| Perplexity Computation | Compute language-model perplexity from token probability distributions and the observed token indices. | https://www.tensortonic.com/problems/perplexity-computation |
| Implement Positional Encoding (sin/cos) | Generate sinusoidal Transformer positional encodings across sequence positions and embedding dimensions. | https://www.tensortonic.com/problems/positional-encoding |
| Precision and Recall at K | Compute recommendation precision and recall at K by comparing ranked predictions with relevant items. | https://www.tensortonic.com/problems/precision-recall-at-k |
| Tabular Q-Learning (Single Update) | Perform one tabular Q-learning update from reward, discount, learning rate, and the best next-state value. | https://www.tensortonic.com/problems/q-learning-update |
| Implement R² Score (Coefficient of Determination) | Compute the coefficient of determination from targets and predictions with explicit constant-target handling. | https://www.tensortonic.com/problems/r2-score |
| Implement ReLU Activation | Apply the ReLU activation element-wise by replacing negative values with zero and preserving nonnegative inputs. | https://www.tensortonic.com/problems/relu-activation |
| Remove Stopwords | Remove tokens found in a supplied stopword collection while preserving the order of remaining words. | https://www.tensortonic.com/problems/remove-stopwords |
| RMSProp Optimizer (Single Update Step) | Implement one RMSProp update in NumPy using an exponential squared-gradient average and adaptive scaling. | https://www.tensortonic.com/problems/rmsprop-optimizer |
| RNN Step Backward (Vanilla RNN) | Backpropagate through one vanilla RNN timestep to compute input, hidden-state, weight, and bias gradients. | https://www.tensortonic.com/problems/rnn-step-backward |
| RNN Step Forward (Tanh Cell) | Implement one vanilla RNN timestep with affine input and recurrent transforms followed by tanh activation. | https://www.tensortonic.com/problems/rnn-step-forward |
| Sample Variance & Standard Deviation | Compute sample variance and standard deviation with Bessel's correction from a numeric collection. | https://www.tensortonic.com/problems/sample-var-std |
| SELU Activation | Apply SELU activation element-wise with scaled positive values and exponential negative values. | https://www.tensortonic.com/problems/selu-activation |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| Compute Silhouette Score | Compute the mean silhouette score from intra-cluster and nearest-cluster distances for labeled samples. | https://www.tensortonic.com/problems/silhouette-score |
| Implement Softmax Function | Implement numerically stable softmax by shifting logits before exponentiation and normalizing probabilities. | https://www.tensortonic.com/problems/softmax-function |
| Implement Swish Activation | Apply the Swish activation element-wise by multiplying each input by its sigmoid value. | https://www.tensortonic.com/problems/swish-activation |
| Implement Tanh Activation | Implement the hyperbolic tangent activation element-wise with outputs bounded between minus one and one. | https://www.tensortonic.com/problems/tanh-activation |
| One-Step TD Value Update | Perform one temporal-difference value update from reward, discount, next-state value, and learning rate. | https://www.tensortonic.com/problems/td-value-update |
| Text Chunking | Split text into ordered chunks under the requested size and overlap rules without dropping content. | https://www.tensortonic.com/problems/text-chunking |
| Implement TF-IDF Vectorizer | Build TF-IDF document vectors from token counts and inverse document frequency across a text corpus. | https://www.tensortonic.com/problems/tfidf-vectorizer |
| Implement Triplet Loss | Compute triplet loss from anchor, positive, and negative embeddings using distances and a margin. | https://www.tensortonic.com/problems/triplet-loss |
| Compute 3D Vector Norm | Compute the Euclidean norm of a 3D vector from the square root of summed squared coordinates. | https://www.tensortonic.com/problems/vector-norm-3d |
| Warmup + Linear Decay LR Schedule | Compute a learning-rate schedule with linear warmup followed by linear decay across training steps. | https://www.tensortonic.com/problems/warmup-decay-lr |
| Implement Wasserstein Critic Loss | Compute Wasserstein critic loss as the difference between mean fake and real critic scores. | https://www.tensortonic.com/problems/wasserstein-critic-loss |
| Word Count Dictionary | Count token occurrences in text and return a dictionary mapping each distinct word to its frequency. | https://www.tensortonic.com/problems/word-count-dict |
| Xavier Initialization | Scale raw weights into the Xavier uniform range using a bound derived from fan-in and fan-out. | https://www.tensortonic.com/problems/xavier-initialization |
| Implement z-Score Standardization | Standardize NumPy features to zero mean and unit variance with explicit handling for constant columns. | https://www.tensortonic.com/problems/zscore-standardization |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/maxie)
<!-- tensortonic:end -->

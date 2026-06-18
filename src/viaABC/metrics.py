import numpy as np

def normalize(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

def l1_distance(x, y):
    """Compute the Manhattan Distance (L1) between two vectors."""
    return np.sum(np.abs(y - x), axis=-1).mean()

def l2_distance(x, y):
    """
    x: (B, T, D)
    y: (1, T, D)
    returns: scalar
    """
    # Broadcast: (B, T, D)
    diff = x - y

    # L2 distance per (batch, time): (B, T)
    dist_bt = np.linalg.norm(diff, axis=-1)

    # scalar
    return dist_bt.mean()

def cosine_similarity(x, y):
    """Compute the Cosine Similarity between two vectors."""
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)

    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)

    # Compute the cosine similarity matrix for each batch
    cosine_sim = np.dot(x_norm, y_norm.T)

    return cosine_sim.item()

def bert_score(x, y):
    """Compute BERTScore (Precision, Recall, F1) between two sets of embeddings."""
    # Normalize the embeddings across the feature dimension (columns)
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    # If y is 3D, normalize each batch individually
    if y.ndim == 3:
        y_norm = y / np.linalg.norm(y, axis=2, keepdims=True)
    else:
        y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)

    # Compute the cosine similarity matrix for each batch
    cosine_sim = np.matmul(x_norm, y_norm.transpose(0, 2, 1)) if y.ndim == 3 else np.dot(x_norm, y_norm.T)

    # Precision: average of the maximum cosine similarity for each token in x
    precision = np.mean(np.max(cosine_sim, axis=-1), axis=-1)  # Max along last axis for each x in each batch
    
    # Recall: average of the maximum cosine similarity for each token in y
    recall = np.mean(np.max(cosine_sim, axis=-2), axis=-1)  # Max along second-to-last axis for each y in each batch

    # F1 score: harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def bert_score_batch(x, y):
    scores = []
    for a, b in zip(x, y):
        precision, recall, f1 = bert_score(a, b)
        scores.append(f1)
    scores = np.array(scores)
    return scores.mean()

def pairwise_cosine(x, y, eps=1e-8):
    """
    x: (B, T, D)
    y: (1, T, D)  or broadcastable to x
    returns: scalar — mean cosine similarity over batch and time
    """
    assert x.ndim == 3 and y.ndim == 3, "Input arrays must be 3D"
    assert y.shape[0] == 1, "y must have batch size 1"
    assert x.shape[1:] == y.shape[1:], "x and y must match in (T, D)"

    # Normalize along feature dimension
    x_norm = x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    y_norm = y / (np.linalg.norm(y, axis=-1, keepdims=True) + eps)

    # Broadcast y over batch: (B, T, D)
    # cosine per timestep: (B, T)
    cos_per_t = np.sum(x_norm * y_norm, axis=-1)

    # mean over time → (B,)
    cos_per_batch = cos_per_t.mean(axis=-1)

    # mean over batch → scalar
    return cos_per_batch.mean()

def pairwise_wasserstein_diagonal(mu_x, sigma_x, mu_y, sigma_y, eps=1e-12):
    """
    Squared 2-Wasserstein distance for matched diagonal Gaussians.

    Inputs are expected to be shaped like latent tensors, usually (B, T, D)
    for patch tokens. `sigma_*` is the standard deviation, not the variance.
    Returns one distance per batch element.
    """
    mu_x = np.asarray(mu_x, dtype=np.float64)
    sigma_x = np.asarray(sigma_x, dtype=np.float64)
    mu_y = np.asarray(mu_y, dtype=np.float64)
    sigma_y = np.asarray(sigma_y, dtype=np.float64)

    if mu_x.shape != sigma_x.shape or mu_y.shape != sigma_y.shape:
        raise ValueError("Wasserstein mu and sigma tensors must have matching shapes.")
    if mu_x.shape[1:] != mu_y.shape[1:]:
        raise ValueError(
            "Wasserstein latent distributions must match after the batch dimension: "
            f"{mu_x.shape} vs {mu_y.shape}."
        )

    if mu_x.shape[0] == 1 and mu_y.shape[0] != 1:
        mu_x = np.broadcast_to(mu_x, mu_y.shape)
        sigma_x = np.broadcast_to(sigma_x, sigma_y.shape)
    elif mu_y.shape[0] == 1 and mu_x.shape[0] != 1:
        mu_y = np.broadcast_to(mu_y, mu_x.shape)
        sigma_y = np.broadcast_to(sigma_y, sigma_x.shape)

    if mu_x.shape != mu_y.shape:
        raise ValueError(f"Wasserstein batch shapes are not broadcastable: {mu_x.shape} vs {mu_y.shape}.")

    sigma_x = np.maximum(sigma_x, eps)
    sigma_y = np.maximum(sigma_y, eps)
    w2_squared = np.sum((mu_x - mu_y) ** 2 + (sigma_x - sigma_y) ** 2, axis=tuple(range(1, mu_x.ndim)))
    return np.sqrt(np.maximum(w2_squared, 0.0)).astype(np.float32)

def maxSim(x, y):
    """
    x: [num_query_tokens, dim] - query embeddings (numpy array)
    y: [num_doc_tokens, dim] - document embeddings (numpy array)
    
    Returns:
        scalar maxSim score (float)
    """
    # remove 1st dimension
    if x.ndim == 3:
        x = x.squeeze(0)  # [Q, D]
    if y.ndim == 3:
        y = y.squeeze(0)

    # L2 normalize embeddings
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)  # [Q, D]
    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)  # [D, D]

    # Compute cosine similarity matrix [Q, D]
    sim_matrix = np.dot(x_norm, y_norm.T)

    # Take max over document tokens for each query token, then sum
    max_sim = np.max(sim_matrix, axis=1).sum()
    
    return float(max_sim)

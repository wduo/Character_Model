import numpy as np

vocabulary_size = 27


def random_distribution():
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b / np.sum(b, 1)[:, None]


print(random_distribution())
print(random_distribution().shape)

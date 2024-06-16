import numpy as np

class Perceptron:
    """
    Perceptron classifier.

    Args:
        eta (float): Learning rate (between 0.0 and 1.0).
        niter (int): Passes over the training dataset.
        random_state (int): Random number generator seed for random weight initialization.
    """
    def __init__(self, eta=0.01, niter=50, random_state=42):
        self.eta = eta
        self.niter = niter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit training data.

        Args:
            X (array-like): Training vectors, shape = [n_samples, n_features].
            y (array-like): Target values, shape = [n_samples].

        Returns:
            self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # Initialize weights
        self.errors_ = []

        for _ in range(self.niter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)  # Count an error if update is not zero
            self.errors_.append(errors)
        
        return self

    def net_input(self, X):
        """
        Calculate net input.

        Args:
            X (array-like): Input vectors, shape = [n_samples, n_features].

        Returns:
            float: Net input.
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Return class label after unit step.

        Args:
            X (array-like): Input vectors, shape = [n_samples, n_features].

        Returns:
            array: Class labels.
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

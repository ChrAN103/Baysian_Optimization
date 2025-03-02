import os
import numpy as np


class DumbAgent:
    def __init__(self, state_size, model_path=None, weights=None):
        if model_path and os.path.isfile(model_path):
            try:
                self.weights = np.loadtxt(model_path)[:state_size]
                print("Using pre-trained model")
            except Exception as e:
                print(f"Error loading model: {e}")
        elif weights is not None:
            self.weights = weights
        else:
            self.weights = np.array(
                [0.210521, -0.601187, -0.204963, -0.964773, 0.659504]
            )[:state_size]

    def act(self, states):
        return max(states.items(), key=lambda x: (np.dot(x[1], self.weights)))[0]
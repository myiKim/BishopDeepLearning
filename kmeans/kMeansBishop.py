import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)

class KMeansBishop:
    def __init__(self, num_clusters, max_iter=3000):
        self.K = num_clusters  # Number of clusters
        self.max_iter = max_iter  # Maximum iterations
        self.mu = None  # Centroids
        self.assignment = None  # Assignments
        self.data = None  # Data points
        self.N = None

    def init_centroids_randomly(self):
        self.mu = self.data[np.random.choice(self.N, self.K, replace=False)]  # Randomly initialize centroids

    def fit(self, data):
        self.data = data  # Store data
        self.N = len(data)  # Number of data points
        self.init_centroids_randomly()  # Initialize centroids
        logging.info("Successfully initialized centroids.")
        
        # Initialize the assignment matrix (cluster indices)
        self.assignment = np.zeros(self.N, dtype=int)
        
        for iteration in range(self.max_iter):  # Maximum iteration limit
            new_assignment = np.zeros_like(self.assignment)  # Temporary assignment matrix
            
            # Assign each point to the nearest centroid
            for i in range(self.N):
                distances = np.linalg.norm(self.data[i] - self.mu, axis=1)  # Calculate distances to centroids
                closest_cluster = np.argmin(distances)  # Index of the nearest centroid
                new_assignment[i] = closest_cluster  # Assign to the closest centroid
            
            # Recalculate centroids based on current assignment
            for k in range(self.K):
                cluster_data = self.data[new_assignment == k]  # Data points in the current cluster
                if len(cluster_data) > 0:
                    self.mu[k] = np.mean(cluster_data, axis=0)  # Update the centroid
            
            # Check for convergence
            if np.all(self.assignment == new_assignment):
                break  # If assignments haven't changed, break
            
            self.assignment = new_assignment.copy()  # Update assignments
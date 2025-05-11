import numpy as np

def process(X, centroids, labels):
    for i in range(X.shape[0]):
        # calculate the distance from each point to each centroid
        distances = np.linalg.norm(X[i] - centroids, axis=1)
        # assign the point to the nearest centroid
        labels[i] = np.argmin(distances)
    return labels
def update_centroids(X, labels, k):
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        # calculate the mean of the points assigned to each centroid
        new_centroids[i] = X[labels == i].mean(axis=0)
    return new_centroids
def kmeans(X, k):
    labels = np.zeros(X.shape[0])
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    while True:
        # assign points to the nearest centroid
        labels = process(X, centroids, labels)
        # update the centroids
        new_centroids = update_centroids(X, labels, k)
        # check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

if __name__ == "__main__":
    print("This is a module for K-Means clustering.")

### Task 6: Image Classification Using K-Means Clustering

K-Means clustering is a popular algorithm used in unsupervised machine learning for partitioning data into `k` distinct clusters based on feature similarity. This task involves classifying images from the MNIST dataset of handwritten digits using K-Means clustering.

#### Key Concepts and Formulas

1. **K-Means Algorithm Overview**:
   - **Initialization**: Choose `k` initial centroids randomly from the dataset.
   - **Assignment Step**: Assign each data point to the nearest centroid based on the Euclidean distance.
   - **Update Step**: Calculate the new centroids as the mean of the data points assigned to each cluster.
   - **Convergence**: Repeat the assignment and update steps until the centroids no longer change significantly or a maximum number of iterations is reached.
2. **Distance Calculation**:
   The Euclidean distance d between two points \(p_i\) and \(c_j\) (where \(p_i\) is a data point and \(c_j\) is a centroid) is calculated as:

![aa](https://github.com/user-attachments/assets/81086c1a-56af-4dfd-b87d-41a1fe7966ac)

  where \(D\) is the number of dimensions (for MNIST, \(D = 784\) because each image is 28x28 pixels).

3. **Centroid Update Formula**:
   The new centroid \(c_j\) for cluster \(j\) is calculated as:

![aaa](https://github.com/user-attachments/assets/f24b22e1-ea0a-4b83-b0d0-7b6c818d4303)


#### Example

- Let's say you choose `k = 10` for the MNIST dataset, which contains images of the digits 0-9. You initialize the centroids, assign the images to clusters based on the nearest centroid, and then update the centroids until they stabilize.

#### Implementation

 example code using Python with `scikit-learn` and `matplotlib` to perform K-Means clustering on the MNIST dataset:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data / 255.0  # Normalize pixel values

# Apply K-Means clustering
k = 10  # Number of clusters (digits 0-9)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Get cluster centers and labels
cluster_centers = kmeans.cluster_centers_.reshape(k, 28, 28)

# Plotting cluster centers
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, center in zip(axes.ravel(), cluster_centers):
    ax.imshow(center, cmap='gray')
    ax.axis('off')
plt.show()

```
![image](https://github.com/user-attachments/assets/4a058186-f9ec-4728-b86d-7d64384e2827)




### Explanation of the Code

#### 1. Loading the Dataset
The MNIST dataset is fetched from OpenML using the `fetch_openml` function. The pixel values, which range from 0 to 255, are normalized to a range of 0 to 1 by dividing by 255.0. This normalization improves the performance of the K-Means algorithm.

#### 2. K-Means Clustering
The K-Means algorithm is initialized with `k=10`, representing the ten digits (0-9). The algorithm is then fitted to the normalized dataset using the `fit` method, which finds the optimal cluster centroids based on the input data.

#### 3. Cluster Centers
After fitting the model, the average images (cluster centers) for each digit are calculated. These cluster centers are reshaped back into a 28x28 format to represent the digit images visually.

#### 4. Visualization
The average images for each digit are displayed in a grid format. The `imshow` function is used to visualize the cluster centers as grayscale images, while the axis labels are hidden for a cleaner look. This visualization helps in understanding the characteristics of each digit as grouped by the clustering process.

### Conclusion

K-Means clustering is a straightforward yet effective method for image classification tasks, particularly in unsupervised learning scenarios. By applying it to the MNIST dataset, we can group similar handwritten digits together, providing insights into the data structure and helping with further tasks in machine learning and computer vision.

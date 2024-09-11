import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.decomposition import PCA

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from grid_search import ClusteringGridSearch


os.chdir('/content/drive/MyDrive/master/an1/sem1/PML/unsupervised')
os.listdir()

"""# 1. Load the data"""


# Load and preprocess images
def load_and_preprocess_images(folder_path):
    images = []
    for i in range(6401):
        if i != 115:
            image_path = os.path.join(folder_path, f'{i}.png')
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)

    return np.array(images)


images_data = load_and_preprocess_images('./data/images')

# Load labels
with open('./data/labels.txt', 'r') as f:
    y = np.array([int(label) for label in f.read().replace('\n', ' ').split()])


"""# 2. Preprocess

### Hog descriptors
"""


def get_hog_descriptors(image):
    return hog(
        image,
        pixels_per_cell=(32, 32),
        orientations=8,
        cells_per_block=(8, 8),
        block_norm='L2-Hys',
        feature_vector=True
    )


X = np.array([get_hog_descriptors(x) for x in images_data])


"""### ResNet50 feature extraction"""


def extract_features(images):
    images = np.array([np.repeat(cv2.resize(img, (224, 224))[..., np.newaxis], 3, -1) for img in images])
    images = preprocess_input(images)
    features = resnet_model.predict(images)

    return features.reshape((6400, -1))


resnet_model = ResNet50(weights='imagenet', include_top=False)
X_resnet = extract_features(images_data)


"""# 3. Train and score supervised model"""

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resnet, y, test_size=0.2, random_state=42)

"""### SVC"""

param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
}

svm_model = SVC()
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Get the best SVM model
best_svm_model = grid_search.best_estimator_
y_pred = best_svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100}%')

confusion_matrix_svc = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:')
print(confusion_matrix_svc)

classification_report_svc = classification_report(y_test, y_pred)
print('\nClassification Report:\n', classification_report_svc)

"""### DummyClassifier"""

random_model = DummyClassifier(strategy='uniform')
random_model.fit(X_train, y_train)
y_pred_random = random_model.predict(X_test)

# Evaluate the model
accuracy_random = accuracy_score(y_test, y_pred_random)
print(f'Random Classifier Accuracy: {accuracy_random * 100}%')

confusion_matrix_random = confusion_matrix(y_test, y_pred_random)
print('\nConfusion Matrix:')
print(confusion_matrix_random)

classification_report_random = classification_report(y_test, y_pred_random)
print('\nClassification Report:\n', classification_report_random)


"""# 4. Train unsupervised model

### Standardize and reduce dimensionality
"""

X = X_resnet

pca = PCA()
pca.fit(X)
explained_variance_ratio = pca.explained_variance_ratio_

# Choose the number of dimensions that captures
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
desired_dimensions = np.argmax(cumulative_variance_ratio >= 0.98) + 1

pca = PCA(n_components=desired_dimensions)
X_reduced = pca.fit_transform(X)

"""### K-means
"""

param_grid = {
    'n_clusters': [4, 5, 6, 7, 8],
    'init': ['k-means++', 'random'],
    'n_init': [10, 20, 30],
}

grid_search = ClusteringGridSearch(estimator=KMeans(), param_grid=param_grid)
best_kmeans_model = grid_search.fit(X_reduced, y)

print(f'Best K-means hyperparameters: {best_kmeans_model.best_params_}')

"""### Agglomerative clustering"""

agl_param_grid = {
    'n_clusters': [4, 5, 6, 7, 8],
    'linkage': ['complete', 'average'],
    'metric': ['euclidean', 'l1', 'l2']
}

grid_search = ClusteringGridSearch(estimator=AgglomerativeClustering(), param_grid=agl_param_grid)
best_agl_model = grid_search.fit(X_reduced)

print(f'Best K-means hyperparameters: {best_agl_model.best_params_}')

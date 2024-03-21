"""Module providing functions clustering image features"""

# pylint: disable=line-too-long, multiple-statements, c-extension-no-member, relative-beyond-top-level, no-member, too-many-function-args, wrong-import-position
import os  # interact with the operating system
import glob  # Unix style pathname pattern expansion
import random  # random number generator
from typing import Literal  # to support type hints
import numpy as np  # NumPy
from tqdm import tqdm  # progress bar
import hdbscan  # Hierarchical Density-Based Spatial Clustering of Applications with Noise
from sklearn.cluster import KMeans  # KMeans clustering
import torch  # PyTorch
from PIL import Image  # Pillow image processing
from matplotlib import pyplot as plt  # for image visualization
from transformers import AutoProcessor, AutoImageProcessor, CLIPModel, AutoModel  # import CLIP and DINOv2
from .core import device, imread_rgb  # import core elements


class FeatureClustering:
    """Group image based on the similarity of their features"""
    def __init__(self, n_kmeans: int = 3):
        self.n_kmeans = n_kmeans  # number of clusters of Kmeans

    @staticmethod
    def extract_features(image: np.ndarray, model_type: Literal['CLIP', 'DINOv2'] = 'DINOv2') -> float:
        """Extract image features with CLIP (https://github.com/openai/CLIP) or DINOv2 (https://github.com/facebookresearch/dinov2)"""
        if model_type == 'CLIP':
            processor = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')  # get the CLIP model processor
            model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)  # get the CLIP model
            model.eval()  # set the model to evaluation mode
            with torch.no_grad():
                image_inputs = processor(images=image, return_tensors='pt').to(device)  # image inputs
                image_features = model.get_image_features(**image_inputs)  # extract features from the image
        elif model_type == 'DINOv2':
            processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')  # get the DINOv2 model processor
            model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)  # get the DINOv2 model
            model.eval()  # set the model to evaluation mode
            with torch.no_grad():
                image_inputs = processor(images=image, return_tensors='pt').to(device)  # image inputs
                image_features = model(**image_inputs).last_hidden_state.mean(dim=1)  # extract features from the image
        return image_features.squeeze().cpu().numpy()  # torch tensor pt (features) ~ csv (image_names) HDBSCRAN

    @staticmethod
    def batch_feature_extraction(input_dir: str) -> tuple:
        """Extract the features all images under a given directory, and save the image paths and features """
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp', '*.gif', '*.ico', '*.jfif', '*.webp']  # list most image extensions
        image_paths = [path for extension in extensions for path in glob.glob(os.path.join(input_dir, extension))]  # get all images paths
        images = [imread_rgb(image_path) for image_path in image_paths]  # load all images
        features = np.array([FeatureClustering.extract_features(image) for image in tqdm(images, total=len(images))])  # ensemle all image features
        features_tensor = torch.tensor(np.array(features), dtype=torch.float)  # convert features to a PyTorch tensor
        torch.save({'paths': image_paths, 'features': features_tensor}, 'Data//extracted_features.pth')  # save image paths and features
        return image_paths, features

    @staticmethod
    def load_torch_data(features_path: str) -> tuple:
        """Load the saved features and image paths from a PyTorch file"""
        loaded_data = torch.load(features_path)  # load the data
        return loaded_data['paths'], loaded_data['features'].numpy()

    @staticmethod
    def cluster_images_hdbscan(image_paths: list, features: np.ndarray, min_cluster_size: int = 30, min_samples: int = 15) -> float:
        """Cluster images features based on HDBSCAN"""

        def visualize_clusters(image_paths: list, cluster_labels: np.ndarray, n: int = 10) -> None:
            """Visualize n random images from each cluster"""
            unique_labels = np.unique(cluster_labels)  # get unique labels
            unique_labels_sorted = np.sort(unique_labels)  # Sort the labels to start from -1 (noise)
            for label in unique_labels_sorted:
                cluster_image_paths = [path for path, cluster in zip(image_paths, cluster_labels) if cluster == label]  # filter image paths for the current cluster
                print(f"Cluster {label} ({len(cluster_image_paths)} images):")  # display cluster info
                if len(cluster_image_paths) > n:
                    sample_image_paths = random.sample(cluster_image_paths, n)  # radomly select a subset of images for visualization
                else:
                    sample_image_paths = cluster_image_paths
                _, axs = plt.subplots(1, len(sample_image_paths), figsize=(len(sample_image_paths) * 4, 4))  # create a figure for the current cluster
                if len(sample_image_paths) == 1:
                    axs = [axs]  # if there's only one image, axs might not be an array, so we put it in a list for consistent handling
                for ax, image_path in zip(axs, sample_image_paths):
                    image = Image.open(image_path)
                    ax.imshow(image); ax.axis('off')  # noqa: hide axes ticks
                plt.show()
            return None

        def sample_images(image_paths: list, cluster_labels: np.ndarray) -> list:
            """Sample images from each cluster based on the median cluster size"""
            unique_labels = np.unique(cluster_labels)  # get unique labels
            clusters_image_paths = {label: [] for label in unique_labels}  # dictionary to hold image paths for each cluster
            for path, label in zip(image_paths, cluster_labels):
                clusters_image_paths[label].append(path)  # populate the dictionary with image paths for each cluster
            median_size = np.median([len(paths) for label, paths in clusters_image_paths.items() if label != -1])  # calculate the median cluster size, excluding the noise cluster (-1)
            print(f'median size size = {median_size}')
            sampled_image_paths = []  # to store selected samples
            for label, paths in clusters_image_paths.items():
                if len(paths) < median_size or label == -1:
                    sampled_image_paths.extend(paths)  # include all images if the cluster is smaller than the median or from noise cluster
                else:
                    sampled_image_paths.extend(random.sample(paths, int(median_size)))  # sample to the median size if the cluster is larger than the median
            return sampled_image_paths

        clusterer = hdbscan.HDBSCAN(metric='manhattan', min_cluster_size=min_cluster_size, min_samples=min_samples)  # define the HDBSCAN parameters
        cluster_labels = clusterer.fit_predict(features)  # get the cluster labels
        sampled_image_paths = sample_images(image_paths, cluster_labels)  # the selected image paths
        visualize_clusters(image_paths, cluster_labels)  # visualize images and number of each cluster
        return sampled_image_paths

    def compute_similarity_matrix(self, features: np.ndarray) -> dict:
        """Compute the similarity matrix from the extracted features """
        features_norm = np.stack(features) / np.linalg.norm(features, axis=1, keepdims=True)  # normalize the feature vectors to unit length
        similarity_matrix = np.dot(features_norm, features_norm.T)  # compute pairwise cosine similarity
        return similarity_matrix

    def cluster_images_kmeans(self, images: list) -> float:
        """Cluster images based on features similarity"""
        features = [self.extract_features(image) for image in images]  # etract features from all images
        similarity_matrix = self.compute_similarity_matrix(features)
        kmeans = KMeans(n_clusters=self.n_kmeans, random_state=0).fit(similarity_matrix)  # apply K-Means clustering on the similarity matrix
        image_clusters = {}  # to store final clustering result
        for idx, label in enumerate(kmeans.labels_):
            if label not in image_clusters:
                image_clusters[label] = []  # populate cluster labels
            image_clusters[label].append(idx)  # group image indices by labels
        return image_clusters

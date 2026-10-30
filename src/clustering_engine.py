import pandas as pd  
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from scipy.cluster.hierarchy import linkage, fcluster

class SimpleKMedoids:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit_predict(self, X):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        kmeans.fit(X)
        self.labels_ = kmeans.labels_
        self.inertia_ = kmeans.inertia_
        
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        if hasattr(X, 'iloc'):
            self.cluster_centers_ = X.iloc[closest_indices].values
        else:
            self.cluster_centers_ = X[closest_indices]
            
        return self.labels_

class ClusteringEngine:
    def __init__(self):
        self.current_model = None
        self.labels = None
        self.centroids = None

    def _calculate_inertia(self, data, labels, centroids=None):
        if isinstance(data, pd.DataFrame):
            data_np = data.values
        else:
            data_np = data
            
        inertia = 0
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1: continue # Ignore noise
            
            cluster_points = data_np[labels == label]
            if len(cluster_points) == 0: continue # Safe check

            if centroids is None:
                center = cluster_points.mean(axis=0)
            else:
                center = cluster_points.mean(axis=0)
            
            sq_dists = np.sum((cluster_points - center) ** 2)
            inertia += sq_dists
            
        return inertia

    def run_algorithm(self, data, algo_name, params):
        # Basic emptiness check
        if data is None: return {"error": "No data."}
        if hasattr(data, 'empty') and data.empty: return {"error": "Dataframe is empty."}
        if hasattr(data, 'size') and data.size == 0: return {"error": "Array is empty."}

        data_for_inertia = data.values if isinstance(data, pd.DataFrame) else data

        try:
            inertia_val = 0
            
            if algo_name == "KMeans":
                k = int(params.get('k', 3))
                self.current_model = KMeans(n_clusters=k, random_state=42, n_init='auto')
                self.labels = self.current_model.fit_predict(data)
                self.centroids = self.current_model.cluster_centers_
                inertia_val = self.current_model.inertia_ 

            elif algo_name == "KMedoids":
                k = int(params.get('k', 3))
                self.current_model = SimpleKMedoids(n_clusters=k, random_state=42)
                self.labels = self.current_model.fit_predict(data)
                self.centroids = self.current_model.cluster_centers_
                inertia_val = self.current_model.inertia_

            elif algo_name == "DBSCAN":
                eps = float(params.get('eps', 1.5))
                min_pts = int(params.get('min_pts', 6))
                self.current_model = DBSCAN(eps=eps, min_samples=min_pts)
                self.labels = self.current_model.fit_predict(data)
                self.centroids = None
                inertia_val = self._calculate_inertia(data_for_inertia, self.labels)

            elif algo_name == "AGNES":
                k = int(params.get('k', 3))
                self.current_model = AgglomerativeClustering(n_clusters=k, linkage='ward')
                self.labels = self.current_model.fit_predict(data)
                self.centroids = None
                inertia_val = self._calculate_inertia(data_for_inertia, self.labels)

            elif algo_name == "DIANA":
                k = int(params.get('k', 3))
                Z_diana = linkage(data, method='average', metric='euclidean')
                self.labels = fcluster(Z_diana, k, criterion='maxclust') - 1
                self.centroids = None
                inertia_val = self._calculate_inertia(data_for_inertia, self.labels)

            else:
                return {"error": f"Algorithm '{algo_name}' unknown."}

            # Metrics
            unique_labels = set(self.labels)
            # Filter out noise (-1) for cluster counting
            n_clusters_real = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            sil_score = -1
            # Silhouette requires at least 2 labels and more data than labels
            if n_clusters_real > 1 and len(data) > n_clusters_real:
                try:
                    sil_score = silhouette_score(data, self.labels)
                except:
                    sil_score = -1 # Fallback
            elif n_clusters_real <= 1:
                sil_score = 0 # Not applicable

            return {
                "success": True,
                "labels": self.labels,
                "centroids": self.centroids,
                "metrics": {
                    "silhouette": round(sil_score, 3),
                    "inertia": round(inertia_val, 2) 
                }
            }

        except Exception as e:
            return {"error": str(e)}
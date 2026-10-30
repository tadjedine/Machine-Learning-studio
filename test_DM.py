# test_clustering_engine.py

# Import des classes depuis les fichiers sources
from src.data_manager import DataManager 
from src.clustering_engine import ClusteringEngine
import pandas as pd # Pour un affichage plus clair

# Configuration de base pour le test
FILE_PATH = "data/horse-colic.csv"  # Changer si votre fichier est différent
K_CLUSTERS = 3  # Nombre de clusters que nous allons tester
# Pour DBSCAN, ces valeurs sont souvent spécifiques au dataset
EPSILON_DBSCAN = 0.45 
MIN_PTS_DBSCAN = 5

# --- 1. PRÉPARATION DES DONNÉES ---
print("--- 1. PRÉPARATION DES DONNÉES ---")
dm = DataManager()
print(dm.load_data(FILE_PATH))
print(dm.handle_missing_values(method='mean'))
print(dm.normalize_data(method='zscore'))

# Étape essentielle: Réduction de dimension (pour simuler la visualisation)
# Si vous avez plus de 4 colonnes, la PCA est vitale.
if dm.get_data().shape[1] > 2:
    print(dm.apply_pca(n_components=2))

data_ready = dm.get_data()
if data_ready is None:
    print("Erreur: Données non prêtes. Fin du test.")
else:
    print(f"Données prêtes pour clustering (Shape: {data_ready.shape})\n")
    
    # --- 2. EXÉCUTION DES ALGORITHMES ---
    engine = ClusteringEngine()
    
    algorithms_to_test = {
        "KMeans": {'k': K_CLUSTERS},
        "KMedoids": {'k': K_CLUSTERS},
        "AGNES": {'k': K_CLUSTERS},
        "DIANA": {'k': K_CLUSTERS},
        "DBSCAN": {'eps': EPSILON_DBSCAN, 'min_pts': MIN_PTS_DBSCAN}
    }
    
    results_comparison = {}
    
    print("--- 2. EXÉCUTION & ÉVALUATION DES 5 MÉTHODES ---")
    
    for algo_name, params in algorithms_to_test.items():
        print(f"\n-> Exécution de {algo_name} avec paramètres: {params}")
        
        # L'objet de test data_ready doit être un DataFrame ou un numpy array
        # On s'assure que c'est un numpy array pour éviter des conflits d'index
        data_np = data_ready.values if isinstance(data_ready, pd.DataFrame) else data_ready
        
        result = engine.run_algorithm(data_np, algo_name, params)
        
        if "error" in result:
            print(f"Échec de {algo_name}: {result['error']}")
            results_comparison[algo_name] = {"Silhouette": "N/A", "Inertia": "N/A", "Labels_Count": "N/A"}
        else:
            labels = result["labels"]
            
            # Calculer le nombre de clusters (en ignorant le bruit -1)
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            results_comparison[algo_name] = {
                "Silhouette": result["metrics"]["silhouette"],
                "Inertia": result["metrics"]["inertia"],
                "Labels_Count": n_clusters
            }
            print(f"  Succès: {n_clusters} clusters trouvés.")
            print(f"  - Silhouette: {result['metrics']['silhouette']}")
            print(f"  - Inertie: {result['metrics']['inertia']}")


    # --- 3. AFFICHAGE DES RÉSULTATS COMPARATIFS (TP4) ---
    print("\n--- 3. TABLEAU DE COMPARAISON FINAL ---")
    df_results = pd.DataFrame(results_comparison).T
    print(df_results)
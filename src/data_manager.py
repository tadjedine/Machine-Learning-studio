# src/data_manager.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class DataManager:
    def __init__(self):
        self.raw_data = None
        self.data = None

    def load_data(self, file_path):
        try:
            if file_path.endswith('.csv'):
                self.raw_data = pd.read_csv(file_path, na_values=['?'])
            elif file_path.endswith('.xlsx'):
                self.raw_data = pd.read_excel(file_path, na_values=['?'])
            else: 
                return "Format non supporté"

            id_cols = [c for c in self.raw_data.columns if c.lower() == 'id']
            if id_cols:
                self.raw_data = self.raw_data.drop(columns=id_cols)
                print(f"Column {id_cols} dropped automatically.")
            # ------------------------------

            self.reset_data()
            return f"Succès : {self.data.shape[0]} lignes chargées"
        except Exception as e: 
            return f"Erreur : {e}"

    def reset_data(self):
        if self.raw_data is not None:
            self.data = self.raw_data.copy()

    def get_data(self): return self.data

    def get_statistics(self):
        """Retourne les stats descriptives (min, max, mean...)."""
        if self.data is not None:
            return self.data.describe(include='all').transpose().round(3)
        return None

    def clean_aberrant_values(self, target_column=None):
        """Transforme les erreurs (ex: 'abc' dans 'Age') en NaN pour les colonnes numériques."""
        if self.data is None: return
        for col in self.data.columns:
            if col == target_column: continue
            # Si la colonne est de type objet (texte)
            if self.data[col].dtype == 'object':
                # On tente de convertir. Si > 50% sont des nombres, on force le nettoyage
                converted = pd.to_numeric(self.data[col], errors='coerce')
                if converted.notnull().sum() > (len(self.data) / 2):
                    self.data[col] = converted

    def apply_label_encoding(self, target_column=None):
        """Transforms categorical text into numbers, but can skip the target."""
        if self.data is None: return
        le = LabelEncoder()
        for col in self.data.columns:
            # Skip the target column so it stays as text for now
            if col == target_column: 
                continue
                
            if self.data[col].dtype == 'object':
                # Convert to string first to handle mixed types/NaNs
                self.data[col] = le.fit_transform(self.data[col].astype(str))

    def handle_missing_values(self, method='mean'):
        if self.data is None or method == 'none': return
        if method == 'drop': 
            self.data.dropna(inplace=True)
        else:
            for col in self.data.columns:
                if self.data[col].dtype in [np.float64, np.int64]:
                    val = self.data[col].mean() if method == 'mean' else self.data[col].median()
                    self.data[col].fillna(val, inplace=True)
                else:
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)

    def normalize_data(self, method='zscore', target_to_ignore=None):
        if self.data is None or method == 'none': return
        num_cols = list(self.data.select_dtypes(include=[np.number]).columns)
        if target_to_ignore in num_cols: num_cols.remove(target_to_ignore)
        if not num_cols: return
        
        scaler = StandardScaler() if method == 'zscore' else MinMaxScaler()
        self.data[num_cols] = scaler.fit_transform(self.data[num_cols])

    def apply_pca(self, n_components=2, target_to_keep=None):
        if self.data is None: return
        target_series = None
        if target_to_keep and target_to_keep in self.data.columns:
            target_series = self.data[target_to_keep].copy()
            feat = self.data.drop(columns=[target_to_keep])
        else: feat = self.data
        
        num_feat = feat.select_dtypes(include=[np.number]).columns
        if len(num_feat) < n_components: return
        
        pca = PCA(n_components=n_components)
        pca_res = pca.fit_transform(feat[num_feat])
        self.data = pd.DataFrame(pca_res, columns=[f'PC{i+1}' for i in range(n_components)])
        if target_series is not None: self.data[target_to_keep] = target_series.values

    def split_data(self, target_column):
        if self.data is None or target_column not in self.data.columns: return None
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return train_test_split(X, y, test_size=0.2, random_state=42)
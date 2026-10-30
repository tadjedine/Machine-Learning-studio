from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

class ClassificationEngine:
    def run_classification(self, X_train, X_test, y_train, y_test, algo_name, k_val=3):
        """
        Exécute l'algorithme de classification et retourne les métriques demandées.
        """
        if algo_name == "k-NN":
            model = KNeighborsClassifier(n_neighbors=k_val)
        elif algo_name == "Naive Bayes":
            model = GaussianNB()
        elif algo_name == "C4.5":
            # Utilisation de criterion='entropy' pour simuler C4.5
            model = DecisionTreeClassifier(criterion='entropy', random_state=42)
        elif algo_name == "SVM":
            model = SVC(kernel='linear', random_state=42)
        else:
            raise ValueError(f"Algorithme {algo_name} inconnu.")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='macro', zero_division=0),
            "cm": confusion_matrix(y_test, y_pred)
        }

    def get_knn_curve_data(self, X_train, X_test, y_train, y_test):
        """
        Génère les données pour la courbe de précision (k de 1 à 10).
        """
        ks = list(range(1, 11))
        precisions = []
        for k in ks:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            precisions.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        return ks, precisions
from scipy.io import loadmat

def main():
    # Charger le fichier MAT
    data = loadmat('Franka_TrainingData.mat')
    
    # Extraire les variables
    U = data.get('U')
    X = data.get('X')
    Y = data.get('Y')
    
    # Afficher les types de données
    print("U:", type(U), "shape:", U.shape if U is not None else None)
    print("X:", type(X), "shape:", X.shape if X is not None else None)
    print("Y:", type(Y), "shape:", Y.shape if Y is not None else None)

if __name__ == "__main__":
    main()

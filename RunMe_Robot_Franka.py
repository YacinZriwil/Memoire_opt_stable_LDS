import numpy as np
from   frgm_restart import optimize
np.random.seed(2024)

def extract_data_by_p(U, X, Y, p_values):
    # Dictionnaires pour stocker les données par valeur de p
    data_by_p = {p: {'U': None, 'X': None, 'Y': None} for p in p_values}
    
    # Itérer sur les valeurs de p pour extraire les sous-ensembles de données
    for i, p in enumerate(p_values):
        if U is not None:
            data_by_p[p]['U'] = U[:, i] if U.shape[1] > i else None
        if X is not None:
            data_by_p[p]['X'] = X[:, i] if X.shape[1] > i else None
        if Y is not None:
            data_by_p[p]['Y'] = Y[:, i] if Y.shape[1] > i else None
    
    return data_by_p

def main():
    path_U = "U.npy"
    path_X = "X.npy"
    path_Y = "Y.npy"

    U, X, Y = np.load(path_U),np.load(path_X),np.load(path_Y)
    
    p_values = [50, 75, 100, 150, 200, 300, 500]

    data = extract_data_by_p(U, X, Y, p_values)
    
    
    for p in p_values:
        print(f"Pour p = {p}")
        U_p = data[p]['U']
        X_p = data[p]['X']
        Y_p = data[p]['Y']
        print(f"U.shape = {U_p.shape}")
        print(f"X.shape = {X_p.shape}")
        print(f"Y.shape = {Y_p.shape}")
        
        print("\n\n\t\t\t==========================")
        A, B = optimize(Y_p, X_p, U_p, max_time= 20, max_iterations = 20)

if __name__ == '__main__':
    main()
# =============================================================================
# Erreur de NeurIPS l'article : 2.1609139459682556
# =============================================================================
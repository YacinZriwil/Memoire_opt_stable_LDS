import numpy as np
import time
import matplotlib.pyplot as plt
from frgm_restart import optimize  # Assurez-vous que cette fonction est correctement implémentée
import engine
np.random.seed(2024)

# Configuration
n_states = 20
n_control = 7
all_samples = [50, 75, 100, 150, 200, 300, 400, 500, 1000, 1500, 2000]

# Placeholders pour les résultats SUB
SCHUR = {
    'error': np.full((len(all_samples), 2), np.nan),
    'time': np.full((len(all_samples), 2), np.nan),
    'maxeval': np.full((len(all_samples), 2), np.nan),
    'A': np.full((len(all_samples), n_states, n_states), np.nan),
    'B': np.full((len(all_samples), n_states, n_control), np.nan)
}

def extract_data_by_p(U, X, Y, p_values):
    data_by_p = {p: {'U': None, 'X': None, 'Y': None} for p in p_values}
    for p in p_values:
        data_by_p[p]['U'] = U[:, :p] if U.shape[1] >= p else None
        data_by_p[p]['X'] = X[:, :p] if X.shape[1] >= p else None
        data_by_p[p]['Y'] = Y[:, :p] if Y.shape[1] >= p else None
    return data_by_p

def adjusted_frobenius_norm(X):
    return np.linalg.norm(X, 'fro')**2 / 2

def plot_results(p_values, errors, method_name):
    plt.figure()
    plt.loglog(p_values, errors, 'o-', label=method_name)
    plt.xlabel('number of measurements')
    plt.ylabel('error percentage')
    plt.title('Error vs Number of Measurements')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig(f'{method_name}_error_plot.png')
    plt.show()

def main():
    path_U = "U.npy"
    path_X = "X.npy"
    path_Y = "Y.npy"

    U, X, Y = np.load(path_U), np.load(path_X), np.load(path_Y)
    print(f"U.shape = {U.shape}")
    print(f"X.shape = {X.shape}")
    print(f"Y.shape = {Y.shape}")
    print("\n=============")
    p_values = all_samples

    data = extract_data_by_p(U, X, Y, p_values)

    errors = []

    for n_training, n_training_samples in enumerate(p_values):
        print(f'\n\nntraining_samples = {n_training_samples} et n_training = {n_training}\n \n')

        U_p = data[n_training_samples]['U']
        X_p = data[n_training_samples]['X']
        Y_p = data[n_training_samples]['Y']

        print(f"U.shape = {U_p.shape}")
        print(f"X.shape = {X_p.shape}")
        print(f"Y.shape = {Y_p.shape}")

        # Calcul AB_ls et prédiction
        XU = np.concatenate((X_p, U_p))
        AB_ls = Y_p @ np.linalg.pinv(XU)
        pred = AB_ls @ XU
        
        ls_error = adjusted_frobenius_norm(Y_p - pred)
        
        start_time = time.time()

        #A_SCHUR, B_SCHUR = optimize(Y_p, X_p, U_p, max_time=20, max_iterations=100)
        A_SCHUR, B_SCHUR, _ = engine.learn_stable_soc_with_inputs(X_p,Y_p, U_p)
        SCHUR['time'][n_training] = [n_training_samples, time.time() - start_time]
        SCHUR['A'][n_training] = A_SCHUR
        SCHUR['B'][n_training] = B_SCHUR

        schur_err = adjusted_frobenius_norm(Y_p - A_SCHUR @ X_p - B_SCHUR @ U_p)
        perc_error = 100 * (schur_err - ls_error) / ls_error
        SCHUR['error'][n_training] = [n_training_samples, perc_error]
        
        maxeval_SCHUR = max(abs(np.linalg.eigvals(A_SCHUR)))
        SCHUR['maxeval'][n_training] = [n_training_samples, maxeval_SCHUR]

        errors.append(perc_error)

        A_ls = AB_ls[:X.shape[0], :X.shape[0]]

        maxeval_LS = max(abs(np.linalg.eigvals(A_ls)))
        
        print(f'\n->    Max eigenvalue is : {maxeval_SCHUR:.4f}')
        print(f'->    Max eigenvalue LS is : {maxeval_LS:.4f}')
        print(f'->    ls_error : {ls_error:.5f}')
        print(f'->    schur_err : {schur_err:.5f}')
        print(f'->    perc_error : {perc_error:.5f}%')
        print("===================================================================================")
    plot_results(p_values, errors, 'SCHUR')

if __name__ == '__main__':
    main()

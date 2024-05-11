import numpy as np
import scipy.io
from numpy.linalg import pinv, eig, norm
import time
import matplotlib.pyplot as plt
import engine  # Supposons que engine est le module avec vos fonctions d'apprentissage
from frgm_restart import optimize  # Assurez-vous que cette fonction est correctement implémentée
np.random.seed(2024)

# Settings
nStates = 17
nControl = 7
AllSamples = [50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3192]

# Initialisation des résultats
def init_results(number_samples):
    return {
        'error': np.full((number_samples, 2), np.nan),
        'time': np.full((number_samples, 2), np.nan),
        'maxeval': np.full((number_samples, 2), np.nan),
        'A': np.full((number_samples, nStates, nStates), np.nan),
        'B': np.full((number_samples, nStates, nControl), np.nan)
    }

# Fonction pour calculer Psi_x
def psi_x(s, u):
    s = np.asarray(s, dtype=np.float64).flatten()
    u = np.asarray(u, dtype=np.float64).flatten()
    return np.concatenate((s[:3], s[6:], u), axis=0)


def main():
    # Boucle principale pour chaque randomShuffling
    for randomShuffling in range(2):
        numberSamples = len(AllSamples)
    
        LS = init_results(numberSamples)
        SUB = init_results(numberSamples)
        SCHUR = init_results(numberSamples)
        
        for nTraining in range(numberSamples):
            nTrainingSamples = AllSamples[nTraining]
            print(f"\n\n {nTrainingSamples} \n\n")
    
            # Charger les données de formation Franka
            data = scipy.io.loadmat("Franka_TrainingData.mat")
            X = data['X']
            U = data['U']
            Y = data['Y']
    
            Ps0_list = np.zeros((nTrainingSamples, nStates + nControl))
            Psi_list = np.zeros((nTrainingSamples, nStates + nControl))
    
            for k in range(nTrainingSamples):
                i = np.random.randint(len(X)) if randomShuffling else k
                Ps0_list[k, :] = psi_x(X[i, :], U[i, :])
                Psi_list[k, :] = psi_x(Y[i, :], U[i, :])
    
            Y_train = Psi_list[:, :nStates].T
            X_train = Ps0_list[:, :nStates].T
            U_train = Ps0_list[:, nStates:].T
    
            # Afficher les types de données
            #print("U_train:", type(U_train), "shape:", U_train.shape if U_train is not None else None)
            #print("X_train:", type(X_train), "shape:", X_train.shape if X_train is not None else None)
            #print("Y_train:", type(Y_train), "shape:", Y_train.shape if Y_train is not None else None)
        
            # Solution LS (non contrainte)
            AB_ls = Y_train @ pinv(np.vstack((X_train, U_train)))
            A_ls = AB_ls[:nStates, :nStates]
            B_ls = AB_ls[:nStates, nStates:]
    
            e_LS = norm(Y_train - A_ls @ X_train - B_ls @ U_train, 'fro')**2 / 2
            maxeval_LS = np.max(np.abs(eig(A_ls)[0]))
            LS['error'][nTraining, :] = [nTrainingSamples, e_LS]
            LS['maxeval'][nTraining, :] = [nTrainingSamples, maxeval_LS]
            LS['A'][nTraining, :, :] = A_ls
            LS['B'][nTraining, :, :] = B_ls
    
            print(f"   LS Max eigenvalue is : {maxeval_LS:.4f}")
            print(f"   LS Reconstruction error : {e_LS:.5f}")
            
            print("\t\t\t-----")
            
            
            # Solution SUB
            time_sub_start = time.time()
            A_SUB, B_SUB, _ = engine.learn_stable_soc_with_inputs(X_train, Y_train, U_train)
            sub_time_elapsed = time.time() - time_sub_start
            
            SUB['time'][nTraining, :] = [nTrainingSamples, sub_time_elapsed]
            SUB['A'][nTraining, :, :] = A_SUB
            SUB['B'][nTraining, :, :] = B_SUB
            
            # Erreur de reconstruction
            e_SUB = norm(Y_train - A_SUB @ X_train - B_SUB @ U_train, 'fro') ** 2 / 2
            maxeval_SUB = np.max(np.abs(eig(A_SUB)[0]))
            if maxeval_SUB <= 1 + 1e-3:
                SUB['error'][nTraining, :] = [nTrainingSamples, e_SUB]
            SUB['maxeval'][nTraining, :] = [nTrainingSamples, maxeval_SUB]
            
            print(f"   SUB Max eigenvalue is : {maxeval_SUB:.4f}")
            print(f"   SUB Reconstruction error : {e_SUB:.5f}")
            
            # Solution SCHUR
            time_SCHUR_start = time.time()
            A_SCHUR, B_SCHUR = optimize(Y = Y_train, X = X_train, U = U_train, max_time=20, max_iterations=100)
            SCHUR_time_elapsed = time.time() - time_SCHUR_start
            
            SCHUR['time'][nTraining, :] = [nTrainingSamples, SCHUR_time_elapsed]
            SCHUR['A'][nTraining, :, :] = A_SCHUR
            SCHUR['B'][nTraining, :, :] = B_SCHUR
            
            # Erreur de reconstruction
            e_SCHUR = norm(Y_train - A_SCHUR @ X_train - B_SCHUR @ U_train, 'fro') ** 2 / 2
            maxeval_SCHUR = np.max(np.abs(eig(A_SCHUR)[0]))
            if maxeval_SCHUR <= 1 + 1e-3:
                SCHUR['error'][nTraining, :] = [nTrainingSamples, e_SCHUR]
            SCHUR['maxeval'][nTraining, :] = [nTrainingSamples, maxeval_SCHUR]
            
            print(f"   SCHUR Max eigenvalue is : {maxeval_SCHUR:.4f}")
            print(f"   SCHUR Reconstruction error : {e_SCHUR:.5f}")
            print("==================================================")
    
    # Calcul des erreurs relatives
    relative_error_sub = (SUB['error'][:, 1] - LS['error'][:, 1]) / LS['error'][:, 1] * 100
    relative_error_schur = (SCHUR['error'][:, 1] - LS['error'][:, 1]) / LS['error'][:, 1] * 100
    
    # Plotting the results
    plt.figure(figsize=(10, 7))
    plt.semilogy(AllSamples, relative_error_sub, 's-', label='SOC')
    plt.semilogy(AllSamples, relative_error_schur, 'o-', label='SCHUR')
    
    plt.xlabel('number of measurements')
    plt.ylabel('relative error percentage')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

    # Plotting the results for computation times
    plt.figure(figsize=(10, 7))
    plt.plot(AllSamples, SUB['time'][:, 1], 's-', label='SOC')
    plt.plot(AllSamples, SCHUR['time'][:, 1], 'o-', label='SCHUR')
    plt.xlabel('Number of measurements')
    plt.ylabel('Computation time (s)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.title('Computation Time Comparison')
    plt.show()
    
    # Plotting the results for maximum eigenvalues
    plt.figure(figsize=(10, 7))
    plt.plot(AllSamples, SUB['maxeval'][:, 1], 's-', label='SOC')
    plt.plot(AllSamples, SCHUR['maxeval'][:, 1], 'o-', label='SCHUR')
    #plt.plot(AllSamples, LS['maxeval'][:, 1], '*', label='LS')

    plt.xlabel('Number of measurements')
    plt.ylabel('Maximum eigenvalue')
    plt.ylim(0.999, 1.001)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.title('Maximum Eigenvalues Comparison')
    plt.show()

if __name__ == "__main__":
    main()
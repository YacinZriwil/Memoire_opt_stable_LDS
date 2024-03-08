import numpy as np
from   frgm_restart import optimize
np.random.seed(2024)

path_U = "U.npy"
path_X = "X.npy"
path_Y = "Y.npy"

U, X, Y = np.load(path_U),np.load(path_X),np.load(path_Y)
best_A = optimize(Y, X, U, max_time= 1500, max_iterations = 300000000)

# =============================================================================
# Erreur de NeurIPS l'article : 2.1609139459682556
# =============================================================================
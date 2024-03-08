import numpy as np
from   math  import atan, cos, sin
from   math  import pi as PI
from  utils  import hyperbola_critical

def compute_G_in_SO2(A):
    # calcul le G tq : Â = G.T @ A @ G avec  Â[1,1] = Â[2,2]
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    
    x = 1/2 * atan((d-a)/(b+c))
    
    alpha = x if x >= 0 else x + PI/2  
    
    G = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
    
    return G

def is_stable_schur(A, epsilon = 1e-4):
    valeurs_propres, vecteurs_propres = np.linalg.eig(A)
    
    for lambda_ in valeurs_propres:
        if abs(lambda_) > 1 + epsilon:
            return False
    
    return True

def compute_bold_B0(Sigma0, U0, V0t):
    bold_B0 = []
    for t in hyperbola_critical(Sigma0):
        tho1, tho2 = t, 1/t
        tmp        = U0 @ np.array([[tho1, 0],[0, tho2]]) @ V0t
        bold_B0.append(tmp)
    
    return bold_B0
    
def compute_bold_Bstar(A_hat, G):
    bold_Bstar = []
    for t in hyperbola_critical([A_hat[0,1], A_hat[1,0]]):
        tho1, tho2 = t, 1/t
        tmp        = G @ np.array([[0, tho1],[tho2, 0]]) @ G.T
        bold_Bstar.append(tmp)
        
    return bold_Bstar
    
def get_nearest_stable_2x2_matrix_schur(A):
    if A.shape == (1, 1):
        if A[0] > 1:
            return 1
        else :
            return A[0]

    elif A.shape == (2,2):
    
        U0, Sigma0, V0t = np.linalg.svd(A)
        
        Uplus, Sigma_plus, Vplus_t    = np.linalg.svd(A - np.eye(2))
        sigma_plus_1, _               = Sigma_plus[0], Sigma_plus[1]
        
        Umoins, Sigma_moins, Vmoins_t = np.linalg.svd(A + np.eye(2))
        sigma_moins_1, _              = Sigma_moins[0], Sigma_moins[1]
    
        G = compute_G_in_SO2(A)
        
        A_hat = G.T @ A @ G
        #print(f"Â[1,1] = Â[2,2] ? -> {abs(A_hat[0][0] - A_hat[1][1]) < 1e-6}")
        
        bold_B0     = compute_bold_B0(Sigma0, U0, V0t)
        
        Bplus       = np.eye(2) + Uplus @ np.array([[sigma_plus_1, 0],[0, 0]]) @ Vplus_t.T # vérifier si c'est bien Vplus ou Vplus.T
        Bmoins      = -np.eye(2) + Umoins @ np.array([[sigma_moins_1, 0],[0, 0]]) @ Vmoins_t.T # vérifier si c'est bien Vplus ou Vplus.T
        
        
        bold_Bplus  = [G @ np.array([[1, A_hat[0,1]] , [0           ,  1]]) @ G.T, 
                       G @ np.array([[1,      0    ] , [A_hat[1,0]  ,  1]]) @ G.T]
        
        bold_Bmoins = [G @ np.array([[-1, A_hat[0,1]], [0           , -1]]) @ G.T, 
                       G @ np.array([[-1,     0     ], [A_hat[1,0]  , -1]]) @ G.T]
        
        bold_Bstar  = compute_bold_Bstar(A_hat, G)
        
        noms               = ["A", "B+", "B-", "bold_B0", "bold_B+",  "bold_B-",  "bold_B*"]
        possible_solutions = [[A], [Bplus], [Bmoins], bold_B0, bold_Bplus, bold_Bmoins, bold_Bstar] # on met A,B+ et B- dans un tableau mm s'il n'y a qu'un élément pour avoir plus facile dans la boucle for en dessous
        
        #print("=================================================================")
    
        min_dist = np.inf
        A_stable = None
        count = 0
        for nom, sol in zip(noms, possible_solutions):
            for X in sol:
                #print(f"{nom} : ")

                count+=1
                if is_stable_schur(X):
                    dist = np.linalg.norm(A - X, 'fro')**2
                    if dist < min_dist:
                        min_dist = dist 
                        A_stable = np.copy(X)
            
                #print("---------------------")
            #print("=================================================================")
        return A_stable
    
    else:
        print("La matrice d'entrée n'est ni de taille 1x1 ni 2x2")

if __name__ == "__main__":
    import time
    A = np.array([[1,2],[1,1]])
    print(A)
    
    start_time = time.time()
    X = get_nearest_stable_2x2_matrix_schur(A)
    end_time = time.time()
    print(f"Temps écoulé : {(end_time - start_time):.6f}")
    print(X)
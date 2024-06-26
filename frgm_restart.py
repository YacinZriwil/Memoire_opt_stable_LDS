import time
import numpy as np
import scipy
from   nearest_schur_stable_2x2     import is_stable_schur
from   utils                        import norm, get_max_abs_eigval, project_psd, adjusted_frobenius_norm, checkdstable
from   modified_real_schur_manifold import ModifiedRealSchur
from   pymanopt.manifolds           import Stiefel
import os
import utilities
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def initialize_schur(X,Y,U, Schur_manifold):
    XU = np.concatenate((X, U), axis=0)
    AB_ls = Y @ np.linalg.pinv(XU)
    A_ls = AB_ls[:X.shape[0], :X.shape[0]]
    
    B_ls = AB_ls[:, X.shape[0]:X.shape[0] + U.shape[0]]

    T, Q = scipy.linalg.schur(A_ls, output='real')
    
    T = Schur_manifold.projection(T, T)
    
    return Q,T, B_ls

def initialize_soc_with_inputs(X,Y, U,**kwargs):
    """
    DEBUG
    """

    stability_relaxation = kwargs.get('stability_relaxation', 0)

    n = len(X)
    S = np.identity(n)
    S_inv = S
    XU = np.concatenate((X, U), axis=0)
    AB_ls = Y @ np.linalg.pinv(XU)

    O,C = scipy.linalg.polar(a=AB_ls[:n,:n], side='right')
    eig_max = get_max_abs_eigval(S_inv @ O @ C @ S, is_symmetric=False)
    B = AB_ls[:n,n:]
    
    if eig_max > 1 - stability_relaxation:
        C = project_psd(Q=C, eps=0, delta=1-stability_relaxation)
        e_old = adjusted_frobenius_norm(
            X=Y - (S_inv @ O @ C @ S) @ X - B @ U)
    else:
        e_old = adjusted_frobenius_norm(X=Y - AB_ls @ XU) 
    
    eig_max = max(1, eig_max)
    A_stab = AB_ls[:n,:n] / eig_max
    _,S_,O_,C_ = checkdstable(
        A=0.9999*A_stab, 
        stab_relax=stability_relaxation)
    S_inv_ = np.linalg.inv(a=S_)
    
    e_temp = adjusted_frobenius_norm(
        X=Y - (S_inv_ @ O_ @ C_ @ S_) @ X - B @ U)
    if e_temp < e_old:
        S, O, C = S_, O_, C_
        e_old = e_temp
    
    return e_old, S, O, C, B    
  

def refine_inputs_solution(X, Y, U, Q, T, B, **kwargs):
    e_0 = kwargs.get('e_0', 0.00001)
    delta = kwargs.get('delta', 0.00001)
    
    
    e_t = e_0
    A = Q @ T @ Q.T  
    n = len(A)

    XU = np.concatenate((X, U), axis=0)
    AB_ls = Y @ np.linalg.pinv(XU)

    A_B = np.concatenate((A, B), axis=1)
    grad = AB_ls - A_B

    AB_new = A_B + e_t * grad
    A_new = AB_new[:n,:n]
    
    # get initial max abs eigenvalue
    eig_max = utilities.get_max_abs_eigval(A_new, is_symmetric=False)
    
    while eig_max < 1 and utilities.adjusted_frobenius_norm(X=AB_new - AB_ls) > 0.01:
        print(f"JE SUIS DANS LE REFINE !!!!!!!!!!!!!!!!!!!!!!!!!")
        e_t += delta
        AB_new = A_B + e_t * grad
        A_new = AB_new[:n,:n]
        eig_max = utilities.get_max_abs_eigval(A_new, is_symmetric=False)
    
    if e_t != e_0:
        A_Btemp = A_B + (e_t - delta) * grad
        A = A_Btemp[:n,:n]
        B = A_Btemp[:n,n:]
    
    return A, B


def objective_function(Q, T, B, Y, X, U):
    return 1/2 * np.linalg.norm(Y - Q@T@Q.T @ X - B@U, 'fro')**2

def euclidean_gradient(Q, T, B, Y, X, U):
    C  = X.T @ Q @ T.T
    minusYplusBU = -Y + B@U
    D = minusYplusBU.T @ Q
    
    grad_B = (minusYplusBU + Q @ T @ Q.T @ X ) @ U.T
    grad_Q = ( (minusYplusBU) @ C + X @ (D + C) @ T ) 
    grad_T = ( D.T + C.T ) @ X.T @ Q
    
    return grad_Q, grad_T, grad_B
   
def displayInfoIteration(i, gamma, Y, X, U, Q, T, B, Orthogonal_manifold):
    loss = objective_function(Q, T, B, Y, X, U)
    QQ, TT, BB = euclidean_gradient(Q, T, B, Y, X, U)
    QQ = Orthogonal_manifold.projection(Q, QQ)
    
    print(f"Iteration {i+1}:" , end = " ")
    print(f"Loss = {loss:.3f}\t gamma = {gamma:.3e}\t ∇Q(f) = {norm(QQ):.3f} \t ∇T(f) = {norm(TT):.3f} \t ∇B(f) = {norm(BB):.3f}")
    
    A_ = Q @ T @ Q.T
    print(f"\t\t\t  A schur-stable ? -> {is_stable_schur(A_)}")

    print("=============================================================================================")
 

def optimize(Y, X, U, gamma = 0.1, min_gamma = 0.0000000001, alpha1 = 0.5, max_iterations = 30, max_time = 20):
    N                   = X.shape[0]
    Schur_manifold      = ModifiedRealSchur(N)
    Orthogonal_manifold = Stiefel(N,N)
    
    # Initialisation
    alpha  = alpha1
    
    Q,T, B = initialize_schur(X,Y,U, Schur_manifold)
    Q1, T1, B1 = Q.copy(), T.copy(), B.copy()
    best_Q, best_T, best_B = Q.copy(), T.copy(), B.copy()
    loss_best = objective_function(best_Q, best_T, best_B, Y, X, U)
    
    start_time = time.time()
    i = 0
    while i < max_iterations and time.time() - start_time < max_time:
        Q2, T2, B2 = Q.copy(), T.copy(), B.copy()
    
        grad_Q1, grad_T1, grad_B1 = euclidean_gradient(Q1, T1, B1, Y, X, U)
        gradR_Q1                  = Orthogonal_manifold.projection(Q1, grad_Q1) # projection de gradE sur le plan tangent -> gradR
        
        Q = Orthogonal_manifold.retraction(Q1, -gamma *norm(Q1)* gradR_Q1)
        T = Schur_manifold.projection(T1,  T1 -gamma * norm(T1)* grad_T1)
        B = B1- gamma * norm(B1)* grad_B1
        
        # Fonction objectif
        loss2 = objective_function(Q2, T2, B2, Y, X, U) # pour voir si on a un meilleur objectif qu'avant la maj de Q,T,B
        gamma1 = gamma
        
        while objective_function(Q, T, B, Y, X, U) > loss2 and gamma1 >= min_gamma: # si loss <= loss2 => on a réussi à décroitre (donc on passe dans le else d'en dessous); 
            gamma1 = 1/5 * gamma1
            Q = Orthogonal_manifold.retraction(Q1, -gamma1  *norm(Q1)* gradR_Q1)
            T = Schur_manifold.projection(T1, T1 - gamma1 * norm(T1)* grad_T1)
            B = B1 - gamma1 * norm(B1)* grad_B1
       
        if gamma1 < min_gamma: # ca veut dire qu'on n'a pas pu diminuer f(x) -> minimum local
            Q1, T1, B1 = Q.copy(), T.copy(), B.copy()
            alpha = alpha1
            
            print("-> Restart ...")
            
        else: # ca veut dire qu'on a pu décroitre -> on peut stocker la valeur de gamma1 qui est la dernière valeur pour laquelle on a pu décroitre
            gamma = gamma1
            alpha_pred = alpha
            alpha = 1/2 * (np.sqrt(alpha**4 + 4 * alpha**2) - alpha**2)
            beta  = alpha*(1-alpha)/(alpha**2 + alpha_pred)
            
            Q1, T1, B1 = Q + beta * (Q - Q2), T + beta*(T-T2), B + beta*(B - B2)
            
        loss_current = objective_function(Q, T, B, Y, X, U)
        if loss_current < loss_best:
            loss_best = loss_current
            #print("Update the best")
            #print(f"Before : loss = {loss_best} \t Now : loss = {loss_current}")
            #print("=============================================================================================")
            best_Q, best_T, best_B = Q.copy(), T.copy(), B.copy()
        
        gamma = 2*gamma
        
        #displayInfoIteration(i, gamma, Y, X, U, Q, T, B, Orthogonal_manifold)
        i+=1
       
    # Fin de l'algorithme
    end_time = time.time()
    best_A, best_B = refine_inputs_solution(X, Y, U, best_Q, best_T, best_B)
    #best_A = best_Q @ best_T @ best_Q.T
    
    loss = objective_function(best_Q, best_T, best_B, Y, X, U)
    schur_err  = adjusted_frobenius_norm(Y - best_A@X - best_B@U)
    print(f"\nLoss finale : {loss:.2f}   et   ||Y - AX - BU|| = {schur_err/norm(Y) *100}%\n========================================") 
    
    XU = np.concatenate((X, U), axis=0)
    AB_ls = Y @ np.linalg.pinv(XU)
    pred = AB_ls @ XU
    
    ls_error = adjusted_frobenius_norm(X=Y - pred)
    print(f"ls_error ={ls_error}")
    print(f"schur_err ={schur_err}")

    perc_error = 100*(schur_err - ls_error)/ls_error
    print(f"% error : {perc_error}")
    print(f"Temps écoulé : {((end_time - start_time)):.2f}s\n========================================")
    print(f"best_A schur-stable ? -> {is_stable_schur(best_A)}")
    
    return best_A, best_B

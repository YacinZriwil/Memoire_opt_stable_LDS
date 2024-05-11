import time
import numpy as np
import scipy
from   nearest_schur_stable_2x2     import is_stable_schur
from   utils                        import norm, get_max_abs_eigval, project_psd, adjusted_frobenius_norm, checkdstable
from   modified_real_schur_manifold import ModifiedRealSchur
from   pymanopt.manifolds           import Stiefel
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
    print(f"Loss = {loss:.2f}\t gamma = {gamma:.2e}\t ∇Q(f) = {norm(QQ):.2f} \t ∇T(f) = {norm(TT):.2f} \t ∇B(f) = {norm(BB):.2f}")
    print("=============================================================================================")
  
def optimize(Y, X, U, gamma = 0.001, min_gamma = 0.0000000001, alpha1 = 1, max_iterations = 10000, max_time = 20):
    N                   = X.shape[0]
    Schur_manifold      = ModifiedRealSchur(N)
    Orthogonal_manifold = Stiefel(N,N)
    
    # Initialisation
    alpha  = alpha1
    _, __, Q, __, B = initialize_soc_with_inputs(X,Y,U)
    T = Schur_manifold.random_point()
    
    Q1, T1, B1 = Q.copy(), T.copy(), B.copy()
    best_Q, best_T, best_B = Q.copy(), T.copy(), B.copy()
    loss_best = objective_function(best_Q, best_T, best_B, Y, X, U)
    
    start_time = time.time()
    i = 0
    while i < max_iterations and time.time() - start_time < max_time:
        Q2, T2, B2 = Q.copy(), T.copy(), B.copy()
    
        grad_Q1, grad_T1, grad_B1 = euclidean_gradient(Q1, T1, B1, Y, X, U)
        gradR_Q1                  = Orthogonal_manifold.projection(Q1, grad_Q1) # projection de gradE sur le plan tangent -> gradR
       
        Q = Orthogonal_manifold.retraction(Q1, -gamma * norm(Q1)* gradR_Q1)
        T = Schur_manifold.projection(T1,  T1 -gamma * norm(T1)* grad_T1)
        B = B1- gamma * norm(B1) * grad_B1
        
        # Fonction objectif
        loss2 = objective_function(Q2, T2, B2, Y, X, U) # pour voir si on a un meilleur objectif qu'avant la maj de Q,T,B
        gamma1 = gamma
        
        while objective_function(Q, T, B, Y, X, U) > loss2 and gamma1 >= min_gamma: # si loss <= loss2 => on a réussi à décroitre (donc on passe dans le if d'en dessous); 
            gamma1 = 2/3 * gamma1
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
        
        displayInfoIteration(i, gamma, Y, X, U, Q, T, B, Orthogonal_manifold)
        i+=1
       
    # Fin de l'algorithme
    end_time = time.time()
    best_A = best_Q @ best_T @ best_Q.T
    
    loss = objective_function(best_Q, best_T, best_B, Y, X, U)
    err  = norm(Y - best_A@X - best_B@U)
    print(f"\nLoss finale : {loss:.2f}   et   ||Y - AX - BU|| = {err/norm(Y) *100}\n========================================") 
    print(f"Temps écoulé : {((end_time - start_time)):.2f}s\n========================================")
    print(f"best_A schur-stable ? -> {is_stable_schur(best_A)}")
    
    return best_A, best_B


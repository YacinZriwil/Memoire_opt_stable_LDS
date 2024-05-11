import numpy as np
from   utils                        import norm, get_max_abs_eigval, project_psd, adjusted_frobenius_norm, checkdstable
import time
from   nearest_schur_stable_2x2     import is_stable_schur
import scipy
from   modified_real_schur_manifold import ModifiedRealSchur
from   pymanopt.manifolds           import Stiefel
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

def f(X, Y, U, Q, T, B):
    # Implémentez la fonction objectif qui dépend de S, U et B.
    return 1/2 * np.linalg.norm(Y - Q@T@Q.T @ X - B@U, 'fro')**2

def grad_f(Q, T, B, Y, X, U):
    C  = X.T @ Q @ T.T    
    
    minusYplusBU = -Y + B@U

    D = minusYplusBU.T @ Q
    
    grad_B = (minusYplusBU + Q @ T @ Q.T @ X ) @ U.T
    
    grad_Q = ( (minusYplusBU) @ C + X @ (D + C) @ T ) 
    
    grad_T = ( D.T + C.T ) @ X.T @ Q
    
    return grad_Q, grad_T, grad_B

def displayInfoIteration(i, gamma, Y, X, U, Q, T, B, Orthogonal_manifold):
    loss = f(X, Y, U, Q, T, B)
    QQ, TT, BB = grad_f(Q, T, B, Y, X, U)
    QQ = Orthogonal_manifold.projection(Q, QQ)
    
    print(f"Iteration {i+1}:" , end = " ")
    print(f"Loss = {loss:.2f}\t gamma = {gamma:.2e}\t ∇Q(f) = {norm(QQ):.2f} \t ∇T(f) = {norm(TT):.2f} \t ∇B(f) = {norm(BB):.2f}")
    print("=============================================================================================")
 
    
def P(Q_old, T_old, B_old, gradR_Q, grad_T, grad_B, gamma, Orthogonal_manifold, Schur_manifold):
    Q = Orthogonal_manifold.retraction(Q_old, -gamma * gradR_Q)
    T = Schur_manifold.projection(T_old,  T_old -gamma * grad_T)
    B = B_old - gamma * grad_B
    
    return Q,T,B


def fast_gradient_method(Y, X, U, alpha_1, gamma, lower_bound_gamma, max_iterations = 30, max_time = 20):
    
    # =============================================================================
    # Initialisation 
    # =============================================================================
    N                   = X.shape[0]
    Schur_manifold      = ModifiedRealSchur(N)
    Orthogonal_manifold = Stiefel(N,N)
    _, __, Q, __, B = initialize_soc_with_inputs(X,Y,U)
    T = Schur_manifold.random_point()
    
    # =============================================================================
    # Starter
    # =============================================================================
    f_X = f(X, Y, U, Q, T, B)
    alpha = alpha_1
    
    start_time = time.time()
    i = 0
    while i < max_iterations and time.time() - start_time < max_time:
        Q_old, T_old, B_old = Q, T, B
        grad_Q, grad_T, grad_B = grad_f(Q, T, B, Y, X, U)
        
        gradR_Q = Orthogonal_manifold.projection(Q_old, grad_Q)
        
    
        Q, T, B = P(Q_old, T_old, B_old, gradR_Q, grad_T, grad_B, gamma, Orthogonal_manifold, Schur_manifold)
        f_X_prime = f(X, Y, U, Q, T, B)
        
        # Réduction de gamma si la fonction objectif n'a pas diminué.
        while f_X_prime > f_X:
            gamma /= 2
            if gamma < lower_bound_gamma:
                Q, T, B = Q_old, T_old, B_old
                break
            Q, T, B = P(Q_old, T_old, B_old, gradR_Q, grad_T, grad_B, gamma, Orthogonal_manifold, Schur_manifold)
            f_X_prime = f(X, Y, U, Q, T, B)
        
        displayInfoIteration(i, gamma, Y, X, U, Q, T, B, Orthogonal_manifold)

        if gamma < lower_bound_gamma:
            gamma = lower_bound_gamma
            alpha = alpha_1
            Q, T, B = Q_old, T_old, B_old
            continue
        
        alpha_next = 0.5 * (1 + np.sqrt(1 + 4 * alpha**2))
        beta = (alpha * (1 - alpha)) / (alpha**2 + alpha_next)
        
        Q = Q + beta * (Q - Q_old)
        T = T + beta * (T - T_old)
        B = B + beta * (B - B_old)
        
        alpha = alpha_next
        gamma = 2 * gamma
        i+=1
        
    # =============================================================================
    # End algo  
    # =============================================================================
    end_time = time.time()
    A = Q @ T @ Q.T
    
    loss = f(X, Y, U, Q, T, B)
    schur_err  = adjusted_frobenius_norm(Y - A@X - B@U)
    print(f"\nLoss finale : {loss:.2f}   et   ||Y - AX - BU|| = {schur_err/norm(Y) *100}%\n========================================") 
    
    XU = np.concatenate((X, U), axis=0)
    AB_ls = Y @ np.linalg.pinv(XU)
    pred = AB_ls @ XU
    
    ls_error = adjusted_frobenius_norm(X=Y - pred)
    perc_error = 100*(schur_err - ls_error)/ls_error
    print(f"% error : {perc_error}")
    print(f"Temps écoulé : {((end_time - start_time)):.2f}s\n========================================")
    print(f"best_A schur-stable ? -> {is_stable_schur(A)}")
    
    return A, B



import typing
import numpy as np
import scipy

def normalize(X):
    # Appliquer la normalisation min-max
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min)
    
    return X_norm

def standardize(X):
    mean = np.mean(X, axis=0)

    # Calculer l'écart-type de chaque caractéristique
    std = np.std(X, axis=0)
    
    # Standardiser les données
    X_standardized = (X - mean) / std
    
    return X_standardized

def get_max_abs_eigval(
    X: np.ndarray, 
    is_symmetric: bool = False) -> float:
    """
    Get maximum value of the absolute eigenvalues of
    a matrix X. Returns `numpy.inf` if the eigenvalue
    computation does not converge. X does not have to 
    be symmetric.
    """
    eigval_operator = (
        np.linalg.eigvalsh 
        if is_symmetric 
        else np.linalg.eigvals)

    try: 
        eig_max = max(abs(eigval_operator(X)))
    except:
        eig_max = np.inf
        
    return eig_max

def project_psd(
    Q: np.ndarray, 
    eps: float = 0, 
    delta: float = np.inf) -> np.ndarray:
    """
    DEBUG
    """
    Q = (Q + Q.T)/2
    E,V = np.linalg.eig(a=Q)
    E_diag = np.diag(v=np.minimum(delta, np.maximum(E, eps)))
    Q_psd = V @ E_diag @ V.T
    return Q_psd

def adjusted_frobenius_norm(X: np.ndarray) -> float:
    """
    Compute the square of the Frobenius norm of a matrix
    and divide the result by 2.
    """
    return np.linalg.norm(X)**2/2

def checkdstable(
    A : np.ndarray,
    stab_relax: bool) -> typing.Tuple[np.ndarray, ...]:
    
    P = scipy.linalg.solve_discrete_lyapunov(a=A.T, q=np.identity(len(A)))
    S = scipy.linalg.sqrtm(A=P)
    S_inv = np.linalg.inv(a=S)
    OC = S @ A @ S_inv
    O,C = scipy.linalg.polar(a=OC, side='right')
    C = project_psd(Q=C, eps=0, delta=1-stab_relax)
    return P, S, O, C

def is_orthogonal(A):
    return np.allclose(A @ A.T, np.eye(A.shape[0]), rtol=1e-02, atol=1e-02)

def norm(matrix):
    return np.linalg.norm(matrix, 'fro')

def is_modified_real_schur_form(T, rtol=1e-02, atol=1e-02):
    n = T.shape[0] 
    if n % 2 == 0:
        for j in range((n-2)//2):
            J = 2*j
            i = 2*(j+1)
            T_J =  T[i:,J:J+1+1]
            
            N,M = T_J.shape
            
            if not np.allclose(T_J, np.zeros((N,M)), rtol, atol):
                return False
    else:
        for j in range((n-2)//2 +1):
            J = 2*j
            i = 2*(j+1)
            T_J =  T[i:,J:J+1 +1]
            N,M = T_J.shape

            if not np.allclose(T_J, np.zeros((N,M)), rtol, atol):
                return False
    
    return True

def hyperbola_critical(sigma):
    """
    Calcul des points critiques (réels) de dist(x, H), où H = {(s, t) : st=1}.
    Renvoie un vecteur d'au plus 4 points r(i) tels que (r(i), 1/r(i)) soient ces points critiques.
    """
    
    coeffs = [1, -sigma[0], 0, sigma[1], -1]
    
    # provient de la résolution de d/dt (...) = 0 (page 17 du pdf de noferini)
    r = np.roots(coeffs)
    
    # on enlève les solutions complexes
    r = r[np.isreal(r)].real
        
    return r


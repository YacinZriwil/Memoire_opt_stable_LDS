import numpy as np
from nearest_schur_stable_2x2    import get_nearest_stable_2x2_matrix_schur
from pymanopt.manifolds.manifold import RiemannianSubmanifold

"""
# ================================================================================================================
# La classe ModifiedRealSchur est en fait un cas particulier de la classe _Euclidean de PyManopt
# Je n'ai pas fait hérité la classe car la variable "dimension" n'est pas recalculé dans la classe _Euclidean
# J'ai donc copié collé la classe _Euclidean et le but est de changer quelques fonctions afin
# d'être dans le cas des matrices dans la forme de Schur réelle modifiée càd : 
#   
#   1. Projection
#   2. Rétraction (qui est en fait la fonction exp car l'exponentielle est un type de rétraction particulier (géodésique))
#   3. random_point
# =================================================================================================================
"""
class ModifiedRealSchur(RiemannianSubmanifold):
    def __init__(self, n : int):
        # Initialisation de votre manifold avec ses paramètres spécifiques
        self.n = n
        self._shape = (n,n)
        dimension = int(n*(n+1)/2 + n//2)
        print(f"DIMENSION = {dimension}")
        
        super().__init__("Manifold des matrices dans la forme de Schur réelle modifiée", dimension)

    @property
    def typical_dist(self):
        return np.sqrt(self.dim)

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return float(
            np.real(
                np.tensordot(
                    tangent_vector_a.conj(),
                    tangent_vector_b,
                    axes=tangent_vector_a.ndim,
                )
            )
        )

    def norm(self, point, tangent_vector):
        return np.linalg.norm(tangent_vector)

    def dist(self, point_a, point_b):
        return np.linalg.norm(point_a - point_b)

    def projection(self, point, vector):
        # projection sur l'espace RS
        T = vector
        if self.n % 2 == 0:
            #on projette T sur l'espace RS
            for j in range((self.n-2)//2):
                J = 2*j
                i = 2*(j+1)
                T[i:,J:J+1+1] = 0 # le deuxième +1 est là car en python matrice[debut_ligne:fin_ligne + 1, debut_colonne:fin_colonne + 1]
                  
        else:
            for j in range((self.n-2)//2 +1):
                J = 2*j
                i = 2*(j+1)
                T[i:,J:J+1 +1] = 0
                
            bloc_1x1 = [T[-1][-1]]
            T[-1][-1] = get_nearest_stable_2x2_matrix_schur(bloc_1x1)
            
        #on stabilise T
        for i in range(0, self.n - 1, 2):  
            bloc_2x2 = T[i:i+2, i:i+2]
            T[i:i+2, i:i+2] = get_nearest_stable_2x2_matrix_schur(bloc_2x2)
        return T

    to_tangent_space = projection

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        return euclidean_hessian

    def exp(self, point, tangent_vector):
        return point + tangent_vector # ∈ RS car point ∈ RS et tangent_vector ∈ RS => point + tangent_vector ∈ RS car RS est un espace vectoriel

    retraction = exp

    def random_point(self):
        a,b = -10, 10
        # Initialiser la matrice à zéro
        T = np.zeros((self.n, self.n), dtype=int)
        
        # Fonction pour générer un bloc 2x2 sans zéros
        def generate_nonzero_2x2_block(a, b):
            block = np.random.randint(a, b+1, size=(2, 2))
            while 0 in block:
                block = np.random.randint(a, b+1, size=(2, 2))
            return block
        
        # Itérer sur la matrice pour remplir les blocs
        i = 0
        while i < self.n:
            if i == self.n-1 and self.n % 2 == 1:
                # Si n est impair et c'est le dernier élément, on met un bloc 1x1
                T[i, i] = np.random.randint(a, b+1)
                while T[i, i] == 0:  # Assurer que le bloc 1x1 n'est pas zéro
                    T[i, i] = np.random.randint(a, b+1)
                i += 1
            else:
                # Mettre un bloc 2x2 sur la diagonale sans zéros
                T[i:i+2, i:i+2] = generate_nonzero_2x2_block(a, b)
                # Remplir les éléments au-dessus du bloc avec des valeurs aléatoires entières
                if i+2 < self.n:  # S'assurer qu'on ne dépasse pas la dimension de la matrice
                    T[i:i+2, i+2:self.n] = np.random.randint(a, b+1, size=(2, self.n-i-2))
                i += 2
        
        return T.astype(np.float64)

    def random_tangent_vector(self, point):
        tangent_vector = self.random_point()
        return tangent_vector / self.norm(point, tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return tangent_vector_a

    def zero_vector(self, point):
        return np.zeros(self._shape)
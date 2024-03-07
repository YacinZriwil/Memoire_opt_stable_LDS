# 1. But du mémoire 
On veut trouver les matrices $A$ et $B$ telles que, pour les observations données $Y$, $X$ et $U$, on ait : $Y \approx AX + BU$. \
La contrainte est que la matrice A doit être stable au sens de Schur car le système dynamique associé est à valeurs discrètes : $y_t \equiv x_{t+1} = A x_t + B u_t$.

## - Stabilité de Schur
$\Omega_S = \{z \in \mathbb{C} : |z| \leq 1 \}$ \
Une matrice $A$ est stable ssi toutes les valeurs propres sont dans $\Omega_S$.

# 2. Résolution du problème
On va se baser sur l'algorithme "Fast Gradient Method with restart" (FGM) qui est une version améliorée de la descente de gradient classique. La différence entre le FGM et l'algorithme implémenté dans ce github est qu'on va faire une FRGM : "Fast Riemannian Gradient Method". 

Le but est donc d'utiliser la théorie sur les manifolds dans le FGM.
 
# 3. Explications des fichiers

## 3.1 frgm_restart.py

## 3.2 ModifiedRealSchurManifold.py
## 3.3 nearest_schur_stable_2x2.py

## 3.4 hyperbola_critical.py


# 4. Données pour la partie pratique
## 4.1 Robot Franka Emika Panda

## 4.2 Dataset UCLA ??

# 1. But du mémoire 
On veut trouver les matrices $A$ et $B$ telles que, pour les observations données $Y$, $X$ et $U$, on ait : $Y \approx AX + BU$. \
La contrainte est que la matrice A doit être stable au sens de Schur car le système dynamique associé est à valeurs discrètes : $y_t \equiv x_{t+1} = A x_t + B u_t$.

## - Stabilité de Schur
$\Omega_S =$ { $z \in \mathbb{C} : |z| \leq 1$ } \
Une matrice $A$ est stable ssi toutes les valeurs propres sont dans $\Omega_S$. \
On va noter l'espace $RS$ qui est l'espace des matrices de Schur réelle modifiée.

# 2. Résolution du problème
On va se baser sur l'algorithme "Fast Gradient Method with restart" (FGM) qui est une version améliorée de la descente de gradient classique. La différence entre le FGM et l'algorithme implémenté dans ce github est qu'on va faire une FRGM : "Fast Riemannian Gradient Method". 

Le but est donc d'utiliser la théorie sur les manifolds dans le FGM.
 
# 3. Explications des fichiers

## 3.1 frgm_restart.py
Ce script contient : 
1. l'initialisation : la matrice orthogonale a été calculé avec le SOC algorithm utilisé dans l'article NeurIPS;
2. le gradient euclidien : le calcul de $\nabla f$ est noté dans les annexes du rapport.
3. algorithme FRGM with restart.

## 3.2 modified_real_schur_manifold.py
Ce script contient une classe qui hérite de RiemannianSubmanifold de PyManopt même si je n'utilise pas ici le solveur de PyManopt. (j'ai déjà testé et j'ai eu quelques soucis avec l'utilisation du gradient avec un produit de manifolds donc j'ai gardé la classe si jamais je reviens sur PyManopt plus tard).

J'ai donc implémenté : 
1. projection  : projette la matrice donnée en entrée sur $RS$ + rend cette matrice stable.
2. random_point: renvoie un point aléatoire appartenant à $RS$ et stable.

## 3.3 nearest_schur_stable_2x2.py
Ce script contient le calcul des 15 matrices possiblement stables pour le cas d'une matrice 2x2 (voir article de Noferini). Par le lemme qu'ils ont démontré, on sait que la matrice la plus proche est forcément l'une des 15 matrices calculées.

## 3.4 RunMe
### Robot Franka Emika Panda
Pour tester le code, on peut utiliser le jeu de données de NeurIPS pour le bras robotisé "Franka". Dans ce github, il y a les fichiers contenant les matrices $U, X, Y$ qui sont les données du le robot (positions, angles, ... ). \
Il faut lancer le fichier RunMe_Robot_Franka.py. 


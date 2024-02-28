import numpy as np

A = np.abs(np.random.rand(3,3) + np.identity(3))
A = A + A.transpose()

chol_A = np.linalg.cholesky(A)

print(f"A : \n{A}")
print("-------------")
print(f"chol_A : \n{chol_A}")
print("-------------")
A_recons = chol_A @ chol_A.transpose()
print(f"A_recons**2 : \n{A_recons}")

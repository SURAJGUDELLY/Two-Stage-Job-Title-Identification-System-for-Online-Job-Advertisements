from numpy import dot
from numpy.linalg import norm

a = (1, 2, 3)
b = (1, 22, 3)
dst = dot(a, b)/(norm(a)*norm(b))
print(dst)

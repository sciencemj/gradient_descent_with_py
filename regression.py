import numpy as np

def regression_analysis(train_data):
  sigma = lambda x: sum([j for j in range(x)])
  square_sigma = lambda x: sum([j**2 for j in range(x)])
  n = len(train_data)
  sum_x = sigma(n)
  sum_x_square = square_sigma(n)
  sum_y = sum(train_data)
  sum_xy = sum([train_data[i]*i for i in range(n)])
  A = np.array([[n, sum_x],[sum_x, sum_x_square]])
  r = np.array([[sum_y,], [sum_xy]])
  A_inverse = np.linalg.inv(A)
  a_b = np.dot(A_inverse, r)
  return lambda x: a_b[0] + a_b[1]*x

if __name__ == "__main__":
  print("regression.py is running")
# Code written by Chetan Bhole @ 2015
# Tutorial for understanding simple versions of Gaussian Processes


import numpy as np
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt


# standard deviation of the signal 
STDDEV = 1
SCALE = 1
# The standard deviation of the noise
STDDEV_NOISE = 0.1

# periodicty parameter
period_p = 2

num_fn_samples = 5


def get_periodic_kernel(vec1, vec2, sigma, scale):
  K = np.zeros(shape=(len(vec1),len(vec2)))
  for i in range(0,len(vec1)):
    for j in range(0, len(vec2)):
      x = period_p*(vec1[i]-vec2[j])/2.0
      K[i,j] = sigma**2 * np.exp(-2.0*(np.sin(x)**2)/(scale**2))
  return K

def get_sq_kernel(vec1, vec2, sigma, scale):
  K = np.zeros(shape=(len(vec1),len(vec2)))
  for i in range(0,len(vec1)):
    for j in range(0, len(vec2)):
      K[i,j] = sigma**2 * np.exp(-0.5*((vec1[i]-vec2[j])/scale)**2)
  return K

def plot_GP(num_fn_samples, x_eq, y_eq, prior_mean, sqK_xeq_xeq, x, y, titlex):
  plt.figure()
  for i in range(0,num_fn_samples):
    plt.plot(x_eq, y_eq[:,i], 'r-')
    
  lines = plt.plot(x_eq, prior_mean)
  plt.setp(lines, linestyle='-')
  plt.setp(lines, linewidth=3, color='b')
  cov_y_eq_diag_196 = 1.96*np.diagonal(sqK_xeq_xeq)
  plt.fill_between(x_eq, prior_mean - cov_y_eq_diag_196, prior_mean + cov_y_eq_diag_196, facecolor='#ffdae0', linewidth=0.0)
  plt.plot(x, y, 'g*', markersize=12)
  plt.title(titlex)


# Let's say x_eq are the equidistant points chosen to enable visualization.
# It also allows generating a smoother plot, the more the merrier.
x_eq = np.arange(-5, 5, 0.1)
x_eq = x_eq.astype(np.float_)
sqK_xeq_xeq = get_sq_kernel(x_eq, x_eq, STDDEV, SCALE)

prior_mean = np.zeros(shape=(len(x_eq)))

# Part I. Plotting some sample prior functions using the mean
# and covariance given above

# Generating sample prior functions from the process
y_eq = np.random.multivariate_normal(prior_mean, sqK_xeq_xeq, num_fn_samples)
y_eq = y_eq.transpose()

plot_GP(num_fn_samples, x_eq, y_eq, prior_mean, sqK_xeq_xeq, [], [], 'prior, no data')


# Part II. If we have some data points i.e. we have some y values
# for the x, the GP gets modified.
x = np.array([-4.5, -3, -1.5, 0, 2])
y = np.array([-2.5, 0, 1, 2.5, -0.8]) 
 
# Calculate the four sub-matrices each of which is 
# a covariance matrices (not necessarily square here) 
# use original x_eq points and data points x

# This submatrix is the same as above
# sqK_xeq_xeq = get_sq_kernel(x_eq, x_eq, STDDEV, SCALE)

sqK_x_x = get_sq_kernel(x, x, STDDEV, SCALE)
# These two matrices should be transposes
# though i guess can change depending on the defined kernel matrix
sqK_x_xeq = get_sq_kernel(x, x_eq, STDDEV, SCALE)
sqK_xeq_x = get_sq_kernel(x_eq, x, STDDEV, SCALE)
 
# Computations to get new modified mean and sample y values
# using conditional data
inv_sqK_x_x = np.linalg.inv(sqK_x_x)
mean_y_eq = sqK_xeq_x.dot(inv_sqK_x_x).dot(y)
cov_y_eq = sqK_xeq_xeq - sqK_xeq_x.dot(inv_sqK_x_x).dot(sqK_x_xeq)
 
y_eq = np.random.multivariate_normal(mean_y_eq, cov_y_eq, num_fn_samples)
y_eq = y_eq.transpose()
 
plot_GP(num_fn_samples, x_eq, y_eq, mean_y_eq, cov_y_eq, x, y, 'with data, noiseless')


# Part III. If the observed data points also have gaussian noise.
# using _n to indicate noise case

inv_sqK_x_x_n = np.linalg.inv(sqK_x_x + STDDEV_NOISE**2 * np.identity(sqK_x_x.shape[0]))
mean_y_eq_n = sqK_xeq_x.dot(inv_sqK_x_x_n).dot(y)
cov_y_eq_n = sqK_xeq_xeq - sqK_xeq_x.dot(inv_sqK_x_x_n).dot(sqK_x_xeq)
 
y_eq = np.random.multivariate_normal(mean_y_eq_n, cov_y_eq_n, num_fn_samples)
y_eq = y_eq.transpose()

plot_GP(num_fn_samples, x_eq, y_eq, mean_y_eq_n, cov_y_eq_n, x, y, 'with data, with noise')



# Part IV. Plotting some sample prior functions using the mean
# and covariance given a periodic kernel

x_eq = np.arange(-5, 5, 0.1)
x_eq = x_eq.astype(np.float_)
sqK_xeq_xeq = get_periodic_kernel(x_eq, x_eq, STDDEV, SCALE)

prior_mean = np.zeros(shape=(len(x_eq)))

# Generating sample prior functions from the process
y_eq = np.random.multivariate_normal(prior_mean, sqK_xeq_xeq, num_fn_samples)
y_eq = y_eq.transpose()

plot_GP(num_fn_samples, x_eq, y_eq, prior_mean, sqK_xeq_xeq, [], [], 'prior, no data')


# 2. If we have some data points i.e. we have some y values
# for the x, the GP gets modified.
x = np.array([-4.5, -3, -1.5, 0, 2])
y = np.array([-2.5, 0, 1, 2.5, -0.8]) 
 
# Calculate the four sub-matrices each of which is 
# a covariance matrices (not necessarily square here) 
# use original x_eq points and data points x

# This submatrix is the same as above
# sqK_xeq_xeq = get_periodic_kernel(x_eq, x_eq, STDDEV, SCALE)

sqK_x_x = get_periodic_kernel(x, x, STDDEV, SCALE)
# These two matrices should be transposes
# though i guess can change depending on the defined kernel matrix
sqK_x_xeq = get_periodic_kernel(x, x_eq, STDDEV, SCALE)
sqK_xeq_x = get_periodic_kernel(x_eq, x, STDDEV, SCALE)
 
# Computations to get new modified mean and sample y values
# using conditional data
inv_sqK_x_x = np.linalg.inv(sqK_x_x)
mean_y_eq = sqK_xeq_x.dot(inv_sqK_x_x).dot(y)
cov_y_eq = sqK_xeq_xeq - sqK_xeq_x.dot(inv_sqK_x_x).dot(sqK_x_xeq)
 
y_eq = np.random.multivariate_normal(mean_y_eq, cov_y_eq, num_fn_samples)
y_eq = y_eq.transpose()
 
plot_GP(num_fn_samples, x_eq, y_eq, mean_y_eq, cov_y_eq, x, y, 'with data, noiseless')


plt.show()





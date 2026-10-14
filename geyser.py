'''
 UCI CS 177: Old Faithful Geyser Data Statistics

 This is a template file for code submission:
 - It begins with an example of how to load the geyser data, plot the data,
   and compute simple statistics of this data.
 - Your task is to fill in the functions compute_variance (part a),
   find_cdf_input (part b), and compute_joint_and_marginal (part c).
 - Do not change the name of the file: it should be submitted as geyser.py
 - Upon submitting your code to Gradescope, an autograder will run and inform
   you if there are any errors compiling your code. It can take a minute, so
   please be patient so you can confirm your code is formatted correctly.
 - Each student should write code independently. Refer to the Canvas page on
   Collaboration and Academic Honesty for more details.

 Description of data:
   Waiting time between eruptions and the duration of the eruption
   for the Old Faithful geyser in Yellowstone National Park, Wyoming, USA.

   eruptions (numeric)  Eruption time in minutes
   waiting   (numeric)  Waiting time to next eruption in minutes

 References:
 - Hardle, W. (1991) Smoothing Techniques with Implementation in S.
   New York: Springer.
 - Azzalini, A. and Bowman, A. W. (1990). A look at some data on the
   Old Faithful geyser. Applied Statistics 39, 357-365.
'''

# Import statements. You may add import statements, but only import from
# numpy, matplotlib, scipy, or math packages.
import numpy as np
import matplotlib.pyplot as plt

# Load data
def load_data():
    with np.load('eruptions.npz') as data:
        return dict(**data)

data = load_data()
S = data['eruptions']  # vector of observed eruption times
T = data['waiting']    # vector of observed waiting times
n = S.shape[0]                # number of observations

# Plot data. This is provided to help you visualize,
# and does not need to be included in your submission.
plt.plot(S, T, 'ok')
plt.xlabel('Eruption Time (minutes)')
plt.ylabel('Waiting Time to Next Eruption (minutes)')
plt.show()

# Compute mean under empirical distribution
meanS = np.sum(S)/n
meanT = np.sum(T)/n

### Part a
def compute_variance(x):
    '''
    Computes the empirical variance of a random variable from samples

    Parameters
    ----------
    x :
        An array of samples of the random variable X
    Returns
    -------
    var :
        The empirical variance of X
    '''

    var = np.var(x)

    return var

print('Part a')
print('Var[S]:', compute_variance(S))
print('Var[T]:', compute_variance(T))

### Part b
def find_cdf_input(x, q):
    '''
    Finds the value such that Fx(value) = q.
    If no such point exists due to multiple data points with repeated values,
    returns the smallest value such that Fx(value) >= q.

    Parameters
    ----------
    x :
        An array of samples of the random variable X
    q :
        The target cdf value
    Returns
    -------
    val :
        The value such that Fx(val) = q

    '''
    val = np.quantile(x, q, method = 'inverted_cdf')
    return val

print('\nPart b')
print('Fs=0.25:', find_cdf_input(S, 0.25), '\t', end=' ')
print('Fs=0.5:', find_cdf_input(S, 0.5), '\t', end=' ')
print('Fs=0.75:', find_cdf_input(S, 0.75))

print('Ft=0.25:', find_cdf_input(T, 0.25), '\t', end=' ')
print('Ft=0.5:', find_cdf_input(T, 0.5), '\t', end=' ')
print('Ft=0.75:', find_cdf_input(T, 0.75))

### Part c
def compute_joint_and_marginal(S, T, threshX=3.5, threshY=70):
    '''
    Thresholds the empirical data in S,T to produce binary random variables X,Y.
    Then computes the joint and marginal probability mass functions of X and Y.

    Parameters
    ----------
    S :
        An array of samples of the random variable S
    T :
        An array of samples of the random variable T
    threshX :
        A value at which to threshold S, so that X = 1 if S > threshX
    threshY :
        A value at which to threshold T, so that Y = 1 if T > threshY
    Returns
    -------
    p_xy :
        A 2x2 numpy array for the joint p(x,y)
    p_x :
        A numpy array of length 2 for the marginal p(x)
    p_y :
        A numpy array of length 2 for the marginal p(y)
    '''

    n = S.shape[0]

    X = (S > threshX).astype(int)
    Y = (T > threshY).astype(int)

    p_xy = np.zeros((2,2))

    for x in (0,1):
        for y in (0,1):
            p_xy[x,y] = np.sum((X == x) & (Y ==y)) / n

    p_x = np.sum(p_xy, axis = 1)
    p_y = np.sum(p_xy, axis = 0)
    return p_xy, p_x, p_y

p_xy, p_x, p_y = compute_joint_and_marginal(S, T)
print('\nPart c')
print('Joint probability of X and Y: p(x,y)')
print(p_xy)
print('Marginal probability of X: p(x)')
print(p_x)
print('Marginal probability of Y: p(y)')
print(p_y)

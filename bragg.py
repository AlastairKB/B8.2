import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import itertools
import scipy
from scipy import constants
from scipy.stats import logistic
from scipy import ndimage as ndi
import scipy.signal as signal
import math
import scipy.misc as misc

def find_spectral_line(image,n):
    """
    Inputs:
        Image: The image in which we are looking for the spectral line
        n: The number of points to return on the spectral line

    Output: 
        x: x-coordinates of the points (array with length n)
        y: y-coordinates of the points (array with length y)
    """

    plt.figure(figsize=(16,14))
    plt.imshow(image,cmap='flag')
    plt.colorbar()
    points = plt.ginput(n=2,show_clicks=True) #Allow user to select two points
    plt.close()
    
    xmin = math.floor(points[0][0])
    xmax = math.floor(points[1][0])
    
    sample = ndi.gaussian_filter(image,sigma=5)

    x = [] #x co-ordinates of the line
    y = [] #y coordinates of the line

    for i in range(n):
        y_width = math.floor(sample.shape[0]/n)
        ymin = math.floor(i*y_width)
        ymax = math.floor((i+1)*y_width)
        y.append((ymax+ymin)/2)
        lineout = np.sum(sample[ymin:ymax,xmin:xmax],axis=0)
        x.append(xmin+np.argmax(lineout))   
    return x,y


def remove_outlier(x):
    """
    This function takes a set of values, x, and removes the value furthest from its two neighbours
    """
    dx = [0]
    for i in range(1,len(x)-1):
        dx.append(
            (x[i]-x[i-1]) + (x[i]-x[i+1])
        )
    
    dx.append(0)
    dx = np.square(dx)
    #dx is the difference in x value between each data point and its neighbour squared
    
    j = np.argmax(dx)
    x[j] = (x[j-1] + x[j+1])/2
    
    return x
    #Find the most anomalous result and replace it with the average between its two neighbours. 

def find_xmin(y,x):
    """
    This function takes a set of coordinates, x and y, fits them to a quadratic, and returns the coordinates of the minimum of those points
    """
    [a,b,c] = np.polyfit(y,x,deg=2) #x = ay^2 + by + c 
    ymin = -b/(2*a)
    xmin = a*ymin**2 + b*ymin + c
    y_c = np.linspace(np.min(y),np.max(y),100)
    x_c = a*y_c**2 + b*y_c + c
    return ymin,xmin
    
#This anonymous function takes E in eV and returns theta in radians
E_to_theta = lambda two_d, E: np.arcsin((constants.h*constants.c)/(two_d*E*constants.e))

#This anonymous function takes theta and returns E in eV
theta_to_E = lambda two_d, theta: (two_d*constants.e*np.sin(theta))/(constants.h*constants*c)

def find_x0(alpha,Pz,theta,xmin):
    x_calculated = x_of_y(alpha,theta,Pz,y=0,y0=0,x0=0)
    x0 = xmin - x_calculated
    #so for example if x_calculated is 500, but the one you measure is 600, then x0 must be at 100 = 600-500
    return x0

def Pz_of_alpha(alpha,theta,delta_xmin):
    """
    Input: 
        alpha: see diagram for this angle
        theta: Bragg angle for the spectral lines
        delta_xmin: distance between the two spectral lines
    Output:
        Pz: perpendicular distance between CCD and the crystal used. 
    """
    a0 = np.sin(alpha)**2-np.sin(theta[0])**2
    a1 = np.sin(alpha)**2-np.sin(theta[1])**2
    b0 = np.sin(2*theta[0])-np.sin(2*alpha)
    b1 = np.sin(2*theta[1])-np.sin(2*alpha)
    
    Pz = 2*delta_xmin*a0*a1/(a1*b0-a0*b1)
    return abs(Pz)


def x_of_y(alpha,theta,Pz,y,y0,x0):
    #Return x given alpha, theta, Pz, y. 
    #If y is a numpy array, it will return a numpy array. 
    y = np.array(y) - y0
    det = 2*np.sin(theta)*np.sqrt((Pz*np.cos(theta))**2+y**2*(np.sin(alpha)**2-np.sin(theta)**2))
    x = (-Pz*np.sin(2*alpha) + det)/(2*(np.sin(alpha)**2 - np.sin(theta)**2)) + x0    
    return x


def plot_initial_guess(image,E1,E2,x,y,E,two_d):
    """
    Inputs:
        Image: The image we are looking at
        E1: Lower bound of the energy range we are looking at
        E2: Upper bound of the energy range we are looking at 
        E.g. for the original data we were looking at 1100-1600eV
        x: the x coordinates of points on the lines
        y: the y coordinates of the points on the lines
        E: energies of the lines [Energy of line 1, Energy of line 2, ...]
        two_d: from braggs law
    Outputs:
        alpha, Pz, x0, y0 (See diagram)
        For initial guess we assume angle gamma=0
    """

    theta1 = E_to_theta(two_d,E1)
    theta2 = E_to_theta(two_d,E2)
    alpha = np.pi/2 -theta1/2 -theta2/2 #Initial estimate of alpha 

    E = np.array(E)
    theta = E_to_theta(two_d,E)
    
    ymin0,xmin0 = find_xmin(y[0],x[0])
    ymin1,xmin1 = find_xmin(y[1],x[1])
    delta_xmin = xmin1-xmin0
    Pz = Pz_of_alpha(alpha,theta,delta_xmin)
    
    y0=(ymin1+ymin0)/2
    x0=find_x0(alpha,Pz,theta[0],xmin0)
    
    plt.figure(figsize=(16,14))
    plt.imshow(image,cmap='binary')
    plt.colorbar()
    colors = ['red','green','blue','aqua','magenta']
    y_c = np.linspace(0,np.shape(image)[0],500) #y calculated
    for i in range(len(E)):
        plt.scatter(x[i],y[i],color = colors[i%5],marker='x')
        x_c = x_of_y(alpha,theta[i],Pz,y_c,y0,x0)
        plt.plot(x_c,y_c,color = colors[i%5])
    plt.show()
    
    print(f"alpha: {alpha}")
    print(f"Pz: {Pz}")
    print(f"x0: {x0}")
    print(f"y0: {y0}")
    return [alpha, Pz, x0, y0, 0],theta



def residuals(params,x,y,theta):
    """
    Input:
        params = [alpha,Pz,x0,y0]
        x: the x coordinates of points on the lines
        y: the y coordinates of the points on the lines
        theta: bragg angles of the lines   

    Output: Residuals
    """
    alpha = params[0]
    Pz = params[1]
    x0 = params[2]
    y0 = params[3]
    gamma = params[4]

    x = np.array(x)
    y = np.array(y)
    x = x*np.cos(gamma) + y*np.sin(gamma)
    y = y*np.cos(gamma) - x*np.sin(gamma)
    
    residuals = np.array([])
    for i in range(len(theta)):
        yi = np.array(y[i])
        xi = np.array(x[i])
        theta_i = theta[i]
        residual_i = xi - x_of_y(alpha,theta_i,Pz,yi,y0,x0)
        residuals = np.append(residuals,residual_i)

    return residuals

def fit_params(initial_guess,x,y,theta,dims):
    """
    This function uses scipy's built in least squares optimizer to fit the parameters to the curve. 
    Inputs:
        initial_guess: Initial guess of parameters, which we get from plot_initial_guess
        x: Array of the x values of each spectral line, [x1, x2, x3] where each xi is a list of x values for that line
        y: Array of the y values of each spectral line, [y1, y2, y3] where each yi is the list of y values for that line
        E: Energies of the two spectral lines 
    Outputs:
        Fitted parameters: [alpha, Pz, x0, y0, gamma]
    """
    alpha = initial_guess[0]
    Pz = initial_guess[1]
    bounds = ([alpha-0.5,Pz*0.5,0,0,-0.1],[alpha+0.5,Pz*1.5,dims[1],dims[0],0.1])
    opt = scipy.optimize.least_squares(residuals,x0=initial_guess,args=(x,y,theta),bounds=bounds)
    print(opt.message)

    loss = np.sum(np.square(opt.fun))
    print(f"loss: {loss}")
    return opt.x


def plot_fitted_params(fitted_params,x,y,theta,dims):
    alpha = fitted_params[0]
    Pz = fitted_params[1]
    x0 = fitted_params[2]
    y0 = fitted_params[3]
    gamma = fitted_params[4]


    colors = ['red','green','blue']
    y_c = np.linspace(0,dims[0],500) #y calculated
    for i in range(len(theta)):
        plt.figure(i,figsize=(12,14))
        plt.scatter(x[i],y[i],color = colors[i%3],marker='x')
        x_c = x_of_y(alpha,theta[i],Pz,y_c,y0,x0)
        x_c = x_c*np.cos(gamma) - y_c*np.sin(gamma)
        y_c = y_c*np.cos(gamma) + x_c*np.sin(gamma)
        plt.plot(x_c,y_c,color = colors[i%3])
        plt.title(f"line {i+1}")
    plt.show()
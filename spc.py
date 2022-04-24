from xml.etree.ElementTree import TreeBuilder
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
from scipy import constants as cs
import math


def remove_background_variation(image,axis=1):
    #For the real data we are looking at, axis=1
    lineout = np.sum(image,axis=axis)
    length = np.shape(image)[axis]
    
    #A profile of how the background varies with the given axis (x or y axis)
    lineout = np.divide(lineout,length)
    lineout = ndi.gaussian_filter1d(lineout,sigma=5)
    
    #The matrix we will subtract from the original image
    correction = np.full((2048,2048),lineout)
    correction = np.transpose(correction)
    
    corrected_image = np.subtract(image,correction)
    return corrected_image

def return_threshold(image):
    return 2*np.median(image) - np.min(image)

theta_to_E = lambda two_d, sin_theta: cs.h*cs.c/(two_d*cs.e*sin_theta)
E_to_theta = lambda two_d, E: np.arcsin((constants.h*constants.c)/(two_d*E*constants.e))


def sum_by_energy(energy_matrix,bins,image,offset=0,normalise=True):
    """
    Inputs:
        energy_matrix: matrix that gives an energy for each pixel
        image: The image we are analysing
        N: the number of bins we are using 
        Normalise: If normalise = True then we return average pixel value instead of sum of pixel values
    Outputs: 
        bins: These are the energy bins we are looking at 
        spectrum: This is the total ADU value of pixels in each energy bin
    """
    spectrum = [] #This will be a list of total ADUs attached to each energy. 
    image = image - float(offset)
    for i in range(1,len(bins)):
        low = bins[i-1]
        high = bins[i]
        coords = np.nonzero(np.logical_and(energy_matrix>=low,energy_matrix<high))
        if normalise==True:
            intensity = np.average(image[coords])
        else:
            intensity = np.sum(image[coords])
        spectrum = np.append(spectrum,intensity)
    return spectrum

def return_energy_matrix(two_d,size,x0,y0,gamma,alpha,Pz,N,return_normalisations=False):
    """
    Inputs: 
        two_d: spacing for Braggs law
        size: size of the matrix
        x0, y0, gamma, alpha, Pz: Fitted parameters from earlier
        N: Number of bins we want to use
        return_normalisations: Returns array you can divide by to normalise to photons per pixel, or to a.u.
    Outputs:
        energy_matrix: Matrix that has a value of energy for every pixel
        bins: used for photons into different energies later
        pixels_by_energy: Number of pixels on the image for each energy bin
        norm_to_au: divide your spectrum of total photon counts in each bin by this to normalise to a.u.
    """
    coordinates = np.mgrid[0:size[0],0:size[1]]
    x = coordinates[1]
    y = coordinates[0]
    x_r = (x-x0)*np.cos(gamma) + (y-y0)*np.sin(gamma)
    y_r = -(x-x0)*np.sin(gamma) + (y-y0)*np.cos(gamma)
    sin_theta = (x_r*np.sin(alpha)+Pz*np.cos(alpha))/np.sqrt(x_r**2 + y_r**2 + Pz**2) #Return sin(theta) for each pixel
    energy_matrix = theta_to_E(two_d,sin_theta) #Matrix who's values are energies of a photon that would hit that pixel
    Emin = np.min(energy_matrix)
    Emax = np.max(energy_matrix)
    bins = np.linspace(Emin,Emax,N)

    if return_normalisations==True:
        pixels_per_bin = np.histogram(energy_matrix,bins=bins)[0] #The number of pixels in each energy bin 
        solid_angle_matrix = Pz/((x_r**2+y_r**2+Pz**2)**(3/2))
        solid_angle_by_energy = sum_by_energy(energy_matrix,bins,solid_angle_matrix,normalise=False)
        dtheta = -np.diff(np.arcsin(E_to_theta(two_d,bins))) #dtheta for each bin 

        sin_theta = E_to_theta(two_d,bins+bins[1]-bins[0]) #Give the value for theta at the middle of each bin 
        sin_theta = sin_theta[:-1] #exclude the last element
        norm_to_au = solid_angle_by_energy/(sin_theta*dtheta) #Divide by this to normalise to a.u. 

        return energy_matrix,bins,pixels_per_bin,norm_to_au 
    else:
        return energy_matrix,bins


def return_spc(image,energy_matrix,bins,normalization=None):
    """
    Inputs: 
        image: image we are analysing 
        threshold: The threshold ADU number we use to detect a photon
    Outputs: 
        bins: These are the energy bins we are looking at 
        photon_count: This is the number of photons we have found in each bin
    """
    threshold = 2*np.median(image) - np.min(image) #Threshold we use for single photon counting

    def filter_func(x):
        if np.sum(x<x[4])==8 and x[4]>threshold: #If the middle pixel is higher than the threshold and highest out of all 9
            return np.sum(x)
        else:
            return 0
    
    photon_ADUs = ndi.generic_filter(image,filter_func,size=(3,3)) #this is a matrix that gives the total ADU value where there is a photon. 
    photon_energies = energy_matrix[photon_ADUs>0] 
    
    spectrum = np.histogram(photon_energies,bins=bins)[0]
    if normalization is None:
        return spectrum
    else:
        spectrum = spectrum/normalization
        return spectrum

def combine_spectra(ADU_spectrum,spc_spectrum,pixels_per_bin,bins,plot=False,percentiles=[5,95]):
    """
    Inputs:
        ADU_spectrum: Total ADU in each bin
        return_spc: Total number of single photons found in each bin 
        normalization: array used to normalise spectrum later 
    Outputs:
        combined_spectrum: Total number of single photons found in each bin, taking into account sum ADU data 
    """
    avg_ADU = ADU_spectrum/pixels_per_bin
    avg_spc_x_E = bins[1:]*spc_spectrum/pixels_per_bin
    #For the polynomial fit it is advantageous to only consider the middle 80% of points.
    #I.e. ignoring the regime of high multi-photon events and low photon events. 
    plt.figure(figsize=(8,8),dpi=300)
    plt.scatter(avg_ADU,avg_spc_x_E,s=3)
    plt.xlabel("Average ADU per pixel")
    plt.ylabel("Photon count per pixel times energy")
    plt.title("Photon count per pixel times energy against average ADU per pixel")
    plt.savefig("avg_ADU_vs_avg_spc_x_E")
    coords = plt.ginput(n=2,timeout=-1)
    plt.close()
    
    b = (coords[0][1]-coords[1][1])/(coords[0][0]-coords[1][0])
    a = coords[0][1]-b*coords[0][0]
    
    combined_spectrum = pixels_per_bin*(a+b*avg_ADU)/bins[1:]
    percent_uncertainty = 1/np.sqrt(combined_spectrum)
        
    zeropoint = -a/b
    ADU_per_photon_per_eV = b
    print(f"Zeropoint: {zeropoint}")
    print(f"ADU per photon per eV: {ADU_per_photon_per_eV}")
    plt.figure(figsize=(14,14))
    plt.plot(avg_ADU,a+b*avg_ADU)
    plt.scatter(avg_ADU,avg_spc_x_E)
    plt.xlabel("Average ADU per pixel")
    plt.ylabel("Photon count times photon energy per pixel")
    
    return combined_spectrum,percent_uncertainty





###############################################################
#Beyond this point is all generating synthetic data
###############################################################


def E_to_probability(E,spec_params,Pmax):
    """
    Inputs: 
        E: array of energies we want to turn into probanilities
        spec_params: array of values describing the bremmstrahlung and the shapes of the two spectral lines. These parameters include:
            E: array of energies we want to turn into probanilities
            Tc: Temperature of electrons in eV
            Z: average charge of ions
            ne: Density of electrons
            E_lines: Photon energy for each of the spectral lines
            I_lines: Intensity of each of the spectral lines
        Pmax: The maximum probability that a given pixel in the most intense region is a photon hit
    """
    [ne,Z,Te,E_lines,I_lines,sigma,gamma] = spec_params
    Te = Te * cs.e #Convert temperature from eV to joules 
    wavelength = cs.h*cs.c/cs.e #Convert energy to wavelength
    wavelength = wavelength/E
    emissivity = (cs.e**2/4*cs.pi*cs.epsilon_0)**3
    emissivity = emissivity * ne**2 * 16*cs.pi*Z * np.exp(-2*cs.pi*cs.hbar*cs.c/(Te*wavelength))
    emissivity = emissivity/(3*cs.m_e*cs.c**2 * np.sqrt(6*cs.pi*Te*cs.m_e) * wavelength**2) #Calculate emmissivity
    
    for idx,line_energy in enumerate(E_lines):
        emissivity += I_lines[idx]*scipy.special.voigt_profile(E-line_energy,sigma,gamma)
    #adding on the emissivity 
    
    probability = emissivity * Pmax/np.max(emissivity)    
    return probability


def synth_data_gen(energy_matrix,spec_params,Pmax,m,e):
    """
    Inputs:
        energy_matrix: matrix assigning an energy to each pixel on the image
        spec_params: array of values describing the bremmstrahlung and the shapes of the two spectral lines. 
        Pmax: The maximum probability that a given pixel in the most intense region is a photon hit
        m: ADU value for major detection (minus background)
        e: ADU value for edge pixel (minus background)
    Output:
        Matrix of synthetic data 

    """
    size = energy_matrix.shape
    background = np.random.normal(loc = 70, scale = 20,size = size)
    rng = np.random.default_rng()
    random_floats = rng.random(size=size) #Produce a random array of floats between 0 and 1
    probability_matrix = E_to_probability(energy_matrix,spec_params,Pmax)
    photon_hits = np.heaviside(probability_matrix-random_floats,1)
    photon_layer = np.zeros(shape=size) #This is ultimately the layer that we will put on top of the noise layer
    
    photon_hit_shapes = np.array([
        [[m,0],
         [0,0]],
        [[m,e],
         [0,0]],
        [[m,e],
         [e,0]],
        [[m,e],
         [e,e]]
    ])
    
    for hit_shape in photon_hit_shapes:
        for i in range(4):
            #Generate new random floats for each one
            random_floats = rng.random(size=size)
            photon_hits = np.heaviside(probability_matrix-random_floats,1)
            hit_shape = np.rot90(hit_shape)
            photon_layer = photon_layer + signal.convolve2d(photon_hits,hit_shape,mode='same')
    return photon_layer+background


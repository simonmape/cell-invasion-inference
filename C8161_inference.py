import skimage
from skimage import io
from skimage import filters
from skimage.measure import block_reduce
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from findiff import Gradient, Divergence, Laplacian, Curl
import tempfile
from pyabc import ABCSMC, RV, Distribution, LocalTransition, QuantileEpsilon, History
from pyabc.visualization import plot_data_callback, plot_kde_2d
from pyabc.sampler import SingleCoreSampler
import pandas as pd
from tqdm import tqdm

#Configure main hyperparameters here
#Number of points in either x or y direction
numX = numY = 500
dx = dy = 1.196
dt= 0.5
grad = Gradient(h=[dx, dy])
div = Divergence(h=[dx, dy])

def makeStack(knockdown):
    #Create empty list to contain the stack 
    stack = []
    #Extract locations associated with knockdown
    locations = wellnameDict[knockdown]
    #Loop over locations to find the stack values
    for idx, location in enumerate(locations):
        #Find file name based on location
        #And downsample immediately while reading the file in the x- and y-
        #coordinate directions
        im = io.imread(baseDir+'00'+location+'-stitched.tif')[:, ::7, ::7,:].copy()
        #Find dimensions of the tif file (number of time points and channels)
        numT = im.shape[0]
        numZ = im.shape[3]
        #Resize the image into the desired shape
        im = resize(im, (numT, numX, numY, numZ))
        #Create array containing the densities
        densities = np.zeros((numT,int(numX/10),int(numY/10)))
        #Loop over times to smooth the resulting densities with
        #a Gaussian kernel
        for t in range(numT):
            dens = filters.gaussian(im[t], sigma=10, mode='nearest',multichannel=True)
            densities[t,:,:] = block_reduce(dens[:,:,0] + dens[:,:,1],(10,10))
        #Append densities to stack
        stack.append(densities)
    return stack
    
    #Main function to compute numerical LHS of PDE
def compute_LHS(stack):
    #Find number of densities in this stack
    numDensities = len(stack)
    #Find number of time points required
    numT = stack[0].shape[0]
    #Create empty LHS list
    LHS = []
    for i in range(numDensities):
        #Create LHS for all 0<t<T_max
        timeDer= np.zeros((numT-2,int(numX/10),int(numY/10)))
        #Extract densities at this stack index
        dens = stack[i]
        #Estimate temporal derivative for each such t
        for t in range(numT-2):
            timeDer[t] = (dens[t+1]-dens[t])/dt
        LHS.append(timeDer)
    return LHS

#Main function to compute numerical RHS of PDE
def compute_RHS(m,p,gm,gp,K):
    #Find number of densities in this stack
    numDensities = len(stack)
    #Store distances in this array
    dist = np.zeros(numDensities)
    #Find number of time points required
    numT = stack[0].shape[0] -2
    #Iterate over the different densities in the stack
    for i in range(numDensities):
        #Create RHS for all 0<t<T_max
        spaceDer= np.zeros((numT,int(numX/10),int(numY/10)))
        #Extract densities at this stack index
        dens = stack[i]
        #Loop over times to compute the RHS at each time
        for t in range(numT-2):
            u = dens[t]
            sat = u/(K+u)
            #Compute proliferation term
            proliferation = p*u*(1-2*gp*sat)*(1-u/K)
            #Compute advection term
            adv = m*(1-2*gm*(u/K))*grad(u)
            #Take divergence to get diffusion term
            diffusion = div(adv)
            spaceDer[t] = diffusion + proliferation
        #Compute distance with respect to the numerical time derivative
        lhs = LHS[i] #Time derivative of the same density in the stack
        dist[i] = (1/numT)*sum([np.linalg.norm(lhs[i]-spaceDer[i])/np.linalg.norm(lhs[i]) for i in range(numT)]) 
    #Return average distance for all the observations in the stack
    return dist.mean()
    
#Set up the main ABC-SMC implementation details
#Function to wrap distance computation between pyABC and
#our implementation
def distance(a,b):
    return abs(a["data"]-b["data"])
#Make wrapper function for pyABC simulator
def simulate(parameter):
    res = compute_RHS(**parameter)
    return {"data": res}
#Set path to temporary location
db_path = "sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db")
#Set prior distribution
m_min, m_max = 0, 5
p_min, p_max = 0, 0.1
K_min, K_max = 0, 15
gm_min, gm_max = -2, 2
gp_min, gp_max = -2, 2

prior = Distribution(
    m=RV("uniform", m_min, m_max - m_min),
    p=RV("uniform", p_min, p_max - p_min),
    K=RV("uniform", K_min, K_max - K_min),
    gm=RV("uniform", gm_min, gm_max - gm_min),
    gp=RV("uniform", gp_min, gp_max - gp_min),
)
#Set up ABC-SMC simulation
abc = ABCSMC(
    models=simulate,
    parameter_priors=prior,
    distance_function=distance,
    population_size=500,
    sampler =  SingleCoreSampler(),
    eps= QuantileEpsilon(alpha=0.25)
)

#RUN ABC-SMC FOR PLATE 1, CELL LINE C8161
baseDir = '/media/martinaperez/Elements/kulesa_screen/20210917_siRNA_C8161_plate1/migration_day1_stitched/'
#Read CSV file with the wellmaps
wellmap_plate1 = pd.read_csv('HT_ABC/wellmap_plate1.csv')
#Extract the name of the knockdown
#this is the first part of the string before a white space or a _
wellmap_plate1['siRNA'] = wellmap_plate1['siRNA'].str.replace('_',' ').str.split().apply(pd.Series)[0]
#Extract the names of the knockdowns
knockdowns = wellmap_plate1['siRNA'].value_counts().index.tolist()
#Make dictionary containing names of knockdowns and their corresponding
#well locations
wellnameDict = {}
for knockdown in knockdowns:
    #Extract the data for this specific knockdown
    df = wellmap_plate1[wellmap_plate1['siRNA'] ==knockdown]
    wellnames = []
    for index, row in df.iterrows():
        WellName = str(row['WellName'])
        wellnames.append(WellName)
    wellnameDict[knockdown] = wellnames
    
#Create empty list to record the ABC-SMC posterior 
#distributions and the parameter weights at the
#last generation
for k, knockdown in enumerate(knockdowns):
    print('Started knockdown ', knockdown, '(', k,' out of', len(knockdowns),')')
    #Make stack for this particular knockdown
    stack = makeStack(knockdown)
    #Compute left hand side for this knockdown
    LHS = compute_LHS(stack)
    #With stack and LHS stack, perform ABC
    abc.new(db_path, {"data": 0})
    history = abc.run(max_nr_populations=10, minimum_epsilon=0)
    #Save posterior distribution for this knockdown to csv file
    history.get_distribution()[0].to_csv('C8161_'+knockdown+'_kd_posterior.csv')
    #Save posterior distribution weights for this knockdown to np array
    np.savetxt('C8161_'+knockdown+'_kd_posterior.csv',history.get_distribution()[1])
    
#RUN ABC-SMC FOR PLATE 2, CELL LINE C8161
baseDir = '/media/martinaperez/Elements/kulesa_screen/20210923_siRNA_C8161_plate2/Images_stitched/'
#Read CSV file with the wellmaps
wellmap_plate2 = pd.read_csv('HT_ABC/wellmap_plate2.csv')
#Extract the name of the knockdown
#this is the first part of the string before a white space or a _
wellmap_plate2['siRNA'] = wellmap_plate2['siRNA'].str.replace('_',' ').str.split().apply(pd.Series)[0]
#Extract the names of the knockdowns
knockdowns = wellmap_plate2['siRNA'].value_counts().index.tolist()
#Make dictionary containing names of knockdowns and their corresponding
#well locations
wellnameDict = {}
for knockdown in knockdowns:
    #Extract the data for this specific knockdown
    df = wellmap_plate2[wellmap_plate1['siRNA'] == knockdown]
    wellnames = []
    for index, row in df.iterrows():
        WellName = str(row['WellName'])
        wellnames.append(WellName)
    wellnameDict[knockdown] = wellnames
    
#Create empty list to record the ABC-SMC posterior 
#distributions and the parameter weights at the
#last generation
for k, knockdown in enumerate(knockdowns):
    print('Started knockdown ', knockdown, '(', k,' out of', len(knockdowns),')')
    #Make stack for this particular knockdown
    stack = makeStack(knockdown)
    #Compute left hand side for this knockdown
    LHS = compute_LHS(stack)
    #With stack and LHS stack, perform ABC
    abc.new(db_path, {"data": 0})
    history = abc.run(max_nr_populations=10, minimum_epsilon=0)
    #Save posterior distribution for this knockdown to csv file
    history.get_distribution()[0].to_csv('C8161_'+knockdown+'_kd_posterior.csv')
    #Save posterior distribution weights for this knockdown to np array
    np.savetxt(history.get_distribution()[1],'C8161_'+knockdown+'_kd_posterior.csv')

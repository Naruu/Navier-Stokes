import time
from math import floor, sqrt
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import medfilt2d

# Estimates T1 together with:
#   RD-NLS: a and b parameters to fit the data to a + b*exp(-TI/T1)
#   RD-NLS-PR: ra and rb parameters to fit the data to |ra + rb*exp(-TI/T1)|                                                  
#
# written by J. Barral, M. Etezadi-Amoli, E. Gudmundson, and N. Stikov, 2009
#  (c) Board of Trustees, Leland Stanford Junior University

# yck, 2019-11-07
# translate matlab to pythoh. hll, 2019-11-09



def getNLSStruct_v2(*args,**kwargs):
    """
    nlsS = getNLSStruct( extra, dispOn, zoom)

    extra.tVec    : defining TIs 
                (not called TIVec because it looks too much like T1Vec)
    extra.T1Vec   : defining T1s
    dispOn        : 1 - display the struct at the end
                    0 (or omitted) - no display
    zoom          : 1 (or omitted) - do a non-zoomed search
                    x>1 - do an iterative zoomed search (x-1) times (slower search)
                    NOTE: When zooming in, convergence is not guaranteed.

    Data Model    : a + b*exp(-TI/T1) 
    
    written by J. Barral, M. Etezadi-Amoli, E. Gudmundson, and N. Stikov, 2009
    (c) Board of Trustees, Leland Stanford Junior University 
    """
    
    extra = args[0]
    if kwargs.get('dispOn'): pass
    else: dispOn = args[1]
    if kwargs.get('zoom'): pass
    else: zoom = args[1]
    
    nlsS = dict()
    nlsS['tVec'] = extra.tVec[:]
    nlsS['N'] = len(nlsS['tVec'])
    nlsS['T1Vec'] = extra.T1Vec[:]
    nlsS['T1Start'] = nlsS['T1Vec'][0]
    nlsS['T1Stop'] = nlsS['T1Vec'][-1]
    nlsS['T1Len'] = len(nlsS['T1Vec'])

    # Set the number of times you zoom the grid search in, 1 = no zoom
    # Setting this greater than 1 will reduce the step size in the grid search
    # (and slow down the fit significantly)

    nargin = len(args) + len(kwargs)
    if nargin < 3:
        nlsS['nbrOfZoom'] = 2
    else :
        nlsS['nbrOfZoom'] = zoom

    if nlsS['nbrOfZoom'] > 1:
        nlsS['T1LenZ'] = 21; #Length of the zoomed search

    # Set the help variables that can be precomputed:
    # alpha is 1/T1,
    # theExp is a matrix of exp(-TI/T1) for different TI and T1,
    # rhoNormVec is a vector containing the norm-squared of rho over TI,
    # where rho = exp(-TI/T1), for different T1's.
 
    alphaVec = 1./nlsS['T1Vec']
    tVec_t = nlsS['tVec'].reshape(nlsS['tVec'].shape[0], -1) # transpose
    tVec_t = tVec_t.astype('float64')
    # matlab vect is colum basen while numpy is just an array.
    nlsS['theExp'] = np.exp((-1)*tVec_t*alphaVec ) # datatype changed to matrix
    nlsS['rhoNormVec'] = sum(np.power(nlsS['theExp'],2), 0) - 1/nlsS['N']*np.power(sum(nlsS['theExp'],0),2)
    
    if dispOn:
        # Display the structure for inspection
        print(nlsS)
    return nlsS


# start here
def rdNlsPr_v2(data, nlsS):
    """
    [T1Est, bMagEst, aMagEst, res] = rdNlsPr(data, nlsS)
    Finds estimates of T1, |a|, and |b| using a nonlinear least
    squares approach together with polarity restoration. 
    The model +-|ra + rb*exp(-t/T1)| is used. 
    The residual is the rms error between the data and the fit. 

    INPUT:
    data - the absolute data to estimate from
    nlsS - struct containing the NLS search parameters and
            the data model to use
        
    written by J. Barral, M. Etezadi-Amoli, E. Gudmundson, and N. Stikov, 2009
    (c) Board of Trustees, Leland Stanford Junior University 
    """
    if len(data)!= nlsS['N']:
        raise ValueError('nlsS.N and data must be of equal length!')

    # Make sure the data come in increasing TI-order
    tVec = sorted(nlsS['tVec'])
    order = nlsS['tVec'].argsort()

    data = np.squeeze(data)
    data = data[order]

    # Initialize variables
    aEstTmp = np.zeros(2)
    bEstTmp = np.zeros(2)
    T1EstTmp = np.zeros(2)
    resTmp = np.zeros(2)

    # Make sure data vector is a column vector
    data = data.reshape(-1, 1)

    # Find the min of the data
    minVal = np.min(data)
    minInd = np.argmin(data)

    # Fit
    try:
        nbrOfZoom = nlsS['nbrOfZoom']
    except:
        nbrOfZoom = 1 # No zoom


    for ii in range(2):
        theExp = nlsS['theExp'][order, :]

        if ii == 1:
            # First, we set all elements up to and including
            # the smallest element to minus
            dataTmp = np.multiply(data,np.append((-1)*np.ones((minInd+1,1)), np.ones((nlsS['N'] - minInd-1,1))))
            
        elif ii == 2:
            # Second, we set all elements up to (not including)
            # the smallest element to minus
            dataTmp = np.multiply(data, np.append((-1)*np.ones((minInd,1)), np.ones((nlsS['N'] - minInd,1))))


        # The sum of the data
        ySum = sum(dataTmp)

        # Compute the vector of rho'*t for different rho,
        # where rho = exp(-TI/T1) and y = dataTmp
        rhoTyVec = np.matmul(dataTmp.transpose(),theExp) - 1/nlsS['N']*np.sum(theExp,axis=0) * ySum

        # rhoNormVec is a vector containing the norm-squared of rho over TI,
        # where rho = exp(-TI/T1), for different T1's.
        rhoNormVec = nlsS['rhoNormVec']

        #Find the max of the maximizing criterion
        cal = np.divide(np.power(abs(rhoTyVec),2),rhoNormVec)
        tmp = max(cal)
        ind = np.argmax(cal)

        T1Vec = nlsS['T1Vec'] # Initialize the variable
        
        if nbrOfZoom > 1: # Do zoomed search
            try:
              T1LenZ = nlsS['T1LenZ'] # For the zoomed search
            except:
              T1LenZ = 21 # For the zoomed search
            
            for k in range(1,nbrOfZoom):
                if ind > 0 and ind < len(T1Vec)-1:
                    T1Vec = np.linspace(T1Vec[ind-1], T1Vec[ind+1], T1LenZ)
                elif ind == 0:
                    T1Vec = np.linspace(T1Vec[ind], T1Vec[ind+2], T1LenZ)
                else:
                    T1Vec = np.linspace(T1Vec[ind-2], T1Vec[ind], T1LenZ)

                # Update the variables
                alphaVec = 1/T1Vec
                tVec = np.array(tVec)
                theExp = np.exp((-1)*np.matmul(tVec.reshape(-1,1), alphaVec.reshape(1, -1)))
                yExpSum = np.squeeze(np.matmul(dataTmp.reshape(1,-1), theExp))
                rhoNormVec = np.sum(np.power(theExp,2), axis=0) - 1/nlsS['N']*np.power(np.sum(theExp,axis=0),2)
                rhoTyVec = yExpSum - 1/nlsS['N']*np.sum(theExp,axis=0)*ySum

                #Find the max of the maximizing criterion
                tmp = np.max( np.divide(np.power(abs(rhoTyVec),2),rhoNormVec))
                ind = np.argmax( np.divide(np.power(abs(rhoTyVec),2),rhoNormVec))


        # The estimated parameters
        T1EstTmp[ii] = T1Vec[ind]
        bEstTmp[ii] = rhoTyVec[ind]/ rhoNormVec[ind]
        aEstTmp[ii] = 1/nlsS['N']*(ySum - bEstTmp[ii]*np.sum(theExp[:,ind]))

        # Compute the residual
        modelValue = aEstTmp[ii] + bEstTmp[ii]*np.exp((-1)* tVec/T1EstTmp[ii])
        resTmp[ii] = 1/sqrt(nlsS['N']) * norm(1 - modelValue./dataTmp)
    """

    % Finally, we choose the point of sign shift as the point giving
    % the best fit to the data, i.e. the one with the smallest residual   
    [res, ind] = min(resTmp);
    aEst = aEstTmp(ind);
    bEst = bEstTmp(ind);
    T1Est = T1EstTmp(ind);
    """
    return (T1Est, bMagEst, aMagEst, res)


plt.close('all')
fname = 'TestSingleSlice'
method = 'RD-NLS-PR'
saveStr = 'T1Fit'+method+'_'+fname


matContents = loadmat(fname+'.mat',  struct_as_record=False, squeeze_me=True)
extra = matContents['extra']
data = matContents['data']

nlsS = getNLSStruct_v2(extra, 1)

data = abs(data)
nbrOfFitParams = 4 # Number of output arguments for the fit

nbrow, nbcol, _ = data.shape
nbslice = 1

dshape = data.shape
data = data.reshape(dshape[0], dshape[1], 1, -1) # Make data a 4-D array regardless of number of slices

dataOriginal = data

mF = 0.1
maskFactor = mF
mask = np.zeros((nbrow, nbcol, nbslice))

u = max(extra.tVec)
v = np.argmax(extra.tVec)

for kk in range(nbslice):
	maskTmp = mask[:,:,kk] # 3D with one z-index to 2D
	maskTmp = medfilt2d(maskTmp); # remove salt and pepper noise
	maskThreshold = maskFactor*np.amax(abs(data[:,:,kk,v])) # numpy.amax returns the maximum from flattened matrix
	maskTmp[np.where((abs(data[:,:,kk,v])> maskThreshold))] = 1
	mask[:, :, kk] = maskTmp
	del maskTmp


maskInds = np.where(mask > 0)
nVoxAll = len(maskInds[0])
# How many voxels to process before printing out status data
# numVoxelsPerUpdate = min(floor(nVoxAll/10), 1000); 
						   
ll_T1 = np.zeros((nVoxAll, nbrOfFitParams))

# Number of status reports
# nSteps = ceil(nVoxAll/numVoxelsPerUpdate); 
"""
for ii in range(data.shape[3]):
    tmpData = np.zeros((len(maskInds[0]),data.shape[3]))
    tmpVol = data[(]:,:,:,ii]
    tmpData[ii,:] = tmpVol_t[maskInds_t]
"""
tmpData = []
maskInds_t = np.where(mask.transpose() > 0)
for ii in range(data.shape[3]):
    tmpVol_t = data[:,:,:,ii].transpose()
    tmpData.append(np.array(tmpVol_t[maskInds_t][:]))
tmpData = np.array(tmpData)
tmpData = tmpData.transpose()

data = tmpData
del tmpVol_t, tmpData

startTime = time.time()
print('Processing {:d} voxels.\n'.format(nVoxAll))
"""
for jj in range(nVoxAll):
    T1Est, bMagEst, aMagEst, res = rdNlsPr_v2( data[jj, :], nlsS)
    ll_T1[jj, :] = [T1Est, bMagEst, aMagEst, res]

del jj, T1Est, bMagEst, aMagEst, res

timeTaken = floor(cputime - startTime)

print('Processed {:d} voxels in {:d} seconds.'.format(nVoxAll, timeTaken))

dims = [*mask.shape, 4]
im = np.zeros(mask.shape)

for ii in range(nbrOfFitParams):
    im[(maskInds] = ll_T1[(:,ii
    T1[:, :, :, ii] = im
end

% Going back from a numVoxels x 4 array to nbrow x nbcol x nbslice
ll_T1 = T1;

% Store ll_T1 and mask in saveStr
% For the complex data, ll_T1 has four parameters 
% for each voxel, namely:
% (1) T1 
% (2) 'b' or 'rb' parameter 
% (3) 'a' or 'ra' parameter
% (4) residual from the fit

save(saveStr, 'll_T1', 'mask', 'nlsS')

% Check the fit
TI = extra.tVec;
nbtp = 20;
timef = linspace(min(TI), max(TI), nbtp);

% Inserting a short pause, otherwise some computers seem
% to get problems
pause(1);

zz = 0;
while(true)

    zz = input(['Enter 1 to check the fit, 0 for no check --- '], 's');
    zz = cast(str2double(zz), 'int16');
    
    if ( isinteger(zz) && zz >= 0 && zz <= nbslice ) 
        break;
    end
    
end

sliceData = squeeze(dataOriginal(:,:,zz,:));
datafit = zeros(nbrow, nbcol, nbtp);

for kk = 1:nbtp
    datafit(:, :, kk) = abs(ll_T1(:, :, zz, 3) + ll_T1(:,:,zz,2).*exp(-timef(kk)./ll_T1(:,:,zz,1)));
end

disp('Click on one point to check the fit. CTRL-click or right-click when done')

plotData_v2( real(sliceData), TI, real(datafit), ll_T1);

close all
"""
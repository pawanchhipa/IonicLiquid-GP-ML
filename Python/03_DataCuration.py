# -*- coding: utf-8 -*-
"""
03_DataCuration.py

Performs Gaussian Process-based data curation for ionic liquid
thermophysical property datasets using COSMO-RS sigma profiles.

The script has three main objectives:
    - Train a GP model on the full experimental dataset.
    - Identify potentially inconsistent observations using
      prediction errors and predefined thresholds.
    - Export flagged observations, cleaned datasets, and
      diagnostic results.

Sections:
    . Imports
    . Configuration
    . Auxiliary Functions
    . Main Script
    . Plots
    . Outlier Detection

Authors: Pawan Chhipa & Dinis O. Abranches
Last Updated: 2026-10-14

"""

# =============================================================================
# 1. Imports
# =============================================================================

# General
import warnings
import time
import os
from pathlib import Path

# Specific
import numpy
import pandas
from sklearn import metrics
from sklearn import preprocessing
from sklearn import decomposition
import gpflow
from matplotlib import pyplot as plt
import re

# =============================================================================
# 2. Configuration
# =============================================================================

# Repository directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

# Input property dataset
PROPERTY_FILE = "ML-Ready-Pure_ILs_Surface_tension.csv"
dbPath = DATA_DIR / PROPERTY_FILE

# Dataset column layout
# Options: "T_P_Prop", "T_Prop", "Prop"
dataLayout = "T_Prop"

# Feature normalization
# Options: None, "Standardization", "MinMax", "LogStand", "Log+bStand"
featureNorm = "Log+bStand"

# Target normalization
# Options: None, "Standardization", "MinMax", "LogStand", "Log+bStand"
labelNorm = "Standardization"

# Gaussian Process configuration
# Kernels: "RBF", "RQ", "Matern32", "Matern52"
gpConfig = {
    "kernel": "RQ",
    "useWhiteKernel": True,
    "trainLikelihood": True,
    "useSparse": False,
    "nInducing": 3000,
}

# Run mode
# False: Data curation
# True: Verification of cleaned data
isVerification = False


# =============================================================================
# 3. Auxiliary Functions
# =============================================================================

def normalize(inputArray,skScaler=None,method='Standardization',reverse=False):
    """
    normalize() normalizes (or unnormalizes) inputArray using the method
    specified and the skScaler provided.

    Parameters
    ----------
    inputArray : numpy array
        Array to be normalized. If dim>1, array is normalized column-wise.
    skScaler : scikit-learn preprocessing object or None
        Scikit-learn preprocessing object previosly fitted to data. If None,
        the object is fitted to inputArray.
        Default: None
    method : string, optional
        Normalization method to be used.
        Methods available:
            . Standardization - classic standardization, (x-mean(x))/std(x)
            . MinMax - scale to range (0,1)
            . LogStand - standardization on the log of the variable,
                         (log(x)-mean(log(x)))/std(log(x))
            . Log+bStand - standardization on the log of variables that can be
                           zero; uses a small buffer,
                           (log(x+b)-mean(log(x+b)))/std(log(x+b))
        Defalt: 'Standardization'
    reverse : bool
        Whether  to normalize (False) or unnormalize (True) inputArray.
        Defalt: False

    Returns
    -------
    inputArray : numpy array
        Normalized (or unnormalized) version of inputArray.
    skScaler : scikit-learn preprocessing object
        Scikit-learn preprocessing object fitted to inputArray. It is the same
        as the inputted skScaler, if it was provided.

    """
    # If inputArray is a labels vector of size (N,), reshape to (N,1)
    if inputArray.ndim==1:
        inputArray=inputArray.reshape((-1,1))
        warnings.warn('Input to normalize() was of shape (N,). It was assumed'\
                      +' to be a column array and converted to a (N,1) shape.')
    # If skScaler is None, train for the first time
    if skScaler is None:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand': aux=numpy.log(inputArray)
        elif method=='Log+bStand': aux=numpy.log(inputArray+10**-3)
        else: raise ValueError('Could not recognize method in normalize().')
        if method!='MinMax':
            skScaler=preprocessing.StandardScaler().fit(aux)
        else:
            skScaler=preprocessing.MinMaxScaler().fit(aux)
    # Do main operation (normalize or unnormalize)
    if reverse:
        # Rescale the data back to its original distribution
        inputArray=skScaler.inverse_transform(inputArray)
        # Check method
        if method=='LogStand': inputArray=numpy.exp(inputArray)
        elif method=='Log+bStand': inputArray=numpy.exp(inputArray)-10**-3
    elif not reverse:
        # Check method
        if method=='Standardization' or method=='MinMax': aux=inputArray
        elif method=='LogStand': aux=numpy.log(inputArray)
        elif method=='Log+bStand': aux=numpy.log(inputArray+10**-3)
        else: raise ValueError('Could not recognize method in normalize().')
        inputArray=skScaler.transform(aux)
    # Return
    return inputArray,skScaler

def buildGP(X_Train,Y_Train,gpConfig={}):
    """
    buildGP() builds and fits a GP model using the training data provided.

    Parameters
    ----------
    X_Train : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).
    Y_Train : numpy array (N,1)
        Training labels (e.g., property of a given molecule).
    gpConfig : dictionary, optional
        Dictionary containing the configuration of the GP. If a key is not
        present in the dictionary, its default value is used.
        Keys:
            . kernel : string
                Kernel to be used. One of:
                    . 'RBF' - gpflow.kernels.RBF()
                    . 'RQ' - gpflow.kernels.RationalQuadratic()
                    . 'Matern32' - gpflow.kernels.Matern32()
                    . 'Matern52' - gpflow.kernels.Matern52()
                The default is 'RQ'.
            . useWhiteKernel : boolean
                Whether to use a White kernel (gpflow.kernels.White).
                The default is True.
            . trainLikelihood : boolean
                Whether to treat the variance of the likelihood of the modeal
                as a trainable (or fitting) parameter. If False, this value is
                fixed at 10^-5.
                The default is True.
        The default is {}.
    Raises
    ------
    UserWarning
        Warning raised if the optimization (fitting) fails to converge.

    Returns
    -------
    model : gpflow.models.gpr.GPR object
        GP model.

    """
    # Unpack gpConfig
    kernel=gpConfig.get('kernel','RQ')
    useWhiteKernel=gpConfig.get('useWhiteKernel','True')
    trainLikelihood=gpConfig.get('trainLikelihood','True')
    # Select and initialize kernel
    if kernel=='RBF':
        gpKernel=gpflow.kernels.SquaredExponential()
    if kernel=='RQ':
        gpKernel=gpflow.kernels.RationalQuadratic()
    if kernel=='Matern32':
        gpKernel=gpflow.kernels.Matern32()
    if kernel=='Matern52':
        gpKernel=gpflow.kernels.Matern52()
    # Add White kernel
    if useWhiteKernel: gpKernel=gpKernel+gpflow.kernels.White()
    # Build GP model 
    if gpConfig.get('useSparse',False):
        M=min(gpConfig.get('nInducing',800),X_Train.shape[0])
        Z = X_Train[numpy.random.default_rng(42).choice(X_Train.shape[0], M, 
            replace=False)].copy()
        model=gpflow.models.SGPR((X_Train,Y_Train),gpKernel,inducing_variable=Z,
            noise_variance=10**-5)
        gpflow.utilities.set_trainable(model.inducing_variable,False)
    else:
        model=gpflow.models.GPR((X_Train,Y_Train),gpKernel,noise_variance=10**-5)
    # Select whether the likelihood variance is trained
    gpflow.utilities.set_trainable(model.likelihood.variance,trainLikelihood)
    # Build optimizer
    optimizer=gpflow.optimizers.Scipy()
    # Fit GP to training data
    aux=optimizer.minimize(model.training_loss,
                           model.trainable_variables,
                           method='L-BFGS-B')
    # Check convergence
    if aux.success==False:
        warnings.warn('GP optimizer failed to converge.')
    # Output
    return model

def gpPredict(model,X):
    """
    gpPredict() returns the prediction and standard deviation of the GP model
    on the X data provided.

    Parameters
    ----------
    model : gpflow.models.gpr.GPR object
        GP model.
    X : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., sigma profile bins).

    Returns
    -------
    Y : numpy array (N,1)
        GP predictions.
    STD : numpy array (N,1)
        GP standard deviations.

    """
    # Do GP prediction, obtaining mean and variance
    GP_Mean,GP_Var=model.predict_f(X)
    # Convert to numpy
    GP_Mean=GP_Mean.numpy()
    GP_Var=GP_Var.numpy()
    # Prepare outputs
    Y=GP_Mean
    STD=numpy.sqrt(GP_Var)
    # Output
    return Y,STD

def classify_ion(smi):
    charges = re.findall(r'\[([^\]]+)\]', smi)
    pos = sum(t.count('+') for t in charges)
    neg = sum(t.count('-') for t in charges)
    return 'cation' if pos > neg else 'anion' if neg > pos else 'neutral'

def fmtMAE(val):
    if val>=1:      return f'{val:.2f}'
    elif val>=0.01: return f'{val:.4f}'
    else:           return f'{val:.2e}'
    
def get_cation_family(smiles):
    """Detect cation core from IL SMILES without RDKit."""
    frags = smiles.split('.')
    cat = ''
    for f in frags:
        if classify_ion(f.strip()) == 'cation':
            cat = f.strip(); break
    if not cat: return 'Other'
    upper = cat.upper()
    if '[P+]' in cat or '[P+' in cat or '[pH]' in cat: return 'Phosphonium'
    if '[S+]' in cat:  return 'Sulfonium'
    hasRing = any(c.isdigit() for c in cat) or '%' in cat
    nArom = cat.count('n')
    if not hasRing:
        if '[N+]' in cat or '[NH' in cat: return 'Ammonium'
        return 'Other'
    if nArom >= 2: return 'Imidazolium'
    if nArom == 1: return 'Pyridinium'
    if '[N+]' in cat or '[nH]' in cat or 'N' in cat:
        carbons = upper.count('C') - upper.count('CL')
        if 'O' in upper and carbons <= 5: return 'Morpholinium'
        if carbons <= 4: return 'Pyrrolidinium'
        if carbons <= 6: return 'Piperidinium'
    return 'Other'

# =============================================================================
# Main Script
# =============================================================================

# Derive column indices from dataLayout
if dataLayout=='T_P_Prop':
    # [name, smiles, Temp, Pressure, Property, sigma...]
    colTemp=2; colProp=4; colSigmaStart=5
    hasTemp=True; hasPressure=True
elif dataLayout=='T_Prop':
    # [name, smiles, Temp, Property, sigma...]
    colTemp=2; colProp=3; colSigmaStart=4
    hasTemp=True; hasPressure=False
elif dataLayout=='Prop':
    # [name, smiles, Property, sigma...]
    colProp=2; colSigmaStart=3
    hasTemp=False; hasPressure=False
else:
    raise ValueError('dataLayout must be T_P_Prop, T_Prop, or Prop.')

# Number of condition columns (T, T+P, or none) before sigma profiles
nCondCols = colProp - colTemp if hasTemp else 0

# Initiate timer
ti=time.time()
# Load CSV file
mlDB=pandas.read_csv(dbPath)
mlDB=mlDB.dropna()
#mlDB=mlDB.iloc[:2000,:]
# Check and drop if the value of property is 0 or non-finite in log space
Y=numpy.log(mlDB.iloc[:,colProp].to_numpy('double').reshape(-1,1))
mask = numpy.isfinite(Y).reshape(-1)
mlDB = mlDB.loc[mask].reset_index(drop=True)
Y = Y[mask]

# Dataset Summary
cations, anions = set(), set()
for smiles in mlDB['component1_smiles'].dropna():
    for frag in smiles.split('.'):
        t = classify_ion(frag.strip())
        if t == 'cation': cations.add(frag)
        elif t == 'anion': anions.add(frag)

prop_col = mlDB.columns[colProp]
prop_vals = mlDB.iloc[:, colProp]
print('='*52)
print(f'  DATABASE SUMMARY – {prop_col}')
print('='*52)
print(f'  Total rows           : {len(mlDB)}')
print('-'*52)
print(f'  Unique Ionic Liquids : {mlDB["component1"].nunique()}')
print(f'  Unique Cations       : {len(cations)}')
print(f'  Unique Anions        : {len(anions)}')
print('-'*52)
if hasTemp:
    print(f'  Temperature (K): {mlDB.iloc[:,colTemp].min():.1f} – \
          {mlDB.iloc[:,colTemp].max():.1f}')
if hasPressure:
    print(f'  Pressure (kPa) : {mlDB.iloc[:,colTemp+1].min():.1f} – \
          {mlDB.iloc[:,colTemp+1].max():.1f}')
print(f'  {prop_col} Range : {prop_vals.min():.4f} – {prop_vals.max():.4f}')
print('-'*52)

# ---- Full-dataset training (no split) ----
# Extract features and labels from the entire dataset
if hasTemp:
    X_All=numpy.concatenate((mlDB.iloc[:,colTemp:colProp].to_numpy('double'),
                             mlDB.iloc[:,colSigmaStart:].to_numpy('double')),axis=1)
else:
    X_All=mlDB.iloc[:,colSigmaStart:].to_numpy('double')

Y_All=mlDB.iloc[:,colProp].to_numpy('double').reshape(-1,1)
# Get target variable name
varName=mlDB.columns[colProp]
# Clean variable name for display
displayName=varName.replace('_',' ').rsplit(' ',1)
if len(displayName)==2:
    propName=re.sub(r'([a-z])([A-Z])',r'\1 \2',displayName[0].strip()).title()
    units=displayName[1].strip()
    displayLabel=f'{propName} ({units})'
else:
    propName=re.sub(r'([a-z])([A-Z])',r'\1 \2',displayName[0].strip()).title()
    units=''
    displayLabel=propName

# ratio = Y_All.max() / Y_All.min()
if labelNorm=='LogStand': useLogScale=True
else: useLogScale=False


# Normalize
if featureNorm is not None:
    X_All_N,skScaler_X=normalize(X_All,method=featureNorm)
else:
    X_All_N=X_All
if labelNorm is not None:
    Y_All_N,skScaler_Y=normalize(Y_All,method=labelNorm)
else:
    Y_All_N=Y_All
# Train GP on full dataset
model=buildGP(X_All_N,Y_All_N,gpConfig=gpConfig)
# Get GP predictions on full dataset
Y_Pred_N,STD_All=gpPredict(model,X_All_N)

# Unnormalize
if labelNorm is not None:
    Y_Pred,__=normalize(Y_Pred_N,skScaler=skScaler_Y,
                        method=labelNorm,reverse=True)
else:
    Y_Pred=Y_Pred_N

#--- Full dataset sorted by GP uncertainty (worst to best) ---
# Build column list based on layout
outCols=['component1','component1_smiles']
if hasTemp: outCols.append(mlDB.columns[colTemp])
if hasPressure: outCols.append(mlDB.columns[colTemp+1])
outCols.append(mlDB.columns[colProp])

             
# outlier DataFrame: features + prediction + uncertainty + error
outlierDF = mlDB[outCols].copy()
outlierDF['Predicted'] = Y_Pred.reshape(-1)
outlierDF['GP_STD']    = STD_All.reshape(-1)

# Pull arrays for error math
actual    = outlierDF[mlDB.columns[colProp]].values
predicted = outlierDF['Predicted'].values

# Error definition matches the parity plot metric
if useLogScale:
    outlierDF['LogError']    = numpy.log(predicted) - numpy.log(actual)
    outlierDF['AbsLogError'] = numpy.abs(outlierDF['LogError'])
    errName = 'AbsLogError'
else:
    outlierDF['Error']    = predicted - actual
    outlierDF['AbsError'] = numpy.abs(outlierDF['Error'])
    errName = 'AbsError'
# Sort worst → best

# ---- Outlier flagging & file saving ----
# Iteration suffix from kernel state (Stage 1: no WK; Stage 2: WK + train likelihood)
iterTag = '_iter2' if gpConfig.get('useWhiteKernel', False) else '_iter1'

# Ensure APE% exists for non-log regimes, then pick threshold by errName
if errName != 'AbsLogError':
    if 'APE%' not in outlierDF.columns:
        outlierDF['APE%'] = numpy.abs((actual - predicted) / actual) * 100
    flagCol, threshold = 'APE%', 5.0
else:
    flagCol, threshold = 'AbsLogError', 1.0
    
outlierDF = outlierDF.sort_values(flagCol, ascending=False).reset_index(drop=True)

outlierMask = outlierDF[flagCol].values > threshold
outliersOnly = outlierDF[outlierMask].reset_index(drop=True)

# Cleaned input: original mlDB minus the outlier rows (keep full columns).
# outlierDF row order was scrambled by sort; map back via the (component1, T, P, prop)
# tuple used in outCols. Build a join key from outCols on both sides.
keyCols = outCols  # ['component1','component1_smiles', (T), (P), prop]
outlierKey = set(map(tuple, outliersOnly[keyCols].to_numpy().tolist()))
mlDBkey = list(map(tuple, mlDB[keyCols].to_numpy().tolist()))
keepMask = numpy.array([k not in outlierKey for k in mlDBkey])
cleanedDB = mlDB[keepMask].reset_index(drop=True)

# Output paths: one flat folder per property, alongside the ORIGINAL input file
# (not nested inside whatever subfolder the cleaned file lives in)
baseName = os.path.splitext(os.path.basename(dbPath))[0]

# Stem extraction:
#  (a) dbPath points to the original CSV → stem from filename
#  (b) dbPath points to a cleaned file inside Curation_<stem>/ → stem from
#      the folder name (filename here is just 'iter1_cleaned' or similar)
parentDir = os.path.dirname(os.path.abspath(dbPath))
parentFolder = os.path.basename(parentDir)
if parentFolder.startswith('Curation_'):
    stemName = parentFolder[len('Curation_'):]
else:
    # Strip any legacy suffix (older naming used <stem>_cleaned_iterN.csv)
    stemMatch = re.match(
        r'(.*?)(_cleaned_iter\d+|_outliers_iter\d+|_sorted_iter\d+)?$',
        baseName)
    stemName = stemMatch.group(1) if stemMatch else baseName

# Anchor: walk up from dbPath until we leave any 'Curation_*' folder so the
# new outputs land back in the parent Databases folder (no re-nesting).
anchorDir = parentDir
while os.path.basename(anchorDir).startswith('Curation_'):
    anchorDir = os.path.dirname(anchorDir)

curationDir = os.path.join(anchorDir, f'Curation_{stemName}')
os.makedirs(curationDir, exist_ok=True)

prefix = f'verification_{iterTag[1:]}' if isVerification else iterTag[1:]
outlierPath = os.path.join(curationDir, f'{prefix}_outliers.csv')
cleanedPath = os.path.join(curationDir, f'{prefix}_cleaned.csv')
sortedPath  = os.path.join(curationDir, f'{prefix}_sorted.csv')

outliersOnly.to_csv(outlierPath, index=False)
cleanedDB.to_csv(cleanedPath, index=False)
outlierDF.to_csv(sortedPath, index=False)

print('-'*52)
print(f'  Mode               : {"VERIFICATION" if isVerification else "CURATION"}')
print(f'  Curation ({iterTag[1:]}): {flagCol} > {threshold}')
print(f'  Outliers flagged    : {len(outliersOnly)} / {len(mlDB)} rows')
print(f'  Cleaned rows        : {len(cleanedDB)}')
print(f'  Outlier file        : {os.path.basename(outlierPath)}')
print(f'  Cleaned file        : {os.path.basename(cleanedPath)}')
print(f'  Sorted file         : {os.path.basename(sortedPath)}')
print('-'*52)

# =============================================================================
# Plots
# =============================================================================

# Pyplot Configuration
plt.rcParams['figure.dpi']=600
plt.rcParams['savefig.dpi']=600
plt.rcParams['text.usetex']=False
plt.rcParams['font.family']='serif'
plt.rcParams['font.serif']='Times New Roman'
plt.rcParams['font.weight']='bold'
plt.rcParams['mathtext.rm']='serif'
plt.rcParams['mathtext.it']='serif:italic'
plt.rcParams['mathtext.bf']='serif:bold'
plt.rcParams['mathtext.fontset']='custom'
plt.rcParams['axes.titlesize']=8
plt.rcParams['axes.labelsize']=8
plt.rcParams['xtick.labelsize']=7
plt.rcParams['ytick.labelsize']=7
plt.rcParams['font.size']=8
plt.rcParams["savefig.pad_inches"]=0.02

# Histogram of the target data
print(f'  Total samples        : {Y_All.shape[0]}')
print(f'  Range ({units})      : {Y_All.min():.4f} – {Y_All.max():.4f}')
plt.figure(figsize=(2.3,2))
if useLogScale:
    plt.hist(Y_All,
             bins=numpy.exp(numpy.histogram_bin_edges(numpy.log(Y_All))),
             color='white',edgecolor='red',hatch='.')
    plt.xscale('log')
else:
    plt.hist(Y_All,color='white',edgecolor='red',hatch='.')
plt.yscale('log')
plt.xlabel(varName,weight='bold')
plt.ylabel('Count',weight='bold')
plt.show()

# Sigma profile plot by cation family
sigmaAxis = mlDB.columns[colSigmaStart:].to_numpy('double')
ilNames = mlDB['component1'].unique()
ilSP = mlDB.groupby('component1')[mlDB.columns[colSigmaStart:]]\
    .mean().loc[ilNames].to_numpy('double')
ilSMILES = mlDB.drop_duplicates('component1').set_index('component1')\
    ['component1_smiles'].loc[ilNames]
ilFamily = numpy.array([get_cation_family(s) for s in ilSMILES])

# Define colors/markers per family
familyStyle = {
    'Imidazolium':   ('red',    'o'),
    'Pyridinium':    ('blue',   's'),
    'Pyrrolidinium': ('green',  '^'),
    'Piperidinium':  ('olive',  'P'),
    'Morpholinium':  ('teal',   '*'),
    'Phosphonium':   ('orange', 'D'),
    'Ammonium':      ('purple', 'v'),
    'Sulfonium':     ('cyan',   'p'),
    'Other':         ('gray',   'x'),
}

plt.figure(figsize=(3.2,2.5))
for i in range(ilSP.shape[0]):
    plt.semilogy(sigmaAxis, ilSP[i,:], '.k', markersize=1, alpha=0.3)
for fam, (col, mk) in familyStyle.items():
    mask = ilFamily == fam
    if mask.sum() == 0: continue
    idx = numpy.where(mask)[0]
    plt.semilogy(sigmaAxis, ilSP[idx[0],:],
                 marker=mk, color=col, markersize=2, linestyle='none',
                 label=f'{fam} ({mask.sum()})')
    for j in idx[1:]:
        plt.semilogy(sigmaAxis, ilSP[j,:],
                     marker=mk, color=col, markersize=2, linestyle='none')
plt.xlabel(r'$\sigma$ / e$\cdot$Å$^{-2}$', weight='bold')
plt.ylabel(r'P($\sigma$)$\cdot$A / Å$^{2}$', weight='bold')
plt.legend(fontsize=5, ncol=2, loc='upper right')
plt.title('Sigma Profile Space by Cation Family', fontsize=7)
plt.show()

# PCA colored by cation family
PCA_IL = decomposition.PCA(n_components=2, svd_solver='full')\
    .fit_transform(ilSP)
plt.figure(figsize=(3.2,2.5))
plt.plot(PCA_IL[:,0], PCA_IL[:,1], '.k', markersize=2, alpha=0.2)
for fam, (col, mk) in familyStyle.items():
    mask = ilFamily == fam
    if mask.sum() == 0: continue
    plt.plot(PCA_IL[mask,0], PCA_IL[mask,1],
             marker=mk, color=col, markersize=3, linestyle='none',
             label=f'{fam} ({mask.sum()})')
plt.xlabel('PC1', weight='bold')
plt.ylabel('PC2', weight='bold')
plt.legend(fontsize=5, ncol=2, loc='best')
plt.title('PCA of IL Sigma Profiles by Cation Family', fontsize=7)
plt.show()

# PCA of all data points
nSigma=mlDB.iloc[:,colSigmaStart:].shape[1]
SP_Matrix=mlDB.iloc[:,colSigmaStart:].to_numpy('double')
PCA=decomposition.PCA(n_components=2,svd_solver="full").fit(SP_Matrix)
PCA_All=PCA.transform(X_All[:,nCondCols:] if hasTemp else X_All)
plt.figure(figsize=(2.3,2))
plt.plot(PCA_All[:,0],PCA_All[:,1],'r.',markersize=2)
plt.xlabel('PC1',weight='bold')
plt.ylabel('PC2',weight='bold')
plt.show()

# Predictions Scatter Plot (full dataset)
plt.figure(figsize=(2.3,2))
allY=numpy.concatenate([Y_All,Y_Pred])
if useLogScale:
    lo,hi=allY.min()*0.8,allY.max()*1.2
else:
    pad=0.002
    lo,hi=allY.min()-pad,allY.max()+pad
plt.xlim(lo,hi)
plt.ylim(lo,hi)
plt.plot([lo,hi],[lo,hi],'k--',linewidth=1,zorder=1)
if useLogScale:
    plt.loglog(Y_All,Y_Pred,'ow',markersize=3,mec='red',zorder=2)
else:
    plt.plot(Y_All,Y_Pred,'ow',markersize=3,mec='red',zorder=2)
plt.xlabel('Exp. '+displayLabel,weight='bold')
plt.ylabel('Pred. '+displayLabel,weight='bold')
# Metrics
if useLogScale:
    R2=metrics.r2_score(numpy.log(Y_All),numpy.log(Y_Pred))
    MALE=numpy.mean(numpy.abs(numpy.log(Y_Pred)-numpy.log(Y_All)))
    plt.text(0.03,0.93,f'MALE = {MALE:.4f}',
             horizontalalignment='left',transform=plt.gca().transAxes,c='r')
else:
    R2=metrics.r2_score(Y_All,Y_Pred)
    MAE=metrics.mean_absolute_error(Y_All,Y_Pred)
    if MAE<0.01:
        MAPE=numpy.mean(numpy.abs((Y_All-Y_Pred)/Y_All))*100
        plt.text(0.03,0.93,f'MAPE = {MAPE:.2f}%',
                 horizontalalignment='left',transform=plt.gca().transAxes,c='r')
    else:
        plt.text(0.03,0.93,f'MAE = {fmtMAE(MAE)} {units}',
                 horizontalalignment='left',transform=plt.gca().transAxes,c='r')
plt.text(0.97,0.03,'$R^2$ = '+'{:.2f}'.format(R2),
         horizontalalignment='right',transform=plt.gca().transAxes,c='r')
plt.show()

# Predictions Scatter Plot (full dataset) with outliers highlighted
plt.figure(figsize=(2.3,2))
allY=numpy.concatenate([Y_All,Y_Pred])
pad=0.002
lo,hi=allY.min()-pad,allY.max()+pad
plt.xlim(lo,hi); plt.ylim(lo,hi)
plt.plot([lo,hi],[lo,hi],'k--',linewidth=1,zorder=1)
#plt.plot(Y_All[~outlierMask],Y_Pred[~outlierMask],'ow',markersize=3,mec='red',zorder=2)
#plt.plot(Y_All[outlierMask],Y_Pred[outlierMask],'^b',markersize=3,zorder=3)
plt.xlabel('Exp. '+displayLabel,weight='bold')
plt.ylabel('Pred. '+displayLabel,weight='bold')
R2=metrics.r2_score(Y_All,Y_Pred)
MAE=metrics.mean_absolute_error(Y_All,Y_Pred)
plt.text(0.03,0.93,f'MAE = {fmtMAE(MAE)} {units}',
         horizontalalignment='left',transform=plt.gca().transAxes,c='r')
plt.text(0.97,0.03,'$R^2$ = '+'{:.2f}'.format(R2),
         horizontalalignment='right',transform=plt.gca().transAxes,c='r')
plt.show()

# =============================================================================
# Outlier Detection
# =============================================================================

# Worst 100 by absolute error
worst_100_abs = mlDB.iloc[numpy.argsort(numpy.abs(Y_All.reshape(-1) - \
                                Y_Pred.reshape(-1)))[-100:][::-1]]
# Worst 100 by log-scale error
worst_200_log = mlDB.iloc[numpy.argsort(numpy.abs(numpy.log(Y_All.reshape(-1)) - \
                              numpy.log(Y_Pred.reshape(-1))))[-200:][::-1]]

# Sigma profile of worst-predicted point
index=numpy.abs(numpy.log(Y_Pred)-numpy.log(Y_All)).argmax()
plt.plot(mlDB.columns[colSigmaStart:].to_numpy('double'),
         mlDB.iloc[index,colSigmaStart:].to_numpy('double'),'--k')
name=mlDB.iloc[index,0]
plt.title(f'Worst outlier: {name}',fontsize=7)
plt.show()
if hasTemp:
    plt.plot(mlDB[mlDB['component1']==name].iloc[:,colTemp],
             mlDB[mlDB['component1']==name].iloc[:,colProp],'k*')
else:
    plt.plot(mlDB[mlDB['component1']==name].iloc[:,colProp],
        numpy.zeros(mlDB[mlDB['component1']==name].shape[0]),'k*')
plt.show()

# Print elapsed time
tf=time.time()
if gpConfig.get('useSparse',False):
    print(f'  Model              : SGPR (M={gpConfig.get("nInducing",800)})')
else:
    print('  Model              : GPR')
print('Time elapsed: '+'{:.2f}'.format(tf-ti)+' s')
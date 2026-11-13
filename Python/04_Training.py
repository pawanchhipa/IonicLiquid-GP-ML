# -*- coding: utf-8 -*-
"""
04_Training.py

Trains Gaussian Process regression models to predict ionic liquid
thermophysical properties using COSMO-RS sigma profiles.

The script has three main objectives:
    - Prepare property-specific training and test datasets.
    - Train GP models using molecular descriptors and available
      temperature and pressure conditions.
    - Evaluate predictions and generate diagnostic plots.

Sections:
    . Imports
    . Configuration
    . Auxiliary Functions
    . Main Script
    . Plots

Authors: Pawan Chhipa & Dinis O. Abranches
Last Updated: 2026-10-14

"""

# =============================================================================
# Imports
# =============================================================================
 
# General
import os
import re
import time
import warnings
from pathlib import Path
# Specific
import gpflow
import numpy
import pandas
from matplotlib import pyplot as plt
from sklearn import decomposition, metrics, preprocessing
from sklearn.model_selection import train_test_split
from numpy import matlib

# =============================================================================
# 2. Configuration
# =============================================================================

# Repository directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

# Input property dataset
PROPERTY_FILE = "ML-Ready-Pure_ILs_Melting_temperature.csv"
dbPath = DATA_DIR / PROPERTY_FILE

# Duplicate record log
LOG_DIR = DATA_DIR / "Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logPath = LOG_DIR / "duplicate_log_Melting_temperature.csv"

# Dataset column layout
# Options: "T_P_Prop", "T_Prop", "Prop"
dataLayout = "Prop"

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
        Z=X_Train[numpy.random.default_rng(42).choice(X_Train.shape[0],
                                                      M,replace=False)].copy()
        model=gpflow.models.SGPR((X_Train,Y_Train),gpKernel,
                                 inducing_variable=Z,noise_variance=10**-5)
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
                           method='L-BFGS-B',
                           options={'maxiter':500})
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
    # Count ring size from SMILES ring digits/% notation
    upper = cat.upper()
    # Phosphonium / Sulfonium — check before nitrogen
    if '[P+]' in cat or '[P+' in cat or '[pH]' in cat: return 'Phosphonium'
    if '[S+]' in cat:  return 'Sulfonium'
    # Count atoms in smallest ring (heuristic from bracket content)
    # Detect ring membership: does the cation have a ring?
    hasRing = any(c.isdigit() for c in cat) or '%' in cat
    # Aromatic nitrogen: lowercase 'n' in the SMILES
    nArom = cat.count('n')
    if not hasRing:
        # No ring → quaternary ammonium
        if '[N+]' in cat or '[NH' in cat: return 'Ammonium'
        return 'Other'
    # Has ring + aromatic nitrogen(s)
    if nArom >= 2:
        # Two aromatic N in ring → imidazolium or triazolium
        return 'Imidazolium'
    if nArom == 1:
        # One aromatic N in ring → pyridinium
        return 'Pyridinium'
    # Has ring but no aromatic N → saturated N-heterocycle
    # Distinguish by ring size using character count between ring digits
    # Find the ring-opening digit
    if '[N+]' in cat or '[nH]' in cat or 'N' in cat:
        # Count carbons between ring open/close to estimate ring size
        # Simple heuristic: count non-bracket uppercase C's in the SMILES
        carbons = upper.count('C') - upper.count('CL')
        # Pyrrolidinium: 4C + 1N = 5-membered
        # Piperidinium:  5C + 1N = 6-membered
        # Morpholinium:  4C + 1N + 1O = 6-membered
        if 'O' in upper and carbons <= 5: return 'Morpholinium'
        if carbons <= 4: return 'Pyrrolidinium'
        if carbons <= 6: return 'Piperidinium'
    return 'Other'

# =============================================================================
# 4. Main Script
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
 
ti=time.time()
# Load CSV file
mlDB=pandas.read_csv(dbPath)
mlDB=mlDB.dropna()

# ---- Diagnose row count before anything ----
print(f'  Rows loaded from CSV         : {len(mlDB)}')
print(f'  Unique ILs in raw CSV        : {mlDB["component1"].nunique()}')

# abs = mlDB[mlDB.duplicated(subset='component1',keep=False)].sort_values('component1').iloc[:,:3]

# ---- Find ILs with same name but different target values (outliers) ----
targetCol = mlDB.columns[colProp]
nBefore = len(mlDB)

# Build subset dynamically: include T and P columns if present
dupSubsetKey = ['component1', 'component1_smiles']
if hasTemp:     dupSubsetKey.append(mlDB.columns[colTemp])
if hasPressure: dupSubsetKey.append(mlDB.columns[colTemp+1])
dupSubset = dupSubsetKey + [targetCol]

# Find ILs with same name+conditions but different target values (true conflicts)
ilTargetCounts = mlDB.groupby(dupSubsetKey)[targetCol].nunique()
conflictILs    = ilTargetCounts[ilTargetCounts > 1].index.tolist()

# All rows belonging to conflicting ILs
dupDF = mlDB[mlDB.duplicated(subset=dupSubset, keep=False)]\
    .copy().sort_values('component1')
nDups = len(dupDF)

# Save exact duplicate records
if not dupDF.empty:
    dupDF.to_csv(logPath, index=False)
    print(f"Duplicate records saved to: {logPath}")
else:
    print("No exact duplicate records found.")

print(f'  Total rows before removal    : {nBefore}')
print(f'  Conflicting ILs found        : {len(conflictILs)}')
print(f'  Rows affected (exact dups)   : {nDups}')
print(f'  Duplicate subset used        : {dupSubset}')  

mlDB = mlDB.drop_duplicates(subset=dupSubset).reset_index(drop=True)
nAfter = len(mlDB)
print(f'  Rows after removal           : {nAfter}')
print(f'  Rows removed                 : {nBefore - nAfter}')

# Stratify Y
Y=numpy.log(mlDB.iloc[:,colProp].to_numpy('double').reshape(-1,1))
#Check and deop if the value of property is not 0
mask = numpy.isfinite(Y).reshape(-1)
mlDB = mlDB.loc[mask].reset_index(drop=True)
Y = Y[mask]
 
#DataSet Summary
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
 
# ---- Unified IL-holdout workflow for ALL layouts ----            
# Stratified 20% IL holdout (stratification)     
# Compute median log-property per IL for stratification at IL level           
uniqueILs = mlDB['component1'].unique()                    
ilMedians = mlDB.groupby('component1')[mlDB.columns[colProp]]\
    .median().reindex(uniqueILs)                                             
ilMedianLog = numpy.log(ilMedians.values.reshape(-1,1))                     
# Find max bins where every bin has >=2 ILs (need >=2 for stratified split)  
for n in range(1,100):                                                       
    sv = pandas.cut(ilMedianLog.reshape(-1,),n,labels=False)                
    if any(numpy.count_nonzero(sv==k)<2 for k in range(n)):                  
        nBinsIL=n-1                                                        
        break                                                              
ilStratVector = pandas.cut(ilMedianLog.reshape(-1,),nBinsIL,labels=False)    
# Build a temporary DataFrame of unique ILs with their strat labels         
ilDF = pandas.DataFrame({'IL':uniqueILs,'stratBin':ilStratVector})           
ilTrain, ilTest = train_test_split(ilDF, test_size=0.20,                     
                                   random_state=42,                       
                                   stratify=ilDF['stratBin'])             
testILs  = set(ilTest['IL'])                                                 
trainILs = set(ilTrain['IL'])                                                
mlDB_Test = mlDB[mlDB['component1'].isin(testILs)].reset_index(drop=True)
mlDB_Train = mlDB[mlDB['component1'].isin(trainILs)].reset_index(drop=True)
print(f'  Unique ILs (total)   : {len(uniqueILs)}')
print(f'  Hold-out test ILs    : {len(testILs)}')
print(f'  Train ILs            : {len(trainILs)}')
print(f'  Train / Test         : {len(mlDB_Train)} / {len(mlDB_Test)} rows')
print('-'*52)

# Extract condition columns (T, T+P) and sigma profiles as numpy arrays
if hasTemp:
    X_Train=numpy.concatenate((mlDB_Train.iloc[:,colTemp:colProp].to_numpy('double'),
                               mlDB_Train.iloc[:,colSigmaStart:].to_numpy('double')),axis=1)
    X_Test=numpy.concatenate((mlDB_Test.iloc[:,colTemp:colProp].to_numpy('double'),
                              mlDB_Test.iloc[:,colSigmaStart:].to_numpy('double')),axis=1)
else:
    X_Train=mlDB_Train.iloc[:,colSigmaStart:].to_numpy('double')                
    X_Test=mlDB_Test.iloc[:,colSigmaStart:].to_numpy('double')
 
Y_Train=mlDB_Train.iloc[:,colProp].to_numpy('double').reshape(-1,1)
Y_Test=mlDB_Test.iloc[:,colProp].to_numpy('double').reshape(-1,1)
# Get target variable denominator
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

if labelNorm=='LogStand': useLogScale=True
else: useLogScale=False

# Normalize
if featureNorm is not None:
    X_Train_N,skScaler_X_Train=normalize(X_Train,method=featureNorm)
    X_Test_N,__=normalize(X_Test,skScaler=skScaler_X_Train,
                          method=featureNorm)
else:
    X_Train_N=X_Train
    X_Test_N=X_Test
if labelNorm is not None:
    Y_Train_N,skScaler_Y_Train=normalize(Y_Train,method=labelNorm)
else:
    Y_Train_N=Y_Train

# Train Gaussian Process model
t0 = time.time()
model = buildGP(X_Train_N, Y_Train_N, gpConfig=gpConfig)

if gpConfig.get("useSparse", False):
    print(
        f"  SGPR training time "
        f"(M={gpConfig.get('nInducing', 3000)}): "
        f"{time.time() - t0:.1f} s"
    )
else:
    print(f"  GPR training time: {time.time() - t0:.1f} s")

# Get GP predictions
Y_Train_Pred_N,STD_Train=gpPredict(model,X_Train_N)
Y_Test_Pred_N,STD_Test=gpPredict(model,X_Test_N)
# Unnormalize
if labelNorm is not None:
    Y_Train_Pred,__=normalize(Y_Train_Pred_N,skScaler=skScaler_Y_Train,
                              method=labelNorm,reverse=True)
    Y_Test_Pred,__=normalize(Y_Test_Pred_N,skScaler=skScaler_Y_Train,
                             method=labelNorm,reverse=True)
else:
    Y_Train_Pred=Y_Train_Pred_N
    Y_Test_Pred=Y_Test_Pred_N
 
index=numpy.abs(numpy.log(Y_Test_Pred)-numpy.log(Y_Test)).argmax()
plt.plot(mlDB_Test.columns[colSigmaStart:].to_numpy('double'),
         mlDB_Test.iloc[index,colSigmaStart:].to_numpy('double'),'--k')
name=mlDB_Test.iloc[index,0]
plt.show()
if hasTemp:
    plt.plot(mlDB_Test[mlDB_Test['component1']==name].iloc[:,colTemp],
             mlDB_Test[mlDB_Test['component1']==name].iloc[:,colProp],'k*')
else:
    plt.plot(mlDB_Test[mlDB_Test['component1']==name].iloc[:,colProp],
        numpy.zeros(mlDB_Test[mlDB_Test['component1']==name].shape[0]),'k*')
plt.show()

# =============================================================================
# 5. Plots
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
print(f'  Train / Test         : {Y_Train.shape[0]} / {Y_Test.shape[0]} samples')
print(f'  Train range ({units}): {Y_Train.min():.4f} – {Y_Train.max():.4f}')
print(f'  Test  range ({units}): {Y_Test.min():.4f} – {Y_Test.max():.4f}')
plt.figure(figsize=(2.3,2))
code='VP'
if useLogScale:
    plt.hist(Y_Train,
             bins=numpy.exp(numpy.histogram_bin_edges(numpy.log(Y_Train))),
             color='white',edgecolor='red',hatch='.')
    plt.hist(Y_Test,
             bins=numpy.exp(numpy.histogram_bin_edges(numpy.log(Y_Test))),
             color='white',edgecolor='blue',hatch='x')
    plt.xscale('log')
else:
    plt.hist(Y_Train,color='white',edgecolor='red',hatch='.')
    plt.hist(Y_Test,color='white',edgecolor='blue',hatch='x')
plt.yscale('log')
plt.xlabel(varName,weight='bold')
plt.ylabel('Count',weight='bold')
plt.show()

# PCA Plot  

# Build per-IL mean sigma profile (one row per unique IL)
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
ssigma = matlib.repmat(sigmaAxis.reshape(1,-1), ilSP.shape[0], 1)
# Background: plot each IL individually with no legend
for i in range(ilSP.shape[0]):
    plt.semilogy(sigmaAxis, ilSP[i,:], '.k', markersize=1, alpha=0.3)
# Overlay by family
for fam, (col, mk) in familyStyle.items():
    mask = ilFamily == fam
    if mask.sum() == 0: continue
    # Plot just one point with label for the legend
    idx = numpy.where(mask)[0]
    plt.semilogy(sigmaAxis, ilSP[idx[0],:],
                 marker=mk, color=col, markersize=2, linestyle='none',
                 label=f'{fam} ({mask.sum()})')
    # Plot the rest without label
    for j in idx[1:]:
        plt.semilogy(sigmaAxis, ilSP[j,:],
                     marker=mk, color=col, markersize=2, linestyle='none')
plt.xlabel(r'$\sigma$ / e$\cdot$Å$^{-2}$', weight='bold')
plt.ylabel(r'P($\sigma$)$\cdot$A / Å$^{2}$', weight='bold')
plt.legend(fontsize=5, ncol=2, loc='upper right')
plt.title('Sigma Profile Space by Cation Family', fontsize=7)
plt.show()

# --- Plot 2: PCA colored by cation family ---
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
                                                                 
nSigma=mlDB.iloc[:,colSigmaStart:].shape[1]                                 
SP_Matrix=mlDB.iloc[:,colSigmaStart:].to_numpy('double')                                       
PCA=decomposition.PCA(n_components=2,svd_solver="full").fit(SP_Matrix)       
PCA_Train=PCA.transform(X_Train[:,nCondCols:] if hasTemp else X_Train)         
PCA_Test=PCA.transform(X_Test[:,nCondCols:] if hasTemp else X_Test)          
plt.figure(figsize=(2.3,2))                                                  
plt.plot(PCA_Train[:,0],PCA_Train[:,1],'r.',markersize=2)                 
plt.plot(PCA_Test[:,0],PCA_Test[:,1],'b.',markersize=2)                    
plt.xlabel('PC1',weight='bold')                                             
plt.ylabel('PC2',weight='bold')
plt.show()                                                                   
 
# Predictions Scatter Plot
plt.figure(figsize=(2.3,2))
# Diagonal first
allY=numpy.concatenate([Y_Train,Y_Test,Y_Train_Pred,Y_Test_Pred])
if useLogScale:
    lo,hi=allY.min()*0.8,allY.max()*1.2
else:
    pad=0.002
    lo,hi=allY.min()-pad,allY.max()+pad
plt.xlim(lo,hi)
plt.ylim(lo,hi)
plt.plot([lo,hi],[lo,hi],'k--',linewidth=1,zorder=1)
# Data points
if useLogScale:
    plt.loglog(Y_Train,Y_Train_Pred,'ow',markersize=3,mec='red',zorder=2)
    plt.loglog(Y_Test,Y_Test_Pred,'^b',markersize=2,zorder=3)
else:
    plt.plot(Y_Train,Y_Train_Pred,'ow',markersize=3,mec='red',zorder=2)
    plt.plot(Y_Test,Y_Test_Pred,'^b',markersize=2,zorder=3)
# Labels
plt.xlabel('Exp. '+displayLabel,weight='bold')
plt.ylabel('Pred. '+displayLabel,weight='bold')
# Metrics + text
if useLogScale:
    R2_Train=metrics.r2_score(numpy.log(Y_Train),numpy.log(Y_Train_Pred))
    R2_Test=metrics.r2_score(numpy.log(Y_Test),numpy.log(Y_Test_Pred))
    MALE_Train=numpy.mean(numpy.abs(numpy.log(Y_Train_Pred)-numpy.log(Y_Train)))
    MALE_Test=numpy.mean(numpy.abs(numpy.log(Y_Test_Pred)-numpy.log(Y_Test)))
    plt.text(0.03,0.93,f'MALE = {MALE_Train:.4f}',
             horizontalalignment='left',transform=plt.gca().transAxes,c='r')
    plt.text(0.03,0.85,f'MALE = {MALE_Test:.4f}',
             horizontalalignment='left',transform=plt.gca().transAxes,c='b')
else:
    R2_Train=metrics.r2_score(Y_Train,Y_Train_Pred)
    R2_Test=metrics.r2_score(Y_Test,Y_Test_Pred)
    MAE_Train=metrics.mean_absolute_error(Y_Train,Y_Train_Pred)
    MAE_Test=metrics.mean_absolute_error(Y_Test,Y_Test_Pred)
    if MAE_Train<0.01:
        MAPE_Train=numpy.mean(numpy.abs((Y_Train-Y_Train_Pred)/Y_Train))*100
        MAPE_Test=numpy.mean(numpy.abs((Y_Test-Y_Test_Pred)/Y_Test))*100
        plt.text(0.03,0.93,f'MAPE = {MAPE_Train:.2f}%',
                 horizontalalignment='left',transform=plt.gca().transAxes,c='r')
        plt.text(0.03,0.85,f'MAPE = {MAPE_Test:.2f}%',
                 horizontalalignment='left',transform=plt.gca().transAxes,c='b')
    else:
        plt.text(0.03,0.93,f'MAE = {fmtMAE(MAE_Train)} {units}',
                 horizontalalignment='left',transform=plt.gca().transAxes,c='r')
        plt.text(0.03,0.85,f'MAE = {fmtMAE(MAE_Test)} {units}',
                 horizontalalignment='left',transform=plt.gca().transAxes,c='b')
plt.text(0.97,0.10,'$R^2$ = '+'{:.2f}'.format(R2_Train),
         horizontalalignment='right',transform=plt.gca().transAxes,c='r')
plt.text(0.97,0.03,'$R^2$ = '+'{:.2f}'.format(R2_Test),
         horizontalalignment='right',transform=plt.gca().transAxes,c='b')
plt.show()

# Print elapsed time
tf=time.time()
print('Time elapsed: '+'{:.2f}'.format(tf-ti)+' s')
mlDB_Train.iloc[numpy.abs(Y_Train-Y_Train_Pred).argmax(),:]
 
worst_100_Train = mlDB_Train.iloc[numpy.argsort(numpy.abs(Y_Train.reshape(-1) - \
                                Y_Train_Pred.reshape(-1)))[-100:][::-1]]
worst_100_log = mlDB_Train.iloc[numpy.argsort(numpy.abs(numpy.log(Y_Train.reshape(-1)) - \
                              numpy.log(Y_Train_Pred.reshape(-1))))[-100:][::-1]]
    
    
# =============================================================================
# Sorted prediction DataFrame (outlier inspection, train + test)
# =============================================================================

# Columns to carry through, from dataLayout
outCols=['component1','component1_smiles']
if hasTemp: outCols.append(mlDB.columns[colTemp])
if hasPressure: outCols.append(mlDB.columns[colTemp+1])
outCols.append(mlDB.columns[colProp])

def buildSortedDF(dbSplit,yTrue,yPred,splitLabel):
    df=dbSplit[outCols].copy()
    df['Split']=splitLabel
    df['Predicted']=yPred.reshape(-1)
    actual=yTrue.reshape(-1)
    pred=yPred.reshape(-1)
    if useLogScale:
        df['LogError']=numpy.log(pred)-numpy.log(actual)
        df['AbsLogError']=numpy.abs(df['LogError'])
    else:
        df['Error']=pred-actual
        df['AbsError']=numpy.abs(df['Error'])
    return df

errName='AbsLogError' if useLogScale else 'AbsError'
sortedDF=pandas.concat(
    [buildSortedDF(mlDB_Train,Y_Train,Y_Train_Pred,'train'),
     buildSortedDF(mlDB_Test,Y_Test,Y_Test_Pred,'test')],
    ignore_index=True)
sortedDF=sortedDF.sort_values(errName,ascending=False).reset_index(drop=True)
# Save sorted dataframe in the same curation folder
savePath = os.path.join(
    os.path.dirname(dbPath),
    'sortedDF.csv'
)

sortedDF.to_csv(savePath, index=False)

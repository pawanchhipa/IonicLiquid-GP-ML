# -*- coding: utf-8 -*-
"""
05_Combinational_Space.py

Generates a combinatorial ionic liquid database from cation and anion
sigma profile databases. Each cation is paired with each anion, and
the ionic stoichiometry is determined from charge balance.

The script has three main objectives:
    - Generate charge-balanced cation-anion combinations.
    - Calculate combined sigma profiles using stoichiometric weights.
    - Compare the combinatorial space with experimental property
      datasets using principal component analysis (PCA).

Sections:
    . Imports
    . Configuration
    . Auxiliary Functions
    . Main Script
    . PCA Plots

Authors: Pawan Chhipa & Dinis O. Abranches
Last Updated: 2026-10-14

"""
# =============================================================================
# 1. Imports
# =============================================================================

# General
import os
from math import gcd
from pathlib import Path
# Specific
import numpy
import pandas
from matplotlib import pyplot as plt
from sklearn import decomposition

# =============================================================================
# 2. Configuration
# =============================================================================

# Repository directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

# Ion sigma profile databases
anionsPath = DATA_DIR / "Anions" / "SP_Database.csv"
cationsPath = DATA_DIR / "Cations" / "SP_Database.csv"

# Combinatorial database output
outPath = DATA_DIR / "Combinatorial_IL_Database.csv"

# Experimental dataset for PCA comparison
# Set to None to skip the property dataset overlay.
PROPERTY_FILE = "ML-Ready-Pure_ILs_Melting_temperature.csv"
propertyDBPath = DATA_DIR / PROPERTY_FILE

# Plot configuration
onlyPlot6 = True

# =============================================================================
# Auxiliary Functions
# =============================================================================

def get_cation_family(smiles):
    """Detect cation core from IL SMILES without RDKit."""
    cat=smiles.strip()
    if not cat: return 'Other'
    upper=cat.upper()
    if '[P+]' in cat or '[P+' in cat or '[pH]' in cat: return 'Phosphonium'
    if '[S+]' in cat: return 'Sulfonium'
    hasRing=any(c.isdigit() for c in cat) or '%' in cat
    nArom=cat.count('n')
    if not hasRing:
        if '[N+]' in cat or '[NH' in cat: return 'Ammonium'
        return 'Other'
    if nArom>=2: return 'Imidazolium'
    if nArom==1: return 'Pyridinium'
    carbons=upper.count('C')-upper.count('CL')
    if 'O' in upper and carbons<=5: return 'Morpholinium'
    if carbons<=4: return 'Pyrrolidinium'
    if carbons<=6: return 'Piperidinium'
    return 'Other'

def get_anion_family(smiles):
    """Classify anion family from SMILES without RDKit."""
    s=smiles.strip()
    upper=s.upper()
    # Halides
    if s in ('[F-]','[Cl-]','[Br-]','[I-]'): return 'Halide'
    # Bis(trifluoromethylsulfonyl)imide (NTf2 / TFSI)
    if 'S(=O)(=O)C(F)(F)F' in s and 'N' in upper: return 'NTf2'
    # Bis(fluorosulfonyl)imide (FSI)
    if 'S(=O)(=O)F' in s and 'N' in upper and 'C(F)' not in s: return 'FSI'
    # Trifluoromethanesulfonate (OTf / triflate)
    if 'S(=O)(=O)[O-]' in s and 'C(F)(F)F' in s and 'N' not in upper:
        return 'Triflate'
    # Tetrafluoroborate
    if '[B-](F)(F)(F)F' in s: return 'BF4'
    # Hexafluorophosphate
    if '[P-](F)(F)(F)(F)(F)F' in s or 'F[P-]' in s: return 'PF6'
    # Dicyanamide
    if 'N(C#N)C#N' in s or 'C(#N)[N-]C#N' in s: return 'DCA'
    # Tricyanomethanide
    if s.count('C#N')>=3: return 'TCM'
    # Thiocyanate
    if '[S-]C#N' in s or 'SC#N' in s or '[N-]=C=S' in s: return 'SCN'
    # Nitrate
    if '[N+](=O)([O-])[O-]' in s or 'O=[N+]([O-])[O-]' in s: return 'Nitrate'
    # Sulfate / hydrogen sulfate
    if 'S(=O)(=O)' in s and 'O' in s and 'F' not in upper and 'N' not in upper:
        return 'Sulfate'
    # Phosphate
    if 'P(=O)' in s and 'O' in s and 'F' not in upper: return 'Phosphate'
    # Carboxylate (acetate, formate, etc.)
    if 'C(=O)[O-]' in s or 'C([O-])=O' in s or '[O-]C=O' in s:
        return 'Carboxylate'
    # Borate (other than BF4)
    if '[B-]' in s or '[B]' in s: return 'Borate'
    return 'Other'

# =============================================================================
# Main Script
# =============================================================================

# Read databases
cationsDB=pandas.read_csv(cationsPath)
anionsDB=pandas.read_csv(anionsPath)
print(f'Cations loaded: {len(cationsDB)}')
print(f'Anions  loaded: {len(anionsDB)}')

# Align sigma axes (union, pad with zeros)
anionSigma=anionsDB.columns[5:].astype('double')
cationSigma=cationsDB.columns[5:].astype('double')
sigmaMin=min(anionSigma.min(),cationSigma.min())
sigmaMax=max(anionSigma.max(),cationSigma.max())
fullSigma=numpy.round(numpy.arange(sigmaMin,sigmaMax+0.001,0.001),3)
# Re-index both DBs onto the common sigma axis
fullCols=list(anionsDB.columns[:5])+fullSigma.tolist()
anionsDB=anionsDB.set_axis(list(anionsDB.columns[:5])+anionSigma.tolist(),
                           axis=1)
cationsDB=cationsDB.set_axis(list(cationsDB.columns[:5])+cationSigma.tolist(),
                             axis=1)
anionsDB=anionsDB.reindex(columns=fullCols,fill_value=0.0)
cationsDB=cationsDB.reindex(columns=fullCols,fill_value=0.0)

# Extract sigma profile matrices
catSP=cationsDB.iloc[:,5:].to_numpy('double')
anSP=anionsDB.iloc[:,5:].to_numpy('double')

# Build combinatorial database
rows=[]
nCat=len(cationsDB)
nAn=len(anionsDB)
print(f'Generating {nCat} x {nAn} = {nCat*nAn} combinations...')
for i in range(nCat):
    if (i+1)%50==0 or i==nCat-1:
        print(f'  Cation {i+1}/{nCat}...')
    catCharge=int(cationsDB.iloc[i,3])   # positive integer
    catName=cationsDB.iloc[i,1]
    catSMILES=cationsDB.iloc[i,2]
    for j in range(nAn):
        anCharge=abs(int(anionsDB.iloc[j,3]))  # positive magnitude
        anName=anionsDB.iloc[j,1]
        anSMILES=anionsDB.iloc[j,2]
        # Charge balance: n_cat*catCharge = n_an*|anCharge|
        g=gcd(catCharge,anCharge)
        nCatNeeded=anCharge//g   # e.g. +1 cat, -2 an → need 2 cations
        nAnNeeded=catCharge//g   # e.g. +1 cat, -2 an → need 1 anion
        totalFrags=nCatNeeded+nAnNeeded
        # Combined sigma profile (weighted average)
        spIL=(nCatNeeded*catSP[i,:]+nAnNeeded*anSP[j,:])/totalFrags
        # IL name and SMILES
        ilName='['+catName+']['+anName+']'
        smilesParts=([catSMILES]*nCatNeeded)+([anSMILES]*nAnNeeded)
        ilSMILES='.'.join(smilesParts)
        rows.append([ilName,ilSMILES]+spIL.tolist())

# Assemble DataFrame
colNames=['component1','component1_smiles']+fullSigma.tolist()
combDB=pandas.DataFrame(rows,columns=colNames)
print(f'Combinatorial database: {len(combDB)} ILs')
# Save
combDB.to_csv(outPath,index=False)
print(f'Saved to: {outPath}')

# =============================================================================
# PCA Plots
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

sigmaAxis=fullSigma
SP_Matrix=combDB.iloc[:,2:].to_numpy('double')

# Pre-classify families from source DBs, then repeat/tile to match combDB order
catFamilyPer=numpy.array([get_cation_family(s) for s in cationsDB.iloc[:,2]])
anFamilyPer=numpy.array([get_anion_family(s) for s in anionsDB.iloc[:,2]])
catFamilyAll=numpy.repeat(catFamilyPer,nAn)   # each cation repeated nAn times
anFamilyAll=numpy.tile(anFamilyPer,nCat)       # anion list tiled nCat times

catFamilyStyle={
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
anFamilyStyle={
    'NTf2':        ('red',       'o'),
    'BF4':         ('blue',      's'),
    'PF6':         ('green',     '^'),
    'Halide':      ('orange',    'D'),
    'DCA':         ('purple',    'v'),
    'Carboxylate': ('olive',     'P'),
    'FSI':         ('teal',      '*'),
    'Triflate':    ('brown',     'h'),
    'SCN':         ('pink',      '<'),
    'Nitrate':     ('gold',      '>'),
    'Sulfate':     ('darkblue',  'd'),
    'Phosphate':   ('darkgreen', '8'),
    'TCM':         ('coral',     'H'),
    'Borate':      ('navy',      'p'),
    'Other':       ('gray',      'x'),
}

# Subsample indices for sigma profile overlay (max 500 background + 30/family)
rng=numpy.random.default_rng(42)
nTotal=SP_Matrix.shape[0]
bgIdx=rng.choice(nTotal,size=min(500,nTotal),replace=False)

def plot_sigma_overlay(familyArr,familyStyle,title):
    """Sigma profile overlay colored by family (subsampled)."""
    plt.figure(figsize=(3.2,2.5))
    for k in bgIdx:
        plt.semilogy(sigmaAxis,SP_Matrix[k,:],'.k',markersize=0.5,alpha=0.15)
    for fam,(col,mk) in familyStyle.items():
        mask=numpy.where(familyArr==fam)[0]
        if len(mask)==0: continue
        sub=rng.choice(mask,size=min(30,len(mask)),replace=False)
        plt.semilogy(sigmaAxis,SP_Matrix[sub[0],:],
                     marker=mk,color=col,markersize=2,linestyle='none',
                     label=f'{fam} ({len(mask)})')
        for k in sub[1:]:
            plt.semilogy(sigmaAxis,SP_Matrix[k,:],
                         marker=mk,color=col,markersize=2,linestyle='none')
    plt.xlabel(r'$\sigma$ / e$\cdot$Å$^{-2}$',weight='bold')
    plt.ylabel(r'P($\sigma$)$\cdot$A / Å$^{2}$',weight='bold')
    plt.legend(fontsize=4,ncol=2,loc='upper right')
    plt.title(title,fontsize=7)
    plt.show()

def plot_pca(PCA_IL,familyArr,familyStyle,title):
    """PCA scatter colored by family."""
    plt.figure(figsize=(3.2,2.5))
    plt.plot(PCA_IL[:,0],PCA_IL[:,1],'.k',markersize=0.5,alpha=0.05)
    for fam,(col,mk) in familyStyle.items():
        mask=familyArr==fam
        if mask.sum()==0: continue
        plt.plot(PCA_IL[mask,0],PCA_IL[mask,1],
                 marker=mk,color=col,markersize=1.5,linestyle='none',
                 alpha=0.4,label=f'{fam} ({mask.sum()})')
    plt.xlabel('PC1',weight='bold')
    plt.ylabel('PC2',weight='bold')
    plt.legend(fontsize=4,ncol=2,loc='best')
    plt.title(title,fontsize=7)
    plt.show()

# Compute PCA once
PCA_obj=decomposition.PCA(n_components=2,svd_solver='full').fit(SP_Matrix)
PCA_IL=PCA_obj.transform(SP_Matrix)

onlyPlot6=True

if not onlyPlot6:
    # --- Plot 1: Sigma profiles by Cation Family ---
    plot_sigma_overlay(catFamilyAll,catFamilyStyle,
                       'Combinatorial IL Sigma Profiles – Cation Family')
    # --- Plot 2: Sigma profiles by Anion Family ---
    plot_sigma_overlay(anFamilyAll,anFamilyStyle,
                       'Combinatorial IL Sigma Profiles – Anion Family')
    # --- Plot 3: PCA by Cation Family ---
    plot_pca(PCA_IL,catFamilyAll,catFamilyStyle,
             'PCA of Combinatorial ILs – Cation Family')
    # --- Plot 4: PCA by Anion Family ---
    plot_pca(PCA_IL,anFamilyAll,anFamilyStyle,
             'PCA of Combinatorial ILs – Anion Family')
    # --- Plot 5: Simple PCA ---
    plt.figure(figsize=(2.3,2))
    plt.plot(PCA_IL[:,0],PCA_IL[:,1],'r.',markersize=1)
    plt.xlabel('PC1',weight='bold')
    plt.ylabel('PC2',weight='bold')
    plt.show()


# --- Plot 6: Combinatorial space with property DB ILs overlaid ---
if propertyDBPath is not None:
    propDB=pandas.read_csv(propertyDBPath).dropna()
    # Detect layout from column count before sigma profiles
    nColsProp=propDB.shape[1]
    testCols=propDB.columns[:6]
    if 'pressure' in testCols.str.lower().to_list() or nColsProp>80 and \
       propDB.iloc[:,4].dtype=='float64' and propDB.iloc[:,3].dtype=='float64':
        propSigmaStart=5   # T_P_Prop
    elif propDB.iloc[:,3].dtype=='float64' and \
         propDB.iloc[:,2].dtype=='float64':
        propSigmaStart=4   # T_Prop
    else:
        propSigmaStart=3   # Prop
    # Extract unique IL sigma profiles from property DB
    propUnique=propDB.drop_duplicates('component1_smiles').reset_index(drop=True)
    # Reindex property sigma profiles onto combinatorial sigma axis
    propSigmaCols=propUnique.columns[propSigmaStart:].astype('double').tolist()
    propSP_df=propUnique.iloc[:,propSigmaStart:].set_axis(propSigmaCols,axis=1)
    propSP_df=propSP_df.reindex(columns=fullSigma.tolist(),fill_value=0.0)
    propSP=propSP_df.to_numpy('double')    
    # Project into combinatorial PCA space
    PCA_prop=PCA_obj.transform(propSP)
    
    propName=os.path.basename(propertyDBPath).replace('ML-Ready-Pure_ILs_','')\
        .replace('.csv','').replace('_',' ')
    print(f'Property overlay ({propName}):')
    print(f'  Unique ILs in property DB : {len(propUnique)}')
    print(f'  Sigma profile columns     : {propSP.shape[1]}')
    plt.figure(figsize=(2.3,2))
    plt.plot(PCA_IL[:,0],PCA_IL[:,1],'r.',markersize=1,alpha=0.15)
    plt.plot(PCA_prop[:,0],PCA_prop[:,1],'b.',markersize=2,
             label=f'{propName} ({len(propUnique)})')
    plt.legend(fontsize=5,loc='best')
    plt.xlabel('PC1',weight='bold')
    plt.ylabel('PC2',weight='bold')
    plt.show()
 
# Summary
print('='*52)
print(f'Total combinations: {len(combDB)}')
print('-'*52)
print('Cation families:')
for fam in numpy.unique(catFamilyAll):
    print(f'  {fam}: {(catFamilyAll==fam).sum()}')
print('-'*52)
print('Anion families:')
for fam in numpy.unique(anFamilyAll):
    print(f'  {fam}: {(anFamilyAll==fam).sum()}')

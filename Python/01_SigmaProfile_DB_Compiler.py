# -*- coding: utf-8 -*-
"""
01_SigmaProfile_DB_Compiler.py

Compiles a sigma profile database from an ion index file and
corresponding COSMO files. The script has two main objectives:
    - Validate the ion database and molecular structures.
    - Generate sigma profiles for all entries using COSMOtherm.

Sections:
    . Imports
    . Configuration
    . Auxiliary Functions
    . Main Script

Authors: Pawan Chhipa & Dinis O. Abranches
Last Updated: 2026-10-14

"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
from pathlib import Path
import tempfile
import time
import subprocess
import glob

# Specific
import numpy
import pandas
import xyz2mol
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import networkx
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt

# =============================================================================
# Configuration
# =============================================================================
# Repository directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

# Select the ion database: "Anions" or "Cations"
ION_TYPE = "Anions"

# Input files
ION_DIR = DATA_DIR / ION_TYPE
indexPath = ION_DIR / "Index.xlsx"
cosmoFolder = ION_DIR / "COSMO_Files"

# Output file
outputPath = ION_DIR / "SP_Database.csv"

# COSMOtherm configuration
cosmoConfig = {
    "parameterization": "BP_TZVP_21.ctd",
    "parameterizationPATH": (
        r"C:\Program Files\BIOVIA\COSMOtherm2025"
        r"\COSMOtherm\CTDATA-FILES"
    ),
    "licencePATH": (
        r"C:\Program Files\BIOVIA\COSMOtherm2025"
        r"\licensefiles"
    ),
    "COSMOthermPATH": (
        r"C:\Program Files\BIOVIA\COSMOtherm2025"
        r"\COSMOtherm\BIN-WINDOWS"
    ),
    "tempFolder": tempfile.mkdtemp(),
}

# =============================================================================
# Auxiliary Functions
# =============================================================================

def getCharge(cosmoFile):
    """
    getCharge() retrieves the charge from the COSMO file given.

    Parameters
    ----------
    cosmoFile : string
        Path to COSMO file.

    Raises
    ------
    ValueError
        Error raised if charge not found or not a float.

    Returns
    -------
    charge : float
        Charge.
    """
    # Initialize flags
    foundSection=False
    foundLine=False
    # Open file
    with open(cosmoFile,'r') as f:
        # Loop over file
        for line in f:
            # Read line
            line = line.strip()
            # Find the screening charge section
            if line.startswith('$screening_charge'):
                foundSection=True
            # Find the total line
            elif foundSection and line.startswith('total'):
                # Obtain the charge
                charge=float(line.split()[2])
                foundLine=True
                break
    # Check that charge was found
    if not foundLine:
        raise ValueError('Charge issue with file: '+cosmoFile)
    # Output
    return charge

def getXYZ(cosmoFile):
    """
    getXYZ() gets XYZ information from cosmoFile.

    Parameters
    ----------
    cosmoFile : string
        Path to COSMO file.

    Raises
    ------
    ValueError
        Error raised if coordinates cannot be read.

    Returns
    -------
    atomList : list of ints
        List where each entry is the element of an atom. Converted from a list
        of strings using [xyz2mol.int_atom(atom) for atom in atomList]
    coordsList : list of lists of floats
        List where each entry is a list containing the xyz coordinates of an
        atom.

    """
    # Initialize atom and coordinates lists
    atomList=[]
    coordsList=[]
    # Open cosmoFile file
    with open(cosmoFile,'r') as file:
        # Find "coordinates from COSMO calculation"
        found=False
        for line in file:
            if line.split()=='coordinates from COSMO calculation'.split():
                found=True
                break
        if not found:
            raise ValueError('Unable to read coordinates: '+cosmoFile)
        # Skip line
        file.readline()
        # Read XYZ lines
        for line in file:
            # Split line
            lineSplit=line.split()
            # Check if line is an xyz line
            if len(lineSplit)!=9: break
            # Append atoms and coords
            atomList.append(lineSplit[7])
            coordsList.append([float(lineSplit[1]),
                               float(lineSplit[2]),
                               float(lineSplit[3])])
    # Convert atomList to xyz2mol standards
    atomList=[xyz2mol.int_atom(atom) for atom in atomList]
    # Output
    return atomList,coordsList

def getGraph(mol):
    """
    getGraph() obtains a networkx graph object from an RDKit mol object.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule of interest.

    Returns
    -------
    G : networkx.classes.graph.Graph
        Graph for the molecule of interest.
    """
    # Initialie graph
    G=networkx.Graph()
    # Add atoms as nodes
    for atom in mol.GetAtoms():
        idx=atom.GetIdx()
        G.add_node(idx,element=atom.GetSymbol())
    # Add bonds as edges (ignore bond type)
    for bond in mol.GetBonds():
        i=bond.GetBeginAtomIdx()
        j=bond.GetEndAtomIdx()
        G.add_edge(i,j)
    # Output
    return G

def runCOSMOtherm(cosmoConfig):
    """
    runCOSMOtherm() runs COSMOtherm on the temp folder. It assumes that a 
    single "cosmo.inp" file is present, which was previously generated with
    generateInput().

    Parameters
    ----------
    cosmoConfig : dictionary
        See generateInput().

    Returns
    -------
    None.

    """
    # Get config
    tempFolder=cosmoConfig.get('tempFolder')
    COSMOthermPATH=cosmoConfig.get('COSMOthermPATH')
    # Run COSMOtherm
    subprocess.call('"'+COSMOthermPATH+'\cosmotherm.exe" '+'cosmo.inp',
                    cwd=tempFolder)
    # Wait until output is generated
    while not os.path.exists(os.path.join(tempFolder,'cosmo.out')):
        time.sleep(1)
    # Output
    return None

def getSigmaProfile(cosmoConfig,databasePath,name):
    """
    getSigmaProfile() obtains the sigma profile of the .cosmo file "name".

    Parameters
    ----------
    cosmoConfig : dictionary
        See generateInput().
    databasePath : string
        Path to the folder containing the COSMO files of interest.
    name : string
        Name of cosmo file without conformer information. See generateInput().

    Returns
    -------
    SP : numpy array
        Array of size Nx2 where the first column contains sigma values and the
        second column contains sigma profile values.

    """
    # Load COSMOtherm configuration
    parameterization=cosmoConfig.get('parameterization')
    parameterizationPATH=cosmoConfig.get('parameterizationPATH')
    licencePATH=cosmoConfig.get('licencePATH')
    tempFolder=cosmoConfig.get('tempFolder')
    # Get conformer list for name
    comp_List=glob.glob(os.path.join(databasePath,name+'_*.cosmo'))
    if not comp_List: # If list is empty, look for name.cosmo
        comp_List=glob.glob(os.path.join(databasePath,name+'.cosmo'))
    if not comp_List: # If list is empty, look for name.mcos
        comp_List=glob.glob(os.path.join(databasePath,name+'.mcos'))
    # Check that files were found
    if not comp_List:
        raise ValueError('Unable to file suitable COSMO files for '
                         +name+' in '+databasePath)
    # Generate input script
    with open(os.path.join(tempFolder,'cosmo.inp'),'w') as file:
        file.write('ctd = '+parameterization
                   +' cdir = "'+parameterizationPATH
                   +'" ldir = "'
                   +licencePATH
                   +'"\n')
        file.write('wtln notempty sprf ehfile\n')
        file.write('!! generated by COSMOthermX !!\n')
        # Write comp_List
        if len(comp_List)==1:
            file.write('f = "'
                       +os.path.basename(comp_List[0])
                       +'" fdir="'
                       +databasePath
                       +'" VPfile\n')
        elif len(comp_List)==2:
            file.write('f = "'
                       +os.path.basename(comp_List[0])
                       +'" fdir="'
                       +databasePath
                       +'" Comp = "'
                       +name
                       +'" [  VPfile\n')
            file.write('f = "'
                       +os.path.basename(comp_List[1])
                       +'" fdir="'
                       +databasePath
                       +'" ]\n')
        else:
            file.write('f = "'
                       +os.path.basename(comp_List[0])
                       +'" fdir="'
                       +databasePath
                       +'" Comp = "'
                       +name
                       +'" [  VPfile\n')
            for n in range(1,len(comp_List)-1):
                file.write('f = "'
                           +os.path.basename(comp_List[n])
                           +'" fdir="'
                           +databasePath
                           +'"\n')
            file.write('f = "'
                       +os.path.basename(comp_List[-1])
                       +'" fdir="'
                       +databasePath
                       +'" ]\n')
        # Write main command
        file.write('x_pure = 1 tc=25.0 ctab wconf ignore_charge')
    # Run COSMOtherm
    runCOSMOtherm(cosmoConfig)
    # Get path to temp folder
    tempFolder=cosmoConfig.get('tempFolder')
    # Get path to output
    outputPath=os.path.join(tempFolder,'cosmo.prf')
    # Initialize SP var
    SP=[]
    # Open output file
    with open(outputPath,'r') as file:
        # Skip comments
        for line in file:
            if line[0]=='#':
                continue
            else:
                aux=[float(line.split()[0]),float(line.split()[1])]
                # Append
                SP.append(aux)
                break
        # Loop over sigma profile table
        for line in file:
            aux=[float(line.split()[0]),float(line.split()[1])]
            # Append
            SP.append(aux)
    # Conver to numpy array
    SP=numpy.array(SP)
    # Delete temp files
    for f in os.listdir(tempFolder): os.remove(os.path.join(tempFolder,f))
    # Output
    return SP

# =============================================================================
# Main Script
# =============================================================================
#dup_keys_Cat = (indexFile["Name"].value_counts().loc[lambda x: x > 1])
# Load index file
indexFile=pandas.read_excel(indexPath)
# Check for duplicates
cond1=indexFile.duplicated(subset=['ID']).any()
cond2=indexFile.duplicated(subset=['Name']).any()
cond3=indexFile.duplicated(subset=['SMILES']).any()
cond4=indexFile.duplicated(subset=['InChI_Key']).any()
if cond1 or cond2 or cond3 or cond4:
    raise ValueError('Duplicates found in Index File.')
# Get list of available cosmo files
fileList=[]
for file in Path(cosmoFolder).iterdir():
    # Check that file is file
    if not file.is_file(): raise ValueError('Not file: '+file.name)
    # Check that file is cosmo file
    if not file.suffix=='.cosmo': raise ValueError('Not COSMO: '+file.name)
    # Append name to fileList
    fileList.append(file.stem)
# Check that there are no duplicates
if len(indexFile.iloc[:,0].to_list())!=len(set(indexFile.iloc[:,0].to_list())):
    raise ValueError('Duplicate entries in Index File.')
if len(fileList)!=len(set(fileList)):
    raise ValueError('Duplicate entries in COSMO Folder.')
# Check that there are no missing entries or files
diff1=set(indexFile.iloc[:,0].to_list())-set(fileList)
diff2=set(fileList)-set(indexFile.iloc[:,0].to_list())
if diff1 or diff2:
    raise ValueError('Index File and COSMO Folder do not match.')
# Initialize list of issues
softIssueList=[]
hardIssueList=[]
# Loop over cosmo files list
for file in fileList:
    # Obtain path to file
    filePath=os.path.join(cosmoFolder,file+'.cosmo')
    # Check charge
    indexCharge=indexFile[indexFile['ID']==file]['Charge'].iloc[0]
    fileCharge=getCharge(filePath)
    if indexCharge!=int(round(-fileCharge,0)):
        raise ValueError('Charge mismatch for: '+file)
    # Obtain rdkit object from cosmo file
    atomList,coordsList=getXYZ(filePath)
    try: molList=xyz2mol.xyz2mol(atomList,coordsList,charge=indexCharge)
    except:
        hardIssueList.append([file,'Could not convert XYZ to MOL'])
        continue
    if not molList:
        hardIssueList.append([file,'Could not convert XYZ to MOL'])
        continue
    # Check size of molList
    if len(molList)>1:
        raise ValueError('More than one possibility for: '+file)
    # Remove explicit hydrogens from SMILES, if any
    SMILES_File=Chem.MolToSmiles(molList[0],canonical=True)
    mol=Chem.MolFromSmiles(SMILES_File)
    # Calculate formal charge directly from the mol object
    charge = Chem.GetFormalCharge(mol)
    # Obtain SMILES_Index
    SMILES_Index=indexFile[indexFile['ID']==file]['SMILES'].iloc[0]
    # Ensure charges from SMILES match charge
    #plusCount=SMILES_Index.count('+')
    #minusCount=SMILES_Index.count('-')
    #charge=plusCount-minusCount
    if charge!=indexCharge: 
        raise ValueError('Charge/SMILES mismatch for: '+file)
    # If Index File ignores chirality, ignore chirality in SMILES_File
    if '@' not in SMILES_Index:
        SMILES_File=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False)
        mol=Chem.MolFromSmiles(SMILES_File)
    # Obtain final SMILES_File
    SMILES_File=Chem.MolToSmiles(mol,canonical=True)
    # Obtain inchi keys and check if match
    inchi_Index=indexFile[indexFile['ID']==file]['InChI_Key'].iloc[0]
    inchi_File=Chem.MolToInchiKey(mol)
    if inchi_File==inchi_Index: continue
    # If Inchi keys are different, check connectivity; solves issues like
    # [N+]#S([O-])([O-]) vs. [N-]S(=O)(=O)
    graphFile=getGraph(mol)
    graphIndex=getGraph(Chem.MolFromSmiles(SMILES_Index))
    cond=networkx.is_isomorphic(graphFile,graphIndex,
            node_match=networkx.algorithms.isomorphism.categorical_node_match(
                'element',None))
    # Decide on whether this is a soft or hard issue
    if cond: softIssueList.append([file,'Mismatch but Isomorphic'])
    else: hardIssueList.append([file,'Mismatch'])
    
# Define list of false positives:
falsePositives=['ANI000001','ANI000005','ANI000018','ANI000052','ANI000053',
                'ANI000056','ANI000069','ANI000074','ANI000085','ANI000101',
                'ANI000109','ANI000144','ANI000145','ANI000149','ANI000157',
                'ANI000163','ANI000173','ANI000177',
                'CAT000385','CAT000390','CAT000574','CAT000598','CAT000627',
                'CAT000629','CAT000698','CAT000709','CAT000773'
                ]
# Print issues
print('\n\nList of Soft Issues:')
for issue in softIssueList: print(issue[1]+' : '+issue[0])
print('\n\nList of Hard Issues:')
for issue in hardIssueList:
    if issue[0] not in falsePositives: print(issue[1]+' : '+issue[0])
# Obtain sigma profiles
SP_List=[]
currentMin=0
currentMax=0
for file in tqdm(fileList,'Compiling: '):
    # Obtain sigma profile
    SP=getSigmaProfile(cosmoConfig,cosmoFolder,file)
    # Update current bounds
    currentMin=min(currentMin,SP[:,0].min())
    currentMax=max(currentMax,SP[:,0].max())
    # Append
    SP_List.append([file,SP])
    

# Place all sigma profiles into a common sigma axis
sigmaAxis=numpy.arange(currentMin,currentMax+0.001,0.001)
spDB_temp=pandas.DataFrame(columns=['I2']+[str(round(x,3)) for x in sigmaAxis])
for entry in SP_List:
    startIndex=numpy.where(numpy.isclose(sigmaAxis,entry[1][0,0]))[0][0]
    endIndex=numpy.where(numpy.isclose(sigmaAxis,entry[1][-1,0]))[0][0]
    tempSP=numpy.zeros(sigmaAxis.shape)
    tempSP[startIndex:endIndex+1]=entry[1][:,1]
    spDB_temp.loc[len(spDB_temp)]=[entry[0]]+tempSP.copy().tolist()
# Concatenate with Index File
if (indexFile.iloc[:,0]!=spDB_temp.iloc[:,0]).any(): raise ValueError('Error')
spDB=pandas.concat((indexFile,spDB_temp),axis=1)
spDB=spDB.drop('I2',axis=1)
# Check for duplicate sigma profiles
tolerance=2
X=spDB.iloc[:,5:].to_numpy()
# Pairwise Euclidean distances
dist_matrix=squareform(pdist(X,metric="euclidean"))
# Find duplicates (excluding self-comparisons)
i,j=numpy.where((dist_matrix<=tolerance)&(dist_matrix>0)&(numpy.triu(
    numpy.ones(dist_matrix.shape),k=1).astype(bool)))
duplicate_pairs = list(zip(i, j))
if duplicate_pairs:
    for pair in duplicate_pairs:
        plt.plot(sigmaAxis,X[pair[0],:],'-k',label=SP_List[pair[0]][0])
        plt.plot(sigmaAxis,X[pair[1],:],'--r',label=SP_List[pair[1]][0])
        plt.legend()
        plt.show()
# Save
spDB.to_csv(outputPath, index=False)

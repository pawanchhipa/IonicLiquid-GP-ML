# -*- coding: utf-8 -*-
"""
02_Property_DB_Compiler.py

Compiles an ML-ready property database by combining experimental ionic
liquid data with cation and anion sigma profile databases.

The script has three main objectives:
    - Identify cation and anion components from IL SMILES.
    - Match ionic components using SMILES and InChI keys.
    - Calculate combined sigma profiles and record unmatched entries.

The input property database contains the IL name, SMILES, and measured
property, with temperature and pressure included where applicable.

Sections:
    . Imports
    . Configuration
    . Main Script

Authors: Pawan Chhipa & Dinis O. Abranches
Last Updated: 2026-10-14

"""

# =============================================================================
# 1. Imports
# =============================================================================

# General
import os
from pathlib import Path
# Specific
import numpy
import pandas
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from tqdm import tqdm


# =============================================================================
# 2. Configuration
# =============================================================================

# Repository directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

# Experimental property dataset
PROPERTY_FILE = "Pure_ILs_Enthalpy.csv"
propertyPath = DATA_DIR / PROPERTY_FILE

# Ion sigma profile databases
anionsPath = DATA_DIR / "Anions" / "SP_Database.csv"
cationsPath = DATA_DIR / "Cations" / "SP_Database.csv"

# Processing logs
errorDir = DATA_DIR / "Logs"
errorDir.mkdir(parents=True, exist_ok=True)

# Property identifier for log filenames
propTag = propertyPath.stem.replace("Pure_ILs_", "")

# =============================================================================
# 3. Main Script
# =============================================================================
# Read databases
propertyDB = pandas.read_csv(propertyPath)

anionsDB = pandas.read_csv(anionsPath)
cationsDB = pandas.read_csv(cationsPath)
# Change sigma axis and pad with zeros if needed
anionSigma = anionsDB.columns[5:].astype("double")
cationSigma = cationsDB.columns[5:].astype("double")
sigmaMin = min(anionSigma.min(), cationSigma.min())
sigmaMax = max(anionSigma.max(), cationSigma.max())
fullSigma = numpy.round(numpy.arange(sigmaMin, sigmaMax + 0.001, 0.001), 3)
fullCols = list(anionsDB.columns[:5]) + fullSigma.tolist()
anionsDB = anionsDB.set_axis(list(anionsDB.columns[:5]) + anionSigma.tolist(),
                             axis=1)
cationsDB = cationsDB.set_axis(
    list(cationsDB.columns[:5]) + cationSigma.tolist(), axis=1
)
anionsDB = anionsDB.reindex(columns=fullCols, fill_value=0.0)
cationsDB = cationsDB.reindex(columns=fullCols, fill_value=0.0)
# Initialize Issues
notUnderstood = []
anionsNotFound = []
cationsNotFound = []
smilesMissing = []
parseFailed = []
hasNeutral = []
# Loop over propertyDB
spList = []
for i, row in tqdm(
    propertyDB.iterrows(), "Processing Data Points: ", total=propertyDB.shape[0]
):
    # Retrieve info
    name = row.iloc[0]
    originalSMILES = row.iloc[1]
    if not isinstance(originalSMILES, str) or not originalSMILES.strip():
        smilesMissing.append(i)
        continue
    # temp=row.iloc[2]
    # press=row.iloc[3]
    prop = row.iloc[2]
    # Split SMILES
    SMILES_List = originalSMILES.split(".")
    # Obtain list of cations and anions
    cationsList = []
    anionsList = []
    neutralsList = []
    failedParses = []
    for SMILES in SMILES_List:
        mol = Chem.MolFromSmiles(SMILES)
        if mol is None:
            failedParses.append(SMILES)
            continue
        charge = Chem.GetFormalCharge(mol)
        if charge < 0:
            anionsList.append(SMILES)
        elif charge > 0:
            cationsList.append(SMILES)
        else:
            neutralsList.append(SMILES)
    if failedParses:
        parseFailed.append(i)
    if neutralsList:
        hasNeutral.append(i)
        continue
    # Check that anions are present in DB
    allAnionsFound = True
    anionsSP = 0
    for anion in anionsList:
        # Try to find direct SMILES match
        match = anionsDB[anionsDB["SMILES"] == anion]
        if not match.empty:
            anionsSP += match.iloc[0, 5:].to_numpy("double")
            continue
        # If no direct match exists, try to match inchi key
        mol = Chem.MolFromSmiles(anion)
        inchi = Chem.MolToInchiKey(mol)
        match = anionsDB[anionsDB["InChI_Key"] == inchi]
        if not match.empty:
            anionsSP += match.iloc[0, 5:].to_numpy("double")
            continue
        # If no match was found, skip
        allAnionsFound = False
        break
    if not allAnionsFound:
        anionsNotFound.append(i)
    # Check that cations are present in DB
    allCationsFound = True
    cationsSP = 0
    for cation in cationsList:
        # Try to find direct SMILES match
        match = cationsDB[cationsDB["SMILES"] == cation]
        if not match.empty:
            cationsSP += match.iloc[0, 5:].to_numpy("double")
            continue
        # If no direct match exists, try to match inchi key
        mol = Chem.MolFromSmiles(cation)
        inchi = Chem.MolToInchiKey(mol)
        match = cationsDB[cationsDB["InChI_Key"] == inchi]
        if not match.empty:
            cationsSP += match.iloc[0, 5:].to_numpy("double")
            continue
        # If no match was found, skip
        allCationsFound = False
        break
    if not allCationsFound:
        cationsNotFound.append(i)
    # If all ions found, compute sigma profile
    if not allCationsFound or not allAnionsFound:
        continue
    spIL = anionsSP / len(SMILES_List) + cationsSP / len(SMILES_List)
    spList.append(list(row) + spIL.tolist())
# Print issues
knownMissingAnions = [""]
for entry in anionsNotFound:
    print("Anion not found for: " + propertyDB.iloc[entry, 0])
for entry in cationsNotFound:
    print("Cation not found for: " + propertyDB.iloc[entry, 0])
for entry in notUnderstood:
    print("IL not understood: " + propertyDB.iloc[entry, 0])

# ---- Problems CSV ----
problems = pandas.concat(
    [
        pandas.DataFrame(
            {"name": propertyDB.iloc[anionsNotFound, 0], 
             "issue": "anion_missing"}
        ),
        pandas.DataFrame(
            {"name": propertyDB.iloc[cationsNotFound, 0], "issue": 
             "cation_missing"}
        ),
        pandas.DataFrame(
            {"name": propertyDB.iloc[hasNeutral, 0], "issue": 
             "neutral_component"}
        ),
        pandas.DataFrame(
            {"name": propertyDB.iloc[notUnderstood, 0], "issue": 
             "not_understood"}
        ),
        pandas.DataFrame(
            {"name": propertyDB.iloc[smilesMissing, 0], "issue": 
             "missing_SMILES"}
        ),
        pandas.DataFrame(
            {"name": propertyDB.iloc[parseFailed, 0], "issue": 
             "rdkit_parse_failed"}
        ),
    ]
)
problems = (
    problems.groupby("name")["issue"]
    .apply(lambda x: ",".join(sorted(set(x))))
    .reset_index()
)
problems.to_csv(
    os.path.join(errorDir, f"problematic_unique_entries_with_issue_{propTag}.csv"),
    index=False,
)

# ---- Side-by-side lists, full IL name, deduplicated ----
anion_missing_ils = sorted(set(propertyDB.iloc[anionsNotFound, 0]))
cation_missing_ils = sorted(set(propertyDB.iloc[cationsNotFound, 0]))
smiles_missing_ils = sorted(set(propertyDB.iloc[smilesMissing, 0]))
parse_failed_ils = sorted(set(propertyDB.iloc[parseFailed, 0]))
neutral_ils = sorted(set(propertyDB.iloc[hasNeutral, 0]))

final_missing = pandas.DataFrame(
    {
        "anion_missing_IL": pandas.Series(anion_missing_ils),
        "cation_missing_IL": pandas.Series(cation_missing_ils),
        "smiles_missing_IL": pandas.Series(smiles_missing_ils),
        "parse_failed_IL": pandas.Series(parse_failed_ils),
    }
)
final_missing.to_csv(
    os.path.join(errorDir, f"missing_anion_cation_lists_{propTag}.csv"), 
    index=False
)

# ---- Cumulative run log (one row appended per run) ----
import datetime

logPath = os.path.join(errorDir, "compiler_run_log.csv")
logRow = pandas.DataFrame(
    [
        {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "property": propTag,
            "source_file": os.path.basename(propertyPath),
            "rows_in": len(propertyDB),
            "rows_kept": len(spList),
            "rows_dropped": len(propertyDB) - len(spList),
            "missing_smiles": len(smilesMissing),
            "missing_anion": len(anionsNotFound),
            "missing_cation": len(cationsNotFound),
            "parse_failed": len(parseFailed),
            "unique_ILs_in": propertyDB.iloc[:, 0].nunique(),
            "problem_ILs": len(problems),
        }
    ]
)
logRow.to_csv(logPath, mode="a", header=not os.path.exists(logPath), index=False)

print(f"\nUnique ILs with missing SMILES: {len(smiles_missing_ils)}")
print(f"Unique ILs with missing anion:  {len(anion_missing_ils)}")
print(f"Unique ILs with missing cation: {len(cation_missing_ils)}")
print(f"Unique ILs failing RDKit parse: {len(parse_failed_ils)}")
print(f"Rows kept: {len(spList)} / {len(propertyDB)}")
print(f"\nUnique ILs with missing anion:  {len(anion_missing_ils)}")
print(f"Unique ILs with missing cation: {len(cation_missing_ils)}")

# Generate ML-ready property database
mlDB = pandas.DataFrame(
    spList,
    columns=list(propertyDB.columns) + fullSigma.tolist(),
)
# Save database
outputPath = DATA_DIR / f"ML-Ready-{propertyPath.name}"
mlDB.to_csv(outputPath, index=False)
print(f"ML-ready database saved to: {outputPath}")

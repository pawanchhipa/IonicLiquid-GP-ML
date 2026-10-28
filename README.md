# Beyond Property Prediction: Artificial Intelligence for Ionic Liquid Data Curation and Chemical Space Exploration

This repository contains the Python scripts and datasets associated with the study of Gaussian Process (GP)-based curation of experimental ionic liquid (IL) thermophysical property data, property prediction using COSMO-RS sigma profiles, and exploration of the combinatorial IL space.

## Repository structure

```text
Python/
    00_Data_Extraction.py
    01_SigmaProfile_DB_Compiler.py
    02_Property_DB_Compiler.py
    03_DataCuration.py
    04_Training.py
    05_Combinational_Space.py
Data/
    ILThermo/
    Anions/
        Index
        SP_Database
    Cations/
        Index
        SP_Database
    Curation_ML-Ready-Pure_ILs_*/
    Logs/
    Pure_ILs_*.csv
    ML-Ready-Pure_ILs_*.csv
    Combinatorial_IL_Database.csv
README.md
```

The asterisk denotes property-specific files and folders. Curation folders contain cleaned datasets, flagged observations, and supporting diagnostics.

## Scripts

| Script | Description |
| --- | --- |
| `00_Data_Extraction.py` | Extracts and processes experimental property records from ILThermo. |
| `01_SigmaProfile_DB_Compiler.py` | Checks ion records and compiles COSMO-RS sigma profiles. |
| `02_Property_DB_Compiler.py` | Combines experimental property data with sigma-profile descriptors. |
| `03_DataCuration.py` | Identifies inconsistent observations using GP-based data curation. |
| `04_Training.py` | Trains and evaluates GP thermophysical property prediction models. |
| `05_Combinational_Space.py` | Generates 286,650 charge-balanced IL combinations from 975 cations and 294 anions, and compares the combinatorial descriptor space with experimentally represented ILs using PCA. |

## Usage

Configure input paths and property-specific settings in each script before running. The numbered scripts follow the data preparation, curation, modelling, and chemical space workflow. Sigma-profile compilation requires COSMOtherm. Selected large datasets use Git Large File Storage (Git LFS).

## Authors of the Associated Research Paper

**Pawan Chhipa¹, Dinis O. Abranches², Oliver Järvik¹**

¹ Department of Energy Technology, Tallinn University of Technology, Ehitajate tee 5, 19086 Tallinn, Estonia.

² CICECO-Aveiro Institute of Materials, Department of Chemistry, University of Aveiro, 3810-193 Aveiro, Portugal.

**Corresponding authors:** [jdinis@ua.pt](mailto:jdinis@ua.pt), [oliver.jarvik@taltech.ee](mailto:oliver.jarvik@taltech.ee)

## Acknowledgements

This work was supported by the Estonian Research Council grant (PRG1784). This work was also developed within the scope of the project CICECO-Aveiro Institute of Materials, UID/50011/2025 (DOI 10.54499/UID/50011/2025) & LA/P/0006/2020 (DOI 10.54499/LA/P/0006/2020), financed by national funds through the FCT/MCTES (PIDDAC).

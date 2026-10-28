# -*- coding: utf-8 -*-
"""
00_Data_Extraction.py

Extracts thermophysical property data for ionic liquids from the NIST
ILThermo database using ILThermoPy. The script has three main objectives:
    - Retrieve experimental property records and associated metadata.
    - Identify property types, standardize units, and classify phases
      and mixtures.
    - Generate property-specific datasets and processing logs.

Sections:
    . Imports
    . Configuration
    . Property and Unit Definitions
    . Data Retrieval and Parsing
    . Unit Validation and Conversion
    . Phase and Mixture Classification
    . Property Dataset Construction
    . Data Extraction Workflow

Authors: Pawan Chhipa & Dinis O. Abranches
Last Updated: 2026-10-14

"""

# =============================================================================
# 1. Imports
# =============================================================================

import os, re, time, random, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

# =============================================================================
# 2. Configuration
# =============================================================================

# Repository root directory
BASE_DIR = Path(__file__).resolve().parent.parent
# ILThermo database output directory
OUTPUT_DIR = BASE_DIR / "Data" / "ILThermo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORKERS, RETRY_ATTEMPTS, RETRY_SLEEP_BASE = 12, 3, 0.8

# =============================================================================
# 3. Property and Unit Definitions
# =============================================================================

UNIT_CONVERSION_MAP = {
    "temperature": {
        "target_unit": "K",
        "conversions": {"C": lambda x: x + 273.15, "K": lambda x: x},
    },
    "pressure": {
        "target_unit": "kPa",
        "conversions": {
            "Pa": lambda x: x / 1000.0,
            "kPa": lambda x: x,
            "MPa": lambda x: x * 1000.0,
            "bar": lambda x: x * 100.0,
            "atm": lambda x: x * 101.325,
        },
    },
    "Equilibrium_pressure": {
        "target_unit": "kPa",
        "conversions": {
            "Pa": lambda x: x / 1000.0,
            "kPa": lambda x: x,
            "MPa": lambda x: x * 1000.0,
            "bar": lambda x: x * 100.0,
            "atm": lambda x: x * 101.325,
        },
    },
    "viscosity": {
        "target_unit": "Pa·s",
        "conversions": {
            "Pa·s": lambda x: x,
            "Pa s": lambda x: x,
            "mPa·s": lambda x: x * 0.001,
            "mPa s": lambda x: x * 0.001,
            "cP": lambda x: x * 0.001,
        },
    },
    "surface_tension": {
        "target_unit": "N/m",
        "conversions": {
            "N/m": lambda x: x,
            "mN/m": lambda x: x * 0.001,
            "dyn/cm": lambda x: x * 0.001,
        },
    },
    "density": {
        "target_unit": "kg/m³",
        "conversions": {"kg/m³": lambda x: x, "g/cm³": lambda x: x * 1000.0},
    },
    "heat_capacity": {
        "target_unit": "J/K/mol",
        "conversions": {
            "J/K/mol": lambda x: x,
            "J/(K·mol)": lambda x: x,
            "J/mol/K": lambda x: x,
            "cal/K/mol": lambda x: x * 4.184,
        },
    },
    "thermal_conductivity": {
        "target_unit": "W/m/K",
        "conversions": {
            "W/m/K": lambda x: x,
            "W/(m·K)": lambda x: x,
            "mW/m/K": lambda x: x * 0.001,
        },
    },
    # Melting temperatures use the same unit conversions as temperature.
    "melting_temperature": {
        "target_unit": "K",
        "conversions": {"C": lambda x: x + 273.15, "K": lambda x: x},
    },
    # Enthalpy: single-point property (one value per substance at a transition)
    "enthalpy": {
        "target_unit": "kJ/mol",
        "conversions": {
            "kJ/mol": lambda x: x,
            "J/mol": lambda x: x / 1000.0,
            "cal/mol": lambda x: x * 4.184 / 1000.0,
            "kcal/mol": lambda x: x * 4.184,
        },
    },
    "enthalpy_vaporization": {
        "target_unit": "kJ/mol",
        "conversions": {
            "kJ/mol": lambda x: x,
            "J/mol": lambda x: x / 1000.0,
            "cal/mol": lambda x: x * 4.184 / 1000.0,
            "kcal/mol": lambda x: x * 4.184,
        },
    },
}

# Match specific property names before general temperature descriptors.
PROPERTY_CHECK_ORDER = [
    "heat_capacity",
    "thermal_conductivity",
    "surface_tension",
    "Equilibrium_pressure",
    "molar_volume",
    "specific_volume",
    "enthalpy",  # <── single-point: must come before generic matches
    "melting_temperature",  # <── single-point: must come BEFORE "temperature"
    "temperature",
    "pressure",
    "viscosity",
    "density",
]

# Types where phase assignment makes no sense
# (condition variables, phase-transition properties, equilibrium measurements)
PHASE_SKIP_TYPES = {
    "temperature",
    "pressure",
    "Equilibrium_pressure",
    "melting_temperature",
    "enthalpy",
}

PROPERTY_PATTERNS = {
    "heat_capacity": re.compile(r"\bheat\s*capacity\b", re.I),
    "thermal_conductivity": re.compile(r"\bthermal\s*conductivity\b", re.I),
    "surface_tension": re.compile(r"\bsurface\s*tension\b|\bgamma\b", re.I),
    "Equilibrium_pressure": re.compile(
        r"\bequilibrium\s*pressure\b|\bvapor\s*pressure\b", re.I
    ),
    "molar_volume": re.compile(r"\bmolar\s*volume\b", re.I),
    "specific_volume": re.compile(r"\bspecific\s*volume\b", re.I),
    # Single-point property patterns
    "enthalpy": re.compile(r"\benthalpy\b", re.I),
    "enthalpy_vaporization": re.compile(
        r"\benthalpy\s*(of\s*)?(vaporization|sublimation|evaporation)\b", re.I
    ),
    "melting_temperature": re.compile(
        r"\bmelting\s*(temperature|point|temp)\b|\bnormal\s*melting\b"
        r"|\bfusion\s*temperature\b", re.I,
    ),
    "temperature": re.compile(r"^temperature\b|\btemperature\s*,", re.I),
    "pressure": re.compile(r"^pressure\b|\bpressure\s*,", re.I),
    "viscosity": re.compile(r"\bviscosity\b", re.I),
    "density": re.compile(r"\bdensity\b|\bspecific\s*density\b|\brho\b", re.I),
}

UNIT_PATTERNS = {
    "J/K/mol": re.compile(r"\bJ/K/mol\b|\bJ/\(K·mol\)\b|\bJ/mol/K\b", re.I),
    "J/K/m³": re.compile(
        r"\bJ/K/m³\b|\bJ/K/m<SUP>3</SUP>|\bJ/K/m\^3\b|\bJ/K/m3\b|\bJ/\(K·m³\)\b", re.I
    ),
    # Enthalpy units (kJ/mol before J/mol — same milli-prefix logic)
    "kJ/mol": re.compile(r"\bkJ/mol\b|\bkJ/\(mol\)", re.I),
    "kcal/mol": re.compile(r"\bkcal/mol\b", re.I),
    "J/mol": re.compile(
        r"\bJ/mol\b(?!/K)", re.I
    ),
    "cal/mol": re.compile(r"\bcal/mol\b(?!.*kcal)", re.I),
    "W/m/K": re.compile(r"\bW/m/K\b|\bW/\(m·K\)\b", re.I),
    "mW/m/K": re.compile(r"\bmW/m/K\b", re.I),
    "Pa·s": re.compile(r"\bPa\s*·\s*s\b|\bPa\s*s\b|\bPa·s\b", re.I),
    "mPa·s": re.compile(r"\bmPa\s*·\s*s\b|\bmPa\s*s\b|\bmPa·s\b|\bcP\b", re.I),
    "kg/m³": re.compile(
        r"\bkg/m³\b|\bkg/m<SUP>3</SUP>|\bkg/m\^3\b|\bkg/m3\b|\bkg/m<sup>3</sup>", re.I
    ),
    "g/cm³": re.compile(
        r"\bg/cm³\b|\bg/cm<SUP>3</SUP>|\bg/cm\^3\b|\bg/cm3\b|\bg/cm<sup>3</sup>", re.I
    ),
    "N/m": re.compile(r"\bN/m\b", re.I),
    "mN/m": re.compile(r"\bmN/m\b", re.I),
    "MPa": re.compile(r"\bMPa\b", re.I),
    "kPa": re.compile(r"\bkPa\b", re.I),
    "bar": re.compile(r"\bbar\b", re.I),
    "atm": re.compile(r"\batm\b", re.I),
    "Pa": re.compile(r"\bPa\b(?!\s*s)", re.I),
    "K": re.compile(r"\bK\b(?!/)", re.I),
    "C": re.compile(r"°C|\bC\b|celsius|degC", re.I),
}

# Check prefixed units before their corresponding base units.
UNIT_CHECK_ORDER = [
    "J/K/mol",
    "J/K/m³",
    "kJ/mol",
    "kcal/mol",
    "J/mol",
    "cal/mol",  # <── enthalpy units (specific before broad)
    "mW/m/K",
    "W/m/K",
    "mPa·s",
    "Pa·s",
    "kg/m³",
    "g/cm³",
    "mN/m",
    "N/m",
    "MPa",
    "kPa",
    "bar",
    "atm",
    "Pa",
    "K",
    "C",
]

PROPERTIES = [
    # {
    #     "property_key": "SwyC",
    #     "property_name": "Equilibrium pressure",
    #     "property_type": "Equilibrium_pressure",
    # },
    # {
    #     "property_key": "PusA",
    #     "property_name": "Viscosity",
    #     "property_type": "viscosity",
    # },
    {
        "property_key": "ETUw",
        "property_name": "Surface tension liquid-gas",
        "property_type": "surface_tension",
    },
    # {
    #     "property_key": "Qneg",
    #     "property_name": "Heat capacity at constant pressure",
    #     "property_type": "heat_capacity",
    # },
    # {
    #     "property_key": "KvgF",
    #     "property_name": "Heat capacity (Cv)",
    #     "property_type": "heat_capacity",
    # },
    # {
    #     "property_key": "zJIE",
    #     "property_name": "Heat capacity (vapor saturation)",
    #     "property_type": "heat_capacity",
    # },
    # {
    #     "property_key": "CXUw",
    #     "property_name": "Enthalpy of transition or fusion",
    #     "property_type": "enthalpy",
    # },
    # {
    #     "property_key": "JkYu",
    #     "property_name": "Density",
    #     "property_type": "density",
    # },
    # {
    #     "property_key": "iaOF",
    #     "property_name": "Enthalpy of vaporization or sublimation",
    #     "property_type": "enthalpy_vaporization",
    # },
    # {
    #     "property_key": "NmYB",
    #     "property_name": "Normal Melting Temperature",
    #     "property_type": "melting_temperature",
    # },
    # {
    #     "property_key": "pAFI",
    #     "property_name": "Thermal Conductivity",
    #     "property_type": "thermal_conductivity",
    # },
]

try:
    import ilthermopy
except Exception as e:
    raise RuntimeError("ilthermopy required") from e

DESIRED_ORDER = [
    "entry_id",
    "property",
    "reference",
    "component1",
    "component1_smiles",
    "component2",
    "component2_smiles",
    "component3",
    "component3_smiles",
    "V1",
    "V1_property",
    "V1_type",
    "V1_unit",
    "V1_phase",
    "V2",
    "V2_property",
    "V2_type",
    "V2_unit",
    "V2_phase",
    "dV2",
    "dV2_property",
    "V3",
    "V3_property",
    "V3_type",
    "V3_unit",
    "V3_phase",
    "V4",
    "V4_property",
    "V4_type",
    "V4_unit",
    "V4_phase",
    "dV4",
    "dV4_property",
    "V5",
    "V5_property",
    "V5_type",
    "V5_unit",
    "V5_phase",
    "dV5",
    "dV5_property",
]

VALUE_COLS = ["V1", "V2", "V3", "V4", "V5"]
PROP_COLS = [f"V{i}_property" for i in range(1, 6)]
TYPE_COLS = [f"V{i}_type" for i in range(1, 6)]
PHASE_COLS = [f"V{i}_phase" for i in range(1, 6)]
UNIT_COLS = [f"V{i}_unit" for i in range(1, 6)]

NUM_RE = re.compile(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?")
PHASE_PATTERNS = {
    p: re.compile(f"\\b{p}\\b", re.I)
    for p in ["liquid", "gas", "vapor", "solid", "crystal", "glass", "amorphous"]
}

def empty(x):
    return pd.isna(x) or str(x).strip() == ""

def parse_num(t):
    m = NUM_RE.search(str(t) if t else "")
    return float(m.group()) if m else None

def detect_unit(t):
    """Detect unit: known units first, then fallback to UNKNOWN unit extraction"""
    if not isinstance(t, str):
        return None

    # Step 1: Check known units in order
    for unit_key in UNIT_CHECK_ORDER:
        if UNIT_PATTERNS[unit_key].search(t):
            return unit_key

    # Step 2: FALLBACK - Extract unknown units from property text
    try:
        text_without_phase = t.split("=>")[0] if "=>" in t else t
        parts = text_without_phase.split(",")
        if len(parts) >= 2:
            unit_candidate = parts[-1].strip()
            if unit_candidate and len(unit_candidate) > 0 and len(unit_candidate) < 60:
                if any(
                    c in unit_candidate
                    for c in [
                        "/",
                        "^",
                        "°",
                        "·",
                        "<",
                        ">",
                        "K",
                        "k",
                        "P",
                        "M",
                        "W",
                        "J",
                        "C",
                        "m",
                        "g",
                        "N",
                        "c",
                        "D",
                        "a",
                        "e",
                        "%",
                        "L",
                    ]
                ):
                    return f"UNKNOWN[{unit_candidate[:45]}]"
    except:
        pass

    return None

def detect_prop_type(pt):
    """Detect property type from property text"""
    if empty(pt):
        return None

    desc = str(pt).split(",")[0].strip().lower()

    for ptype in PROPERTY_CHECK_ORDER:
        if PROPERTY_PATTERNS[ptype].search(desc):
            return ptype

    return None

def parse_prop_col(pt):
    """Parse property column: extract unit and phase"""
    if empty(pt):
        return ("", None, None, "")
    pt = str(pt).strip()
    phase = None
    if "=>" in pt:
        pt, ph = pt.split("=>", 1)
        ph = ph.strip().lower()
        phase = next((p for p, pat in PHASE_PATTERNS.items() if pat.search(ph)), None)
    unit = detect_unit(pt)
    return (pt, phase, unit, pt)

def convert_val(v, ptype, unit):
    if v is None or unit is None or ptype not in UNIT_CONVERSION_MAP:
        return v, False
    conv_map = UNIT_CONVERSION_MAP[ptype]
    if unit not in conv_map["conversions"] or unit == conv_map["target_unit"]:
        return v, False
    try:
        return conv_map["conversions"][unit](v), True
    except:
        return v, False

# =============================================================================
# 4. Data Retrieval and Parsing
# =============================================================================

def get_entry(eid, mx=RETRY_ATTEMPTS):
    for _ in range(mx):
        try:
            return ilthermopy.GetEntry(eid)
        except:
            time.sleep(RETRY_SLEEP_BASE + random.uniform(0, 0.8))
    return None

def process_entry(eid, plbl):
    entry = get_entry(eid)
    if entry is None:
        return None
    try:
        data = entry.data.copy()
    except:
        try:
            data = pd.DataFrame([entry.data])
        except:
            return None
    data["entry_id"], data["reference"], data["property"] = (
        eid,
        str(getattr(entry, "ref", "")),
        plbl,
    )
    if hdr := getattr(entry, "header", None):
        try:
            for k, v in hdr.items():
                if k in data.columns:
                    data[f"{k}_property"] = v
        except Exception as e:
            print(f"  ⚠ Header parse error for {eid}: {e}")
    if comps := getattr(entry, "components", None):
        try:
            for i, c in enumerate(comps, 1):
                data[f"component{i}"] = getattr(c, "name", None)
                data[f"component{i}_smiles"] = getattr(c, "smiles", None)
                data[f"component{i}_id"] = getattr(c, "id", None)
                data[f"component{i}_formula"] = getattr(c, "formula", None)
                data[f"component{i}_mw"] = getattr(c, "mw", None)
                data[f"component{i}_smiles_error"] = getattr(c, "smiles_error", None)
        except Exception as e:
            print(f"  ⚠ Component parse error for {eid}: {e}")
    return data


def normalize_concat(dfs):
    if not dfs:
        return pd.DataFrame()
    cols = list(set(c for df in dfs for c in df.columns))
    return pd.concat([df.reindex(columns=cols) for df in dfs], ignore_index=True)


def reorder_cols(df, desired=DESIRED_ORDER):
    if df.empty:
        return df
    cols = [c for c in desired if c in df.columns] + [
        c for c in df.columns if c not in desired
    ]
    return df.reindex(columns=cols)

def analyze_all_columns(df):
    """Identify property types, units, and phases in V1–V5."""
    df = df.copy()

    for i in range(1, 6):
        vcol, pcol = f"V{i}", f"V{i}_property"
        type_col, unit_col, phase_col = f"V{i}_type", f"V{i}_unit", f"V{i}_phase"

        for idx, row in df.iterrows():
            if not empty(row.get(pcol)):
                prop_text = str(row.get(pcol))
                clean_text = (
                    prop_text.split("=>")[0].strip() if "=>" in prop_text else prop_text
                )
                detected_type = detect_prop_type(clean_text)
                df.at[idx, type_col] = detected_type
                _, phase, unit, _ = parse_prop_col(prop_text)
                df.at[idx, unit_col] = unit
                # Transition properties do not require a phase assignment.
                if detected_type not in PHASE_SKIP_TYPES:
                    df.at[idx, phase_col] = phase
    return df

# =============================================================================
# 5. Unit Validation and Conversion
# =============================================================================

def check_units(df):
    """Check if units match expected for each property type, fix mislabels in same pass"""
    df = df.copy()
    logs = []
    for i in range(1, 6):
        vcol, type_col, unit_col = f"V{i}", f"V{i}_type", f"V{i}_unit"
        for idx, row in df.iterrows():
            ptype = row.get(type_col)
            unit = row.get(unit_col)
            val = row.get(vcol)
            # Correct known unit label inconsistencies
            if ptype == "viscosity" and unit == "Pa":
                unit = "Pa·s"
                df.at[idx, unit_col] = unit
            if ptype == "surface_tension" and unit == "Pa":
                unit = "N/m"
                df.at[idx, unit_col] = unit
            # Now check
            if empty(ptype) or empty(unit) or empty(val):
                continue
            expected_unit = UNIT_CONVERSION_MAP.get(ptype, {}).get("target_unit")
            if expected_unit is None:
                continue
            if str(unit).startswith("UNKNOWN["):
                status, flag = "unknown_unit", "UNKNOWN"
            elif unit == expected_unit:
                status, flag = "ok", "OK"
            else:
                status, flag = "unit_mismatch", "WARNING"
            logs.append(
                {
                    "row": idx + 1,
                    "entry_id": row.get("entry_id"),
                    "comp": row.get("component1"),
                    "col": vcol,
                    "property_type": ptype,
                    "expected_unit": expected_unit,
                    "found_unit": unit,
                    "value": val,
                    "status": status,
                    "flag": flag,
                }
            )
    return df, pd.DataFrame(logs)

def apply_conv(df):
    """Apply conversions and identify different unit categories"""
    df, logs = df.copy(), []
    unconvertible_rows = set()
    unknown_unit_rows = set()

    for i in range(1, 6):
        vcol, type_col, unit_col, pcol = (
            f"V{i}",
            f"V{i}_type",
            f"V{i}_unit",
            f"V{i}_property",
        )
        for idx, row in df.iterrows():
            ptype = row.get(type_col)
            unit = row.get(unit_col)
            val = parse_num(row.get(vcol))

            if val is None or unit is None or ptype is None:
                continue

            # ── Identify property types without supported conversions. ──
            if ptype not in UNIT_CONVERSION_MAP:
                if ptype in ("molar_volume", "specific_volume"):
                    logs.append(
                        {
                            "row": idx + 1,
                            "entry_id": row.get("entry_id"),
                            "comp": row.get("component1"),
                            "col": vcol,
                            "property_type": ptype,
                            "orig_val": val,
                            "orig_unit": unit,
                            "conv_val": None,
                            "tgt_unit": "N/A",
                            "status": "unconvertible_property_type",
                            "flag": "UNCONVERTIBLE",
                        }
                    )
                    unconvertible_rows.add(idx)
                continue

            if str(unit).startswith("UNKNOWN["):
                logs.append(
                    {
                        "row": idx + 1,
                        "entry_id": row.get("entry_id"),
                        "comp": row.get("component1"),
                        "col": vcol,
                        "property_type": ptype,
                        "orig_val": val,
                        "orig_unit": unit,
                        "conv_val": None,
                        "tgt_unit": UNIT_CONVERSION_MAP[ptype]["target_unit"],
                        "status": "unknown_unit_detected",
                        "flag": "UNKNOWN",
                    }
                )
                unknown_unit_rows.add(idx)
                continue

            if ptype == "heat_capacity" and unit == "J/K/m³":
                logs.append(
                    {
                        "row": idx + 1,
                        "entry_id": row.get("entry_id"),
                        "comp": row.get("component1"),
                        "col": vcol,
                        "property_type": ptype,
                        "orig_val": val,
                        "orig_unit": unit,
                        "conv_val": None,
                        "tgt_unit": UNIT_CONVERSION_MAP[ptype]["target_unit"],
                        "status": "unconvertible_variant",
                        "flag": "UNCONVERTIBLE",
                    }
                )
                unconvertible_rows.add(idx)
                continue

            cval, conv = convert_val(val, ptype, unit)
            if conv:
                tgt = UNIT_CONVERSION_MAP[ptype]["target_unit"]
                logs.append(
                    {
                        "row": idx + 1,
                        "entry_id": row.get("entry_id"),
                        "comp": row.get("component1"),
                        "col": vcol,
                        "property_type": ptype,
                        "orig_val": val,
                        "orig_unit": unit,
                        "conv_val": cval,
                        "tgt_unit": tgt,
                        "status": "converted",
                        "flag": "CONVERTED",
                    }
                )
                df.at[idx, vcol] = cval
            elif unit not in UNIT_CONVERSION_MAP.get(ptype, {}).get("conversions", {}):
                logs.append(
                    {
                        "row": idx + 1,
                        "entry_id": row.get("entry_id"),
                        "comp": row.get("component1"),
                        "col": vcol,
                        "property_type": ptype,
                        "orig_val": val,
                        "orig_unit": unit,
                        "conv_val": None,
                        "tgt_unit": UNIT_CONVERSION_MAP[ptype]["target_unit"],
                        "status": "unrecognized_unit",
                        "flag": "ERROR",
                    }
                )

    return df, pd.DataFrame(logs), unconvertible_rows, unknown_unit_rows

# =============================================================================
# 6. Phase and Mixture Classification
# =============================================================================

def filter_phase(df):
    """Filter by phase"""
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    phase_cols_ok = [c for c in PHASE_COLS if c in df.columns]
    type_cols_ok = [c for c in TYPE_COLS if c in df.columns]

    def has_liquid(row):
        for pcol, tcol in zip(phase_cols_ok, type_cols_ok):
            ptype = row.get(tcol)
            pval = row.get(pcol)
            if (
                ptype not in PHASE_SKIP_TYPES
                and pd.notna(pval)
                and str(pval).lower() == "liquid"
            ):
                return True
        return False

    def has_any_phase(row):
        for pcol, tcol in zip(phase_cols_ok, type_cols_ok):
            ptype = row.get(tcol)
            pval = row.get(pcol)
            if ptype not in PHASE_SKIP_TYPES and pd.notna(pval):
                return True
        return False

    def get_phases(row):
        phases = []
        for pcol, tcol in zip(phase_cols_ok, type_cols_ok):
            ptype = row.get(tcol)
            pval = row.get(pcol)
            if ptype not in PHASE_SKIP_TYPES and pd.notna(pval):
                phases.append(str(pval).lower())
        return list(set(phases))

    liquid_df = df[df.apply(has_liquid, axis=1)].copy()
    non_liquid_df = df[
        df.apply(lambda r: has_any_phase(r) and not has_liquid(r), axis=1)
    ].copy()
    no_phase_df = df[df.apply(lambda r: not has_any_phase(r), axis=1)].copy()

    phase_log = []
    for idx, row in df.iterrows():
        phases = get_phases(row)
        if phases:
            phase_log.append(
                {
                    "row": idx + 1,
                    "entry_id": row.get("entry_id"),
                    "comp": row.get("component1"),
                    "phases_detected": ", ".join(phases),
                    "is_liquid": "yes" if has_liquid(row) else "no",
                }
            )

    return liquid_df, non_liquid_df, no_phase_df, pd.DataFrame(phase_log)

# MIXTURE CLASSIFICATION

def is_pure(r):
    """No component2, no component3 → pure substance"""
    return (
        empty(r.get("component2"))
        and empty(r.get("component2_smiles"))
        and empty(r.get("component3"))
        and empty(r.get("component3_smiles"))
    )

def is_binary(r):
    """Has component2, but no component3 → binary mixture"""
    has_c2 = not empty(r.get("component2")) or not empty(r.get("component2_smiles"))
    has_c3 = not empty(r.get("component3")) or not empty(r.get("component3_smiles"))
    return has_c2 and not has_c3

def is_ternary(r):
    """Has component3 → ternary (or higher) mixture"""
    return not empty(r.get("component3")) or not empty(r.get("component3_smiles"))

# OUTPUT FORMAT FUNCTIONS

# Types that are ALWAYS condition variables (never the measured property)
# Note: Equilibrium_pressure is NOT here — it's a condition for some properties
# but the measured value for others. make_4col handles this via prop_type param.
CONDITION_TYPES = {"temperature", "pressure"}

# Types that are single-point (never appear alongside a T/P axis)
SINGLE_POINT_TYPES = {"melting_temperature", "enthalpy"}

# =============================================================================
# 7. Property Dataset Construction
# =============================================================================

def make_2col(pdf, prop_name, prop_type):
    """For single-point properties (melting T, enthalpy).

    Output: component1, component1_smiles, {PropertyName}_{Unit}
    No T/P axis — the measured value IS the property.
    """
    target_unit = UNIT_CONVERSION_MAP.get(prop_type, {}).get("target_unit", "unknown")
    clean_name = prop_name.replace(" ", "_")
    combined_col_name = f"{clean_name}_{target_unit}"

    if pdf.empty:
        return pd.DataFrame(
            columns=["component1", "component1_smiles", combined_col_name]
        )

    tmp = pdf.copy()
    tmp[combined_col_name] = None
    tmp["has_valid_prop"] = False

    for i in range(1, 6):
        type_col, unit_col, vcol = f"V{i}_type", f"V{i}_unit", f"V{i}"
        for idx, row in tmp.iterrows():
            if (
                pd.isna(tmp.at[idx, combined_col_name])
                and row.get(type_col) == prop_type
            ):
                unit = row.get(unit_col)
                if unit and not str(unit).startswith("UNKNOWN"):
                    val = parse_num(row.get(vcol))
                    if val is not None:
                        tmp.at[idx, combined_col_name] = val
                        tmp.at[idx, "has_valid_prop"] = True

    return tmp[tmp["has_valid_prop"]][
        ["component1", "component1_smiles", combined_col_name]
    ].copy()

def make_4col(pdf, prop_name, prop_type, include_pressure=False):
    """For multi-point properties (viscosity, density, etc.).

    4-col: component1, component1_smiles, Temperature_K, {Property}_{Unit}
    5-col: component1, component1_smiles, Temperature_K, Pressure_kPa, {Property}_{Unit}

    The property column is identified by matching prop_type exactly.
    If no exact match found, falls back to: any V column that isn't
    temperature, pressure, or a single-point type.
    """
    target_unit = UNIT_CONVERSION_MAP.get(prop_type, {}).get("target_unit", "unknown")
    temp_col = "Temperature_K"
    pres_col = "Pressure_kPa"
    prop_col = f"{prop_name.replace(' (Cp)', '').replace(' (Cv)', '')}_{target_unit}"
    if include_pressure:
        out_cols = ["component1", "component1_smiles", temp_col, pres_col, prop_col]
    else:
        out_cols = ["component1", "component1_smiles", temp_col, prop_col]

    if pdf.empty:
        return pd.DataFrame(columns=out_cols)

    tmp = pdf.copy()
    tmp[temp_col] = None
    tmp["has_valid_temp"] = False
    tmp[prop_col] = None
    tmp["has_valid_prop"] = False
    if include_pressure:
        tmp[pres_col] = None
        tmp["has_valid_pressure"] = False

    # ── Exclude condition variables and single-point properties. ────────────
    skip_types = CONDITION_TYPES | SINGLE_POINT_TYPES

    # Extract temperature
    for i in range(1, 6):
        type_col, unit_col, vcol = f"V{i}_type", f"V{i}_unit", f"V{i}"
        for idx, row in tmp.iterrows():
            if pd.isna(tmp.at[idx, temp_col]) and row.get(type_col) == "temperature":
                unit = row.get(unit_col)
                if unit and not str(unit).startswith("UNKNOWN"):
                    val = parse_num(row.get(vcol))
                    if val is not None:
                        tmp.at[idx, temp_col] = val
                        tmp.at[idx, "has_valid_temp"] = True

    # Extract pressure (optional)
    if include_pressure:
        for i in range(1, 6):
            type_col, unit_col, vcol = f"V{i}_type", f"V{i}_unit", f"V{i}"
            for idx, row in tmp.iterrows():
                if pd.isna(tmp.at[idx, pres_col]) and row.get(type_col) == "pressure":
                    unit = row.get(unit_col)
                    if unit and not str(unit).startswith("UNKNOWN"):
                        val = parse_num(row.get(vcol))
                        if val is not None:
                            convs = UNIT_CONVERSION_MAP["pressure"]["conversions"]
                            tmp.at[idx, pres_col] = (
                                convs[unit](val) if unit in convs else val
                            )
                            tmp.at[idx, "has_valid_pressure"] = True

    # Extract property value
    # Select the requested property, with a fallback for other measured variables.
    for i in range(1, 6):
        type_col, unit_col, vcol = f"V{i}_type", f"V{i}_unit", f"V{i}"
        for idx, row in tmp.iterrows():
            if pd.isna(tmp.at[idx, prop_col]):
                ptype = row.get(type_col)
                unit = row.get(unit_col)
                # Allow exact matches and the existing non-condition fallback.
                if ptype is not None and (
                    ptype == prop_type or ptype not in (skip_types | {prop_type})
                ):
                    if unit and not str(unit).startswith("UNKNOWN"):
                        val = parse_num(row.get(vcol))
                        if val is not None:
                            tmp.at[idx, prop_col] = val
                            tmp.at[idx, "has_valid_prop"] = True

    return tmp[tmp["has_valid_temp"] & tmp["has_valid_prop"]][out_cols].copy()

# PROPERTY OUTPUT CONFIGURATION
# Configure output layout and phase filtering by property.
# format:           "2col" → make_2col  |  "4col" → make_4col
# phase_filter:     True → separate liquid/non-liquid/no-phase
#                   False → skip phase filtering (single-point)
# include_pressure: True → also generate 5col file with Pressure_kPa
# Add new property definitions to this mapping.

PROPERTY_OUTPUT_CONFIG = {
    # Multi-point properties (T-dependent, liquid phase)
    "viscosity": {"format": "4col", "phase_filter": True, "include_pressure": True},
    "density": {"format": "4col", "phase_filter": True, "include_pressure": True},
    "surface_tension": {
        "format": "4col",
        "phase_filter": True,
        "include_pressure": False,
    },
    "heat_capacity": {"format": "4col", "phase_filter": True, "include_pressure": True},
    "thermal_conductivity": {
        "format": "4col",
        "phase_filter": True,
        "include_pressure": True,
    },
    "Equilibrium_pressure": {
        "format": "4col",
        "phase_filter": True,
        "include_pressure": False,
    },
    # Single-point properties (one value per substance, no T/P axis)
    "melting_temperature": {
        "format": "2col",
        "phase_filter": False,
        "include_pressure": False,
    },
    "enthalpy": {"format": "2col", "phase_filter": False, "include_pressure": False},
    "enthalpy_vaporization": {
        "format": "4col",
        "phase_filter": False,
        "include_pressure": False,
    },
}

# SEARCH HELPERS
_PLIST = {}

# =============================================================================
# 8. Data Extraction Workflow
# =============================================================================

def robust_search(pk, pn):
    try:
        if not _PLIST:
            _PLIST["p"] = ilthermopy.PropertyList()
        key = {k.lower(): v for k, v in _PLIST["p"].prop2key.items()}[pn.lower()]
        return ilthermopy.Search(prop_key=key), f"{pn} -> {key}"
    except KeyError:
        return (
            None,
            f"Property name not on server: {pn!r} (run ilthermopy.ShowPropertyList())",
        )
    except Exception as e:
        return None, f"Failed: {repr(e)}"


def get_ids(res):
    if hasattr(res, "empty") and res.empty:
        return []
    if hasattr(res, "columns") and "id" in list(res.columns):
        return res["id"].tolist()
    try:
        return list(res)
    except:
        return []

def main():
    for p in PROPERTIES:
        pk, pn, pt = (
            p.get("property_key"),
            p.get("property_name"),
            p.get("property_type"),
        )
        lbl = pn or str(pk)
        cfg = PROPERTY_OUTPUT_CONFIG.get(
            pt, {"format": "4col", "phase_filter": True, "include_pressure": False}
        )

        print(f"\n{'='*60}")
        print(f"{lbl} (type={pt})")
        print(
            f"  Output: {cfg['format']}  |  Phase filter: {cfg['phase_filter']}  |  Pressure: {cfg['include_pressure']}"
        )
        print(f"{'='*60}")

        # ── STEP 1: Fetch all entries ─────────────────────────
        res, info = robust_search(pk, pn)
        if res is None:
            print(f"❌ Search failed: {info}")
            continue

        eids = get_ids(res)
        if not eids:
            print("❌ No entries found")
            continue
        print(f"✓ Found {len(eids)} entries")

        rows = []
        with ThreadPoolExecutor(max_workers=WORKERS) as exe:
            futures = {exe.submit(process_entry, eid, lbl): eid for eid in eids}
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc=f"Fetching {lbl}"
            ):
                try:
                    dfi = fut.result()
                    if dfi is not None and not dfi.empty:
                        rows.append(dfi)
                except Exception as e:
                    print(f"  ⚠ Entry processing error: {e}")

        combined = normalize_concat(rows)
        if combined.empty:
            print(f"✓ Collected {len(combined)} rows")
            continue

        # ── STEP 2: Detect types, units, phases ──────────────
        combined = reorder_cols(combined)
        combined = analyze_all_columns(combined)

        combined, unit_check_log = check_units(combined)
        ok_c = (
            len(unit_check_log[unit_check_log["status"] == "ok"])
            if not unit_check_log.empty
            else 0
        )
        unk_c = (
            len(unit_check_log[unit_check_log["status"] == "unknown_unit"])
            if not unit_check_log.empty
            else 0
        )
        warn_c = (
            len(unit_check_log[unit_check_log["status"] == "unit_mismatch"])
            if not unit_check_log.empty
            else 0
        )
        print(f"✓ Units: {ok_c} OK, {unk_c} unknown, {warn_c} warnings")

        # ── STEP 3: Convert units ────────────────────────────
        combined, conv_log, unconvertible_rows, unknown_unit_rows = apply_conv(combined)
        conv_c = (
            len(conv_log[conv_log["status"] == "converted"])
            if not conv_log.empty
            else 0
        )
        print(f"✓ Conversions: {conv_c} converted")

        unconvertible_df = (
            combined.iloc[list(unconvertible_rows)].copy()
            if unconvertible_rows
            else pd.DataFrame()
        )
        unknown_df = (
            combined.iloc[list(unknown_unit_rows)].copy()
            if unknown_unit_rows
            else pd.DataFrame()
        )
        convertible_df = combined.iloc[
            [
                i
                for i in range(len(combined))
                if i not in unconvertible_rows and i not in unknown_unit_rows
            ]
        ].copy()

        # ── STEP 4: Classify pure compounds and mixtures ──
        pure_df = convertible_df[convertible_df.apply(is_pure, axis=1)].copy()
        binary_df = convertible_df[convertible_df.apply(is_binary, axis=1)].copy()
        ternary_df = convertible_df[convertible_df.apply(is_ternary, axis=1)].copy()
        print(
            f"✓ Mixture split — Pure: {len(pure_df)}, Binary: {len(binary_df)}, Ternary: {len(ternary_df)}"
        )

        # ── STEP 5: Phase filtering (only if config says so) ─
        if cfg["phase_filter"]:
            liquid_df, non_liquid_df, no_phase_df, phase_log = filter_phase(pure_df)
            print(
                f"✓ Pure phase split — Liquid: {len(liquid_df)}, Non-liquid: {len(non_liquid_df)}, No phase: {len(no_phase_df)}"
            )
        else:
            liquid_df = pd.DataFrame()
            non_liquid_df = pd.DataFrame()
            no_phase_df = pure_df
            phase_log = pd.DataFrame()
            print("✓ Single-point property — skipping phase filter")

        # ── STEP 6: Save diagnostic/intermediate files ────────
        base = re.sub(r"[\s/\\:]", "_", lbl.lower())

        files = {
            # Full data
            f"{base}_all_data.csv": combined,
            # Unit categories
            f"{base}_convertible_units.csv": convertible_df,
            f"{base}_unconvertible_units.csv": unconvertible_df,
            f"{base}_unknown_units.csv": unknown_df,
            # Mixture categories
            f"{base}_pure.csv": pure_df,
            f"{base}_binary_mixtures.csv": binary_df,
            f"{base}_ternary_mixtures.csv": ternary_df,
            # Phase splits (empty for single-point properties)
            f"{base}_pure_liquid.csv": liquid_df,
            f"{base}_pure_non_liquid.csv": non_liquid_df,
            f"{base}_pure_no_phase.csv": no_phase_df,
            # Logs
            f"{base}_unit_check_log.csv": unit_check_log,
            f"{base}_conversion_log.csv": conv_log,
            f"{base}_phase_log.csv": phase_log,
        }

        for fname, df in files.items():
            if df.empty:
                continue
            path = os.path.join(OUTPUT_DIR, fname)
            df.to_csv(path, index=False)
            print(f"  {fname:50s} ({len(df):6d} rows)")

        # ── STEP 7: Generate clean output file for PURE only ─
        #  Select the output layout for each property.
        #    "2col" → make_2col(pure_df)           → component, smiles, value
        #    "4col" → make_4col(pure_liquid_df)     → component, smiles, T, value
        #           + make_4col(include_pressure)   → component, smiles, T, P, value
        print("\n  ─── Clean output (pure compounds only) ───")

        if cfg["format"] == "2col":
            # Single-point: use ALL pure compounds (no phase needed)
            two_col_df = make_2col(pure_df, pn, pt)
            fname = f"{base}_pure_2col.csv"
            two_col_df.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
            print(f"  ✓ {fname:48s} ({len(two_col_df):6d} rows)")

        elif cfg["format"] == "4col":
            # Multi-point: generate TWO versions
            #   1. Strict:    liquid-phase only (Liquid-phase records)
            #   2. Inclusive:  liquid + no-phase-info

            # ── Include records without phase labels in the inclusive dataset. ──
            liquid_plus_nophase = (
                pd.concat([liquid_df, no_phase_df], ignore_index=True)
                if not no_phase_df.empty
                else liquid_df
            )

            # Strict 4col (liquid only)
            four_col_df = make_4col(liquid_df, pn, pt, include_pressure=False)
            fname4 = f"{base}_pure_liquid_4col.csv"
            four_col_df.to_csv(os.path.join(OUTPUT_DIR, fname4), index=False)
            print(f"  ✓ {fname4:48s} ({len(four_col_df):6d} rows)  [liquid only]")

            # Inclusive 4col (liquid + untagged)
            four_col_incl = make_4col(
                liquid_plus_nophase, pn, pt, include_pressure=False
            )
            fname4i = f"{base}_pure_liquid_incl_4col.csv"
            four_col_incl.to_csv(os.path.join(OUTPUT_DIR, fname4i), index=False)
            print(
                f"  ✓ {fname4i:48s} ({len(four_col_incl):6d} rows)  [liquid + no-phase]"
            )

            # Optionally generate 5col (T + P + property)
            if cfg["include_pressure"]:
                five_col_df = make_4col(liquid_df, pn, pt, include_pressure=True)
                fname5 = f"{base}_pure_liquid_5col.csv"
                five_col_df.to_csv(os.path.join(OUTPUT_DIR, fname5), index=False)
                print(f"  ✓ {fname5:48s} ({len(five_col_df):6d} rows)  [liquid only]")

                five_col_incl = make_4col(
                    liquid_plus_nophase, pn, pt, include_pressure=True
                )
                fname5i = f"{base}_pure_liquid_incl_5col.csv"
                five_col_incl.to_csv(os.path.join(OUTPUT_DIR, fname5i), index=False)
                print(
                    f"  ✓ {fname5i:48s} ({len(five_col_incl):6d} rows)  [liquid + no-phase]"
                )

if __name__ == "__main__":
    main()
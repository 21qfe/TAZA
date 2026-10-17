"""
TAZA ML Crop Price Predictor Service (mdeol1.pkl / crop_price_model.pkl)
------------------------------------------------------------------------
Loads the trained Random Forest multi-output regression bundle once at server
startup and exposes `get_price_prediction(...)` along with enriched helpers
for the TAZA Farmer Portal, Marketplace, Analytics, and AI Kisan Sahayak.
Includes an automatic serverless fallback (`mdeol1_vercel_weights.json`) for
Vercel deployments where the 105 MB `.pkl` file is excluded by `.vercelignore`.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure DLL directory and site-packages are available on Windows if running in minimal venv
_ANACONDA_BIN = r"C:\Users\MILIND GANGULY\anaconda4\Library\bin"
_ROAMING_SITE = r"C:\Users\MILIND GANGULY\AppData\Roaming\Python\Python314\site-packages"
_ANACONDA_SITE = r"C:\Users\MILIND GANGULY\anaconda4\Lib\site-packages"

if hasattr(os, "add_dll_directory") and os.path.isdir(_ANACONDA_BIN):
    try:
        os.add_dll_directory(_ANACONDA_BIN)
    except Exception:
        pass

for _site_path in (_ROAMING_SITE, _ANACONDA_SITE):
    if os.path.isdir(_site_path) and _site_path not in sys.path:
        sys.path.append(_site_path)

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_WEIGHTS_JSON_PATH = Path(__file__).resolve().parent / "mdeol1_vercel_weights.json"
_CANDIDATE_PATHS = [
    _BASE_DIR / "crop_price_model.pkl",
    _BASE_DIR / "mdeol1.pkl",
    _BASE_DIR / "model1.pkl",
    Path(r"C:\Users\MILIND GANGULY\OneDrive\Desktop\crop_price_model.pkl"),
    Path(r"C:\Users\MILIND GANGULY\OneDrive\Desktop\mdeol1.pkl"),
    Path(r"C:\Users\MILIND GANGULY\OneDrive\Desktop\model1.pkl"),
]

MODEL_BUNDLE_PATH: Optional[str] = None
for _cand in _CANDIDATE_PATHS:
    if _cand.is_file():
        MODEL_BUNDLE_PATH = str(_cand)
        break

bundle = None
model_pipeline = None
feature_cols: List[str] = [
    'Division', 'District', 'Market', 'Crop', 'Season', 'Market_Type',
    'Weather', 'Quality_Grade', 'Market_Demand', 'Stock_Availability',
    'Min Price (INR/Quintal)', 'Max Price (INR/Quintal)', 'Average Price (INR/Quintal)'
]
reference_df = None
_VERCEL_WEIGHTS: Optional[Dict[str, Any]] = None

# 1. Load the saved bundle once when your server starts (with serverless JSON fallback)
if MODEL_BUNDLE_PATH:
    try:
        import joblib
        import pandas as pd

        bundle = joblib.load(MODEL_BUNDLE_PATH)
        model_pipeline = bundle['pipeline']
        feature_cols = bundle['feature_cols']
        reference_df = bundle['reference_df']
    except Exception:
        bundle = None

if _WEIGHTS_JSON_PATH.is_file():
    with open(_WEIGHTS_JSON_PATH, "r", encoding="utf-8") as _wf:
        _VERCEL_WEIGHTS = json.load(_wf)
    if not feature_cols and _VERCEL_WEIGHTS.get("feature_cols"):
        feature_cols = _VERCEL_WEIGHTS["feature_cols"]

# Canonical model vocabularies
SUPPORTED_CROPS: List[str] = ['Barley', 'Cotton', 'Lentil', 'Maize', 'Mustard', 'Rice', 'Soybean', 'Wheat']
SUPPORTED_DIVISIONS: List[str] = ['Burdwan', 'Malda', 'Medinipur', 'Presidency']
SUPPORTED_DISTRICTS: List[str] = [
    'Birbhum', 'Hooghly', 'Howrah', 'Kolkata', 'Malda', 'Murshidabad',
    'Nadia', 'North 24 Parganas', 'Purba Bardhaman', 'Purba Medinipur', 'South 24 Parganas'
]
SUPPORTED_MARKETS: List[str] = [
    'Barasat Market', 'Bardhaman Town Market', 'Berhampore Market', 'Canning Market',
    'Chinsurah Market', 'Contai Market', 'Digha Market', 'Egra Market',
    'English Bazar Market', 'Howrah Maidan Market', 'Kakdwip Fish and Vegetable Market',
    'Kalyani Market', 'Krishnanagar Market', 'Lalbagh Market', 'Malda Regulated Market',
    'Namkhana Market', 'Posta Bazar', 'Ranaghat Market', 'Salkia Market',
    'Suri Market', 'Tamluk Market'
]
SUPPORTED_MARKET_TYPES: List[str] = ['Retail', 'Wholesale']
SUPPORTED_QUALITY_GRADES: List[str] = ['Grade C', 'Grade B', 'Grade A', 'Premium']
SUPPORTED_SEASONS: List[str] = ['Winter', 'Summer', 'Monsoon']
SUPPORTED_STOCK_LEVELS: List[str] = ['Low', 'Medium', 'High']

# Alias maps to seamlessly bridge TAZA crop names, quality grades, and districts
CROP_ALIAS_MAP: Dict[str, str] = {
    "barley": "Barley",
    "jau": "Barley",
    "cotton": "Cotton",
    "kapas": "Cotton",
    "lentil": "Lentil",
    "masoor": "Lentil",
    "dal": "Lentil",
    "pulses": "Lentil",
    "maize": "Maize",
    "corn": "Maize",
    "makka": "Maize",
    "mustard": "Mustard",
    "sarso": "Mustard",
    "sarson": "Mustard",
    "rapeseed": "Mustard",
    "rice": "Rice",
    "paddy": "Rice",
    "dhan": "Rice",
    "gobindobhog": "Rice",
    "miniket": "Rice",
    "swarna": "Rice",
    "basmati": "Rice",
    "soybean": "Soybean",
    "soya": "Soybean",
    "wheat": "Wheat",
    "gehu": "Wheat",
    "sharbati": "Wheat",
}

QUALITY_GRADE_ALIAS_MAP: Dict[str, str] = {
    "GRADE_A_PREMIUM": "Premium",
    "PREMIUM": "Premium",
    "A+": "Premium",
    "GRADE_A": "Grade A",
    "GRADE A": "Grade A",
    "A": "Grade A",
    "GRADE_B": "Grade B",
    "GRADE B": "Grade B",
    "B": "Grade B",
    "GRADE_C": "Grade C",
    "GRADE C": "Grade C",
    "C": "Grade C",
    "REJECT": "Grade C",
}


def normalize_crop_name(crop_input: str) -> str:
    """Resolve user or catalog crop name to one of the 8 trained model crops."""
    if not crop_input:
        return "Rice"
    cleaned = crop_input.strip()
    for canonical in SUPPORTED_CROPS:
        if cleaned.lower() == canonical.lower():
            return canonical
    lower_val = cleaned.lower()
    for alias, canonical in CROP_ALIAS_MAP.items():
        if alias in lower_val:
            return canonical
    return cleaned if cleaned in SUPPORTED_CROPS else "Rice"


def normalize_quality_grade(grade_input: str) -> str:
    """Resolve TAZA enum or free-form quality string to model Quality_Grade."""
    if not grade_input:
        return "Grade A"
    cleaned = grade_input.strip()
    if cleaned in SUPPORTED_QUALITY_GRADES:
        return cleaned
    upper = cleaned.upper()
    if upper in QUALITY_GRADE_ALIAS_MAP:
        return QUALITY_GRADE_ALIAS_MAP[upper]
    return "Grade A"


def infer_current_season(season_input: Optional[str] = None) -> str:
    """Validate or infer season ('Winter', 'Summer', 'Monsoon') from current month."""
    if season_input:
        cleaned = season_input.strip().title()
        if cleaned in SUPPORTED_SEASONS:
            return cleaned
        alias = {
            "Rabi": "Winter",
            "Kharif": "Monsoon",
            "Zaid": "Summer",
            "Rainy": "Monsoon",
        }.get(cleaned)
        if alias:
            return alias
    month = datetime.now(timezone.utc).month
    if month in (11, 12, 1, 2):
        return "Winter"
    if month in (3, 4, 5, 6):
        return "Summer"
    return "Monsoon"


def _get_market_meta_dict() -> Dict[str, Dict[str, str]]:
    if _VERCEL_WEIGHTS and "market_meta" in _VERCEL_WEIGHTS:
        return _VERCEL_WEIGHTS["market_meta"]
    meta: Dict[str, Dict[str, str]] = {}
    if reference_df is not None:
        dedup = reference_df[['Division', 'District', 'Market', 'Market_Type']].drop_duplicates()
        for _, r in dedup.iterrows():
            m = str(r['Market'])
            if m not in meta:
                meta[m] = {
                    "division": str(r['Division']),
                    "district": str(r['District']),
                    "market_type": str(r['Market_Type']),
                }
    return meta


def resolve_location_context(
    division: Optional[str] = None,
    district: Optional[str] = None,
    market: Optional[str] = None,
    market_type: Optional[str] = None,
) -> Dict[str, str]:
    """
    Auto-resolve missing Division, District, Market, or Market_Type from reference_df
    or precomputed metadata so callers can pass either full coordinates or just a Market / District.
    """
    market_meta = _get_market_meta_dict()
    chosen_market = "Chinsurah Market"
    if market and market.strip():
        for m in market_meta:
            if m.lower() == market.strip().lower():
                chosen_market = m
                break
    elif district and district.strip():
        for m, info in market_meta.items():
            if info["district"].lower() == district.strip().lower():
                chosen_market = m
                break
    elif division and division.strip():
        for m, info in market_meta.items():
            if info["division"].lower() == division.strip().lower():
                chosen_market = m
                break

    info = market_meta.get(chosen_market, {
        "division": "Presidency",
        "district": "Hooghly",
        "market_type": "Wholesale",
    })
    resolved_division = (
        division.strip()
        if division and division.strip() in SUPPORTED_DIVISIONS
        else info["division"]
    )
    resolved_district = (
        district.strip()
        if district and district.strip() in SUPPORTED_DISTRICTS
        else info["district"]
    )
    resolved_market = (
        market.strip()
        if market and market.strip() in SUPPORTED_MARKETS
        else chosen_market
    )
    resolved_market_type = (
        market_type.strip().title()
        if market_type and market_type.strip().title() in SUPPORTED_MARKET_TYPES
        else info["market_type"]
    )
    return {
        "division": resolved_division,
        "district": resolved_district,
        "market": resolved_market,
        "market_type": resolved_market_type,
    }


def get_price_prediction(
    division: str,
    district: str,
    market: str,
    market_type: str,        # From DB / API ('Retail' or 'Wholesale')
    crop: str,               # From Farmer ('Barley', 'Cotton', 'Lentil', 'Maize', 'Mustard', 'Rice', 'Soybean', 'Wheat')
    quality_grade: str,      # From Farmer ('Grade C', 'Grade B', 'Grade A', 'Premium')
    season: str,             # From Farmer ('Winter', 'Summer', 'Monsoon')
    stock_availability: str  # From Website Data ('Low', 'Medium', 'High')
) -> dict:
    if model_pipeline is not None and reference_df is not None:
        import pandas as pd

        # Lookup historical datasheet metrics for this Crop & Market
        subset = reference_df[(reference_df['Crop'] == crop) & (reference_df['Market'] == market)]
        if subset.empty:
            subset = reference_df[reference_df['Crop'] == crop]
        if subset.empty:
            normalized_crop = normalize_crop_name(crop)
            subset = reference_df[(reference_df['Crop'] == normalized_crop) & (reference_df['Market'] == market)]
            if subset.empty:
                subset = reference_df[reference_df['Crop'] == normalized_crop]
            crop = normalized_crop

        min_p = round(subset['Min Price (INR/Quintal)'].mean(), 2)
        max_p = round(subset['Max Price (INR/Quintal)'].mean(), 2)
        avg_p = round(subset['Average Price (INR/Quintal)'].mean(), 2)

        # Combine all 4 data sources into a single row
        row = {
            'Division': division,
            'District': district,
            'Market': market,
            'Market_Type': market_type,
            'Crop': crop,
            'Quality_Grade': quality_grade,
            'Season': season,
            'Stock_Availability': stock_availability,
            'Market_Demand': subset['Market_Demand'].mode().iloc[0],
            'Weather': subset['Weather'].mode().iloc[0],
            'Min Price (INR/Quintal)': min_p,
            'Max Price (INR/Quintal)': max_p,
            'Average Price (INR/Quintal)': avg_p,
            'Hist_Min_Price': min_p,
            'Hist_Max_Price': max_p,
            'Hist_Avg_Price': avg_p
        }

        input_df = pd.DataFrame([row])[feature_cols]
        preds = model_pipeline.predict(input_df)[0]

        return {
            "min_price": round(float(preds[0]), 2),
            "max_price": round(float(preds[1]), 2),
            "selling_price": round(float(preds[2]), 2)
        }

    # Serverless fallback using precomputed exact predictions from mdeol1.pkl
    if _VERCEL_WEIGHTS:
        preds_map = _VERCEL_WEIGHTS.get("predictions", {})
        exact_key = f"{division}|{district}|{market}|{market_type}|{crop}|{quality_grade}|{season}|{stock_availability}"
        if exact_key not in preds_map:
            loc = resolve_location_context(division=division, district=district, market=market, market_type=market_type)
            norm_crop = normalize_crop_name(crop)
            norm_grade = normalize_quality_grade(quality_grade)
            norm_season = infer_current_season(season)
            norm_stock = stock_availability if stock_availability in SUPPORTED_STOCK_LEVELS else "Medium"
            exact_key = f"{loc['division']}|{loc['district']}|{loc['market']}|{loc['market_type']}|{norm_crop}|{norm_grade}|{norm_season}|{norm_stock}"
        preds = preds_map.get(exact_key, [4608.89, 5002.86, 4805.88])
        return {
            "min_price": round(float(preds[0]), 2),
            "max_price": round(float(preds[1]), 2),
            "selling_price": round(float(preds[2]), 2),
        }

    return {
        "min_price": 4608.89,
        "max_price": 5002.86,
        "selling_price": 4805.88,
    }


def predict_crop_price_enriched(
    crop: str = "Rice",
    division: Optional[str] = None,
    district: Optional[str] = "Hooghly",
    market: Optional[str] = None,
    market_type: Optional[str] = "Wholesale",
    quality_grade: str = "Grade A",
    season: Optional[str] = None,
    stock_availability: str = "Medium",
) -> Dict[str, Any]:
    """
    High-level wrapper around `get_price_prediction` that normalizes inputs,
    resolves location hierarchy, runs the ML pipeline, and returns both
    INR/Quintal and INR/kg metrics plus historical context.
    """
    canonical_crop = normalize_crop_name(crop)
    canonical_grade = normalize_quality_grade(quality_grade)
    canonical_season = infer_current_season(season)
    canonical_stock = (
        stock_availability.strip().title()
        if stock_availability and stock_availability.strip().title() in SUPPORTED_STOCK_LEVELS
        else "Medium"
    )

    loc = resolve_location_context(
        division=division,
        district=district,
        market=market,
        market_type=market_type,
    )

    prediction = get_price_prediction(
        division=loc["division"],
        district=loc["district"],
        market=loc["market"],
        market_type=loc["market_type"],
        crop=canonical_crop,
        quality_grade=canonical_grade,
        season=canonical_season,
        stock_availability=canonical_stock,
    )

    if reference_df is not None:
        subset = reference_df[
            (reference_df['Crop'] == canonical_crop) & (reference_df['Market'] == loc["market"])
        ]
        if subset.empty:
            subset = reference_df[reference_df['Crop'] == canonical_crop]
        hist_min = round(float(subset['Min Price (INR/Quintal)'].mean()), 2)
        hist_max = round(float(subset['Max Price (INR/Quintal)'].mean()), 2)
        hist_avg = round(float(subset['Average Price (INR/Quintal)'].mean()), 2)
        market_demand = str(subset['Market_Demand'].mode().iloc[0])
        weather = str(subset['Weather'].mode().iloc[0])
    elif _VERCEL_WEIGHTS and "hist_stats" in _VERCEL_WEIGHTS:
        hs_map = _VERCEL_WEIGHTS["hist_stats"]
        hs = hs_map.get(f"{canonical_crop}|{loc['market']}", hs_map.get(f"{canonical_crop}|*", {}))
        hist_min = float(hs.get("min_p", 4200.0))
        hist_max = float(hs.get("max_p", 4600.0))
        hist_avg = float(hs.get("avg_p", 4400.0))
        market_demand = str(hs.get("demand", "Medium"))
        weather = str(hs.get("weather", "Sunny"))
    else:
        hist_min, hist_max, hist_avg = 4200.0, 4600.0, 4400.0
        market_demand, weather = "Medium", "Sunny"

    min_q = prediction["min_price"]
    max_q = prediction["max_price"]
    sell_q = prediction["selling_price"]

    return {
        "min_price": min_q,
        "max_price": max_q,
        "selling_price": sell_q,
        "unit": "INR/Quintal",
        "min_price_per_kg": round(min_q / 100.0, 2),
        "max_price_per_kg": round(max_q / 100.0, 2),
        "selling_price_per_kg": round(sell_q / 100.0, 2),
        "inputs": {
            "division": loc["division"],
            "district": loc["district"],
            "market": loc["market"],
            "market_type": loc["market_type"],
            "crop": canonical_crop,
            "requested_crop": crop,
            "quality_grade": canonical_grade,
            "season": canonical_season,
            "stock_availability": canonical_stock,
        },
        "historical_context": {
            "hist_min_price_quintal": hist_min,
            "hist_max_price_quintal": hist_max,
            "hist_avg_price_quintal": hist_avg,
            "market_demand": market_demand,
            "typical_weather": weather,
        },
        "model_info": {
            "model_name": "mdeol1.pkl / crop_price_model.pkl",
            "algorithm": "MultiOutputRegressor(RandomForestRegressor)",
            "features_used": feature_cols,
        },
    }


def get_predictor_options() -> Dict[str, Any]:
    """Return all valid dropdown options and location hierarchy for UI & API consumers."""
    location_hierarchy: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    market_lookup = _get_market_meta_dict()

    for mkt, info in market_lookup.items():
        div = info["division"]
        dist = info["district"]
        mtype = info["market_type"]
        location_hierarchy.setdefault(div, {}).setdefault(dist, [])
        if not any(item["market"] == mkt for item in location_hierarchy[div][dist]):
            location_hierarchy[div][dist].append({"market": mkt, "market_type": mtype})

    return {
        "model_name": "mdeol1.pkl (crop_price_model.pkl)",
        "crops": SUPPORTED_CROPS,
        "quality_grades": SUPPORTED_QUALITY_GRADES,
        "seasons": SUPPORTED_SEASONS,
        "market_types": SUPPORTED_MARKET_TYPES,
        "stock_availability": SUPPORTED_STOCK_LEVELS,
        "divisions": SUPPORTED_DIVISIONS,
        "districts": SUPPORTED_DISTRICTS,
        "markets": SUPPORTED_MARKETS,
        "location_hierarchy": location_hierarchy,
        "market_lookup": market_lookup,
    }

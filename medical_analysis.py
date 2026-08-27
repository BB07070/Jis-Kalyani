# medical_analysis.py - Private local medical report parser with improved table handling

import re
import json
import os
import warnings
from typing import List, Dict, Any, Optional, Tuple

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="fitz")

# Use pymupdf instead of fitz
try:
    import pymupdf  # New name (recommended)
except ImportError:
    try:
        import fitz  # Fallback
        print("⚠️ Using deprecated 'fitz' import. Please install pymupdf: pip install pymupdf")
    except ImportError:
        raise ImportError("Please install pymupdf: pip install pymupdf")

# For OCR (optional)
try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR not available. Install: pip install pytesseract Pillow")


class MedicalAnalysisError(Exception):
    """Custom exception for medical analysis errors"""
    pass


# ============================================================================
# REFERENCE RANGES FOR COMMON LAB TESTS
# ============================================================================

REFERENCE_RANGES = {
    # Complete Blood Count (CBC)
    "hemoglobin": {"low": 13.0, "high": 17.0, "unit": "g/dL"},
    "hematocrit": {"low": 36.0, "high": 52.0, "unit": "%"},
    "rbc": {"low": 4.2, "high": 6.1, "unit": "M/µL"},
    "wbc": {"low": 4000, "high": 11000, "unit": "/μL"},
    "wbc_count": {"low": 4000, "high": 11000, "unit": "/μL"},
    "platelets": {"low": 150, "high": 400, "unit": "K/µL"},
    "platelet_count": {"low": 150, "high": 400, "unit": "K/µL"},
    "neutrophils": {"low": 40, "high": 70, "unit": "%"},
    "lymphocytes": {"low": 20, "high": 40, "unit": "%"},
    
    # Blood Sugar / Diabetes
    "glucose": {"low": 70, "high": 100, "unit": "mg/dL"},
    "blood_sugar": {"low": 70, "high": 100, "unit": "mg/dL"},
    "blood_sugar_fasting": {"low": 70, "high": 99, "unit": "mg/dL"},
    "a1c": {"low": 4.0, "high": 5.6, "unit": "%"},
    "hba1c": {"low": 4.0, "high": 5.6, "unit": "%"},
    
    # Lipid Panel
    "cholesterol": {"low": 125, "high": 200, "unit": "mg/dL"},
    "total_cholesterol": {"low": 125, "high": 200, "unit": "mg/dL"},
    "ldl": {"low": 0, "high": 130, "unit": "mg/dL"},
    "ldl_cholesterol": {"low": 0, "high": 130, "unit": "mg/dL"},
    "hdl": {"low": 40, "high": 60, "unit": "mg/dL"},
    "hdl_cholesterol": {"low": 40, "high": 60, "unit": "mg/dL"},
    "triglycerides": {"low": 0, "high": 150, "unit": "mg/dL"},
    
    # Blood Pressure
    "blood_pressure_systolic": {"low": 90, "high": 120, "unit": "mmHg"},
    "blood_pressure_diastolic": {"low": 60, "high": 80, "unit": "mmHg"},
    
    # Metabolic Panel
    "creatinine": {"low": 0.6, "high": 1.2, "unit": "mg/dL"},
    "bun": {"low": 7, "high": 20, "unit": "mg/dL"},
    "sodium": {"low": 135, "high": 145, "unit": "mEq/L"},
    "potassium": {"low": 3.5, "high": 5.0, "unit": "mEq/L"},
    "calcium": {"low": 8.5, "high": 10.5, "unit": "mg/dL"},
    "albumin": {"low": 3.4, "high": 5.4, "unit": "g/dL"},
    
    # Liver
    "ast": {"low": 10, "high": 40, "unit": "U/L"},
    "alt": {"low": 7, "high": 56, "unit": "U/L"},
    "alkaline_phosphatase": {"low": 44, "high": 147, "unit": "U/L"},
    "bilirubin": {"low": 0.1, "high": 1.2, "unit": "mg/dL"},
    
    # Thyroid
    "tsh": {"low": 0.4, "high": 4.0, "unit": "mIU/L"},
    "t4": {"low": 4.5, "high": 12.5, "unit": "µg/dL"},
    "t3": {"low": 80, "high": 200, "unit": "ng/dL"},
    
    # Vitamins
    "vitamin_b12": {"low": 200, "high": 900, "unit": "pg/mL"},
    "vitamin_d": {"low": 20, "high": 50, "unit": "ng/mL"},
    "ferritin": {"low": 12, "high": 300, "unit": "ng/mL"},
    "iron": {"low": 60, "high": 170, "unit": "µg/dL"},
}


# ============================================================================
# BODY SYSTEM MAPPING
# ============================================================================

SYSTEM_MAP = {
    "hemoglobin": "blood",
    "hematocrit": "blood",
    "rbc": "blood",
    "wbc": "blood",
    "wbc_count": "blood",
    "platelets": "blood",
    "platelet_count": "blood",
    "neutrophils": "blood",
    "lymphocytes": "blood",
    "glucose": "metabolic",
    "blood_sugar": "metabolic",
    "blood_sugar_fasting": "metabolic",
    "a1c": "metabolic",
    "hba1c": "metabolic",
    "cholesterol": "metabolic",
    "total_cholesterol": "metabolic",
    "ldl": "metabolic",
    "ldl_cholesterol": "metabolic",
    "hdl": "metabolic",
    "hdl_cholesterol": "metabolic",
    "triglycerides": "metabolic",
    "blood_pressure_systolic": "heart",
    "blood_pressure_diastolic": "heart",
    "creatinine": "kidneys",
    "bun": "kidneys",
    "sodium": "metabolic",
    "potassium": "metabolic",
    "calcium": "metabolic",
    "albumin": "metabolic",
    "ast": "liver",
    "alt": "liver",
    "alkaline_phosphatase": "liver",
    "bilirubin": "liver",
    "tsh": "thyroid",
    "t4": "thyroid",
    "t3": "thyroid",
    "vitamin_b12": "blood",
    "vitamin_d": "metabolic",
    "ferritin": "blood",
    "iron": "blood",
}


# ============================================================================
# IMPROVED TEST NAME VARIATIONS
# ============================================================================

TEST_NAME_VARIATIONS = {
    "hemoglobin": ["hemoglobin", "hgb", "hb", "haemoglobin"],
    "wbc_count": ["wbc", "wbc count", "white blood cells", "leukocytes", "white cells"],
    "platelet_count": ["platelets", "platelet count", "plt", "thrombocytes"],
    "blood_sugar": ["blood sugar", "glucose", "blood glucose", "glu", "sugar"],
    "blood_sugar_fasting": ["fasting blood sugar", "fasting glucose", "fbs", "fbg"],
    "a1c": ["a1c", "hba1c", "glycated hemoglobin", "hemoglobin a1c"],
    "hba1c": ["hba1c", "a1c", "glycated hemoglobin"],
    "total_cholesterol": ["total cholesterol", "cholesterol", "chol", "tc", "cholesterol total"],
    "ldl_cholesterol": ["ldl", "ldl cholesterol", "low density lipoprotein", "ldl-c", "ldl direct"],
    "hdl_cholesterol": ["hdl", "hdl cholesterol", "high density lipoprotein", "hdl-c"],
    "triglycerides": ["triglycerides", "trig", "tg", "tryglicerides", "triglyceride"],
    "blood_pressure_systolic": ["systolic", "systolic bp", "sbp", "blood pressure systolic"],
    "blood_pressure_diastolic": ["diastolic", "diastolic bp", "dbp", "blood pressure diastolic"],
    "creatinine": ["creatinine", "creat", "serum creatinine"],
    "bun": ["bun", "blood urea nitrogen", "urea nitrogen"],
    "sodium": ["sodium", "na", "serum sodium"],
    "potassium": ["potassium", "k", "serum potassium"],
    "calcium": ["calcium", "ca", "serum calcium"],
    "albumin": ["albumin", "alb", "serum albumin"],
    "ast": ["ast", "sgot", "aspartate aminotransferase"],
    "alt": ["alt", "sgpt", "alanine aminotransferase"],
    "alkaline_phosphatase": ["alkaline phosphatase", "alp", "alk phos"],
    "bilirubin": ["bilirubin", "total bilirubin", "bili"],
    "tsh": ["tsh", "thyroid stimulating hormone"],
    "t4": ["t4", "thyroxine", "free t4", "total t4"],
    "t3": ["t3", "triiodothyronine", "free t3", "total t3"],
    "vitamin_b12": ["vitamin b12", "b12", "cobalamin"],
    "vitamin_d": ["vitamin d", "vit d", "25-hydroxyvitamin d"],
    "ferritin": ["ferritin", "serum ferritin"],
    "iron": ["iron", "fe", "serum iron"],
}


# ============================================================================
# IMPROVED TEXT EXTRACTION
# ============================================================================

def extract_text_from_pdf(content: bytes, filename: str = "") -> str:
    """Extract text from a PDF file using pymupdf with improved table handling"""
    try:
        # Try pymupdf first
        try:
            doc = pymupdf.open(stream=content, filetype="pdf")
        except NameError:
            doc = fitz.open(stream=content, filetype="pdf")
        
        text = ""
        for page_num, page in enumerate(doc):
            # Get text with better formatting
            page_text = page.get_text()
            
            # Get text as a table-like structure if possible
            # This helps with PDFs that use tables
            try:
                # Try to extract tables using pymupdf's table detection
                table_text = page.get_text("text")
                if table_text:
                    text += table_text + "\n"
                else:
                    text += page_text + "\n"
            except:
                text += page_text + "\n"
            
        doc.close()
        
        # Clean up the text
        text = clean_extracted_text(text)
        return text
        
    except Exception as e:
        raise MedicalAnalysisError(f"Failed to extract text from PDF: {str(e)}")


def clean_extracted_text(text: str) -> str:
    """Clean up extracted text for better parsing"""
    # Remove HTML-like tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove table tags
    text = re.sub(r'</?table[^>]*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'</?tr[^>]*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'</?td[^>]*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'</?th[^>]*>', ' ', text, flags=re.IGNORECASE)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()


def extract_text_from_image(content: bytes) -> str:
    """Extract text from an image using OCR"""
    if not OCR_AVAILABLE:
        raise MedicalAnalysisError("OCR not available. Install: pip install pytesseract Pillow")
    
    try:
        image = Image.open(io.BytesIO(content))
        # Configure Tesseract for better medical text recognition
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz./%:+- '
        text = pytesseract.image_to_string(image, config=custom_config)
        return clean_extracted_text(text)
    except Exception as e:
        raise MedicalAnalysisError(f"Failed to extract text from image: {str(e)}")


def extract_text_from_txt(content: bytes) -> str:
    """Extract text from a plain text file"""
    try:
        text = content.decode('utf-8', errors='ignore')
        return clean_extracted_text(text)
    except Exception as e:
        raise MedicalAnalysisError(f"Failed to extract text from TXT file: {str(e)}")


def extract_text(content: bytes, filename: str = "") -> str:
    """Extract text from various file types"""
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(content, filename)
    elif filename_lower.endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif')):
        return extract_text_from_image(content)
    elif filename_lower.endswith('.txt'):
        return extract_text_from_txt(content)
    elif filename_lower.endswith('.csv'):
        return extract_text_from_txt(content)
    else:
        try:
            return content.decode('utf-8', errors='ignore')
        except:
            raise MedicalAnalysisError(f"Unsupported file type: {filename}")


# ============================================================================
# IMPROVED LAB VALUE PARSING
# ============================================================================

def normalize_test_name(raw_name: str) -> Optional[str]:
    """Normalize a test name to a standard key"""
    raw_name_lower = raw_name.lower().strip()
    
    # Remove common prefixes/suffixes
    raw_name_lower = re.sub(r'^(test|result|lab|value)\s+', '', raw_name_lower)
    raw_name_lower = re.sub(r'\s+(test|result|lab|value)$', '', raw_name_lower)
    
    # Direct match
    for key, variations in TEST_NAME_VARIATIONS.items():
        for var in variations:
            if var.lower() == raw_name_lower:
                return key
    
    # Partial match (more flexible)
    for key, variations in TEST_NAME_VARIATIONS.items():
        for var in variations:
            var_lower = var.lower()
            if var_lower in raw_name_lower or raw_name_lower in var_lower:
                return key
    
    return None


def parse_lab_values(text: str) -> List[Dict[str, Any]]:
    """Parse lab values from extracted text with improved table support"""
    results = []
    
    # Clean the text first
    text = clean_extracted_text(text)
    
    # Split into lines for processing
    lines = text.split('\n')
    
    # Track processed test names to avoid duplicates
    processed_tests = set()
    
    # Common patterns for lab results (ordered by specificity)
    patterns = [
        # Pattern 1: "Test Name: Result (Range) Status" - Most common in medical reports
        r'(?P<name>[A-Za-z\s\-]+?)\s*[:;]\s*(?P<value>[\d,]+\.?[\d]*)\s*(?P<unit>[a-zA-Z/µ%]+)?\s*(?:\((?P<refLow>[\d,]+\.?[\d]*)\s*[-–]\s*(?P<refHigh>[\d,]+\.?[\d]*)\))?\s*(?P<status>[A-Za-z\s]+)?',
        
        # Pattern 2: "Test Name Result (Range) Status" - No colon
        r'(?P<name>[A-Za-z\s\-]+?)\s+(?P<value>[\d,]+\.?[\d]*)\s*(?P<unit>[a-zA-Z/µ%]+)?\s*(?:\((?P<refLow>[\d,]+\.?[\d]*)\s*[-–]\s*(?P<refHigh>[\d,]+\.?[\d]*)\))?\s*(?P<status>[A-Za-z\s]+)?',
        
        # Pattern 3: "Test Name: Result (Range)" - Simpler
        r'(?P<name>[A-Za-z\s\-]+?)\s*[:;]\s*(?P<value>[\d,]+\.?[\d]*)\s*(?:\((?P<refLow>[\d,]+\.?[\d]*)\s*[-–]\s*(?P<refHigh>[\d,]+\.?[\d]*)\))',
        
        # Pattern 4: "Test Name Result (Range)" - Simpler
        r'(?P<name>[A-Za-z\s\-]+?)\s+(?P<value>[\d,]+\.?[\d]*)\s*(?:\((?P<refLow>[\d,]+\.?[\d]*)\s*[-–]\s*(?P<refHigh>[\d,]+\.?[\d]*)\))',
        
        # Pattern 5: "Test Name: Result" - No range
        r'(?P<name>[A-Za-z\s\-]+?)\s*[:;]\s*(?P<value>[\d,]+\.?[\d]*)\s*(?P<unit>[a-zA-Z/µ%]+)?',
        
        # Pattern 6: BP specific - "BP: 130/85 mmHg"
        r'(?P<name>blood\s*pressure|bp)\s*[:;]\s*(?P<systolic>\d+)\s*/\s*(?P<diastolic>\d+)\s*(?P<unit>mmHg)?',
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for blood pressure pattern first
        bp_match = re.search(r'(?P<name>blood\s*pressure|bp)\s*[:;]\s*(?P<systolic>\d+)\s*/\s*(?P<diastolic>\d+)\s*(?P<unit>mmHg)?', line, re.IGNORECASE)
        if bp_match:
            systolic = int(bp_match.group('systolic'))
            diastolic = int(bp_match.group('diastolic'))
            
            # Add systolic
            results.append({
                "name": "Blood Pressure Systolic",
                "raw_name": "Blood Pressure (Systolic)",
                "value": systolic,
                "unit": "mmHg",
                "refLow": 90,
                "refHigh": 120,
                "system": "heart"
            })
            
            # Add diastolic
            results.append({
                "name": "Blood Pressure Diastolic",
                "raw_name": "Blood Pressure (Diastolic)",
                "value": diastolic,
                "unit": "mmHg",
                "refLow": 60,
                "refHigh": 80,
                "system": "heart"
            })
            continue
        
        # Check each pattern
        for pattern in patterns:
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                # Handle BP separately (already handled above)
                if 'systolic' in match.groupdict():
                    continue
                
                raw_name = match.group('name').strip()
                
                # Handle value with commas
                value_str = match.group('value')
                if value_str:
                    value_str = value_str.replace(',', '')
                    try:
                        value = float(value_str)
                    except ValueError:
                        continue
                else:
                    continue
                
                # Get unit
                unit = match.group('unit') if 'unit' in match.groupdict() and match.group('unit') else ""
                
                # Get reference range
                ref_low = match.group('refLow') if 'refLow' in match.groupdict() and match.group('refLow') else None
                ref_high = match.group('refHigh') if 'refHigh' in match.groupdict() and match.group('refHigh') else None
                
                # Get status
                status = match.group('status') if 'status' in match.groupdict() and match.group('status') else ""
                
                # Clean up the name
                raw_name = raw_name.strip()
                raw_name = re.sub(r'^[:\s]+', '', raw_name)
                raw_name = re.sub(r'[:\s]+$', '', raw_name)
                
                # Normalize the test name
                test_key = normalize_test_name(raw_name)
                
                # If no test_key found, try to match with status
                if not test_key and status:
                    test_key = normalize_test_name(status)
                
                if test_key:
                    # Get reference ranges from database if not provided
                    ref_data = REFERENCE_RANGES.get(test_key, {})
                    if ref_low is None:
                        ref_low = ref_data.get("low")
                    if ref_high is None:
                        ref_high = ref_data.get("high")
                    if not unit:
                        unit = ref_data.get("unit", "")
                    
                    # Convert ref values to float
                    try:
                        ref_low = float(ref_low) if ref_low is not None else None
                        ref_high = float(ref_high) if ref_high is not None else None
                    except (ValueError, TypeError):
                        ref_low = None
                        ref_high = None
                    
                    # Create a unique key to avoid duplicates
                    result_key = f"{test_key}_{value}"
                    if result_key not in processed_tests:
                        processed_tests.add(result_key)
                        results.append({
                            "name": test_key.capitalize(),
                            "raw_name": raw_name,
                            "value": value,
                            "unit": unit,
                            "refLow": ref_low,
                            "refHigh": ref_high,
                            "status": status.strip() if status else "",
                            "system": SYSTEM_MAP.get(test_key, "general")
                        })
                    break  # Stop after first match for this line
    
    return results


# ============================================================================
# FLAGGING AND ANALYSIS FUNCTIONS
# ============================================================================

def flag_abnormal_results(lab_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flag results that are outside reference ranges"""
    flagged = []
    
    for lab in lab_results:
        ref_low = lab.get("refLow")
        ref_high = lab.get("refHigh")
        value = lab.get("value")
        
        # Create a copy
        result = lab.copy()
        
        # Check if status field indicates normal/abnormal
        status = lab.get("status", "").lower()
        if status:
            if "high" in status or "elevated" in status:
                result["flag"] = "high"
                flagged.append(result)
                continue
            elif "low" in status or "decreased" in status:
                result["flag"] = "low"
                flagged.append(result)
                continue
            elif "normal" in status or "normal" in status:
                result["flag"] = "normal"
                flagged.append(result)
                continue
        
        # Check against reference ranges
        if ref_low is not None and ref_high is not None:
            if value < ref_low:
                result["flag"] = "low"
            elif value > ref_high:
                result["flag"] = "high"
            else:
                result["flag"] = "normal"
        else:
            result["flag"] = "unknown"
        
        flagged.append(result)
    
    return flagged


def calculate_health_score(lab_results: List[Dict[str, Any]]) -> Optional[int]:
    """Calculate a health score based on lab results"""
    if not lab_results:
        return None
    
    total = len(lab_results)
    abnormal = sum(1 for l in lab_results if l.get("flag") in ["high", "low"])
    
    if total == 0:
        return None
    
    # Weighted scoring: each abnormal reduces score
    # Base score starts at 100, each abnormal reduces by 8 points (minimum 0)
    score = max(0, 100 - (abnormal * 8))
    return int(score)


def generate_recommendations(lab_results: List[Dict[str, Any]]) -> List[str]:
    """Generate recommendations based on lab results"""
    recommendations = []
    
    abnormal_results = [l for l in lab_results if l.get("flag") in ["high", "low"]]
    
    if abnormal_results:
        recommendations.append("⚠️ The following results require attention:")
        for lab in abnormal_results:
            flag = lab.get("flag")
            name = lab.get("name")
            value = lab.get("value")
            unit = lab.get("unit", "")
            ref_low = lab.get("refLow")
            ref_high = lab.get("refHigh")
            
            if flag == "high":
                recommendations.append(f"  • {name}: {value} {unit} is ABOVE the reference range ({ref_low}-{ref_high})")
            elif flag == "low":
                recommendations.append(f"  • {name}: {value} {unit} is BELOW the reference range ({ref_low}-{ref_high})")
        
        recommendations.append("")
        recommendations.append("💡 Recommended next steps:")
        recommendations.append("  • Schedule a follow-up appointment with your healthcare provider")
        recommendations.append("  • Discuss these results and any symptoms you may be experiencing")
        recommendations.append("  • Ask about additional tests or monitoring if needed")
        recommendations.append("  • Consider lifestyle modifications as advised by your doctor")
    else:
        recommendations.append("✅ All your results are within normal ranges.")
        recommendations.append("💡 General recommendations:")
        recommendations.append("  • Continue maintaining a healthy lifestyle")
        recommendations.append("  • Regular exercise and balanced diet")
        recommendations.append("  • Follow up with your healthcare provider as recommended")
        recommendations.append("  • Keep track of any new symptoms or changes in your health")
    
    recommendations.append("")
    recommendations.append("⚠️ IMPORTANT: This analysis is for informational purposes only.")
    recommendations.append("Always consult with a qualified healthcare provider for medical decisions.")
    
    return recommendations


def generate_summary(lab_results: List[Dict[str, Any]], abnormal: List[Dict[str, Any]]) -> str:
    """Generate a summary of the analysis"""
    total = len(lab_results)
    abnormal_count = len(abnormal)
    
    if total == 0:
        return "No lab values were detected in this report."
    
    if abnormal_count == 0:
        return f"Analysis of {total} lab values shows all results are within normal reference ranges."
    else:
        # Group abnormal by system
        system_counts = {}
        for a in abnormal:
            system = a.get("system", "general")
            system_counts[system] = system_counts.get(system, 0) + 1
        
        system_summary = ", ".join([f"{count} in {system}" for system, count in system_counts.items()])
        return f"Analysis of {total} lab values found {abnormal_count} abnormal result(s) ({system_summary})."


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_locally(filename: str, content: bytes) -> Dict[str, Any]:
    """
    Main entry point for local medical report analysis.
    
    Args:
        filename: Name of the file being analyzed
        content: Raw bytes of the file content
    
    Returns:
        Dict containing analysis results
    """
    try:
        # Step 1: Extract text
        text = extract_text(content, filename)
        
        if not text or not text.strip():
            raise MedicalAnalysisError("No text could be extracted from the file.")
        
        # Step 2: Parse lab values
        lab_results = parse_lab_values(text)
        
        if not lab_results:
            return {
                "labs": [],
                "abnormal": [],
                "health_score": None,
                "recommendations": [
                    "No lab values were found in this report. Please ensure the file contains lab results in a readable format.",
                    "Try using a text-based PDF or ensure the PDF text is selectable."
                ],
                "primary_finding": "No lab values detected",
                "summary": "No lab values were detected in this report.",
                "total_labs": 0,
                "normal_labs": 0,
                "abnormal_count": 0
            }
        
        # Step 3: Flag abnormal results
        flagged_results = flag_abnormal_results(lab_results)
        
        # Step 4: Calculate health score
        health_score = calculate_health_score(flagged_results)
        
        # Step 5: Get abnormal results
        abnormal_results = [l for l in flagged_results if l.get("flag") in ["high", "low"]]
        
        # Step 6: Generate recommendations
        recommendations = generate_recommendations(flagged_results)
        
        # Step 7: Generate summary
        summary = generate_summary(flagged_results, abnormal_results)
        
        # Step 8: Determine primary finding
        if abnormal_results:
            primary_finding = f"{len(abnormal_results)} abnormal result(s) detected"
            # Add first abnormal result for context
            first_abnormal = abnormal_results[0]
            primary_finding += f" - {first_abnormal.get('name')} is {first_abnormal.get('flag')}"
        else:
            primary_finding = "All results within normal range"
        
        return {
            "labs": flagged_results,
            "abnormal": abnormal_results,
            "health_score": health_score,
            "recommendations": recommendations,
            "primary_finding": primary_finding,
            "summary": summary,
            "total_labs": len(flagged_results),
            "normal_labs": len([l for l in flagged_results if l.get("flag") == "normal"]),
            "abnormal_count": len(abnormal_results)
        }
        
    except MedicalAnalysisError:
        raise
    except Exception as e:
        raise MedicalAnalysisError(f"Analysis failed: {str(e)}")


# ============================================================================
# FORMAT RESULTS FOR DISPLAY
# ============================================================================

def format_results_for_display(results: Dict[str, Any]) -> str:
    """Format analysis results for display in the UI"""
    output = []
    
    output.append("=" * 60)
    output.append("NEUROGUARD - MEDICAL REPORT ANALYSIS")
    output.append("=" * 60)
    output.append("")
    
    output.append(f"📊 SUMMARY: {results.get('summary', 'N/A')}")
    output.append(f"📈 Primary Finding: {results.get('primary_finding', 'N/A')}")
    output.append(f"💚 Health Score: {results.get('health_score', 'N/A')}%")
    output.append(f"📋 Total Labs: {results.get('total_labs', 0)}")
    output.append(f"✅ Normal: {results.get('normal_labs', 0)}")
    output.append(f"⚠️ Abnormal: {results.get('abnormal_count', 0)}")
    output.append("")
    
    # Lab results table
    if results.get('labs'):
        output.append("-" * 60)
        output.append("📊 LAB RESULTS")
        output.append("-" * 60)
        output.append(f"{'Test':<22} {'Value':<18} {'Range':<22} {'Flag':<10}")
        output.append("-" * 60)
        
        for lab in results.get('labs', []):
            name = lab.get('name', 'Unknown')[:20]
            value = f"{lab.get('value', 'N/A')} {lab.get('unit', '')}"
            ref = f"{lab.get('refLow', 'N/A')} - {lab.get('refHigh', 'N/A')}"
            flag = lab.get('flag', 'unknown').upper()
            
            # Color indicators
            if flag == 'HIGH':
                flag = '🔴 HIGH'
            elif flag == 'LOW':
                flag = '🔵 LOW'
            elif flag == 'NORMAL':
                flag = '🟢 NORMAL'
            else:
                flag = '⚪ UNKNOWN'
            
            output.append(f"{name:<22} {value:<18} {ref:<22} {flag:<10}")
        
        output.append("-" * 60)
        output.append("")
    
    # Recommendations
    output.append("💡 RECOMMENDATIONS")
    output.append("-" * 60)
    for rec in results.get('recommendations', []):
        output.append(f"  {rec}")
    output.append("-" * 60)
    output.append("")
    output.append("⚠️ This analysis is for informational purposes only.")
    output.append("⚠️ Always consult with a qualified healthcare provider.")
    output.append("=" * 60)
    
    return '\n'.join(output)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING MEDICAL ANALYSIS MODULE")
    print("=" * 60)
    
    # Test with sample text from your PDF
    sample_text = """
    Laboratory Test Results
    
    Test Name           Result      Normal Range       Status
    Hemoglobin          13.8 g/dL   13.0 - 17.0        Normal
    WBC Count           6,200 /μL   4,000 - 11,000     Normal
    Blood Sugar         112 mg/dL   70 - 100           Slightly High
    Total Cholesterol   210 mg/dL   < 200              Borderline High
    Blood Pressure      130/85 mmHg 120/80             Elevated
    """
    
    # Parse and analyze
    results = parse_lab_values(sample_text)
    flagged = flag_abnormal_results(results)
    health_score = calculate_health_score(flagged)
    recommendations = generate_recommendations(flagged)
    abnormal = [l for l in flagged if l.get("flag") in ["high", "low"]]
    summary = generate_summary(flagged, abnormal)
    
    analysis_results = {
        "labs": flagged,
        "abnormal": abnormal,
        "health_score": health_score,
        "recommendations": recommendations,
        "primary_finding": f"{len(abnormal)} abnormal result(s) detected" if abnormal else "All results within normal range",
        "summary": summary,
        "total_labs": len(flagged),
        "normal_labs": len([l for l in flagged if l.get("flag") == "normal"]),
        "abnormal_count": len(abnormal)
    }
    
    # Display results
    print(format_results_for_display(analysis_results))
    
    print("\n📦 Dependency Status:")
    try:
        import pymupdf
        print("  ✅ PyMuPDF (pymupdf) - Installed")
    except ImportError:
        print("  ❌ PyMuPDF (pymupdf) - Not installed")
    
    if OCR_AVAILABLE:
        print("  ✅ Tesseract OCR - Available")
    else:
        print("  ⚠️ Tesseract OCR - Not installed (optional)")
    
    print("\n✅ Module loaded successfully!")
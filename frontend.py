# frontend.py - NeuroGuard Patient Portal with Private Local Analysis & Anny 3D Digital Twin

from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime
import hashlib
import random
import json
import os
import re
import tempfile
import shutil
import traceback
import uuid
import httpx
from dotenv import load_dotenv

# Import database
from database import (
    Claim, get_db, init_database, seed_database,
    Patient, Admin, Report, AIAnalysis, DigitalTwin,
    hash_password, verify_password
)

# Import local medical analysis
from medical_analysis import MedicalAnalysisError, analyze_locally

# Load environment variables
load_dotenv()

# Anny is temporarily disabled while its local model dependencies are repaired.
# The report-analysis visual anatomy model continues to work independently.
ANNY_AVAILABLE = False

app = FastAPI(
    title="NeuroGuard Patient Portal",
    description="AI-powered healthcare for patients with private local analysis & 3D Digital Twin",
    version="3.0.0"
)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# ============================================================================
# ANNY 3D MODEL CONFIGURATION
# ============================================================================
ANNY_MODEL = None
ANNY_MODEL_INITIALIZED = False
ANNY_MODEL_INITIALIZING = False
ANNY_ERROR_MESSAGE = None

def get_anny_model():
    """Initialize and return the Anny model (lazy loading for performance)"""
    global ANNY_MODEL, ANNY_MODEL_INITIALIZED, ANNY_MODEL_INITIALIZING, ANNY_ERROR_MESSAGE
    
    if not ANNY_AVAILABLE:
        ANNY_ERROR_MESSAGE = "Anny dependencies not installed. Run: pip install torch anny trimesh"
        return None
    
    if ANNY_MODEL_INITIALIZED:
        return ANNY_MODEL
    
    if ANNY_MODEL_INITIALIZING:
        import time
        time.sleep(2)
        return get_anny_model()
    
    ANNY_MODEL_INITIALIZING = True
    try:
        print("Initializing Anny 3D model (this may take a few seconds)...")
        ANNY_MODEL = anny.Anny(local_changes="default", facial_actions="all").to(dtype=torch.float32)
        ANNY_MODEL_INITIALIZED = True
        ANNY_ERROR_MESSAGE = None
        print("Anny model loaded successfully")
    except Exception as e:
        print(f"Error initializing Anny: {str(e)}")
        traceback.print_exc()
        ANNY_ERROR_MESSAGE = str(e)
        ANNY_MODEL = None
    finally:
        ANNY_MODEL_INITIALIZING = False
    
    return ANNY_MODEL

# Create directory for generated models
MODELS_DIR = "generated_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================================
# ANNY MODEL GENERATION FUNCTION
# ============================================================================

def generate_patient_3d_model(patient_data: dict) -> tuple:
    """
    Generate a 3D model file (.glb) for a patient based on their data.
    Returns (success, file_path_or_error_message)
    """
    try:
        # Check if Anny is available
        if not ANNY_AVAILABLE:
            return False, "Anny dependencies not installed. Run: pip install torch anny trimesh"
        
        # Get the Anny model
        model = get_anny_model()
        
        if model is None:
            error_msg = ANNY_ERROR_MESSAGE or "Failed to initialize Anny model"
            return False, error_msg
        
        # Map patient data to Anny phenotype parameters
        age = patient_data.get("age", 30)
        age_param = min(max((age - 1) / 99, 0), 1)
        
        gender = patient_data.get("gender", "Male")
        gender_param = 1.0 if gender == "Female" else 0.0
        
        height_cm = patient_data.get("height", 170)
        weight_kg = patient_data.get("weight", 70)
        
        if height_cm > 0 and weight_kg > 0:
            bmi = weight_kg / ((height_cm/100) ** 2)
        else:
            bmi = 22
        
        weight_param = min(max((bmi - 15) / 25, 0), 1)
        height_param = min(max((height_cm - 150) / 50, 0), 1)
        muscle_param = 0.5
        
        phenotype_kwargs = {
            "age": float(age_param),
            "gender": float(gender_param),
            "weight": float(weight_param),
            "height": float(height_param),
            "muscle": float(muscle_param)
        }
        
        # Set up pose (neutral standing)
        pose_parameters = torch.eye(4)[None, None].repeat(1, model.bone_count, 1, 1)
        
        # Generate the mesh
        output = model(
            pose_parameters=pose_parameters,
            phenotype_kwargs=phenotype_kwargs,
            local_changes_kwargs={},
            facial_actions={}
        )
        
        # Get vertices and faces
        vertices = output["vertices"].squeeze(dim=0).detach().cpu().numpy()
        faces = model.faces.detach().cpu().numpy()
        
        # Create a mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Generate unique filename
        patient_id = patient_data.get("patient_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        glb_filename = f"patient_{patient_id}_{timestamp}.glb"
        glb_path = os.path.join(MODELS_DIR, glb_filename)
        
        # Export as GLB (for Three.js)
        mesh.export(glb_path)
        
        # Clean up old files for this patient
        for f in os.listdir(MODELS_DIR):
            if f.startswith(f"patient_{patient_id}_") and f != glb_filename:
                try:
                    os.remove(os.path.join(MODELS_DIR, f))
                except:
                    pass
        
        print(f"3D model generated: {glb_path}")
        return True, glb_path
        
    except Exception as e:
        print(f"Error generating 3D model: {str(e)}")
        traceback.print_exc()
        return False, str(e)

# ============================================================================
# INITIALIZATION
# ============================================================================

@app.on_event("startup")
async def startup_event():
    init_database()
    seed_database()
    print("Server started successfully!")
    print("Local, private report parser enabled (no report data leaves this server)")
    if not ANNY_AVAILABLE:
        print("Anny 3D model is temporarily disabled.")

# ============================================================================
# HTML LAYOUT
# ============================================================================

LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroGuard - {title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- Three.js for 3D rendering -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <style>
        * {{ font-family: 'Inter', sans-serif; }}
        body {{ background: #f0f4f8; }}
        .gradient-bg {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .card-hover:hover {{ transform: translateY(-4px); box-shadow: 0 20px 40px rgba(0,0,0,0.1); transition: all 0.3s; }}
        .fade-in {{ animation: fadeIn 0.6s ease-in; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        .health-score-circle {{ width: 120px; height: 120px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 2.5rem; font-weight: bold; }}
        .upload-zone {{ border: 3px dashed #cbd5e0; border-radius: 1rem; padding: 3rem; text-align: center; transition: all 0.3s; cursor: pointer; }}
        .upload-zone:hover {{ border-color: #667eea; background: #f7fafc; }}
        .upload-zone.dragover {{ border-color: #667eea; background: #ebf4ff; }}
        .animate-spin {{ animation: spin 1s linear infinite; }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        .lab-result-item {{ transition: all 0.3s; }}
        .lab-result-item:hover {{ background: #f7fafc; }}
        #twin3d-container {{
            position: relative;
            width: 100%;
            height: 500px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 1rem;
            overflow: hidden;
        }}
        #twin3d {{
            width: 100%;
            height: 100%;
            display: block;
        }}
        .twin-controls {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(255,255,255,0.9);
            padding: 10px;
            border-radius: 8px;
            z-index: 10;
        }}
        .twin-legend {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.9);
            padding: 10px;
            border-radius: 8px;
            z-index: 10;
            font-size: 12px;
        }}
        .loading-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(255,255,255,0.8);
            z-index: 20;
        }}
        .organ-card {{ border: 1px solid #cffafe; border-radius: 1rem; padding: .65rem; background: linear-gradient(160deg,#ecfeff,#fff); transition: transform .2s, box-shadow .2s; cursor: pointer; }}
        .organ-card:hover {{ transform: translateY(-4px); box-shadow: 0 12px 25px rgba(8,145,178,.18); }}
        .organ-card.selected {{ outline: 3px solid #0891b2; transform: translateY(-4px); box-shadow: 0 16px 30px rgba(8,145,178,.22); }}
        .organ-model {{ width: 100%; height: 150px; border-radius: .75rem; background: radial-gradient(circle at 50% 45%,#fff,#cffafe); }}
        .organ-canvas {{ width: 100%; height: 150px; display: block; border-radius: .75rem; background: radial-gradient(circle at 50% 45%,#ecfeff,#bae6fd); cursor: grab; }}
        .organ-badge {{ position: absolute; top: .5rem; right: .5rem; background: #e11d48; color: white; font-size: .65rem; font-weight: 700; border-radius: 999px; padding: .2rem .45rem; }}
    </style>
</head>
<body>
    {navbar}
    <main class="container mx-auto px-4 py-8 max-w-7xl">
        {content}
    </main>
    {footer}
    {scripts}
</body>
</html>
"""

FOOTER = """
<footer class="bg-gray-800 text-white mt-16">
    <div class="container mx-auto px-4 py-8">
        <div class="grid md:grid-cols-4 gap-8">
            <div><h3 class="font-bold text-lg mb-4">NeuroGuard</h3><p class="text-gray-400 text-sm">AI-powered healthcare intelligence platform</p></div>
            <div><h4 class="font-semibold mb-3">Patient</h4><ul class="space-y-2 text-sm text-gray-400"><li><a href="/patient/dashboard" class="hover:text-white">Dashboard</a></li><li><a href="/patient/analyze" class="hover:text-white">AI Analysis</a></li><li><a href="/patient/twin" class="hover:text-white">Digital Twin</a></li></ul></div>
            <div><h4 class="font-semibold mb-3">Security</h4><ul class="space-y-2 text-sm text-gray-400"><li><a href="#" class="hover:text-white">Privacy Policy</a></li><li><a href="#" class="hover:text-white">HIPAA Compliance</a></li><li><a href="#" class="hover:text-white">Security</a></li></ul></div>
            <div><h4 class="font-semibold mb-3">Contact</h4><ul class="space-y-2 text-sm text-gray-400"><li><i class="fas fa-envelope mr-2"></i>support@neuroguard.com</li><li><i class="fas fa-phone mr-2"></i>+1 (800) 555-0199</li></ul></div>
        </div>
        <div class="border-t border-gray-700 mt-8 pt-4 text-center text-sm text-gray-400">&copy; 2026 NeuroGuard Healthcare Platform. All rights reserved.</div>
    </div>
</footer>
"""

def get_navbar(user_type: str = None, user_name: str = None):
    if user_type == "patient":
        return f"""
<nav class="bg-white shadow-lg sticky top-0 z-50">
    <div class="container mx-auto px-4 py-3">
        <div class="flex justify-between items-center">
            <a href="/patient/dashboard" class="text-2xl font-bold text-blue-600 flex items-center"><i class="fas fa-brain mr-2"></i>NeuroGuard</a>
            <div class="hidden md:flex space-x-6">
                <a href="/patient/dashboard" class="text-gray-700 hover:text-blue-600 transition"><i class="fas fa-home mr-1"></i>Dashboard</a>
                <a href="/patient/analyze" class="text-gray-700 hover:text-blue-600 transition"><i class="fas fa-microscope mr-1"></i>Analyze Report</a>
                <a href="/patient/twin" class="text-gray-700 hover:text-blue-600 transition"><i class="fas fa-robot mr-1"></i>Digital Twin</a>
                <a href="/patient/reports" class="text-gray-700 hover:text-blue-600 transition"><i class="fas fa-file-medical mr-1"></i>My Reports</a>
                <a href="/patient/copilot" class="text-gray-700 hover:text-blue-600 transition"><i class="fas fa-comments mr-1"></i>Health Copilot</a>
                <a href="/patient/profile" class="text-gray-700 hover:text-blue-600 transition"><i class="fas fa-user-circle mr-1"></i>Profile</a>
            </div>
            <div class="flex items-center gap-3">
                <span class="text-sm text-gray-600 hidden md:inline"><i class="fas fa-user mr-1"></i>{user_name}</span>
                <a href="/logout" class="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition text-sm"><i class="fas fa-sign-out-alt mr-1"></i>Logout</a>
            </div>
        </div>
    </div>
</nav>"""
    elif user_type == "admin":
        return f"""
<nav class="bg-gray-900 shadow-lg sticky top-0 z-50">
    <div class="container mx-auto px-4 py-3">
        <div class="flex justify-between items-center">
            <a href="/admin/dashboard" class="text-2xl font-bold text-white flex items-center"><i class="fas fa-brain mr-2"></i>NeuroGuard Admin</a>
            <div class="hidden md:flex space-x-6">
                <a href="/admin/dashboard" class="text-gray-300 hover:text-white transition"><i class="fas fa-chart-bar mr-1"></i>Dashboard</a>
                <a href="/admin/patients" class="text-gray-300 hover:text-white transition"><i class="fas fa-users mr-1"></i>Patients</a>
                <a href="/admin/claims" class="text-gray-300 hover:text-white transition"><i class="fas fa-file-invoice mr-1"></i>Claims</a>
                <a href="/admin/fraud" class="text-gray-300 hover:text-white transition"><i class="fas fa-shield-alt mr-1"></i>Fraud Detection</a>
            </div>
            <div class="flex items-center gap-3">
                <span class="text-sm text-gray-400 hidden md:inline"><i class="fas fa-user-shield mr-1"></i>{user_name}</span>
                <a href="/logout" class="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition text-sm"><i class="fas fa-sign-out-alt mr-1"></i>Logout</a>
            </div>
        </div>
    </div>
</nav>"""
    else:
        return """
<nav class="bg-white shadow-lg sticky top-0 z-50">
    <div class="container mx-auto px-4 py-3">
        <div class="flex justify-between items-center">
            <a href="/" class="text-2xl font-bold text-blue-600 flex items-center"><i class="fas fa-brain mr-2"></i>NeuroGuard</a>
            <div class="hidden md:flex space-x-6">
                <a href="/" class="text-gray-700 hover:text-blue-600 transition">Home</a>
                <a href="#features" class="text-gray-700 hover:text-blue-600 transition">Features</a>
            </div>
            <div class="flex items-center gap-3">
                <a href="/login" class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition"><i class="fas fa-sign-in-alt mr-1"></i>Login</a>
            </div>
        </div>
    </div>
</nav>"""

def render_page(content: str, title: str = "NeuroGuard", user_type: str = None, user_name: str = None, scripts: str = "") -> str:
    return LAYOUT.format(
        title=title,
        navbar=get_navbar(user_type, user_name),
        content=content,
        footer=FOOTER,
        scripts=scripts
    )

# ============================================================================
# PAGE CONTENT FUNCTIONS
# ============================================================================

def home_page() -> str:
    return """
<div class="fade-in">
    <section class="text-center py-20">
        <div class="max-w-4xl mx-auto">
            <div class="flex justify-center mb-6"><div class="w-24 h-24 bg-blue-100 rounded-2xl flex items-center justify-center"><i class="fas fa-brain text-5xl text-blue-600"></i></div></div>
            <h1 class="text-5xl font-bold text-gray-900 mb-6">Your Health, <span class="gradient-bg bg-clip-text text-transparent">Intelligently Managed</span></h1>
            <p class="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">NeuroGuard uses AI to analyze your medical reports, create your digital twin, and provide personalized health insights.</p>
            <div class="flex justify-center gap-4 flex-wrap">
                <a href="/signup" class="px-8 py-3 gradient-bg text-white rounded-lg hover:opacity-90 transition"><i class="fas fa-user-plus mr-2"></i>Get Started</a>
            </div>
        </div>
    </section>
    <section id="features" class="py-16">
        <h2 class="text-3xl font-bold text-center text-gray-800 mb-12"><i class="fas fa-cogs text-blue-600 mr-2"></i>What We Offer</h2>
        <div class="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            <div class="bg-white p-8 rounded-2xl shadow-lg card-hover"><div class="w-16 h-16 gradient-bg rounded-2xl flex items-center justify-center mb-4"><i class="fas fa-microscope text-2xl text-white"></i></div><h3 class="text-xl font-semibold mb-2">Private Report Analysis</h3><p class="text-gray-600">Extract supported lab values locally, compare them with reference ranges, and review clear flags.</p></div>
            <div class="bg-white p-8 rounded-2xl shadow-lg card-hover"><div class="w-16 h-16 gradient-bg rounded-2xl flex items-center justify-center mb-4"><i class="fas fa-robot text-2xl text-white"></i></div><h3 class="text-xl font-semibold mb-2">Digital Twin</h3><p class="text-gray-600">Your virtual health twin that tracks vital signs, predicts health risks, and suggests preventive measures.</p></div>
            <div class="bg-white p-8 rounded-2xl shadow-lg card-hover"><div class="w-16 h-16 gradient-bg rounded-2xl flex items-center justify-center mb-4"><i class="fas fa-file-medical text-2xl text-white"></i></div><h3 class="text-xl font-semibold mb-2">Health Records</h3><p class="text-gray-600">All your medical reports, lab results, and health data in one secure, accessible place.</p></div>
        </div>
    </section>
    <section class="gradient-bg rounded-3xl py-12 px-8 text-white">
        <div class="grid md:grid-cols-4 gap-8 max-w-5xl mx-auto text-center">
            <div><div class="text-4xl font-bold">10K+</div><div class="text-sm opacity-90">Patients Served</div></div>
            <div><div class="text-4xl font-bold">98%</div><div class="text-sm opacity-90">AI Accuracy</div></div>
            <div><div class="text-4xl font-bold">50K+</div><div class="text-sm opacity-90">Reports Analyzed</div></div>
            <div><div class="text-4xl font-bold">4.9⭐</div><div class="text-sm opacity-90">Patient Rating</div></div>
        </div>
    </section>
</div>"""

def login_page_content() -> str:
    return """
<div class="max-w-4xl mx-auto mt-8">
    <div class="text-center mb-8"><h1 class="text-3xl font-bold text-gray-800">Welcome Back</h1><p class="text-gray-600">Choose your login type</p></div>
    <div class="grid md:grid-cols-2 gap-8">
        <div class="bg-white p-8 rounded-2xl shadow-lg card-hover">
            <div class="text-center"><div class="w-20 h-20 gradient-bg rounded-2xl flex items-center justify-center mx-auto mb-4"><i class="fas fa-user text-3xl text-white"></i></div><h3 class="text-2xl font-bold text-gray-800">Patient</h3><p class="text-gray-500 text-sm mb-6">Access your health data, AI analysis, and digital twin</p></div>
            <form action="/login/patient" method="post" class="space-y-4">
                <div><label class="block text-sm font-medium text-gray-700 mb-1"><i class="fas fa-envelope mr-1"></i>Email</label><input type="email" name="email" required placeholder="patient@neuroguard.com" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"></div>
                <div><label class="block text-sm font-medium text-gray-700 mb-1"><i class="fas fa-lock mr-1"></i>Password</label><input type="password" name="password" required placeholder="········" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"></div>
                <button type="submit" class="w-full py-3 gradient-bg text-white rounded-lg hover:opacity-90 transition"><i class="fas fa-sign-in-alt mr-2"></i>Sign In as Patient</button>
            </form>
            <div class="mt-4 p-3 bg-blue-50 rounded-lg"><p class="text-center text-xs text-gray-500">Demo: patient@neuroguard.com / patient123</p></div>
            <p class="text-center text-sm text-gray-500 mt-4">Don't have an account? <a href="/signup" class="text-blue-600 hover:underline">Sign Up</a></p>
        </div>
        <div class="bg-white p-8 rounded-2xl shadow-lg card-hover">
            <div class="text-center"><div class="w-20 h-20 bg-gray-900 rounded-2xl flex items-center justify-center mx-auto mb-4"><i class="fas fa-user-shield text-3xl text-white"></i></div><h3 class="text-2xl font-bold text-gray-800">Admin</h3><p class="text-gray-500 text-sm mb-6">Manage patients, claims, and fraud detection</p></div>
            <form action="/login/admin" method="post" class="space-y-4">
                <div><label class="block text-sm font-medium text-gray-700 mb-1"><i class="fas fa-envelope mr-1"></i>Email</label><input type="email" name="email" required placeholder="admin@neuroguard.com" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-gray-900"></div>
                <div><label class="block text-sm font-medium text-gray-700 mb-1"><i class="fas fa-lock mr-1"></i>Password</label><input type="password" name="password" required placeholder="········" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-gray-900"></div>
                <button type="submit" class="w-full py-3 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition"><i class="fas fa-sign-in-alt mr-2"></i>Sign In as Admin</button>
            </form>
            <div class="mt-4 p-3 bg-gray-50 rounded-lg"><p class="text-center text-xs text-gray-500">Demo: admin@neuroguard.com / admin123</p></div>
        </div>
    </div>
</div>"""

def signup_page_content() -> str:
    return """
<div class="max-w-2xl mx-auto mt-8">
    <div class="bg-white p-8 rounded-2xl shadow-lg card-hover">
        <div class="text-center mb-8">
            <div class="w-20 h-20 gradient-bg rounded-2xl flex items-center justify-center mx-auto mb-4">
                <i class="fas fa-user-plus text-3xl text-white"></i>
            </div>
            <h1 class="text-3xl font-bold text-gray-800">Create Your Account</h1>
            <p class="text-gray-600 mt-1">Join NeuroGuard and take control of your health</p>
        </div>
        
        <form action="/signup/patient" method="post" class="space-y-4" id="signup-form">
            <!-- Role Selection -->
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">
                    <i class="fas fa-user-tag mr-1"></i>I am a <span class="text-red-500">*</span>
                </label>
                <select name="role" required id="role-select" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500" onchange="toggleAdminCode()">
                    <option value="patient">Patient</option>
                    <option value="admin">Admin / Healthcare Provider</option>
                </select>
            </div>
            
            <!-- Admin Verification Code (Hidden by default) -->
            <div id="admin-code-section" class="hidden">
                <div class="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        <i class="fas fa-key mr-1"></i>Admin Verification Code <span class="text-red-500">*</span>
                    </label>
                    <input type="password" name="admin_code" id="admin-code" 
                           placeholder="Enter your admin verification code"
                           class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-yellow-500">
                    <p class="text-xs text-gray-500 mt-1">
                        <i class="fas fa-info-circle mr-1"></i>
                        Contact your system administrator for the verification code.
                    </p>
                </div>
            </div>
            
            <!-- User Details -->
            <div class="grid md:grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        <i class="fas fa-user mr-1"></i>Full Name <span class="text-red-500">*</span>
                    </label>
                    <input type="text" name="name" required placeholder="John Doe" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        <i class="fas fa-envelope mr-1"></i>Email <span class="text-red-500">*</span>
                    </label>
                    <input type="email" name="email" required placeholder="you@example.com" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        <i class="fas fa-lock mr-1"></i>Password <span class="text-red-500">*</span>
                    </label>
                    <input type="password" name="password" required placeholder="········" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        <i class="fas fa-calendar mr-1"></i>Age <span class="text-red-500">*</span>
                    </label>
                    <input type="number" name="age" required placeholder="35" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        <i class="fas fa-venus-mars mr-1"></i>Gender <span class="text-red-500">*</span>
                    </label>
                    <select name="gender" required class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                        <option value="">Select Gender</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                    </select>
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        <i class="fas fa-phone mr-1"></i>Phone Number <span class="text-red-500">*</span>
                    </label>
                    <input type="tel" name="phone" required placeholder="+1 (555) 123-4567" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        <i class="fas fa-weight mr-1"></i>Weight (kg) <span class="text-red-500">*</span>
                    </label>
                    <input type="number" name="weight" required placeholder="70" step="0.1" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700 mb-1">
                        <i class="fas fa-ruler-vertical mr-1"></i>Height (cm) <span class="text-red-500">*</span>
                    </label>
                    <input type="number" name="height" required placeholder="170" step="0.1" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                </div>
            </div>
            
            <button type="submit" class="w-full py-3 gradient-bg text-white rounded-lg hover:opacity-90 transition">
                <i class="fas fa-user-plus mr-2"></i>Create Account
            </button>
        </form>
        
        <p class="text-center text-sm text-gray-500 mt-4">
            Already have an account? <a href="/login" class="text-blue-600 hover:underline">Login</a>
        </p>
        
        <div class="mt-4 p-3 bg-blue-50 rounded-lg">
            <p class="text-center text-xs text-gray-500">
                <i class="fas fa-info-circle mr-1"></i>
                By signing up, you agree to our Terms of Service and Privacy Policy.
                Your data is encrypted and secure.
            </p>
        </div>
    </div>
</div>

<script>
function toggleAdminCode() {
    const roleSelect = document.getElementById('role-select');
    const adminSection = document.getElementById('admin-code-section');
    const adminCodeInput = document.getElementById('admin-code');
    
    if (roleSelect.value === 'admin') {
        adminSection.classList.remove('hidden');
        adminSection.style.display = 'block';
        adminCodeInput.required = true;
    } else {
        adminSection.classList.add('hidden');
        adminSection.style.display = 'none';
        adminCodeInput.required = false;
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    toggleAdminCode();
});
</script>
"""

# ============================================================================
# THREE.JS 3D VIEWER HTML
# ============================================================================

def build_twin_viewer_html(model_url: str, patient_data: dict) -> tuple:
    """
    Returns (viewer_html, viewer_js) for displaying the 3D model with Three.js
    """
    
    viewer_html = f"""
    <div class="bg-white p-6 rounded-2xl shadow-lg mb-8">
        <div class="flex justify-between items-center mb-4">
            <div>
                <h4 class="font-semibold text-gray-800">
                    <i class="fas fa-user-astronaut text-purple-600 mr-2"></i>3D Digital Twin
                </h4>
                <p class="text-sm text-gray-500">Interactive anatomical twin with report-linked anomaly highlights</p>
            </div>
            <span class="text-xs text-gray-500">
                <i class="fas fa-info-circle mr-1"></i>
                Open anatomy model · CC BY-SA 4.0
            </span>
        </div>
        
        <div id="twin3d-container">
            <div id="twin3d" class="w-full h-full"></div>
            
            <!-- Loading overlay -->
            <div id="loading-overlay" class="loading-overlay">
                <div class="text-center">
                    <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                    <p class="text-gray-600 mt-4">Loading your 3D model...</p>
                </div>
            </div>
            
            <!-- Controls -->
            <div class="twin-controls">
                <div class="flex gap-2">
                    <button onclick="resetView()" class="px-3 py-1 bg-gray-100 rounded-lg text-xs text-gray-700 hover:bg-gray-200">
                        <i class="fas fa-undo mr-1"></i>Reset
                    </button>
                    <button onclick="rotateLeft()" class="px-3 py-1 bg-gray-100 rounded-lg text-xs text-gray-700 hover:bg-gray-200">
                        <i class="fas fa-rotate-left mr-1"></i>Rotate
                    </button>
                    <button onclick="zoomIn()" class="px-3 py-1 bg-gray-100 rounded-lg text-xs text-gray-700 hover:bg-gray-200">
                        <i class="fas fa-search-plus mr-1"></i>Zoom
                    </button>
                </div>
            </div>
            
            <!-- Health stats overlay -->
            <div class="twin-legend">
                <p class="font-bold text-gray-800 mb-2">Health Stats</p>
                <div class="space-y-1">
                    <p class="text-gray-600">Age: <span class="font-semibold">{patient_data.get('age', 'N/A')}</span></p>
                    <p class="text-gray-600">Gender: <span class="font-semibold">{patient_data.get('gender', 'N/A')}</span></p>
                    <p class="text-gray-600">Height: <span class="font-semibold">{patient_data.get('height', 'N/A')} cm</span></p>
                    <p class="text-gray-600">Weight: <span class="font-semibold">{patient_data.get('weight', 'N/A')} kg</span></p>
                </div>
            </div>
        </div>
        
        <div class="mt-4 flex items-center justify-between">
            <div>
            <p class="text-xs text-gray-500">
                    <i class="fas fa-info-circle mr-1"></i>
                    Your 3D model reflects your body composition and health metrics.
                    Not a substitute for professional medical advice.
                </p>
            </div>
            <button onclick="regenerateModel()" class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition text-sm">
                <i class="fas fa-sync-alt mr-2"></i>Regenerate Model
            </button>
        </div>
    </div>
    """

    # A dependable organ explorer is available even while Anny is disabled.
    # Each card is an organ-focused 3D view of the open anatomy mesh; clicking
    # a card opens a larger view and highlights the system linked to a flag.
    viewer_html += """
    <div class="mt-6 rounded-2xl border border-cyan-100 bg-white p-6 shadow-lg">
      <div class="flex flex-wrap items-end justify-between gap-3"><div><p class="text-xs font-bold uppercase tracking-widest text-cyan-700">Organ explorer</p><h3 class="mt-1 text-2xl font-bold text-slate-900">Inspect five key systems</h3><p class="text-sm text-slate-500">Click any organ to open its focused 3D view. Red badge indicates a linked report flag.</p></div><span class="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-600">Drag to rotate · scroll to zoom</span></div>
      <div class="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <button class="organ-card text-left" data-organ="Brain" data-system="thyroid"><div class="relative"><canvas class="organ-canvas" data-organ-canvas="brain"></canvas><span class="organ-badge hidden" data-badge="thyroid">FLAG</span></div><span class="mt-2 block font-semibold text-slate-800">Brain / thyroid</span><span class="text-xs text-slate-500">Neural focus</span></button>
        <button class="organ-card text-left" data-organ="Heart" data-system="heart"><div class="relative"><canvas class="organ-canvas" data-organ-canvas="heart"></canvas><span class="organ-badge hidden" data-badge="heart">FLAG</span></div><span class="mt-2 block font-semibold text-slate-800">Heart</span><span class="text-xs text-slate-500">Cardiac focus</span></button>
        <button class="organ-card text-left" data-organ="Lungs" data-system="immune"><div class="relative"><canvas class="organ-canvas" data-organ-canvas="lungs"></canvas><span class="organ-badge hidden" data-badge="immune">FLAG</span></div><span class="mt-2 block font-semibold text-slate-800">Lungs</span><span class="text-xs text-slate-500">Respiratory focus</span></button>
        <button class="organ-card text-left" data-organ="Kidneys" data-system="kidneys"><div class="relative"><canvas class="organ-canvas" data-organ-canvas="kidneys"></canvas><span class="organ-badge hidden" data-badge="kidneys">FLAG</span></div><span class="mt-2 block font-semibold text-slate-800">Kidneys</span><span class="text-xs text-slate-500">Renal focus</span></button>
        <button class="organ-card text-left" data-organ="Stomach" data-system="metabolic"><div class="relative"><canvas class="organ-canvas" data-organ-canvas="stomach"></canvas><span class="organ-badge hidden" data-badge="metabolic">FLAG</span></div><span class="mt-2 block font-semibold text-slate-800">Stomach</span><span class="text-xs text-slate-500">Digestive focus</span></button>
      </div>
    </div>
    """
    
    viewer_js = f"""
    <script>
    const anomalySystems = {json.dumps(patient_data.get('anomaly_systems', []))};
    anomalySystems.forEach(system => document.querySelectorAll(`[data-badge="${{system}}"]`).forEach(badge => badge.classList.remove('hidden')));
    document.querySelectorAll('.organ-card').forEach(card => card.addEventListener('click', () => {{
      document.querySelectorAll('.organ-card').forEach(item => item.classList.remove('selected')); card.classList.add('selected');
      const viewer = card.querySelector('model-viewer');
      if (viewer) {{ viewer.setAttribute('auto-rotate', ''); viewer.setAttribute('camera-controls', ''); }}
    }}));
    function drawOrgan(canvas, kind) {{
      const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(35, canvas.clientWidth/canvas.clientHeight,.1,100), renderer=new THREE.WebGLRenderer({{canvas,antialias:true,alpha:true}}); renderer.setSize(canvas.clientWidth,canvas.clientHeight,false); camera.position.z=5;
      const group=new THREE.Group(); scene.add(group); const material=new THREE.MeshStandardMaterial({{color: kind==='heart'?0xef4444:kind==='lungs'?0x38bdf8:kind==='brain'?0xf0abfc:kind==='kidneys'?0xa78bfa:0xf59e0b,roughness:.28,metalness:.08}});
      if(kind==='heart') {{ const a=new THREE.Mesh(new THREE.SphereGeometry(.8,28,20),material), b=new THREE.Mesh(new THREE.SphereGeometry(.8,28,20),material); a.position.x=-.45;b.position.x=.45;group.add(a,b); const stem=new THREE.Mesh(new THREE.CylinderGeometry(.16,.22,.8,16),material);stem.position.y=.9;group.add(stem); }}
      else if(kind==='lungs') {{ [-.55,.55].forEach(x=>{{const m=new THREE.Mesh(new THREE.SphereGeometry(.7,24,18),material);m.scale.set(.72,1.2,.55);m.position.x=x;group.add(m)}}); const tube=new THREE.Mesh(new THREE.CylinderGeometry(.12,.12,1,16),material);tube.position.y=1;group.add(tube); }}
      else if(kind==='brain') {{ const m=new THREE.Mesh(new THREE.SphereGeometry(.85,32,24),material);m.scale.set(1.2,.85,1);group.add(m); for(let i=-2;i<=2;i++){{const ring=new THREE.Mesh(new THREE.TorusGeometry(.5,.035,8,32),new THREE.MeshStandardMaterial({{color:0x9333ea}}));ring.rotation.x=Math.PI/2;ring.position.x=i*.18;ring.position.z=.15;group.add(ring)}} }}
      else if(kind==='kidneys') {{ [-.55,.55].forEach(x=>{{const m=new THREE.Mesh(new THREE.SphereGeometry(.55,24,18),material);m.scale.set(.65,1,.45);m.position.set(x,0,0);group.add(m)}}); }}
      else {{ const m=new THREE.Mesh(new THREE.TorusGeometry(.55,.28,20,32,Math.PI*1.7),material);m.rotation.z=-.45;group.add(m); }}
      scene.add(new THREE.AmbientLight(0xffffff,.75));const light=new THREE.DirectionalLight(0xffffff,1.2);light.position.set(2,3,4);scene.add(light); let down=false,last=0; canvas.onpointerdown=e=>{{down=true;last=e.clientX}};canvas.onpointerup=()=>down=false;canvas.onpointermove=e=>{{if(down){{group.rotation.y+=(e.clientX-last)*.018;last=e.clientX}}}}; (function loop(){{requestAnimationFrame(loop);group.rotation.y+=.004;renderer.render(scene,camera)}})();
    }}
    document.querySelectorAll('[data-organ-canvas]').forEach(c => drawOrgan(c, c.dataset.organCanvas));
    // Three.js setup
    let scene, camera, renderer, controls, model;
    let isModelLoaded = false;
    
    // Initialize Three.js
    function initThree() {{
        const container = document.getElementById('twin3d');
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        // Create scene
        scene = new THREE.Scene();
        
        // Add background gradient
        const bgColor = new THREE.Color(0x667eea);
        scene.background = bgColor;
        
        // Add fog for depth
        scene.fog = new THREE.Fog(0x667eea, 2, 10);
        
        // Create camera
        camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.set(0, 1.5, 3);
        camera.lookAt(0, 0.5, 0);
        
        // Create renderer
        renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(width, height);
        renderer.shadowMap.enabled = true;
        container.appendChild(renderer.domElement);
        
        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 5, 5);
        scene.add(directionalLight);
        
        const pointLight = new THREE.PointLight(0xffffff, 0.5);
        pointLight.position.set(-3, 2, -3);
        scene.add(pointLight);
        
        // Add controls
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.25;
        controls.target.set(0, 0.5, 0);
        
        // Load the model
        loadModel();
        
        // Start animation
        animate();
    }}
    
    // Load 3D model
    function loadModel() {{
        const loader = new THREE.GLTFLoader();
        
        loader.load(
            '{model_url}',
            function(gltf) {{
                model = gltf.scene;
                
                // Center the model
                const box = new THREE.Box3().setFromObject(model);
                const center = box.getCenter(new THREE.Vector3());
                model.position.x -= center.x;
                model.position.z -= center.z;
                model.position.y -= box.min.y;
                
                // Scale to fit
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = 2 / maxDim;
                model.scale.set(scale, scale, scale);
                
                // Add to scene
                scene.add(model);
                
                // Hide loading overlay
                document.getElementById('loading-overlay').style.display = 'none';
                isModelLoaded = true;
                
                console.log('3D model loaded successfully!');
            }},
            function(xhr) {{
                // Loading progress
                console.log((xhr.loaded / xhr.total * 100) + '% loaded');
            }},
            function(error) {{
                console.error('Error loading model:', error);
                // Show error message
                document.getElementById('loading-overlay').innerHTML = `
                    <div class="text-center">
                        <i class="fas fa-exclamation-triangle text-3xl text-red-500 mb-2"></i>
                        <p class="text-red-600">Could not load 3D model</p>
                        <button onclick="regenerateModel()" class="mt-3 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                            Try Again
                        </button>
                    </div>
                `;
            }}
        );
    }}
    
    // Animation loop
    function animate() {{
        requestAnimationFrame(animate);
        
        if (controls) {{
            controls.update();
        }}
        
        if (renderer && scene && camera) {{
            renderer.render(scene, camera);
        }}
    }}
    
    // Control functions
    function resetView() {{
        if (controls) {{
            controls.reset();
            camera.position.set(0, 1.5, 3);
            camera.lookAt(0, 0.5, 0);
        }}
    }}
    
    function rotateLeft() {{
        if (model) {{
            model.rotation.y += Math.PI / 4;
        }}
    }}
    
    function zoomIn() {{
        if (camera) {{
            camera.position.z -= 0.5;
        }}
    }}
    
    function regenerateModel() {{
        // Reload the page to regenerate
        location.reload();
    }}
    
    // Initialize on load
    window.addEventListener('load', initThree);
    
    // Handle resize
    window.addEventListener('resize', function() {{
        const container = document.getElementById('twin3d');
        if (container && camera && renderer) {{
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }}
    }});
    </script>
    """
    
    return viewer_html, viewer_js

# ============================================================================
# FASTAPI ROUTES - PUBLIC
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return render_page(home_page(), "Welcome to NeuroGuard")

@app.get("/login", response_class=HTMLResponse)
async def login():
    return render_page(login_page_content(), "Login")

@app.get("/signup", response_class=HTMLResponse)
async def signup():
    """Sign Up page"""
    return render_page(signup_page_content(), "Sign Up")

@app.post("/signup/patient")
async def signup_patient(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    phone: str = Form(...),
    weight: float = Form(...),
    height: float = Form(...),
    role: str = Form(...),
    admin_code: str = Form(None),
    db: Session = Depends(get_db)
):
    """Handle patient sign up with role selection and admin verification"""
    
    # ADMIN VERIFICATION
    ADMIN_VERIFICATION_CODE = "NEUROGUARD2026"
    
    if role == "admin":
        if not admin_code:
            return HTMLResponse("""
            <div class="max-w-md mx-auto mt-16">
                <div class="bg-white p-8 rounded-xl shadow-lg border-2 border-red-500">
                    <h2 class="text-2xl font-bold text-red-600 text-center mb-4">Verification Required</h2>
                    <p class="text-center text-gray-600 mb-4">Please enter the admin verification code.</p>
                    <a href="/signup" class="block text-center text-blue-600 hover:underline">Try Again</a>
                </div>
            </div>
            """, status_code=400)
        
        if admin_code != ADMIN_VERIFICATION_CODE:
            return HTMLResponse("""
            <div class="max-w-md mx-auto mt-16">
                <div class="bg-white p-8 rounded-xl shadow-lg border-2 border-red-500">
                    <h2 class="text-2xl font-bold text-red-600 text-center mb-4">Invalid Verification Code</h2>
                    <p class="text-center text-gray-600 mb-4">The admin verification code you entered is incorrect.</p>
                    <p class="text-center text-sm text-gray-500 mb-4">Please contact your system administrator.</p>
                    <a href="/signup" class="block text-center text-blue-600 hover:underline">Try Again</a>
                </div>
            </div>
            """, status_code=401)
    
    # CHECK IF EMAIL ALREADY EXISTS
    existing = db.query(Patient).filter(Patient.email == email).first()
    if existing:
        return HTMLResponse("""
        <div class="max-w-md mx-auto mt-16">
            <div class="bg-white p-8 rounded-xl shadow-lg border-2 border-red-500">
                <h2 class="text-2xl font-bold text-red-600 text-center mb-4">Sign Up Failed</h2>
                <p class="text-center text-gray-600 mb-4">An account with this email already exists.</p>
                <a href="/signup" class="block text-center text-blue-600 hover:underline">Try Again</a>
                <a href="/login" class="block text-center text-blue-600 hover:underline mt-2">Login Instead</a>
            </div>
        </div>
        """, status_code=400)
    
    # GENERATE USER ID
    user_id = f"P{str(uuid.uuid4())[:8].upper()}"
    
    # CREATE PATIENT
    patient = Patient(
        user_id=user_id,
        email=email,
        password_hash=hash_password(password),
        name=name,
        age=age,
        gender=gender,
        phone=phone,
        blood_type="",
        address={},
        emergency_contact="",
        medical_conditions=[],
        allergies=[],
        medications=[],
        last_visit=datetime.now(),
        is_active=True
    )
    
    db.add(patient)
    db.commit()
    db.refresh(patient)
    
    # CREATE DIGITAL TWIN
    twin = DigitalTwin(
        patient_id=patient.id,
        health_score=0,
        vital_signs={
            "blood_pressure": "120/80",
            "heart_rate": 72,
            "temperature": 98.6,
            "weight": weight,
            "height": height,
            "bmi": weight / ((height/100) ** 2) if height > 0 else 0
        },
        health_metrics={
            "a1c": 0,
            "cholesterol": 0,
            "ldl": 0,
            "hdl": 0,
            "triglycerides": 0
        },
        activity_data={
            "steps": 0,
            "active_minutes": 0,
            "sleep_hours": 0,
            "calories_burned": 0
        },
        risk_assessment={
            "diabetes_risk": "Unknown",
            "heart_disease_risk": "Unknown",
            "stroke_risk": "Unknown"
        },
        predictions={
            "next_year_health": "Complete your health profile to get predictions",
            "recommendations": [
                "Complete your health profile",
                "Upload your first medical report",
                "Add your medical conditions in profile"
            ]
        }
    )
    db.add(twin)
    db.commit()
    
    # Generate 3D model for the patient
    patient_data = {
        "patient_id": patient.user_id,
        "age": age,
        "gender": gender,
        "weight": weight,
        "height": height,
        "bmi": weight / ((height/100) ** 2) if height > 0 else 22
    }
    
    success, result = generate_patient_3d_model(patient_data)
    if success:
        print(f"3D model generated: {result}")
    else:
        print(f"3D model generation failed: {result}")
    
    # CREATE RESPONSE AND REDIRECT
    if role == "admin":
        admin = Admin(
            email=email,
            password_hash=hash_password(password),
            name=name,
            role="Administrator",
            department="Healthcare"
        )
        db.add(admin)
        db.commit()
        
        resp = RedirectResponse(url="/admin/dashboard", status_code=303)
        resp.set_cookie(key="user_type", value="admin")
        resp.set_cookie(key="user_id", value=str(admin.id))
        resp.set_cookie(key="user_name", value=admin.name)
        
    else:
        resp = RedirectResponse(url="/patient/dashboard", status_code=303)
        resp.set_cookie(key="user_type", value="patient")
        resp.set_cookie(key="user_id", value=patient.user_id)
        resp.set_cookie(key="user_name", value=patient.name)
    
    return resp

# ============================================================================
# FASTAPI ROUTES - LOGIN
# ============================================================================

@app.post("/login/patient")
async def login_patient(
    email: str = Form(...), 
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    patient = db.query(Patient).filter(Patient.email == email).first()
    
    if patient and verify_password(password, patient.password_hash):
        resp = RedirectResponse(url="/patient/dashboard", status_code=303)
        resp.set_cookie(key="user_type", value="patient")
        resp.set_cookie(key="user_id", value=patient.user_id)
        resp.set_cookie(key="user_name", value=patient.name)
        return resp
    
    return HTMLResponse("""
    <div class="max-w-md mx-auto mt-16">
        <div class="bg-white p-8 rounded-xl shadow-lg border-2 border-red-500">
            <h2 class="text-2xl font-bold text-red-600 text-center mb-4">Login Failed</h2>
            <p class="text-center text-gray-600 mb-4">Invalid email or password</p>
            <a href="/login" class="block text-center text-blue-600 hover:underline">Try Again</a>
        </div>
    </div>
    """, status_code=401)

@app.post("/login/admin")
async def login_admin(
    email: str = Form(...), 
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    admin = db.query(Admin).filter(Admin.email == email).first()
    
    if admin and verify_password(password, admin.password_hash):
        resp = RedirectResponse(url="/admin/dashboard", status_code=303)
        resp.set_cookie(key="user_type", value="admin")
        resp.set_cookie(key="user_id", value=str(admin.id))
        resp.set_cookie(key="user_name", value=admin.name)
        return resp
    
    return HTMLResponse("""
    <div class="max-w-md mx-auto mt-16">
        <div class="bg-white p-8 rounded-xl shadow-lg border-2 border-red-500">
            <h2 class="text-2xl font-bold text-red-600 text-center mb-4">Login Failed</h2>
            <p class="text-center text-gray-600 mb-4">Invalid email or password</p>
            <a href="/login" class="block text-center text-blue-600 hover:underline">Try Again</a>
        </div>
    </div>
    """, status_code=401)

@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/", status_code=303)
    resp.delete_cookie("user_type")
    resp.delete_cookie("user_id")
    resp.delete_cookie("user_name")
    return resp

# ============================================================================
# PATIENT ROUTES - DASHBOARD
# ============================================================================

@app.get("/patient/dashboard", response_class=HTMLResponse)
async def patient_dashboard(
    request: Request,
    db: Session = Depends(get_db)
):
    user_id = request.cookies.get("user_id")
    user_name = request.cookies.get("user_name")
    
    if not user_id:
        return RedirectResponse(url="/login")
    
    patient = db.query(Patient).filter(Patient.user_id == user_id).first()
    if not patient:
        return HTMLResponse("<p>Patient not found</p>")
    
    twin = db.query(DigitalTwin).filter(DigitalTwin.patient_id == patient.id).first()
    report_count = db.query(Report).filter(Report.patient_id == patient.id).count()
    analysis_count = db.query(AIAnalysis).filter(AIAnalysis.patient_id == patient.id).count()
    
    return render_page(f"""
<div class="fade-in">
    <div class="gradient-bg rounded-2xl p-8 text-white mb-8">
        <div class="flex justify-between items-center">
            <div><h1 class="text-3xl font-bold">Welcome back, {patient.name}!</h1><p class="text-blue-100 mt-1">Your health is our priority.</p></div>
            <div class="bg-white/20 backdrop-blur-sm px-6 py-3 rounded-xl"><i class="fas fa-calendar-alt mr-2"></i>{datetime.now().strftime('%B %d, %Y')}</div>
        </div>
    </div>
    <div class="grid md:grid-cols-4 gap-6 mb-8">
        <div class="bg-white p-6 rounded-2xl shadow-lg card-hover">
            <div class="flex items-center justify-between">
                <div><p class="text-gray-500 text-sm">Health Score</p><p class="text-3xl font-bold text-blue-600">{twin.health_score if twin else 0}%</p></div>
                <div class="w-14 h-14 gradient-bg rounded-2xl flex items-center justify-center"><i class="fas fa-heartbeat text-2xl text-white"></i></div>
            </div>
        </div>
        <div class="bg-white p-6 rounded-2xl shadow-lg card-hover">
            <div class="flex items-center justify-between">
                <div><p class="text-gray-500 text-sm">Reports</p><p class="text-3xl font-bold text-purple-600">{report_count}</p></div>
                <div class="w-14 h-14 bg-purple-100 rounded-2xl flex items-center justify-center"><i class="fas fa-file-medical text-2xl text-purple-600"></i></div>
            </div>
        </div>
        <div class="bg-white p-6 rounded-2xl shadow-lg card-hover">
            <div class="flex items-center justify-between">
                <div><p class="text-gray-500 text-sm">AI Analyses</p><p class="text-3xl font-bold text-green-600">{analysis_count}</p></div>
                <div class="w-14 h-14 bg-green-100 rounded-2xl flex items-center justify-center"><i class="fas fa-brain text-2xl text-green-600"></i></div>
            </div>
        </div>
        <div class="bg-white p-6 rounded-2xl shadow-lg card-hover">
            <div class="flex items-center justify-between">
                <div><p class="text-gray-500 text-sm">Last Visit</p><p class="text-2xl font-bold text-gray-800">{patient.last_visit.strftime('%b %d') if patient.last_visit else 'N/A'}</p></div>
                <div class="w-14 h-14 bg-blue-100 rounded-2xl flex items-center justify-center"><i class="fas fa-clock text-2xl text-blue-600"></i></div>
            </div>
        </div>
    </div>
    <div class="grid md:grid-cols-3 gap-6 mb-8">
        <a href="/patient/analyze" class="block bg-white p-6 rounded-2xl shadow-lg card-hover border-l-4 border-blue-500">
            <div class="flex items-center gap-4">
                <div class="w-12 h-12 gradient-bg rounded-xl flex items-center justify-center"><i class="fas fa-microscope text-white text-xl"></i></div>
                <div><h4 class="font-semibold text-gray-800">Analyze Report</h4><p class="text-sm text-gray-500">Upload and analyze medical reports with AI</p></div>
                <i class="fas fa-arrow-right text-blue-600 ml-auto"></i>
            </div>
        </a>
        <a href="/patient/twin" class="block bg-white p-6 rounded-2xl shadow-lg card-hover border-l-4 border-purple-500">
            <div class="flex items-center gap-4">
                <div class="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center"><i class="fas fa-robot text-purple-600 text-xl"></i></div>
                <div><h4 class="font-semibold text-gray-800">Digital Twin</h4><p class="text-sm text-gray-500">View your virtual health twin</p></div>
                <i class="fas fa-arrow-right text-purple-600 ml-auto"></i>
            </div>
        </a>
        <a href="/patient/reports" class="block bg-white p-6 rounded-2xl shadow-lg card-hover border-l-4 border-green-500">
            <div class="flex items-center gap-4">
                <div class="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center"><i class="fas fa-folder-open text-green-600 text-xl"></i></div>
                <div><h4 class="font-semibold text-gray-800">My Reports</h4><p class="text-sm text-gray-500">View all your medical reports</p></div>
                <i class="fas fa-arrow-right text-green-600 ml-auto"></i>
            </div>
        </a>
    </div>
</div>
""", "Dashboard", "patient", user_name)

# ============================================================================
# PATIENT ANALYZE ROUTE - WITH LOCAL ANALYSIS
# ============================================================================

@app.get("/patient/analyze", response_class=HTMLResponse)
async def patient_analyze(
    request: Request,
    db: Session = Depends(get_db)
):
    user_id = request.cookies.get("user_id")
    user_name = request.cookies.get("user_name")
    
    if not user_id:
        return RedirectResponse(url="/login")
    
    patient = db.query(Patient).filter(Patient.user_id == user_id).first()
    if not patient:
        return HTMLResponse("<p>Patient not found</p>")
    
    # Get analysis history
    analyses = db.query(AIAnalysis).filter(AIAnalysis.patient_id == patient.id).order_by(AIAnalysis.analysis_date.desc()).all()
    
    history_html = ""
    if analyses:
        for analysis in analyses:
            history_html += f"""
            <div class="border-b border-gray-100 py-4 hover:bg-gray-50 px-4 rounded-lg transition">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="font-semibold text-gray-800">{analysis.primary_diagnosis or 'N/A'}</p>
                        <p class="text-sm text-gray-500">
                            <i class="fas fa-calendar mr-1"></i>{analysis.analysis_date.strftime('%Y-%m-%d') if analysis.analysis_date else 'N/A'}
                            <span class="mx-2">•</span>
                            <i class="fas fa-file mr-1"></i>{analysis.report_title or 'Medical Report'}
                        </p>
                        <p class="text-xs text-gray-400 mt-1"><i class="fas fa-shield-heart mr-1"></i>Local rules engine</p>
                    </div>
                    <span class="px-3 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                        <i class="fas fa-robot mr-1"></i>AI
                    </span>
                </div>
            </div>
            """
    else:
        history_html = """
        <div class="text-center py-8 text-gray-500">
            <i class="fas fa-microscope text-4xl mb-3 block"></i>
            <p>No analysis history. Upload your first report!</p>
        </div>
        """
    
    return render_page(f"""
<div class="fade-in">
    <div class="flex justify-between items-center mb-6">
        <div>
            <h1 class="text-3xl font-bold text-gray-800">
                <i class="fas fa-microscope text-blue-600 mr-2"></i>AI Report Analysis
            </h1>
            <p class="text-gray-600">Extract supported lab values, flag reference-range results, and explore your health view.</p>
            <p class="text-xs text-green-600 mt-1"><i class="fas fa-shield-heart mr-1"></i>Your report stays on this server — no paid API or AI provider.</p>
        </div>
    </div>

    <div class="bg-white p-8 rounded-2xl shadow-lg mb-8">
        <h3 class="font-bold text-gray-800 mb-4">
            <i class="fas fa-upload text-blue-600 mr-2"></i>Upload Medical Report
        </h3>
        <form id="upload-form" enctype="multipart/form-data">
            <div class="upload-zone" id="drop-zone">
                <i class="fas fa-cloud-upload-alt text-5xl text-blue-400 mb-4 block"></i>
                <p class="text-gray-600 font-medium">Drag & drop your medical report here</p>
                <p class="text-sm text-gray-400">or click to browse (PDF, JPG, PNG, WEBP, BMP, TIFF, TXT, CSV — max 10 MB)</p>
                <input type="file" id="file-input" name="file" accept=".pdf,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.txt,.csv" class="hidden">
                <button type="button" onclick="document.getElementById('file-input').click()" class="mt-4 px-6 py-2 gradient-bg text-white rounded-lg hover:opacity-90 transition">
                    <i class="fas fa-folder-open mr-2"></i>Browse Files
                </button>
            </div>
            <div id="file-info" class="hidden mt-4 p-4 bg-blue-50 rounded-lg">
                <p class="text-sm text-gray-700"><i class="fas fa-file mr-2"></i><span id="file-name"></span></p>
                <button type="submit" class="mt-3 px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition">
                    <i class="fas fa-microscope mr-2"></i>Analyze locally
                </button>
            </div>
        </form>
        <div id="analysis-result" class="hidden mt-6"></div>
        <div id="loading" class="hidden mt-6 text-center py-8">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p class="text-gray-600 mt-4">Extracting values and applying reference-range rules...</p>
            <p class="text-sm text-gray-400 mt-2">This may take a few moments</p>
        </div>
    </div>

    <div class="bg-white p-6 rounded-2xl shadow-lg">
        <h3 class="font-bold text-gray-800 mb-4">
            <i class="fas fa-history text-purple-600 mr-2"></i>Analysis History
        </h3>
        {history_html}
    </div>
</div>

<script>
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {{
    e.preventDefault();
    dropZone.classList.add('dragover');
}});
dropZone.addEventListener('dragleave', () => {{
    dropZone.classList.remove('dragover');
}});
dropZone.addEventListener('drop', (e) => {{
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {{
        fileInput.files = e.dataTransfer.files;
        handleFileSelect(e.dataTransfer.files[0]);
    }}
}});

fileInput.addEventListener('change', (e) => {{
    if (e.target.files.length) {{
        handleFileSelect(e.target.files[0]);
    }}
}});

function handleFileSelect(file) {{
    document.getElementById('file-info').classList.remove('hidden');
    document.getElementById('file-name').textContent = file.name;
}}

document.getElementById('upload-form').addEventListener('submit', async function(e) {{
    e.preventDefault();
    const fileInput = document.getElementById('file-input');
    if (!fileInput.files.length) {{
        alert('Please select a file first');
        return;
    }}
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('analysis-result').classList.add('hidden');
    
    try {{
        const response = await fetch('/api/patient/analyze', {{
            method: 'POST',
            body: formData
        }});
        
        const result = await response.json();
        document.getElementById('loading').classList.add('hidden');
        
        if (result.status === 'success') {{
            displayAnalysisResult(result);
        }} else {{
            alert('Analysis failed: ' + result.message);
        }}
    }} catch (error) {{
        document.getElementById('loading').classList.add('hidden');
        alert('Error: ' + error.message);
    }}
}});

function esc(value) {{ const n = document.createElement('span'); n.textContent = value ?? ''; return n.innerHTML; }}

function displayAnalysisResult(result) {{
    const data = result.data || {{}};
    const labs = data.lab_results || [];
    const abnormal = labs.filter(l => l.flag !== 'normal');
    const rows = labs.map(l => {{
        const colour = l.flag === 'high' ? 'red' : l.flag === 'low' ? 'amber' : 'green';
        return `<tr class="border-b"><td class="py-3 font-medium">${{esc(l.name)}}</td><td>${{esc(l.value)}} ${{esc(l.unit)}}</td><td class="text-gray-500">${{esc(l.refLow)}}–${{esc(l.refHigh)}}</td><td><span class="px-2 py-1 rounded-full text-xs bg-${{colour}}-100 text-${{colour}}-700">${{esc(l.flag)}}</span></td></tr>`;
    }}).join('') || '<tr><td colspan="4" class="py-6 text-center text-gray-500">No supported lab values were found. Try a text-based PDF with a results table.</td></tr>';
    
    document.getElementById('analysis-result').innerHTML = `
      <div class="bg-white border border-blue-100 rounded-2xl p-6 shadow-sm fade-in">
        <div class="flex flex-wrap justify-between gap-3 mb-5"><div><h3 class="text-xl font-bold text-gray-800">Report review complete</h3><p class="text-sm text-gray-600">${{esc(data.analysis?.report_summary)}}</p></div><span class="self-start px-3 py-1 rounded-full bg-blue-50 text-blue-700 text-sm">Local rules engine</span></div>
        <div class="grid lg:grid-cols-2 gap-6"><div><h4 class="font-semibold text-gray-800 mb-3">Extracted results</h4><div class="overflow-x-auto"><table class="w-full text-sm"><thead class="text-left text-gray-500"><tr><th>Test</th><th>Value</th><th>Range</th><th>Flag</th></tr></thead><tbody>${{rows}}</tbody></table></div><p class="mt-4 text-xs text-gray-500"><i class="fas fa-info-circle mr-1"></i>Flags compare the extracted value with the shown/default range; they are not a diagnosis.</p></div>
        <div><h4 class="font-semibold text-gray-800 mb-3">2D reference-range view</h4><div class="h-56"><canvas id="lab-chart"></canvas></div><h4 class="font-semibold text-gray-800 mt-6 mb-2">3D body systems</h4><model-viewer id="body-model-viewer" class="h-64 w-full rounded-xl bg-gradient-to-b from-cyan-50 to-slate-100" src="/assets/anatomy/body.glb" alt="3D body model" camera-controls auto-rotate rotation-per-second="16deg" shadow-intensity="1"></model-viewer><p class="mt-2 text-xs text-gray-500">Open anatomy model. Red organ badges in the Digital Twin link to flagged systems.</p></div></div>
        <div class="mt-5 bg-amber-50 border border-amber-200 p-4 rounded-xl text-sm text-amber-900"><i class="fas fa-triangle-exclamation mr-2"></i><strong>For information only.</strong> Discuss results, symptoms, and next steps with a qualified clinician.</div>
      </div>`;
    const container = document.getElementById('analysis-result'); container.classList.remove('hidden');
    if (labs.length && window.Chart) new Chart(document.getElementById('lab-chart'), {{ type: 'bar', data: {{ labels: labs.map(l => l.name), datasets: [{{label:'Value',data:labs.map(l=>l.value),backgroundColor:labs.map(l=>l.flag === 'normal' ? '#22c55e' : '#ef4444')}}] }}, options: {{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{maxRotation:55,minRotation:25}}}}}} }} }});
    if (window.THREE && document.getElementById('body-model')) {{
        const el = document.getElementById('body-model'), scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(45, el.clientWidth/el.clientHeight, .1, 100); const renderer = new THREE.WebGLRenderer({{antialias:true, alpha:true}}); renderer.setSize(el.clientWidth, el.clientHeight); el.appendChild(renderer.domElement); camera.position.set(0,0,8);
        const redSystems = new Set(abnormal.map(l => l.system)); const parts = [['head','thyroid',0,2.2],['chest','heart',0,.7],['abdomen','metabolic',0,-.7],['left kidney','kidneys',-1,-.9],['right kidney','kidneys',1,-.9],['blood','blood',0,-1.8]]; parts.forEach(([label,system,x,y]) => {{const m = new THREE.Mesh(new THREE.SphereGeometry(system==='heart'?.55:.43,24,18),new THREE.MeshStandardMaterial({{color:redSystems.has(system)?0xef4444:0x38bdf8,roughness:.35}}));m.position.set(x,y,0);scene.add(m);}}); scene.add(new THREE.AmbientLight(0xffffff,.85)); const light=new THREE.DirectionalLight(0xffffff,1);light.position.set(3,5,5);scene.add(light); renderer.render(scene,camera);
    }}
}}
</script>
""", "AI Report Analysis", "patient", user_name)

# ============================================================================
# PRIVATE LOCAL ANALYSIS API ENDPOINT
# ============================================================================

@app.post("/api/patient/analyze")
async def api_analyze_report_locally(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Analyze a text PDF locally: no external API and no synthetic findings."""
    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    patient = db.query(Patient).filter(Patient.user_id == user_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    try:
        content = await file.read()
        result = analyze_locally(file.filename, content)
        labs = result["labs"]
        abnormal = result["abnormal"]
        score = result["health_score"]
        report = Report(patient_id=patient.id, title=file.filename or "Medical report", date=datetime.now(),
                        doctor="Local parser", hospital="NeuroGuard", diagnosis=result["primary_finding"],
                        summary=result["summary"], file_name=file.filename, file_size=len(content),
                        extracted_labs=labs, is_analyzed=True)
        db.add(report)
        db.flush()
        analysis = AIAnalysis(patient_id=patient.id, report_id=report.id, report_title=file.filename,
                              primary_diagnosis=result["primary_finding"], confidence=1.0,
                              secondary_diagnoses=[f"{item['name']}: {item['flag']}" for item in abnormal],
                              recommendations=result["recommendations"], risk_factors=[],
                              report_summary=result["summary"])
        db.add(analysis)
        twin = db.query(DigitalTwin).filter(DigitalTwin.patient_id == patient.id).first()
        if twin:
            metrics = twin.health_metrics or {}
            for lab in labs:
                metrics[lab["name"].lower().replace(" ", "_")] = lab["value"]
            twin.health_metrics = metrics
            if score is not None:
                twin.health_score = score
            twin.last_updated = datetime.now()
        db.commit()
        return {"status": "success", "message": "Report reviewed locally", "source": "local-rules",
                "data": {"analysis": {"findings": {"primary_diagnosis": result["primary_finding"], "confidence": 1.0,
                "secondary_diagnoses": analysis.secondary_diagnoses, "risk_factors": []}, "recommendations": result["recommendations"],
                "report_summary": result["summary"], "health_score": score}, "lab_results": labs}}
    except MedicalAnalysisError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception:
        db.rollback()
        return {"status": "error", "message": "The report could not be analyzed. Please try another supported PDF."}

# ============================================================================
# PATIENT DIGITAL TWIN ROUTE - WITH ANNY 3D MODEL
# ============================================================================

@app.get("/patient/twin", response_class=HTMLResponse)
async def patient_twin(
    request: Request,
    db: Session = Depends(get_db)
):
    user_id = request.cookies.get("user_id")
    user_name = request.cookies.get("user_name")
    
    if not user_id:
        return RedirectResponse(url="/login")
    
    patient = db.query(Patient).filter(Patient.user_id == user_id).first()
    if not patient:
        return HTMLResponse("<p>Patient not found</p>")
    
    twin = db.query(DigitalTwin).filter(DigitalTwin.patient_id == patient.id).first()
    if not twin:
        twin = DigitalTwin(
            patient_id=patient.id,
            health_score=0,
            vital_signs={},
            health_metrics={},
            activity_data={},
            risk_assessment={},
            predictions={}
        )
        db.add(twin)
        db.commit()
        db.refresh(twin)
    
    vitals = twin.vital_signs or {}
    metrics = twin.health_metrics or {}
    recent_reports = db.query(Report).filter(Report.patient_id == patient.id).order_by(Report.uploaded_at.desc()).limit(10).all()
    anomaly_systems = sorted({lab.get("system") for report in recent_reports for lab in (report.extracted_labs or []) if lab.get("flag") in {"high", "low"} and lab.get("system")})
    activity = twin.activity_data or {}
    risks = twin.risk_assessment or {}
    predictions = twin.predictions or {}
    
    # Get patient data for 3D model
    patient_data = {
        "patient_id": patient.user_id,
        "age": patient.age or 30,
        "gender": patient.gender or "Male",
        "weight": vitals.get("weight", 70),
        "height": vitals.get("height", 170),
        "bmi": vitals.get("bmi", 22),
        "anomaly_systems": anomaly_systems
    }
    
    # Generate or find existing 3D model
    model_url = None
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith(f"patient_{patient.user_id}_") and f.endswith(".glb")]
    
    if ANNY_AVAILABLE and model_files:
        # Use latest existing model
        model_files.sort()
        model_url = f"/models/{model_files[-1]}"
    elif ANNY_AVAILABLE:
        # Generate new model only when Anny is enabled.
        success, result = generate_patient_3d_model(patient_data)
        if success:
            model_url = f"/models/{os.path.basename(result)}"
        else:
            print(f"3D model generation failed: {result}")
    else:
        model_url = "/assets/anatomy/body.glb"
    
    twin_viewer_html = ""
    twin_viewer_js = ""
    
    if model_url:
        twin_viewer_html, twin_viewer_js = build_twin_viewer_html(model_url, patient_data)
    else:
        # Anny is intentionally disabled; use the local open anatomy model.
        model_url = "/assets/anatomy/body.glb"
        twin_viewer_html, twin_viewer_js = build_twin_viewer_html(model_url, patient_data)

    return render_page(f"""
<div class="fade-in">
    <div class="flex justify-between items-center mb-6">
        <div>
            <h1 class="text-3xl font-bold text-gray-800">
                <i class="fas fa-robot text-purple-600 mr-2"></i>Your Digital Twin
            </h1>
            <p class="text-gray-600">AI-powered 3D representation of your body and health</p>
        </div>
        <div class="bg-purple-100 px-4 py-2 rounded-lg">
            <i class="fas fa-sync-alt mr-2 text-purple-600"></i>
            <span class="text-sm text-purple-600">Updated: {twin.last_updated.strftime('%Y-%m-%d') if twin.last_updated else 'N/A'}</span>
        </div>
    </div>

    <div class="bg-white p-8 rounded-2xl shadow-lg mb-8 text-center">
        <h3 class="text-sm text-gray-500 font-medium">Overall Health Score</h3>
        <div class="health-score-circle gradient-bg text-white mx-auto mt-4">
            {twin.health_score or 0}%
        </div>
        <p class="mt-2 text-sm text-gray-600">Based on your health metrics and lifestyle data</p>
    </div>

    {twin_viewer_html}

    <div class="grid md:grid-cols-4 gap-4 mb-8">
        <div class="bg-white p-4 rounded-xl shadow-lg">
            <p class="text-sm text-gray-500">Blood Pressure</p>
            <p class="text-xl font-bold text-gray-800">{vitals.get('blood_pressure', 'N/A')}</p>
        </div>
        <div class="bg-white p-4 rounded-xl shadow-lg">
            <p class="text-sm text-gray-500">Heart Rate</p>
            <p class="text-xl font-bold text-gray-800">{vitals.get('heart_rate', 'N/A')} bpm</p>
        </div>
        <div class="bg-white p-4 rounded-xl shadow-lg">
            <p class="text-sm text-gray-500">BMI</p>
            <p class="text-xl font-bold text-gray-800">{vitals.get('bmi', 'N/A')}</p>
        </div>
        <div class="bg-white p-4 rounded-xl shadow-lg">
            <p class="text-sm text-gray-500">Temperature</p>
            <p class="text-xl font-bold text-gray-800">{vitals.get('temperature', 'N/A')}°F</p>
        </div>
    </div>

    <div class="grid md:grid-cols-2 gap-6 mb-8">
        <div class="bg-white p-6 rounded-2xl shadow-lg">
            <h4 class="font-semibold text-gray-800 mb-4">
                <i class="fas fa-flask text-blue-600 mr-2"></i>Lab Metrics
            </h4>
            <div class="space-y-3">
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">A1C</span>
                    <span class="font-bold">{metrics.get('a1c', 'N/A')}%</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">Cholesterol</span>
                    <span class="font-bold">{metrics.get('cholesterol', 'N/A')} mg/dL</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">LDL</span>
                    <span class="font-bold">{metrics.get('ldl', 'N/A')} mg/dL</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">HDL</span>
                    <span class="font-bold">{metrics.get('hdl', 'N/A')} mg/dL</span>
                </div>
            </div>
        </div>

        <div class="bg-white p-6 rounded-2xl shadow-lg">
            <h4 class="font-semibold text-gray-800 mb-4">
                <i class="fas fa-heartbeat text-red-500 mr-2"></i>Activity & Lifestyle
            </h4>
            <div class="space-y-3">
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">Daily Steps</span>
                    <span class="font-bold">{activity.get('steps', 'N/A')}</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">Active Minutes</span>
                    <span class="font-bold">{activity.get('active_minutes', 'N/A')} min</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">Sleep</span>
                    <span class="font-bold">{activity.get('sleep_hours', 'N/A')} hours</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">Calories Burned</span>
                    <span class="font-bold">{activity.get('calories_burned', 'N/A')}</span>
                </div>
            </div>
        </div>
    </div>

    <div class="grid md:grid-cols-2 gap-6">
        <div class="bg-white p-6 rounded-2xl shadow-lg">
            <h4 class="font-semibold text-gray-800 mb-4">
                <i class="fas fa-exclamation-triangle text-yellow-500 mr-2"></i>Risk Assessment
            </h4>
            <div class="space-y-3">
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">Diabetes Risk</span>
                    <span class="px-3 py-1 bg-yellow-100 text-yellow-800 text-sm rounded-full">{risks.get('diabetes_risk', 'N/A')}</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">Heart Disease Risk</span>
                    <span class="px-3 py-1 bg-green-100 text-green-800 text-sm rounded-full">{risks.get('heart_disease_risk', 'N/A')}</span>
                </div>
                <div class="flex justify-between items-center">
                    <span class="text-gray-600">Stroke Risk</span>
                    <span class="px-3 py-1 bg-green-100 text-green-800 text-sm rounded-full">{risks.get('stroke_risk', 'N/A')}</span>
                </div>
            </div>
        </div>

        <div class="bg-white p-6 rounded-2xl shadow-lg">
            <h4 class="font-semibold text-gray-800 mb-4">
                <i class="fas fa-chart-line text-blue-600 mr-2"></i>Health Predictions
            </h4>
            <p class="text-sm text-gray-700 mb-3">
                <i class="fas fa-arrow-right text-blue-500 mr-1"></i>
                {predictions.get('next_year_health', 'N/A')}
            </p>
            <div class="mt-3">
                <p class="text-sm text-gray-500 font-medium">Recommendations:</p>
                <ul class="list-disc list-inside text-sm text-gray-700 mt-1">
                    {''.join([f'<li>{r}</li>' for r in predictions.get('recommendations', [])])}
                </ul>
            </div>
        </div>
    </div>
</div>
""", "Digital Twin", "patient", user_name, scripts=twin_viewer_js)

# ============================================================================
# SERVE GENERATED 3D MODELS
# ============================================================================

app.mount("/models", StaticFiles(directory=MODELS_DIR), name="models")

# ============================================================================
# API ENDPOINT TO REGENERATE 3D MODEL
# ============================================================================

@app.post("/api/patient/regenerate-twin")
async def regenerate_twin(
    request: Request,
    db: Session = Depends(get_db)
):
    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    patient = db.query(Patient).filter(Patient.user_id == user_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    twin = db.query(DigitalTwin).filter(DigitalTwin.patient_id == patient.id).first()
    
    # Get latest patient data
    vitals = twin.vital_signs if twin else {}
    
    patient_data = {
        "patient_id": patient.user_id,
        "age": patient.age or 30,
        "gender": patient.gender or "Male",
        "weight": vitals.get("weight", 70),
        "height": vitals.get("height", 170),
        "bmi": vitals.get("bmi", 22)
    }
    
    # Delete old model files
    for f in os.listdir(MODELS_DIR):
        if f.startswith(f"patient_{patient.user_id}_") and f.endswith(".glb"):
            try:
                os.remove(os.path.join(MODELS_DIR, f))
            except:
                pass
    
    # Generate new model
    success, result = generate_patient_3d_model(patient_data)
    
    if success:
        return {"status": "success", "model_url": f"/models/{os.path.basename(result)}"}
    else:
        return {"status": "error", "message": result}

# ============================================================================
# PATIENT REPORTS ROUTE
# ============================================================================

@app.get("/patient/reports", response_class=HTMLResponse)
async def patient_reports(
    request: Request,
    db: Session = Depends(get_db)
):
    user_id = request.cookies.get("user_id")
    user_name = request.cookies.get("user_name")
    
    if not user_id:
        return RedirectResponse(url="/login")
    
    patient = db.query(Patient).filter(Patient.user_id == user_id).first()
    if not patient:
        return HTMLResponse("<p>Patient not found</p>")
    
    reports = db.query(Report).filter(Report.patient_id == patient.id).order_by(Report.uploaded_at.desc()).all()
    
    if not reports:
        reports_html = """
        <div class="bg-white p-12 rounded-2xl shadow-lg text-center">
            <i class="fas fa-folder-open text-6xl text-gray-300 mb-4 block"></i>
            <p class="text-gray-500 text-lg">No reports uploaded yet</p>
            <p class="text-gray-400 text-sm">Upload your first medical report to get started</p>
            <a href="/patient/analyze" class="mt-4 inline-block px-6 py-2 gradient-bg text-white rounded-lg hover:opacity-90 transition">
                <i class="fas fa-upload mr-2"></i>Upload Report
            </a>
        </div>
        """
    else:
        reports_html = ""
        for report in reports:
            reports_html += f"""
            <div class="bg-white p-6 rounded-2xl shadow-lg card-hover">
                <div class="flex justify-between items-start">
                    <div class="flex-1">
                        <div class="flex items-center gap-3 mb-2">
                            <i class="fas fa-file-pdf text-red-500 text-2xl"></i>
                            <h4 class="font-semibold text-gray-800">{report.title or 'Untitled'}</h4>
                        </div>
                        <p class="text-sm text-gray-500">
                            <i class="fas fa-stethoscope mr-1"></i>{report.doctor or 'Unknown'}
                            <span class="mx-2">•</span>
                            <i class="fas fa-hospital mr-1"></i>{report.hospital or 'N/A'}
                            <span class="mx-2">•</span>
                            <i class="fas fa-calendar mr-1"></i>{report.date.strftime('%Y-%m-%d') if report.date else 'N/A'}
                        </p>
                        <p class="text-sm text-gray-600 mt-2">{report.summary[:150] if report.summary else 'No summary available'}...</p>
                    </div>
                    <div class="flex flex-col gap-2">
                        <span class="px-3 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                            <i class="fas fa-check-circle mr-1"></i>{'Analyzed' if report.is_analyzed else 'Pending'}
                        </span>
                        <a href="/patient/reports/{report.id}" class="px-3 py-1 text-center bg-cyan-600 text-white text-xs rounded-full hover:bg-cyan-700">
                            <i class="fas fa-eye mr-1"></i>View report
                        </a>
                    </div>
                </div>
            </div>
            """
    
    return render_page(f"""
<div class="fade-in">
    <div class="flex justify-between items-center mb-6">
        <div>
            <h1 class="text-3xl font-bold text-gray-800">
                <i class="fas fa-file-medical text-green-600 mr-2"></i>My Medical Reports
            </h1>
            <p class="text-gray-600">All your medical records in one place</p>
        </div>
        <a href="/patient/analyze" class="px-6 py-2 gradient-bg text-white rounded-lg hover:opacity-90 transition">
            <i class="fas fa-plus mr-2"></i>Upload New
        </a>
    </div>
    <div class="grid gap-4">
        {reports_html}
    </div>
</div>
""", "My Reports", "patient", user_name)


@app.get("/patient/reports/{report_id}", response_class=HTMLResponse)
async def patient_report_detail(report_id: str, request: Request, db: Session = Depends(get_db)):
    """Show a previous report only when it belongs to the signed-in patient."""
    user_id = request.cookies.get("user_id")
    user_name = request.cookies.get("user_name")
    if not user_id:
        return RedirectResponse(url="/login")
    patient = db.query(Patient).filter(Patient.user_id == user_id).first()
    report = db.query(Report).filter(Report.id == report_id, Report.patient_id == patient.id if patient else "").first()
    if not report:
        return HTMLResponse("<p>Report not found.</p>", status_code=404)
    analysis = db.query(AIAnalysis).filter(AIAnalysis.report_id == report.id).first()
    labs = report.extracted_labs or []
    rows = ""
    for lab in labs:
        flag = str(lab.get("flag", "normal"))
        colour = "green" if flag == "normal" else "red" if flag == "high" else "yellow"
        rows += f"<tr class='border-b'><td class='py-3 font-medium'>{lab.get('name', 'Unknown')}</td><td>{lab.get('value', '—')} {lab.get('unit', '')}</td><td>{lab.get('refLow', '—')}–{lab.get('refHigh', '—')}</td><td><span class='px-2 py-1 rounded-full text-xs bg-{colour}-100 text-{colour}-800'>{flag}</span></td></tr>"
    table = f"<div class='overflow-x-auto'><table class='w-full text-sm'><thead class='text-left text-gray-500'><tr><th class='pb-2'>Test</th><th class='pb-2'>Value</th><th class='pb-2'>Range</th><th class='pb-2'>Flag</th></tr></thead><tbody>{rows}</tbody></table></div>" if rows else "<p class='text-sm text-gray-500'>Detailed extracted lab values were not stored for this older report. New reports include this data.</p>"
    recommendations = "".join(f"<li>{item}</li>" for item in (analysis.recommendations if analysis else [])) or "<li>Review this report with your clinician.</li>"
    return render_page(f"""
<div class="fade-in max-w-5xl mx-auto">
  <a href="/patient/reports" class="text-cyan-700 hover:underline"><i class="fas fa-arrow-left mr-2"></i>Back to reports</a>
  <div class="mt-5 rounded-3xl bg-gradient-to-br from-cyan-50 to-white border border-cyan-100 p-7 shadow-sm">
    <p class="text-xs font-bold uppercase tracking-widest text-cyan-700">Saved report review</p>
    <h1 class="mt-2 text-3xl font-bold text-gray-900">{report.title}</h1>
    <p class="mt-2 text-sm text-gray-600"><i class="fas fa-calendar mr-1"></i>{report.date.strftime('%d %b %Y') if report.date else 'Date unavailable'} <span class="mx-2">•</span> {report.doctor or 'Local parser'}</p>
    <p class="mt-5 text-gray-700">{report.summary or 'No summary is available for this report.'}</p>
  </div>
  <div class="mt-6 grid lg:grid-cols-3 gap-6"><section class="lg:col-span-2 bg-white rounded-2xl shadow-lg p-6"><h2 class="font-bold text-xl text-gray-800"><i class="fas fa-flask text-cyan-600 mr-2"></i>Extracted lab results</h2><div class="mt-4">{table}</div></section><aside class="bg-white rounded-2xl shadow-lg p-6"><h2 class="font-bold text-xl text-gray-800">Recommended next step</h2><ul class="mt-4 list-disc list-inside space-y-2 text-sm text-gray-700">{recommendations}</ul><a class="mt-6 block rounded-xl bg-cyan-600 py-2 text-center text-white hover:bg-cyan-700" href="/patient/copilot"><i class="fas fa-comments mr-2"></i>Ask Health Copilot</a></aside></div>
  <div class="mt-6 rounded-xl bg-amber-50 border border-amber-200 p-4 text-sm text-amber-900"><strong>For information only.</strong> This stored report review is not medical advice or a diagnosis.</div>
</div>
""", "Report Detail", "patient", user_name)


def _patient_copilot_context(patient: Patient, db: Session) -> dict:
    """Build the only context the copilot may access: this patient's local data."""
    reports = db.query(Report).filter(Report.patient_id == patient.id).order_by(Report.uploaded_at.desc()).limit(5).all()
    twin = db.query(DigitalTwin).filter(DigitalTwin.patient_id == patient.id).first()
    return {
        "patient_name": patient.name,
        "conditions": patient.medical_conditions or [],
        "medications": patient.medications or [],
        "allergies": patient.allergies or [],
        "health_score": twin.health_score if twin else None,
        "metrics": twin.health_metrics if twin else {},
        "reports": [{"title": r.title, "date": r.date.strftime("%d %b %Y") if r.date else "Unknown date", "summary": r.summary or "", "labs": r.extracted_labs or []} for r in reports],
    }


def _local_copilot_reply(message: str, context: dict) -> str:
    """Useful offline fallback when a local Ollama model is not installed."""
    text = message.lower()
    reports = context["reports"]
    labs = [lab for report in reports for lab in report["labs"]]
    flagged = [lab for lab in labs if lab.get("flag") != "normal"]
    if any(word in text for word in ("report", "result", "lab", "abnormal", "high", "low")):
        if not reports:
            return "I do not see any saved reports yet. Upload a text-based PDF on Analyze Report, then I can help you review the stored results."
        if not labs:
            return f"Your most recent saved report is “{reports[0]['title']}” from {reports[0]['date']}. It was saved before detailed lab values were stored, so I can show its summary on My Reports but cannot compare individual values."
        if not flagged:
            return f"I found {len(labs)} extracted result(s) across your saved reports, and none are currently marked outside their listed reference range. This is not a medical diagnosis—please review the complete report with your clinician."
        details = "; ".join(f"{item.get('name')}: {item.get('value')} {item.get('unit', '')} ({item.get('flag')})" for item in flagged[:4])
        return f"I found {len(flagged)} saved range flag(s): {details}. These flags compare values with a reference range only; they do not establish a diagnosis. Would you like me to explain one of these tests in plain language?"
    if any(word in text for word in ("medicine", "medication", "drug")):
        meds = context["medications"]
        return "Your profile lists: " + (", ".join(meds) if meds else "no current medications") + ". I can help you keep this list organized, but I cannot tell you to start, stop, or change medication—please ask your prescriber."
    if any(word in text for word in ("allergy", "allergies")):
        allergies = context["allergies"]
        return "Your profile lists: " + (", ".join(allergies) if allergies else "no allergies") + ". Confirm allergies with your clinical team and seek urgent help for severe allergic symptoms."
    score = context["health_score"]
    return f"I can help you review your saved reports, explain extracted lab terms, and summarize your profile. Your current local health-review score is {score if score is not None else 'not available'}. What would you like to explore?"


@app.get("/patient/copilot", response_class=HTMLResponse)
async def patient_copilot(request: Request, db: Session = Depends(get_db)):
    user_id, user_name = request.cookies.get("user_id"), request.cookies.get("user_name")
    if not user_id:
        return RedirectResponse(url="/login")
    patient = db.query(Patient).filter(Patient.user_id == user_id).first()
    if not patient:
        return HTMLResponse("<p>Patient not found.</p>", status_code=404)
    report_count = db.query(Report).filter(Report.patient_id == patient.id).count()
    return render_page(f"""
<div class="fade-in mx-auto max-w-5xl">
  <div class="rounded-3xl bg-gradient-to-br from-cyan-600 to-blue-700 p-7 text-white shadow-lg"><p class="text-xs font-bold uppercase tracking-widest text-cyan-100">Private health assistant</p><h1 class="mt-2 text-3xl font-bold"><i class="fas fa-sparkles mr-2"></i>NeuroGuard Health Copilot</h1><p class="mt-2 max-w-2xl text-cyan-50">Ask about your own saved report summaries, extracted lab results, medications, allergies, and digital-twin data.</p></div>
  <div class="mt-6 grid gap-6 lg:grid-cols-3"><aside class="rounded-2xl bg-white p-6 shadow-lg"><h2 class="font-bold text-gray-800">Your copilot can see</h2><ul class="mt-4 space-y-3 text-sm text-gray-600"><li><i class="fas fa-file-medical text-cyan-600 mr-2"></i>{report_count} saved report(s)</li><li><i class="fas fa-flask text-cyan-600 mr-2"></i>Extracted lab values and range flags</li><li><i class="fas fa-user text-cyan-600 mr-2"></i>Your profile, allergies, and medications</li></ul><div class="mt-6 rounded-xl bg-amber-50 p-3 text-xs text-amber-900"><strong>Not medical advice.</strong> The copilot cannot diagnose conditions or prescribe treatment. Seek urgent care for emergencies.</div></aside>
  <section class="lg:col-span-2 overflow-hidden rounded-2xl bg-white shadow-lg"><div id="chat-messages" class="h-[420px] space-y-4 overflow-y-auto bg-slate-50 p-5"><div class="max-w-[85%] rounded-2xl rounded-tl-sm bg-white p-4 text-sm text-slate-700 shadow-sm"><strong class="text-cyan-700">Health Copilot</strong><br>Hello {patient.name}. I can help review your saved health information. For example, ask: "What did my last report show?"</div></div><form id="copilot-form" class="border-t p-4"><div class="flex gap-3"><input id="copilot-input" maxlength="800" required class="flex-1 rounded-xl border px-4 py-3 focus:border-cyan-500 focus:outline-none" placeholder="Ask about your saved reports…"><button class="rounded-xl bg-cyan-600 px-5 py-3 font-semibold text-white hover:bg-cyan-700"><i class="fas fa-paper-plane"></i><span class="ml-2 hidden sm:inline">Send</span></button></div></form></section></div>
</div>
<script>
const chat=document.getElementById('chat-messages'); const form=document.getElementById('copilot-form'); const input=document.getElementById('copilot-input');
function addMessage(text, mine=false) {{ const div=document.createElement('div'); div.className=`max-w-[85%] rounded-2xl p-4 text-sm shadow-sm ${{mine?'ml-auto rounded-tr-sm bg-cyan-600 text-white':'rounded-tl-sm bg-white text-slate-700'}}`; div.textContent=text; chat.appendChild(div); chat.scrollTop=chat.scrollHeight; }}
form.addEventListener('submit', async e => {{ e.preventDefault(); const message=input.value.trim(); if(!message) return; addMessage(message,true); input.value=''; input.disabled=true; try {{ const response=await fetch('/api/patient/copilot',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{message}})}}); const data=await response.json(); addMessage(data.reply || data.detail || 'I could not answer that right now.'); }} catch {{ addMessage('Connection problem. Please try again.'); }} finally {{input.disabled=false;input.focus();}} }});
</script>
""", "Health Copilot", "patient", user_name)


@app.post("/api/patient/copilot")
async def patient_copilot_message(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    patient = db.query(Patient).filter(Patient.user_id == user_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    payload = await request.json()
    message = str(payload.get("message", "")).strip()
    if not message or len(message) > 800:
        raise HTTPException(status_code=400, detail="Send a message of up to 800 characters.")
    context = _patient_copilot_context(patient, db)
    # Optional fully local AI: set OLLAMA_ENABLED=true after installing Ollama.
    if os.getenv("OLLAMA_ENABLED", "false").lower() == "true":
        try:
            prompt = ("You are a careful medical-record explainer. Use only this patient context. "
                      "Never diagnose, prescribe, or claim certainty. Encourage clinician review.\nContext: "
                      f"{json.dumps(context)}\nPatient question: {message}")
            async with httpx.AsyncClient(timeout=25) as client:
                response = await client.post(os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat"), json={"model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"), "stream": False, "messages": [{"role": "user", "content": prompt}]})
            reply = response.json().get("message", {}).get("content", "").strip()
            if reply:
                return {"reply": reply[:3000], "source": "local-ollama"}
        except (httpx.HTTPError, ValueError, KeyError):
            pass
    return {"reply": _local_copilot_reply(message, context), "source": "local-rules"}


# ============================================================================
# PATIENT PROFILE ROUTE
# ============================================================================

@app.get("/patient/profile", response_class=HTMLResponse)
async def patient_profile(
    request: Request,
    db: Session = Depends(get_db)
):
    user_id = request.cookies.get("user_id")
    user_name = request.cookies.get("user_name")
    
    if not user_id:
        return RedirectResponse(url="/login")
    
    p = db.query(Patient).filter(Patient.user_id == user_id).first()
    if not p:
        return HTMLResponse("<p>Patient not found</p>")
    
    # Profile completion calculation
    completion_fields = {
        "name": 10 if p.name and p.name.strip() else 0,
        "email": 10 if p.email and p.email.strip() else 0,
        "phone": 10 if p.phone and p.phone.strip() else 0,
        "age": 10 if p.age and p.age > 0 else 0,
        "gender": 10 if p.gender and p.gender.strip() else 0,
        "blood_type": 10 if p.blood_type and p.blood_type.strip() else 0,
        "address": 10 if p.address and isinstance(p.address, dict) and p.address.get("street") else 0,
        "emergency_contact": 10 if p.emergency_contact and p.emergency_contact.strip() else 0,
        "medical_conditions": 10 if p.medical_conditions and len(p.medical_conditions) > 0 else 0,
        "allergies": 10 if p.allergies and len(p.allergies) > 0 else 0,
        "medications": 10 if p.medications and len(p.medications) > 0 else 0,
    }
    
    completion_percentage = sum(completion_fields.values())
    
    if completion_percentage >= 80:
        completion_level = "Excellent"
        completion_color = "text-green-600"
        completion_bg = "bg-green-100"
        completion_icon = "fas fa-check-circle"
    elif completion_percentage >= 60:
        completion_level = "Good"
        completion_color = "text-blue-600"
        completion_bg = "bg-blue-100"
        completion_icon = "fas fa-check-circle"
    elif completion_percentage >= 40:
        completion_level = "In Progress"
        completion_color = "text-yellow-600"
        completion_bg = "bg-yellow-100"
        completion_icon = "fas fa-spinner"
    else:
        completion_level = "Needs Attention"
        completion_color = "text-red-600"
        completion_bg = "bg-red-100"
        completion_icon = "fas fa-exclamation-circle"
    
    missing_fields = []
    if not p.name or not p.name.strip():
        missing_fields.append("Full Name")
    if not p.phone or not p.phone.strip():
        missing_fields.append("Phone Number")
    if not p.age or p.age <= 0:
        missing_fields.append("Age")
    if not p.gender or not p.gender.strip():
        missing_fields.append("Gender")
    if not p.blood_type or not p.blood_type.strip():
        missing_fields.append("Blood Type")
    if not p.emergency_contact or not p.emergency_contact.strip():
        missing_fields.append("Emergency Contact")
    if not p.address or not isinstance(p.address, dict) or not p.address.get("street"):
        missing_fields.append("Address")
    if not p.medical_conditions or len(p.medical_conditions) == 0:
        missing_fields.append("Medical Conditions")
    if not p.allergies or len(p.allergies) == 0:
        missing_fields.append("Allergies")
    if not p.medications or len(p.medications) == 0:
        missing_fields.append("Medications")
    
    medical_conditions_str = ", ".join(p.medical_conditions) if p.medical_conditions else ""
    allergies_str = ", ".join(p.allergies) if p.allergies else ""
    medications_str = ", ".join(p.medications) if p.medications else ""
    
    return render_page(f"""
<div class="fade-in">
    <h1 class="text-3xl font-bold text-gray-800 mb-6">
        <i class="fas fa-user-circle text-blue-600 mr-2"></i>My Profile
    </h1>
    
    <div class="grid md:grid-cols-3 gap-6">
        <div class="md:col-span-1">
            <div class="bg-white p-6 rounded-2xl shadow-lg text-center">
                <div class="w-32 h-32 gradient-bg rounded-full flex items-center justify-center mx-auto text-white text-5xl font-bold">
                    {p.name[0] if p.name else '?'}
                </div>
                <h3 class="text-xl font-bold mt-4">{p.name or 'N/A'}</h3>
                <p class="text-gray-500">Patient ID: {p.user_id or 'N/A'}</p>
                <p class="text-sm text-gray-500">Member since: {p.created_at.strftime('%Y-%m-%d') if p.created_at else 'N/A'}</p>
                
                <div class="mt-6 pt-4 border-t border-gray-200">
                    <div class="flex items-center justify-between mb-2">
                        <span class="text-sm font-medium text-gray-700">Profile Completion</span>
                        <span class="text-sm font-bold {completion_color}">{completion_percentage}%</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-3">
                        <div class="gradient-bg rounded-full h-3 transition-all duration-500" style="width: {completion_percentage}%"></div>
                    </div>
                    <div class="flex items-center justify-center gap-2 mt-3">
                        <i class="{completion_icon} {completion_color}"></i>
                        <span class="text-sm font-medium {completion_color}">{completion_level}</span>
                    </div>
                    
                    {f'''
                    <div class="mt-4 p-3 bg-yellow-50 rounded-lg text-left">
                        <p class="text-xs font-semibold text-yellow-800 mb-2">
                            <i class="fas fa-lightbulb mr-1"></i>Complete your profile:
                        </p>
                        <ul class="text-xs text-yellow-700 space-y-1">
                            {''.join([f'<li>• {field}</li>' for field in missing_fields[:5]])}
                            {f'<li class="text-gray-500">• And {len(missing_fields) - 5} more...</li>' if len(missing_fields) > 5 else ''}
                            {'' if missing_fields else '<li class="text-green-600">✅ All fields complete!</li>'}
                        </ul>
                    </div>
                    ''' if missing_fields else '''
                    <div class="mt-4 p-3 bg-green-50 rounded-lg">
                        <p class="text-xs font-semibold text-green-800">
                            <i class="fas fa-check-circle mr-1"></i>Perfect! Your profile is complete.
                        </p>
                    </div>
                    '''}
                    
                    <div class="grid grid-cols-2 gap-2 mt-4">
                        <div class="bg-blue-50 p-2 rounded-lg">
                            <p class="text-xs text-gray-500">Completed</p>
                            <p class="text-sm font-bold text-blue-600">{len(completion_fields) - len(missing_fields)}/11</p>
                        </div>
                        <div class="bg-gray-50 p-2 rounded-lg">
                            <p class="text-xs text-gray-500">Missing</p>
                            <p class="text-sm font-bold text-red-600">{len(missing_fields)}</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="md:col-span-2">
            <div class="bg-white p-6 rounded-2xl shadow-lg">
                <div class="flex justify-between items-center mb-4">
                    <h4 class="font-semibold text-gray-800">
                        <i class="fas fa-edit text-blue-600 mr-2"></i>Edit Profile
                    </h4>
                    <span class="text-xs text-gray-500">All fields are optional</span>
                </div>
                
                <form action="/patient/update-profile" method="post" class="space-y-4">
                    <div class="grid md:grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">
                                <i class="fas fa-user mr-1"></i>Full Name
                            </label>
                            <input type="text" name="name" value="{p.name or ''}" 
                                   class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">
                                <i class="fas fa-phone mr-1"></i>Phone Number
                            </label>
                            <input type="text" name="phone" value="{p.phone or ''}" 
                                   class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">
                                <i class="fas fa-calendar mr-1"></i>Age
                            </label>
                            <input type="number" name="age" value="{p.age if p.age else ''}" 
                                   class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">
                                <i class="fas fa-venus-mars mr-1"></i>Gender
                            </label>
                            <select name="gender" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                                <option value="">Select Gender</option>
                                <option value="Male" {'selected' if p.gender == 'Male' else ''}>Male</option>
                                <option value="Female" {'selected' if p.gender == 'Female' else ''}>Female</option>
                                <option value="Other" {'selected' if p.gender == 'Other' else ''}>Other</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">
                                <i class="fas fa-tint mr-1"></i>Blood Type
                            </label>
                            <select name="blood_type" class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                                <option value="">Select Blood Type</option>
                                <option value="A+" {'selected' if p.blood_type == 'A+' else ''}>A+</option>
                                <option value="A-" {'selected' if p.blood_type == 'A-' else ''}>A-</option>
                                <option value="B+" {'selected' if p.blood_type == 'B+' else ''}>B+</option>
                                <option value="B-" {'selected' if p.blood_type == 'B-' else ''}>B-</option>
                                <option value="AB+" {'selected' if p.blood_type == 'AB+' else ''}>AB+</option>
                                <option value="AB-" {'selected' if p.blood_type == 'AB-' else ''}>AB-</option>
                                <option value="O+" {'selected' if p.blood_type == 'O+' else ''}>O+</option>
                                <option value="O-" {'selected' if p.blood_type == 'O-' else ''}>O-</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">
                                <i class="fas fa-address-book mr-1"></i>Emergency Contact
                            </label>
                            <input type="text" name="emergency_contact" value="{p.emergency_contact or ''}" 
                                   placeholder="Name - +1 (555) 123-4567"
                                   class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">
                                <i class="fas fa-map-marker-alt mr-1"></i>Address (Street, City, State)
                            </label>
                            <input type="text" name="address" value="{p.address.get('street') if p.address and isinstance(p.address, dict) else ''}" 
                                   placeholder="123 Main St, Boston, MA"
                                   class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                        </div>
                    </div>
                    
                    <hr class="my-4">
                    
                    <h4 class="font-semibold text-gray-800">
                        <i class="fas fa-notes-medical text-green-600 mr-2"></i>Medical Information
                    </h4>
                    <div class="space-y-3">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">
                                <i class="fas fa-stethoscope mr-1"></i>Medical Conditions (comma separated)
                            </label>
                            <input type="text" name="medical_conditions" value="{medical_conditions_str}" 
                                   placeholder="Diabetes, Hypertension, Asthma"
                                   class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                            <p class="text-xs text-gray-400 mt-1">Separate multiple conditions with commas</p>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">
                                <i class="fas fa-allergies mr-1"></i>Allergies (comma separated)
                            </label>
                            <input type="text" name="allergies" value="{allergies_str}" 
                                   placeholder="Penicillin, Peanuts, Dust"
                                   class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                            <p class="text-xs text-gray-400 mt-1">Separate multiple allergies with commas</p>
                        </div>
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1">
                                <i class="fas fa-pills mr-1"></i>Current Medications (comma separated)
                            </label>
                            <input type="text" name="medications" value="{medications_str}" 
                                   placeholder="Metformin 500mg, Lisinopril 10mg"
                                   class="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500">
                            <p class="text-xs text-gray-400 mt-1">Separate multiple medications with commas</p>
                        </div>
                    </div>
                    
                    <button type="submit" class="w-full py-3 gradient-bg text-white rounded-lg hover:opacity-90 transition">
                        <i class="fas fa-save mr-2"></i>Update Profile
                    </button>
                </form>
            </div>
        </div>
    </div>
</div>
""", "My Profile", "patient", user_name)

# ============================================================================
# UPDATE PROFILE ROUTE
# ============================================================================

@app.post("/patient/update-profile")
async def update_profile(
    request: Request,
    name: str = Form(None),
    phone: str = Form(None),
    age: int = Form(None),
    gender: str = Form(None),
    blood_type: str = Form(None),
    emergency_contact: str = Form(None),
    address: str = Form(None),
    medical_conditions: str = Form(None),
    allergies: str = Form(None),
    medications: str = Form(None),
    db: Session = Depends(get_db)
):
    user_id = request.cookies.get("user_id")
    
    if not user_id:
        return RedirectResponse(url="/login")
    
    patient = db.query(Patient).filter(Patient.user_id == user_id).first()
    if not patient:
        return HTMLResponse("<p>Patient not found</p>", status_code=404)
    
    try:
        if name is not None:
            patient.name = name
        if phone is not None:
            patient.phone = phone
        if age is not None:
            patient.age = age
        if gender is not None:
            patient.gender = gender
        if blood_type is not None:
            patient.blood_type = blood_type
        if emergency_contact is not None:
            patient.emergency_contact = emergency_contact
        if address is not None:
            patient.address = {"street": address}
        if medical_conditions is not None:
            patient.medical_conditions = [c.strip() for c in medical_conditions.split(",") if c.strip()]
        if allergies is not None:
            patient.allergies = [a.strip() for a in allergies.split(",") if a.strip()]
        if medications is not None:
            patient.medications = [m.strip() for m in medications.split(",") if m.strip()]
        
        patient.updated_at = datetime.now()
        db.commit()
        
        # Regenerate 3D model after profile update
        twin = db.query(DigitalTwin).filter(DigitalTwin.patient_id == patient.id).first()
        if twin:
            vitals = twin.vital_signs or {}
            patient_data = {
                "patient_id": patient.user_id,
                "age": patient.age or 30,
                "gender": patient.gender or "Male",
                "weight": vitals.get("weight", 70),
                "height": vitals.get("height", 170),
                "bmi": vitals.get("bmi", 22)
            }
            success, result = generate_patient_3d_model(patient_data)
            if success:
                print(f"3D model regenerated: {result}")
        
        response = RedirectResponse(url="/patient/profile?success=1", status_code=303)
        return response
        
    except Exception as e:
        db.rollback()
        return HTMLResponse(f"""
        <div class="max-w-md mx-auto mt-16">
            <div class="bg-white p-8 rounded-xl shadow-lg border-2 border-red-500">
                <h2 class="text-2xl font-bold text-red-600 text-center mb-4">Update Failed</h2>
                <p class="text-center text-gray-600 mb-4">Error: {str(e)}</p>
                <a href="/patient/profile" class="block text-center text-blue-600 hover:underline">Try Again</a>
            </div>
        </div>
        """, status_code=400)

# ============================================================================
# ADMIN ROUTES
# ============================================================================

@app.get("/admin/dashboard", response_class=HTMLResponse)
async def admin_dashboard(
    request: Request,
    db: Session = Depends(get_db)
):
    user_id = request.cookies.get("user_id")
    user_name = request.cookies.get("user_name")
    
    if not user_id:
        return RedirectResponse(url="/login")
    
    admin_user = db.query(Admin).filter(Admin.id == user_id).first()
    if not admin_user:
        return RedirectResponse(url="/login")
    
    total_patients = db.query(Patient).count()
    total_reports = db.query(Report).count()
    total_ai_analyses = db.query(AIAnalysis).count()
    total_fraud_alerts = db.query(Claim).filter(Claim.is_fraudulent == True).count()
    active_patients = db.query(Patient).filter(Patient.is_active == True).count()
    
    recent_patients = db.query(Patient).order_by(Patient.created_at.desc()).limit(5).all()
    
    recent_rows = ""
    for p in recent_patients:
        recent_rows += f"""
        <tr class="border-b hover:bg-gray-50">
            <td class="px-4 py-2 text-sm">{p.user_id}</td>
            <td class="px-4 py-2 text-sm font-medium">{p.name}</td>
            <td class="px-4 py-2 text-sm">{p.email}</td>
            <td class="px-4 py-2 text-sm">{p.age if p.age else "N/A"}</td>
            <td class="px-4 py-2 text-sm">
                <span class="px-2 py-1 {'bg-green-100 text-green-800' if p.is_active else 'bg-gray-100 text-gray-800'} text-xs rounded-full">
                    {'Active' if p.is_active else 'Inactive'}
                </span>
            </td>
        </tr>
        """
    
    return render_page(f"""
<div class="fade-in">
    <div class="gradient-bg rounded-2xl p-8 text-white mb-8">
        <h1 class="text-3xl font-bold">Admin Dashboard</h1>
        <p class="text-blue-100 mt-1">Welcome back, {admin_user.name}</p>
    </div>
    <div class="grid md:grid-cols-4 gap-6 mb-8">
        <div class="bg-white p-6 rounded-2xl shadow-lg"><p class="text-gray-500 text-sm">Total Patients</p><p class="text-3xl font-bold text-gray-800">{total_patients}</p></div>
        <div class="bg-white p-6 rounded-2xl shadow-lg"><p class="text-gray-500 text-sm">Total Reports</p><p class="text-3xl font-bold text-gray-800">{total_reports}</p></div>
        <div class="bg-white p-6 rounded-2xl shadow-lg"><p class="text-gray-500 text-sm">AI Analyses</p><p class="text-3xl font-bold text-gray-800">{total_ai_analyses}</p></div>
        <div class="bg-white p-6 rounded-2xl shadow-lg"><p class="text-gray-500 text-sm">Fraud Alerts</p><p class="text-3xl font-bold text-red-600">{total_fraud_alerts}</p></div>
    </div>
    <div class="bg-white p-6 rounded-2xl shadow-lg">
        <h3 class="font-bold text-gray-800 mb-4">Quick Actions</h3>
        <div class="grid md:grid-cols-3 gap-4">
            <a href="/admin/patients" class="block p-4 bg-blue-50 rounded-xl hover:bg-blue-100 transition text-center"><i class="fas fa-users text-2xl text-blue-600 mb-2 block"></i><span class="text-sm font-medium text-gray-700">Manage Patients</span></a>
            <a href="/admin/claims" class="block p-4 bg-green-50 rounded-xl hover:bg-green-100 transition text-center"><i class="fas fa-file-invoice text-2xl text-green-600 mb-2 block"></i><span class="text-sm font-medium text-gray-700">View Claims</span></a>
            <a href="/admin/fraud" class="block p-4 bg-red-50 rounded-xl hover:bg-red-100 transition text-center"><i class="fas fa-shield-alt text-2xl text-red-600 mb-2 block"></i><span class="text-sm font-medium text-gray-700">Fraud Detection</span></a>
        </div>
    </div>
</div>
""", "Admin Dashboard", "admin", user_name)

@app.get("/admin/patients", response_class=HTMLResponse)
async def admin_patients(
    request: Request,
    db: Session = Depends(get_db)
):
    user_id = request.cookies.get("user_id")
    user_name = request.cookies.get("user_name")
    
    if not user_id:
        return RedirectResponse(url="/login")
    
    patients = db.query(Patient).all()
    
    rows = ""
    for p in patients:
        rows += f'<tr class="border-b hover:bg-gray-50"><td class="px-4 py-2 text-sm">{p.user_id}</td><td class="px-4 py-2 text-sm font-medium">{p.name}</td><td class="px-4 py-2 text-sm">{p.email}</td><td class="px-4 py-2 text-sm">{p.age if p.age else "N/A"}</td><td class="px-4 py-2 text-sm"><span class="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">{"Active" if p.is_active else "Inactive"}</span></td></tr>'
    
    return render_page(f"""
<div class="fade-in">
    <h1 class="text-3xl font-bold text-gray-800 mb-6"><i class="fas fa-users text-gray-900 mr-2"></i>Patient Management</h1>
    <div class="bg-white p-6 rounded-2xl shadow-lg">
        <div class="overflow-x-auto">
            <table class="w-full">
                <thead class="bg-gray-50">
                    <tr><th class="px-4 py-2 text-left text-xs font-medium text-gray-500">ID</th><th class="px-4 py-2 text-left text-xs font-medium text-gray-500">Name</th><th class="px-4 py-2 text-left text-xs font-medium text-gray-500">Email</th><th class="px-4 py-2 text-left text-xs font-medium text-gray-500">Age</th><th class="px-4 py-2 text-left text-xs font-medium text-gray-500">Status</th></tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    </div>
</div>
""", "Patient Management", "admin", user_name)

@app.get("/admin/claims")
async def admin_claims(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login")
    user_name = request.cookies.get("user_name")
    return render_page("""
<div class="fade-in"><h1 class="text-3xl font-bold text-gray-800 mb-6"><i class="fas fa-file-invoice text-gray-900 mr-2"></i>Claims Management</h1><div class="bg-white p-6 rounded-2xl shadow-lg"><p class="text-gray-600">Claims management features coming soon.</p></div></div>
""", "Claims Management", "admin", user_name)

@app.get("/admin/fraud")
async def admin_fraud(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login")
    user_name = request.cookies.get("user_name")
    return render_page("""
<div class="fade-in"><h1 class="text-3xl font-bold text-gray-800 mb-6"><i class="fas fa-shield-alt text-gray-900 mr-2"></i>Fraud Detection</h1><div class="bg-white p-6 rounded-2xl shadow-lg"><p class="text-gray-600">Fraud detection features coming soon.</p></div></div>
""", "Fraud Detection", "admin", user_name)

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"NeuroGuard running at http://localhost:{port}")
    print("Anny 3D model is temporarily disabled.")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

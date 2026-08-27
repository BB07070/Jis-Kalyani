# api_routes.py - API Routes with Key Management

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from api_key_manager import api_manager, APIService
from security_middleware import rate_limit
import httpx

router = APIRouter(prefix="/api", tags=["API"])

# ============================================================================
# API KEY MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/keys/status")
async def get_api_status():
    """Get status of all API integrations"""
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "apis": api_manager.get_health_status(),
        "usage": api_manager.get_usage_stats()
    }

@router.get("/keys/health")
async def api_health_check():
    """Health check for API integrations"""
    return {
        "status": "healthy",
        "services": {
            "fhir": api_manager.is_key_valid(APIService.FHIR),
            "health_gorilla": api_manager.is_key_valid(APIService.HEALTH_GORILLA),
            "groq": api_manager.is_key_valid(APIService.GROQ)
        }
    }

# ============================================================================
# API ENDPOINTS WITH KEY MANAGEMENT
# ============================================================================

@router.post("/fhir/patient/{patient_id}")
@rate_limit(limit=20, window=60)  # 20 requests per minute
async def fetch_fhir_patient(request: Request, patient_id: str):
    """Fetch patient data from FHIR API"""
    
    # Check if FHIR API is configured
    if not api_manager.is_key_valid(APIService.FHIR):
        return JSONResponse(
            status_code=503,
            content={"detail": "FHIR API not configured"}
        )
    
    try:
        client_id = api_manager.get_api_key(APIService.FHIR)
        base_url = api_manager.get_base_url(APIService.FHIR)
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{base_url}/Patient/{patient_id}",
                headers={"Authorization": f"Bearer {client_id}"}
            )
            response.raise_for_status()
            return response.json()
            
    except httpx.HTTPError as e:
        api_manager.log_error(APIService.FHIR, str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/groq/analyze")
@rate_limit(limit=10, window=60)  # 10 requests per minute
async def groq_analyze(request: Request, text: str):
    """Analyze text using Groq AI"""
    
    if not api_manager.is_key_valid(APIService.GROQ):
        return JSONResponse(
            status_code=503,
            content={"detail": "Groq API not configured"}
        )
    
    try:
        api_key = api_manager.get_api_key(APIService.GROQ)
        base_url = api_manager.get_base_url(APIService.GROQ)
        model = api_manager.api_keys.get("groq", {}).get("model", "llama3-70b-8192")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a medical assistant."},
                        {"role": "user", "content": text}
                    ],
                    "temperature": 0.1
                }
            )
            response.raise_for_status()
            return response.json()
            
    except httpx.HTTPError as e:
        api_manager.log_error(APIService.GROQ, str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/health-gorilla/order")
@rate_limit(limit=30, window=60)
async def create_lab_order(request: Request, order_data: dict):
    """Create a lab order using Health Gorilla"""
    
    if not api_manager.is_key_valid(APIService.HEALTH_GORILLA):
        return JSONResponse(
            status_code=503,
            content={"detail": "Health Gorilla API not configured"}
        )
    
    try:
        api_key = api_manager.get_api_key(APIService.HEALTH_GORILLA)
        base_url = api_manager.get_base_url(APIService.HEALTH_GORILLA)
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/orders",
                json=order_data,
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            return response.json()
            
    except httpx.HTTPError as e:
        api_manager.log_error(APIService.HEALTH_GORILLA, str(e))
        raise HTTPException(status_code=500, detail=str(e))
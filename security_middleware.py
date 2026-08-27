# security_middleware.py - Security Middleware

import os

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import hashlib
import hmac
from config import config

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for request validation and rate limiting
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.secret_key = config.SECRET_KEY
        self.rate_limits = {}
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Rate limiting
        if not self._check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."}
            )
        
        # Validate API key for API endpoints
        if request.url.path.startswith("/api/"):
            if not self._validate_api_key(request):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"}
                )
        
        # Add security headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check rate limit for client IP"""
        current_time = time.time()
        
        if client_ip not in self.rate_limits:
            self.rate_limits[client_ip] = {
                "requests": [],
                "limit": 100,  # 100 requests per minute
                "window": 60   # 60 second window
            }
        
        # Clean old requests
        limit_data = self.rate_limits[client_ip]
        limit_data["requests"] = [
            req_time for req_time in limit_data["requests"]
            if current_time - req_time < limit_data["window"]
        ]
        
        # Check limit
        if len(limit_data["requests"]) >= limit_data["limit"]:
            return False
        
        # Add current request
        limit_data["requests"].append(current_time)
        return True
    
    def _validate_api_key(self, request: Request) -> bool:
        """Validate API key from request header"""
        # Skip validation for public endpoints
        public_endpoints = ["/api/public", "/api/health"]
        if any(request.url.path.startswith(endpoint) for endpoint in public_endpoints):
            return True

        # Browser-facing patient routes authenticate via the sign-in session.
        # A second API key would otherwise have to be exposed in browser code.
        if request.url.path.startswith("/api/patient/") and request.cookies.get("user_id"):
            return True
        
        # Check for API key in header
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return False
        
        # In production, validate against stored keys
        # For now, check against environment
        valid_api_keys = [
            config.SECRET_KEY,
            os.getenv("API_KEY", "")
        ]
        
        return api_key in valid_api_keys

# ============================================================================
# RATE LIMITER DECORATOR
# ============================================================================

from functools import wraps

def rate_limit(limit: int = 60, window: int = 60):
    """Rate limit decorator"""
    def decorator(func):
        requests = {}
        
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host if request.client else "unknown"
            current_time = time.time()
            
            if client_ip not in requests:
                requests[client_ip] = []
            
            # Clean old requests
            requests[client_ip] = [
                req_time for req_time in requests[client_ip]
                if current_time - req_time < window
            ]
            
            if len(requests[client_ip]) >= limit:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Limit: {limit} requests per {window} seconds."
                )
            
            requests[client_ip].append(current_time)
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator

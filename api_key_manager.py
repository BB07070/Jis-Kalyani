# api_key_manager.py - API Key Management

import os
import json
from typing import Optional, Dict
from datetime import datetime
from enum import Enum

class APIService(Enum):
    FHIR = "fhir"
    HEALTH_GORILLA = "health_gorilla"
    GROQ = "groq"

class APIKeyManager:
    """
    Manages API keys with rotation, validation, and usage tracking
    """
    
    def __init__(self):
        self.api_keys = {}
        self.key_usage = {}
        self.load_keys()
    
    def load_keys(self):
        """Load API keys from environment"""
        self.api_keys = {
            "fhir": {
                "client_id": os.getenv("FHIR_CLIENT_ID"),
                "client_secret": os.getenv("FHIR_CLIENT_SECRET"),
                "base_url": os.getenv("FHIR_API_URL", "https://api.1uphealthdemo.com/r4")
            },
            "health_gorilla": {
                "api_key": os.getenv("HEALTH_GORILLA_API_KEY"),
                "base_url": os.getenv("HEALTH_GORILLA_URL", "https://api.healthgorilla.com/v1")
            },
            "groq": {
                "api_key": os.getenv("GROQ_API_KEY"),
                "base_url": os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1"),
                "model": os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
            }
        }
        
        # Initialize usage tracking
        for service in self.api_keys:
            self.key_usage[service] = {
                "last_used": None,
                "request_count": 0,
                "errors": 0,
                "last_error": None
            }
    
    def get_api_key(self, service: APIService) -> Optional[str]:
        """Get API key for a service"""
        service_name = service.value
        key_data = self.api_keys.get(service_name, {})
        
        # Track usage
        self.key_usage[service_name]["last_used"] = datetime.now().isoformat()
        self.key_usage[service_name]["request_count"] += 1
        
        # Return the appropriate key
        if service_name == "fhir":
            return key_data.get("client_id")
        else:
            return key_data.get("api_key")
    
    def get_api_secret(self, service: APIService) -> Optional[str]:
        """Get API secret for a service"""
        if service == APIService.FHIR:
            return self.api_keys.get("fhir", {}).get("client_secret")
        return None
    
    def get_base_url(self, service: APIService) -> str:
        """Get base URL for a service"""
        service_name = service.value
        return self.api_keys.get(service_name, {}).get("base_url", "")
    
    def is_key_valid(self, service: APIService) -> bool:
        """Check if API key is configured"""
        service_name = service.value
        key_data = self.api_keys.get(service_name, {})
        
        if service_name == "fhir":
            return bool(key_data.get("client_id") and key_data.get("client_secret"))
        else:
            return bool(key_data.get("api_key"))
    
    def log_error(self, service: APIService, error: str):
        """Log API error"""
        service_name = service.value
        self.key_usage[service_name]["errors"] += 1
        self.key_usage[service_name]["last_error"] = f"{datetime.now().isoformat()}: {error}"
    
    def get_usage_stats(self) -> dict:
        """Get API usage statistics"""
        return self.key_usage
    
    def get_health_status(self) -> dict:
        """Get health status of all APIs"""
        return {
            "fhir": {
                "configured": self.is_key_valid(APIService.FHIR),
                "last_used": self.key_usage["fhir"]["last_used"],
                "errors": self.key_usage["fhir"]["errors"]
            },
            "health_gorilla": {
                "configured": self.is_key_valid(APIService.HEALTH_GORILLA),
                "last_used": self.key_usage["health_gorilla"]["last_used"],
                "errors": self.key_usage["health_gorilla"]["errors"]
            },
            "groq": {
                "configured": self.is_key_valid(APIService.GROQ),
                "last_used": self.key_usage["groq"]["last_used"],
                "errors": self.key_usage["groq"]["errors"]
            }
        }

# Create singleton instance
api_manager = APIKeyManager()

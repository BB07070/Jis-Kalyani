# config.py - Configuration Management

import os
from dotenv import load_dotenv
from typing import Optional
import secrets
import hashlib

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # ============================================================================
    # DATABASE
    # ============================================================================
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./neuroguard.db")
    
    # ============================================================================
    # SECURITY KEYS
    # ============================================================================
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret-key")
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", "32-byte-encryption-key-here-12345")
    
    # ============================================================================
    # API KEYS
    # ============================================================================
    # 1upHealth FHIR
    FHIR_API_URL: str = os.getenv("FHIR_API_URL", "https://api.1uphealthdemo.com/r4")
    FHIR_CLIENT_ID: Optional[str] = os.getenv("FHIR_CLIENT_ID")
    FHIR_CLIENT_SECRET: Optional[str] = os.getenv("FHIR_CLIENT_SECRET")
    
    # Health Gorilla
    HEALTH_GORILLA_URL: str = os.getenv("HEALTH_GORILLA_URL", "https://api.healthgorilla.com/v1")
    HEALTH_GORILLA_API_KEY: Optional[str] = os.getenv("HEALTH_GORILLA_API_KEY")
    
    # Groq AI
    GROQ_API_URL: str = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    GROQ_VISION_MODEL: str = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
    
    # ============================================================================
    # APP SETTINGS
    # ============================================================================
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")
    
    # ============================================================================
    # VALIDATION
    # ============================================================================
    @classmethod
    def validate_api_keys(cls) -> dict:
        """Check which API keys are configured"""
        return {
            "fhir": bool(cls.FHIR_CLIENT_ID and cls.FHIR_CLIENT_SECRET),
            "health_gorilla": bool(cls.HEALTH_GORILLA_API_KEY),
            "groq": bool(cls.GROQ_API_KEY),
        }
    
    @classmethod
    def get_encryption_key_bytes(cls) -> bytes:
        """Get encryption key as bytes"""
        key = cls.ENCRYPTION_KEY
        # Ensure key is 32 bytes (AES-256)
        if len(key) < 32:
            key = hashlib.sha256(key.encode()).hexdigest()[:32]
        return key[:32].encode()
    
    @classmethod
    def generate_keys(cls):
        """Generate new keys for development"""
        print("🔑 Generating new keys...")
        print(f"SECRET_KEY: {secrets.token_urlsafe(32)}")
        print(f"JWT_SECRET_KEY: {secrets.token_urlsafe(32)}")
        print(f"ENCRYPTION_KEY: {secrets.token_urlsafe(32)}")

# Singleton config instance
config = Config()

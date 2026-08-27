# encryption.py - Encryption Service

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import base64
import hashlib
from config import config

class EncryptionService:
    """
    AES-256 encryption service for sensitive data
    """
    
    def __init__(self):
        self.key = config.get_encryption_key_bytes()
    
    def encrypt(self, data: str) -> str:
        """Encrypt data using AES-256-CBC"""
        if not data:
            return ""
        
        try:
            # Generate random IV
            iv = get_random_bytes(16)
            
            # Create cipher
            cipher = AES.new(self.key, AES.MODE_CBC, iv)
            
            # Encrypt with PKCS7 padding
            padded_data = pad(data.encode('utf-8'), AES.block_size)
            ciphertext = cipher.encrypt(padded_data)
            
            # Combine IV and ciphertext
            combined = iv + ciphertext
            
            # Return base64 encoded
            return base64.b64encode(combined).decode('utf-8')
        except Exception as e:
            print(f"Encryption error: {e}")
            return data  # Fallback to plain text
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""
        if not encrypted_data:
            return ""
        
        try:
            # Decode from base64
            combined = base64.b64decode(encrypted_data)
            
            # Extract IV and ciphertext
            iv = combined[:16]
            ciphertext = combined[16:]
            
            # Create cipher
            cipher = AES.new(self.key, AES.MODE_CBC, iv)
            
            # Decrypt and unpad
            padded_data = cipher.decrypt(ciphertext)
            data = unpad(padded_data, AES.block_size)
            
            return data.decode('utf-8')
        except Exception as e:
            print(f"Decryption error: {e}")
            return encrypted_data  # Return as is if decryption fails
    
    def encrypt_field(self, data: dict, field: str) -> dict:
        """Encrypt a specific field in a dictionary"""
        if field in data and data[field]:
            data[field] = self.encrypt(str(data[field]))
        return data
    
    def decrypt_field(self, data: dict, field: str) -> dict:
        """Decrypt a specific field in a dictionary"""
        if field in data and data[field]:
            data[field] = self.decrypt(data[field])
        return data


# Sensitive fields that should be encrypted
SENSITIVE_FIELDS = [
    "ssn", "medicare_id", "medicaid_id", 
    "insurance_id", "password_hash", "token"
]

# Create singleton instance
encryption = EncryptionService()
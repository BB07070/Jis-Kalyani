# database.py - Local SQLite Database for NeuroGuard

import os
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, JSON, Text, ForeignKey, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import uuid
import hashlib
import secrets

# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), "neuroguard.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# ============================================================================
# PASSWORD HASHING
# ============================================================================

def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt"""
    salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256((salt + password).encode())
    return f"{salt}:{hash_obj.hexdigest()}"

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    try:
        salt, hash_val = hashed.split(":")
        new_hash = hashlib.sha256((salt + password).encode()).hexdigest()
        return new_hash == hash_val
    except:
        return False

# ============================================================================
# DATABASE MODELS
# ============================================================================

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    
    # Personal Information (Basic fields from signup)
    name = Column(String(255), nullable=False)
    age = Column(Integer)
    gender = Column(String(50))
    phone = Column(String(50))
    
    # Extended Profile Fields (Added in Profile page)
    blood_type = Column(String(10))
    address = Column(JSON)  # Stores {street, city, state, zip}
    emergency_contact = Column(String(255))
    
    # Medical Information (Added in Profile page)
    medical_conditions = Column(JSON, default=list)
    allergies = Column(JSON, default=list)
    medications = Column(JSON, default=list)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_visit = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    reports = relationship("Report", back_populates="patient", cascade="all, delete-orphan")
    twin_data = relationship("DigitalTwin", back_populates="patient", uselist=False, cascade="all, delete-orphan")
    ai_analyses = relationship("AIAnalysis", back_populates="patient", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Patient {self.name} ({self.email})>"
    
    def get_profile_completion(self) -> dict:
        """Calculate profile completion percentage"""
        fields = {
            "name": 10 if self.name and self.name.strip() else 0,
            "email": 10 if self.email and self.email.strip() else 0,
            "phone": 10 if self.phone and self.phone.strip() else 0,
            "age": 10 if self.age and self.age > 0 else 0,
            "gender": 10 if self.gender and self.gender.strip() else 0,
            "blood_type": 10 if self.blood_type and self.blood_type.strip() else 0,
            "address": 10 if self.address and isinstance(self.address, dict) and self.address.get("street") else 0,
            "emergency_contact": 10 if self.emergency_contact and self.emergency_contact.strip() else 0,
            "medical_conditions": 10 if self.medical_conditions and len(self.medical_conditions) > 0 else 0,
            "allergies": 10 if self.allergies and len(self.allergies) > 0 else 0,
            "medications": 10 if self.medications and len(self.medications) > 0 else 0,
        }
        total = sum(fields.values())
        return {
            "percentage": total,
            "fields": fields,
            "completed_count": sum(1 for v in fields.values() if v > 0),
            "total_fields": len(fields),
            "missing_fields": [key.replace("_", " ").title() for key, val in fields.items() if val == 0]
        }
    
    def get_completion_level(self) -> dict:
        """Get completion level with color and icon"""
        completion = self.get_profile_completion()
        percentage = completion["percentage"]
        
        if percentage >= 80:
            return {"level": "Excellent", "color": "text-green-600", "bg": "bg-green-100", "icon": "fas fa-check-circle"}
        elif percentage >= 60:
            return {"level": "Good", "color": "text-blue-600", "bg": "bg-blue-100", "icon": "fas fa-check-circle"}
        elif percentage >= 40:
            return {"level": "In Progress", "color": "text-yellow-600", "bg": "bg-yellow-100", "icon": "fas fa-spinner"}
        else:
            return {"level": "Needs Attention", "color": "text-red-600", "bg": "bg-red-100", "icon": "fas fa-exclamation-circle"}


class Admin(Base):
    __tablename__ = "admins"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    role = Column(String(100), default="Administrator")
    department = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<Admin {self.name} ({self.email})>"


class Report(Base):
    __tablename__ = "reports"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False)
    
    title = Column(String(255), nullable=False)
    date = Column(DateTime)
    doctor = Column(String(255))
    hospital = Column(String(255))
    diagnosis = Column(Text)
    summary = Column(Text)
    file_name = Column(String(255))
    file_size = Column(Integer)
    extracted_labs = Column(JSON, default=list)
    
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_analyzed = Column(Boolean, default=False)
    
    patient = relationship("Patient", back_populates="reports")
    ai_analysis = relationship("AIAnalysis", back_populates="report", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Report {self.title}>"


class AIAnalysis(Base):
    __tablename__ = "ai_analyses"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False)
    report_id = Column(String(36), ForeignKey("reports.id"), nullable=False)
    
    report_title = Column(String(255))
    analysis_date = Column(DateTime, default=datetime.utcnow)
    
    primary_diagnosis = Column(String(255))
    confidence = Column(Float)
    secondary_diagnoses = Column(JSON, default=list)
    recommendations = Column(JSON, default=list)
    risk_factors = Column(JSON, default=list)
    report_summary = Column(Text)
    
    patient = relationship("Patient", back_populates="ai_analyses")
    report = relationship("Report", back_populates="ai_analysis")
    
    def __repr__(self):
        return f"<AIAnalysis {self.primary_diagnosis}>"


class DigitalTwin(Base):
    __tablename__ = "digital_twins"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.id"), unique=True, nullable=False)
    
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    health_score = Column(Integer, default=0)
    
    vital_signs = Column(JSON, default=dict)
    health_metrics = Column(JSON, default=dict)
    activity_data = Column(JSON, default=dict)
    risk_assessment = Column(JSON, default=dict)
    predictions = Column(JSON, default=dict)
    
    patient = relationship("Patient", back_populates="twin_data")
    
    def __repr__(self):
        return f"<DigitalTwin - {self.patient_id}>"


class Claim(Base):
    __tablename__ = "claims"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("patients.id"), nullable=False)
    
    claim_number = Column(String(50), unique=True, index=True)
    provider_name = Column(String(255))
    provider_npi = Column(String(20))
    service_date = Column(DateTime)
    submission_date = Column(DateTime, default=datetime.utcnow)
    
    total_amount = Column(Float, default=0.0)
    paid_amount = Column(Float, default=0.0)
    
    diagnosis_codes = Column(JSON, default=list)
    procedure_codes = Column(JSON, default=list)
    
    is_fraudulent = Column(Boolean, default=False)
    fraud_score = Column(Float, default=0.0)
    fraud_explanation = Column(JSON, default=dict)
    risk_level = Column(String(50), default="LOW")
    
    status = Column(String(50), default="submitted")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Claim {self.claim_number}>"


class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255))
    user_type = Column(String(50))
    action = Column(String(100))
    resource = Column(String(100))
    resource_id = Column(String(255))
    details = Column(JSON)
    ip_address = Column(String(45))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AuditLog {self.action}>"


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def init_database():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
    # Lightweight migration for existing local SQLite installations.
    columns = {column["name"] for column in inspect(engine).get_columns("reports")}
    if "extracted_labs" not in columns:
        with engine.begin() as connection:
            connection.exec_driver_sql("ALTER TABLE reports ADD COLUMN extracted_labs JSON")
    print("Database tables created successfully")
    print(f"Database file: {DB_PATH}")

def get_db():
    """Dependency for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def seed_database():
    """Seed database with sample data"""
    db = SessionLocal()
    try:
        if db.query(Patient).first():
            print("Database already has data, skipping seed...")
            return
        
        print("Seeding database with sample data...")
        
        patients = [
            {
                "user_id": "patient1",
                "email": "patient@neuroguard.com",
                "password": "patient123",
                "name": "John Doe",
                "age": 45,
                "gender": "Male",
                "blood_type": "A+",
                "phone": "+1 (555) 123-4567",
                "address": {"street": "123 Main St", "city": "Boston", "state": "MA", "zip": "02101"},
                "medical_conditions": ["Type 2 Diabetes", "Hypertension"],
                "allergies": ["Penicillin", "Peanuts"],
                "medications": ["Metformin 500mg", "Lisinopril 10mg"],
                "emergency_contact": "Jane Doe - +1 (555) 987-6543",
                "last_visit": datetime.now()
            },
            {
                "user_id": "demo",
                "email": "demo@neuroguard.com",
                "password": "demo123",
                "name": "Demo Patient",
                "age": 35,
                "gender": "Female",
                "blood_type": "O-",
                "phone": "+1 (555) 234-5678",
                "address": {"street": "456 Oak Ave", "city": "New York", "state": "NY", "zip": "10001"},
                "medical_conditions": ["Asthma", "Anxiety"],
                "allergies": ["Dust", "Pollen"],
                "medications": ["Albuterol 90mcg", "Sertraline 50mg"],
                "emergency_contact": "Mike Smith - +1 (555) 876-5432",
                "last_visit": datetime.now()
            }
        ]
        
        patient_objects = []
        for p_data in patients:
            patient = Patient(
                user_id=p_data["user_id"],
                email=p_data["email"],
                password_hash=hash_password(p_data["password"]),
                name=p_data["name"],
                age=p_data["age"],
                gender=p_data["gender"],
                blood_type=p_data["blood_type"],
                phone=p_data["phone"],
                address=p_data["address"],
                medical_conditions=p_data["medical_conditions"],
                allergies=p_data["allergies"],
                medications=p_data["medications"],
                emergency_contact=p_data["emergency_contact"],
                last_visit=p_data["last_visit"]
            )
            db.add(patient)
            patient_objects.append(patient)
        
        admin = Admin(
            email="admin@neuroguard.com",
            password_hash=hash_password("admin123"),
            name="Dr. Smith",
            role="Administrator",
            department="Healthcare Management"
        )
        db.add(admin)
        db.commit()
        
        for patient in patient_objects:
            if patient.user_id == "patient1":
                twin = DigitalTwin(
                    patient_id=patient.id,
                    health_score=78,
                    vital_signs={"blood_pressure": "128/82", "heart_rate": 72, "temperature": 98.6, "weight": 182.5, "height": 70, "bmi": 28.5},
                    health_metrics={"a1c": 6.8, "cholesterol": 185, "ldl": 95, "hdl": 55, "triglycerides": 150},
                    activity_data={"steps": 8500, "active_minutes": 45, "sleep_hours": 7.5, "calories_burned": 2450},
                    risk_assessment={"diabetes_risk": "Medium", "heart_disease_risk": "Low", "stroke_risk": "Low"},
                    predictions={"next_year_health": "Stable with continued management", "recommendations": ["Increase physical activity", "Continue regular A1C monitoring", "Maintain healthy diet"]}
                )
            else:
                twin = DigitalTwin(
                    patient_id=patient.id,
                    health_score=82,
                    vital_signs={"blood_pressure": "118/76", "heart_rate": 68, "temperature": 98.4, "weight": 145.0, "height": 65, "bmi": 24.0},
                    health_metrics={"a1c": 5.2, "cholesterol": 165, "ldl": 85, "hdl": 60, "triglycerides": 100},
                    activity_data={"steps": 10200, "active_minutes": 60, "sleep_hours": 8.0, "calories_burned": 2200},
                    risk_assessment={"asthma_risk": "Medium", "heart_disease_risk": "Low", "stroke_risk": "Low"},
                    predictions={"next_year_health": "Good with current management", "recommendations": ["Continue using Albuterol", "Avoid triggers", "Regular check-ups"]}
                )
            db.add(twin)
        
        db.commit()
        print("Database seeded successfully!")
        
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

def reset_database():
    """Reset database"""
    print("Resetting database...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Database reset successfully")
    seed_database()

# ============================================================================
# RUN DATABASE SETUP
# ============================================================================

if __name__ == "__main__":
    print("Setting up NeuroGuard Database...")
    print("=" * 50)
    init_database()
    print("=" * 50)
    seed_database()
    print("=" * 50)
    print("\nDatabase file created: neuroguard.db")
    print("Setup complete!")
    print("\nYou can now run: python frontend.py")

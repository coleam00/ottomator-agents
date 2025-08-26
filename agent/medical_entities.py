"""
Medical entity types and extraction for episodic memory with Graphiti.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


# Enums for medical domain
class SymptomSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class SymptomFrequency(str, Enum):
    RARELY = "rarely"
    OCCASIONALLY = "occasionally"
    FREQUENTLY = "frequently"
    CONSTANTLY = "constantly"


class TreatmentStatus(str, Enum):
    PLANNED = "planned"
    ONGOING = "ongoing"
    COMPLETED = "completed"
    DISCONTINUED = "discontinued"


# Custom Entity Types for Graphiti
class Patient(BaseModel):
    """A patient entity with medical history information."""
    age: Optional[int] = Field(None, description="Age of the patient")
    gender: Optional[str] = Field(None, description="Gender of the patient")
    medical_history: Optional[List[str]] = Field(default_factory=list, description="List of past medical conditions")
    current_medications: Optional[List[str]] = Field(default_factory=list, description="Current medications")
    allergies: Optional[List[str]] = Field(default_factory=list, description="Known allergies")
    lifestyle_factors: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Lifestyle factors (smoking, exercise, etc.)")


class Symptom(BaseModel):
    """A medical symptom experienced by a patient."""
    location: Optional[str] = Field(None, description="Body location of the symptom")
    severity: Optional[SymptomSeverity] = Field(None, description="Severity level")
    frequency: Optional[SymptomFrequency] = Field(None, description="How often the symptom occurs")
    duration: Optional[str] = Field(None, description="How long the symptom has persisted")
    triggers: Optional[List[str]] = Field(default_factory=list, description="Known triggers")
    relieving_factors: Optional[List[str]] = Field(default_factory=list, description="What helps relieve the symptom")
    associated_symptoms: Optional[List[str]] = Field(default_factory=list, description="Other symptoms that occur together")


class Condition(BaseModel):
    """A medical condition or diagnosis."""
    icd_code: Optional[str] = Field(None, description="ICD-10 code if available")
    category: Optional[str] = Field(None, description="Medical category (cardiovascular, respiratory, etc.)")
    chronic: Optional[bool] = Field(None, description="Whether this is a chronic condition")
    diagnosed_date: Optional[datetime] = Field(None, description="When the condition was diagnosed")
    status: Optional[str] = Field(None, description="Current status (active, resolved, managed)")
    risk_factors: Optional[List[str]] = Field(default_factory=list, description="Associated risk factors")


class Treatment(BaseModel):
    """A medical treatment or intervention."""
    treatment_type: Optional[str] = Field(None, description="Type of treatment (medication, therapy, surgery, lifestyle)")
    dosage: Optional[str] = Field(None, description="Dosage information if applicable")
    frequency: Optional[str] = Field(None, description="How often the treatment is administered")
    duration: Optional[str] = Field(None, description="Expected duration of treatment")
    start_date: Optional[datetime] = Field(None, description="When treatment started")
    end_date: Optional[datetime] = Field(None, description="When treatment ended or expected to end")
    status: Optional[TreatmentStatus] = Field(None, description="Current status of treatment")
    effectiveness: Optional[float] = Field(None, description="Effectiveness score (0-1)")
    side_effects: Optional[List[str]] = Field(default_factory=list, description="Reported side effects")


class Medication(BaseModel):
    """A specific medication."""
    generic_name: Optional[str] = Field(None, description="Generic drug name")
    brand_name: Optional[str] = Field(None, description="Brand name if applicable")
    drug_class: Optional[str] = Field(None, description="Pharmacological class")
    dosage: Optional[str] = Field(None, description="Dosage strength")
    route: Optional[str] = Field(None, description="Route of administration (oral, IV, etc.)")
    frequency: Optional[str] = Field(None, description="Dosing frequency")
    contraindications: Optional[List[str]] = Field(default_factory=list, description="Known contraindications")
    interactions: Optional[List[str]] = Field(default_factory=list, description="Drug interactions")


class TestResult(BaseModel):
    """A medical test or lab result."""
    test_type: Optional[str] = Field(None, description="Type of test (blood, imaging, etc.)")
    test_name: Optional[str] = Field(None, description="Specific test name")
    result_value: Optional[str] = Field(None, description="Test result value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    normal_range: Optional[str] = Field(None, description="Normal reference range")
    abnormal: Optional[bool] = Field(None, description="Whether result is abnormal")
    test_date: Optional[datetime] = Field(None, description="When test was performed")
    interpretation: Optional[str] = Field(None, description="Clinical interpretation")


# Custom Edge Types for Medical Relationships
class HasSymptom(BaseModel):
    """Relationship between a patient and a symptom."""
    onset_date: Optional[datetime] = Field(None, description="When symptom started")
    reported_date: Optional[datetime] = Field(None, description="When symptom was reported")
    severity_progression: Optional[str] = Field(None, description="How severity has changed over time")
    impact_on_daily_life: Optional[str] = Field(None, description="Impact on daily activities")


class HasCondition(BaseModel):
    """Relationship between a patient and a medical condition."""
    diagnosed_by: Optional[str] = Field(None, description="Healthcare provider who diagnosed")
    diagnosis_confidence: Optional[float] = Field(None, description="Confidence in diagnosis (0-1)")
    is_primary: Optional[bool] = Field(None, description="Whether this is the primary diagnosis")
    complications: Optional[List[str]] = Field(default_factory=list, description="Known complications")


class ReceivesTreatment(BaseModel):
    """Relationship between a patient and a treatment."""
    prescribed_by: Optional[str] = Field(None, description="Healthcare provider who prescribed")
    adherence_level: Optional[float] = Field(None, description="Treatment adherence (0-1)")
    response: Optional[str] = Field(None, description="Patient's response to treatment")
    monitoring_required: Optional[bool] = Field(None, description="Whether monitoring is needed")


class CausesSymptom(BaseModel):
    """Relationship indicating a condition causes a symptom."""
    mechanism: Optional[str] = Field(None, description="Pathophysiological mechanism")
    likelihood: Optional[float] = Field(None, description="Likelihood of causation (0-1)")
    evidence_level: Optional[str] = Field(None, description="Level of medical evidence")


class TreatsCondition(BaseModel):
    """Relationship indicating a treatment addresses a condition."""
    efficacy_rate: Optional[float] = Field(None, description="Treatment efficacy rate (0-1)")
    first_line: Optional[bool] = Field(None, description="Whether this is first-line treatment")
    alternative_to: Optional[List[str]] = Field(default_factory=list, description="Alternative treatments")


# Medical entity extraction patterns
MEDICAL_PATTERNS = {
    "symptoms": {
        "pain": ["pain", "ache", "hurt", "sore", "tender", "burning", "stinging", "throbbing"],
        "gastrointestinal": ["nausea", "vomiting", "diarrhea", "constipation", "bloating", "gas", "heartburn"],
        "respiratory": ["cough", "shortness of breath", "wheezing", "congestion", "phlegm"],
        "neurological": ["headache", "dizziness", "vertigo", "numbness", "tingling", "weakness"],
        "general": ["fatigue", "fever", "chills", "sweating", "weight loss", "weight gain"],
        "psychological": ["anxiety", "depression", "insomnia", "stress", "mood changes"]
    },
    "body_parts": [
        "head", "neck", "chest", "abdomen", "back", "arm", "leg", "foot", "hand",
        "stomach", "heart", "lung", "liver", "kidney", "throat", "ear", "eye", "nose"
    ],
    "conditions": {
        "chronic": ["diabetes", "hypertension", "asthma", "arthritis", "COPD", "heart disease"],
        "acute": ["flu", "cold", "infection", "injury", "fracture", "sprain"],
        "mental_health": ["depression", "anxiety disorder", "PTSD", "bipolar disorder"]
    },
    "treatments": {
        "medications": ["medication", "drug", "pill", "tablet", "injection", "prescription"],
        "procedures": ["surgery", "therapy", "procedure", "treatment", "intervention"],
        "lifestyle": ["diet", "exercise", "rest", "hydration", "stress management"]
    },
    "severity_indicators": {
        "mild": ["mild", "slight", "minor", "little"],
        "moderate": ["moderate", "medium", "some", "noticeable"],
        "severe": ["severe", "intense", "extreme", "unbearable", "excruciating"]
    },
    "temporal_indicators": {
        "onset": ["started", "began", "first noticed", "onset"],
        "duration": ["for", "since", "lasting", "continued", "persistent"],
        "frequency": ["daily", "weekly", "constantly", "occasionally", "sometimes"]
    }
}


class MedicalEntityExtractor:
    """Extract medical entities from text."""
    
    def __init__(self):
        self.patterns = MEDICAL_PATTERNS
        
    def extract_symptoms(self, text: str) -> List[Dict[str, Any]]:
        """Extract symptom mentions from text."""
        text_lower = text.lower()
        symptoms = []
        
        # Look for symptom keywords
        for category, keywords in self.patterns["symptoms"].items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Extract context around the symptom
                    context = self._extract_context(text, keyword)
                    
                    symptom = {
                        "name": keyword,
                        "category": category,
                        "severity": self._extract_severity(context),
                        "location": self._extract_body_part(context),
                        "duration": self._extract_duration(context),
                        "context": context
                    }
                    symptoms.append(symptom)
        
        return symptoms
    
    def extract_conditions(self, text: str) -> List[Dict[str, Any]]:
        """Extract medical condition mentions."""
        text_lower = text.lower()
        conditions = []
        
        for category, condition_list in self.patterns["conditions"].items():
            for condition in condition_list:
                if condition in text_lower:
                    context = self._extract_context(text, condition)
                    
                    conditions.append({
                        "name": condition,
                        "category": category,
                        "context": context
                    })
        
        return conditions
    
    def extract_treatments(self, text: str) -> List[Dict[str, Any]]:
        """Extract treatment mentions."""
        text_lower = text.lower()
        treatments = []
        
        for category, treatment_keywords in self.patterns["treatments"].items():
            for keyword in treatment_keywords:
                if keyword in text_lower:
                    context = self._extract_context(text, keyword)
                    
                    treatments.append({
                        "name": keyword,
                        "category": category,
                        "context": context
                    })
        
        return treatments
    
    def extract_body_parts(self, text: str) -> List[str]:
        """Extract body part mentions."""
        text_lower = text.lower()
        body_parts = []
        
        for part in self.patterns["body_parts"]:
            if part in text_lower:
                body_parts.append(part)
        
        return body_parts
    
    def _extract_context(self, text: str, keyword: str, window: int = 50) -> str:
        """Extract context around a keyword."""
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        index = text_lower.find(keyword_lower)
        if index == -1:
            return ""
        
        start = max(0, index - window)
        end = min(len(text), index + len(keyword) + window)
        
        return text[start:end].strip()
    
    def _extract_severity(self, context: str) -> Optional[str]:
        """Extract severity from context."""
        context_lower = context.lower()
        
        for severity, indicators in self.patterns["severity_indicators"].items():
            for indicator in indicators:
                if indicator in context_lower:
                    return severity
        
        return None
    
    def _extract_body_part(self, context: str) -> Optional[str]:
        """Extract body part from context."""
        context_lower = context.lower()
        
        for part in self.patterns["body_parts"]:
            if part in context_lower:
                return part
        
        return None
    
    def _extract_duration(self, context: str) -> Optional[str]:
        """Extract duration information from context."""
        # Look for patterns like "for 3 days", "since yesterday", etc.
        duration_pattern = r'(for|since|lasting|about)\s+(\d+\s+)?(days?|weeks?|months?|years?|hours?|yesterday|today)'
        match = re.search(duration_pattern, context.lower())
        
        if match:
            return match.group(0)
        
        return None
    
    def extract_all_entities(self, text: str) -> Dict[str, Any]:
        """Extract all medical entities from text."""
        return {
            "symptoms": self.extract_symptoms(text),
            "conditions": self.extract_conditions(text),
            "treatments": self.extract_treatments(text),
            "body_parts": self.extract_body_parts(text)
        }


# Entity type mappings for Graphiti
def get_medical_entity_types() -> Dict[str, type[BaseModel]]:
    """Get medical entity types for Graphiti configuration."""
    return {
        "Patient": Patient,
        "Symptom": Symptom,
        "Condition": Condition,
        "Treatment": Treatment,
        "Medication": Medication,
        "TestResult": TestResult
    }


def get_medical_edge_types() -> Dict[str, type[BaseModel]]:
    """Get medical edge types for Graphiti configuration."""
    return {
        "HasSymptom": HasSymptom,
        "HasCondition": HasCondition,
        "ReceivesTreatment": ReceivesTreatment,
        "CausesSymptom": CausesSymptom,
        "TreatsCondition": TreatsCondition
    }


def get_medical_edge_type_map() -> Dict[tuple, List[str]]:
    """Get edge type mappings for different entity pairs."""
    return {
        ("Patient", "Symptom"): ["HasSymptom"],
        ("Patient", "Condition"): ["HasCondition"],
        ("Patient", "Treatment"): ["ReceivesTreatment"],
        ("Patient", "Medication"): ["ReceivesTreatment"],
        ("Condition", "Symptom"): ["CausesSymptom"],
        ("Treatment", "Condition"): ["TreatsCondition"],
        ("Medication", "Condition"): ["TreatsCondition"],
        ("Symptom", "Symptom"): ["RelatedTo"],
        ("Entity", "Entity"): ["RelatedTo"]  # Fallback for any entity types
    }


# Global extractor instance
medical_entity_extractor = MedicalEntityExtractor()
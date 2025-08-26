"""
Medical fact extraction module for episodic memory.
"""

import os
import logging
import json
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MedicalFact:
    """Represents a medical fact extracted from conversation."""
    subject: str
    predicate: str
    object: str
    confidence: float
    source: str
    context: Optional[str] = None
    temporal_info: Optional[str] = None


class MedicalFactExtractor:
    """Extract medical facts from conversations using LLM."""
    
    def __init__(self):
        """Initialize the fact extractor."""
        self.confidence_threshold = float(os.getenv("FACT_EXTRACTION_CONFIDENCE", "0.7"))
    
    async def extract_facts_from_conversation(
        self,
        user_message: str,
        assistant_response: str,
        session_context: Optional[Dict[str, Any]] = None
    ) -> List[MedicalFact]:
        """
        Extract medical facts from a conversation turn.
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response
            session_context: Optional context from the session
        
        Returns:
            List of extracted medical facts
        """
        facts = []
        
        # Extract facts using pattern matching first
        pattern_facts = self._extract_pattern_based_facts(user_message, assistant_response)
        facts.extend(pattern_facts)
        
        # Extract temporal facts
        temporal_facts = self._extract_temporal_facts(user_message)
        facts.extend(temporal_facts)
        
        # Extract causal relationships
        causal_facts = self._extract_causal_facts(user_message, assistant_response)
        facts.extend(causal_facts)
        
        # Filter by confidence threshold
        filtered_facts = [f for f in facts if f.confidence >= self.confidence_threshold]
        
        logger.info(f"Extracted {len(filtered_facts)} facts above confidence threshold")
        return filtered_facts
    
    def _extract_pattern_based_facts(
        self,
        user_message: str,
        assistant_response: str
    ) -> List[MedicalFact]:
        """Extract facts using regex patterns."""
        facts = []
        
        # Symptom patterns
        symptom_patterns = [
            (r"I have (?:a |an )?(\w+\s*\w*)", "HAS_SYMPTOM"),
            (r"experiencing (\w+\s*\w*)", "HAS_SYMPTOM"),
            (r"suffering from (\w+\s*\w*)", "HAS_SYMPTOM"),
            (r"(\w+) in (?:my |the )?(\w+)", "SYMPTOM_LOCATION"),
            (r"(\w+) pain", "HAS_SYMPTOM"),
        ]
        
        for pattern, predicate in symptom_patterns:
            matches = re.finditer(pattern, user_message.lower())
            for match in matches:
                if predicate == "SYMPTOM_LOCATION" and match.lastindex >= 2:
                    symptom = match.group(1) if match.group(1) else "unknown"
                    location = match.group(2) if match.group(2) else "unknown"
                    facts.append(MedicalFact(
                        subject=symptom,
                        predicate="LOCATED_IN",
                        object=location,
                        confidence=0.8,
                        source="pattern_extraction",
                        context=match.group(0)
                    ))
                elif match.lastindex >= 1:
                    symptom = match.group(1) if match.group(1) else "unknown"
                    facts.append(MedicalFact(
                        subject="Patient",
                        predicate=predicate,
                        object=symptom,
                        confidence=0.8,
                        source="pattern_extraction",
                        context=match.group(0)
                    ))
        
        # Severity patterns
        severity_patterns = [
            (r"(\w+) (?:is |feels )?(mild|moderate|severe|extreme)", "HAS_SEVERITY"),
            (r"(mild|moderate|severe|extreme) (\w+)", "HAS_SEVERITY"),
        ]
        
        for pattern, predicate in severity_patterns:
            matches = re.finditer(pattern, user_message.lower())
            for match in matches:
                # Safely extract groups
                groups = match.groups()
                if not groups or match.lastindex is None:
                    continue
                
                # Determine which group contains severity and which contains symptom
                symptom = None
                severity = None
                
                # Check each group safely
                for i, group in enumerate(groups, 1):
                    if group:
                        if any(sev in group.lower() for sev in ["mild", "moderate", "severe", "extreme"]):
                            severity = group
                        else:
                            symptom = group
                
                # Only create fact if we have both components
                if symptom and severity:
                    facts.append(MedicalFact(
                        subject=symptom,
                        predicate=predicate,
                        object=severity,
                        confidence=0.85,
                        source="pattern_extraction",
                        context=match.group(0)
                    ))
        
        # Treatment recommendations from assistant
        treatment_patterns = [
            (r"recommend (\w+\s*\w*)", "RECOMMENDS"),
            (r"try (\w+\s*\w*)", "SUGGESTS"),
            (r"consider (\w+\s*\w*)", "SUGGESTS"),
            (r"prescribe (\w+\s*\w*)", "PRESCRIBES"),
        ]
        
        for pattern, predicate in treatment_patterns:
            matches = re.finditer(pattern, assistant_response.lower())
            for match in matches:
                if match.lastindex >= 1:
                    treatment = match.group(1) if match.group(1) else "unknown"
                    facts.append(MedicalFact(
                        subject="Assistant",
                        predicate=predicate,
                        object=treatment,
                        confidence=0.9,
                        source="pattern_extraction",
                        context=match.group(0)
                    ))
        
        return facts
    
    def _extract_temporal_facts(self, text: str) -> List[MedicalFact]:
        """Extract temporal information from text."""
        facts = []
        
        # Duration patterns
        duration_patterns = [
            (r"for (\d+) (days?|weeks?|months?|years?)", "DURATION"),
            (r"started (\d+) (days?|weeks?|months?) ago", "ONSET"),
            (r"since (yesterday|last week|last month)", "ONSET"),
            (r"lasting (\d+) (hours?|days?)", "DURATION"),
        ]
        
        for pattern, predicate in duration_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                temporal_info = match.group(0)
                facts.append(MedicalFact(
                    subject="Symptom",
                    predicate=predicate,
                    object=temporal_info,
                    confidence=0.9,
                    source="temporal_extraction",
                    context=match.group(0),
                    temporal_info=temporal_info
                ))
        
        # Frequency patterns
        frequency_patterns = [
            (r"(daily|weekly|monthly)", "FREQUENCY"),
            (r"(\d+) times (?:a |per )(day|week|month)", "FREQUENCY"),
            (r"(constantly|occasionally|frequently|rarely)", "FREQUENCY"),
        ]
        
        for pattern, predicate in frequency_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                frequency = match.group(0)
                facts.append(MedicalFact(
                    subject="Symptom",
                    predicate=predicate,
                    object=frequency,
                    confidence=0.85,
                    source="temporal_extraction",
                    context=match.group(0)
                ))
        
        return facts
    
    def _extract_causal_facts(
        self,
        user_message: str,
        assistant_response: str
    ) -> List[MedicalFact]:
        """Extract causal relationships."""
        facts = []
        
        # Trigger patterns
        trigger_patterns = [
            (r"(\w+) (?:triggers|causes|leads to) (\w+)", "CAUSES"),
            (r"(\w+) (?:when|after) (\w+)", "TRIGGERED_BY"),
            (r"(\w+) makes (?:the |my )?(\w+) worse", "AGGRAVATES"),
            (r"(\w+) helps (?:with |relieve )?(?:the |my )?(\w+)", "RELIEVES"),
        ]
        
        for pattern, predicate in trigger_patterns:
            # Check both messages
            for text in [user_message, assistant_response]:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    if match.lastindex >= 2:
                        subject = match.group(1) if match.group(1) else "unknown"
                        object_val = match.group(2) if match.group(2) else "unknown"
                        facts.append(MedicalFact(
                            subject=subject,
                            predicate=predicate,
                            object=object_val,
                            confidence=0.75,
                            source="causal_extraction",
                            context=match.group(0)
                        ))
        
        return facts
    
    def validate_facts(
        self,
        facts: List[MedicalFact],
        medical_knowledge: Optional[Dict[str, Any]] = None
    ) -> List[MedicalFact]:
        """
        Validate extracted facts against medical knowledge.
        
        Args:
            facts: List of extracted facts
            medical_knowledge: Optional medical knowledge base for validation
        
        Returns:
            List of validated facts with adjusted confidence scores
        """
        validated_facts = []
        
        for fact in facts:
            # Basic validation rules
            valid = True
            
            # Check for contradictions
            if fact.predicate == "HAS_SEVERITY":
                # Severity should be one of known values
                valid_severities = ["mild", "moderate", "severe", "extreme", "critical"]
                if fact.object.lower() not in valid_severities:
                    valid = False
            
            # Check for medical plausibility
            if fact.predicate == "LOCATED_IN":
                # Symptom should be in a valid body part
                valid_body_parts = ["head", "chest", "abdomen", "back", "neck", "arm", "leg", "stomach"]
                if fact.object.lower() not in valid_body_parts:
                    fact.confidence *= 0.8  # Reduce confidence
            
            if valid:
                validated_facts.append(fact)
        
        return validated_facts
    
    def consolidate_facts(
        self,
        facts: List[MedicalFact]
    ) -> List[MedicalFact]:
        """
        Consolidate duplicate or related facts.
        
        Args:
            facts: List of facts to consolidate
        
        Returns:
            Consolidated list of facts
        """
        consolidated = {}
        
        for fact in facts:
            # Create a key for deduplication
            key = f"{fact.subject}_{fact.predicate}_{fact.object}"
            
            if key in consolidated:
                # Keep the fact with higher confidence
                if fact.confidence > consolidated[key].confidence:
                    consolidated[key] = fact
            else:
                consolidated[key] = fact
        
        return list(consolidated.values())
    
    def format_facts_for_graphiti(
        self,
        facts: List[MedicalFact]
    ) -> List[Tuple[str, str, str]]:
        """
        Format facts for Graphiti storage.
        
        Args:
            facts: List of medical facts
        
        Returns:
            List of (subject, predicate, object) tuples
        """
        return [(f.subject, f.predicate, f.object) for f in facts]


# Global extractor instance
medical_fact_extractor = MedicalFactExtractor()
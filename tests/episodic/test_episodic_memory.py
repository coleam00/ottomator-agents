"""
Tests for episodic memory functionality with Graphiti.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from agent.episodic_memory import EpisodicMemoryService, EpisodicMemoryQueue
from agent.medical_entities import MedicalEntityExtractor
from agent.fact_extractor import MedicalFactExtractor, MedicalFact


@pytest.fixture
def episodic_service():
    """Create episodic memory service instance."""
    with patch('agent.episodic_memory.graph_client') as mock_graph:
        service = EpisodicMemoryService()
        service.graph_client = mock_graph
        return service


@pytest.fixture
def entity_extractor():
    """Create medical entity extractor instance."""
    return MedicalEntityExtractor()


@pytest.fixture
def fact_extractor():
    """Create medical fact extractor instance."""
    return MedicalFactExtractor()


class TestEpisodicMemoryService:
    """Test episodic memory service functionality."""
    
    @pytest.mark.asyncio
    async def test_create_conversation_episode(self, episodic_service):
        """Test creating a conversation episode."""
        # Mock graph client
        episodic_service.graph_client.add_episode = AsyncMock()
        
        # Test data
        session_id = str(uuid4())
        user_message = "I have a severe headache in my forehead"
        assistant_response = "I understand you have a severe headache. When did it start?"
        tools_used = [{"tool_name": "vector_search"}]
        metadata = {"user_id": "test_user"}
        
        # Create episode
        episode_id = await episodic_service.create_conversation_episode(
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            tools_used=tools_used,
            metadata=metadata
        )
        
        # Verify episode was created
        assert episode_id is not None
        assert episode_id.startswith(f"conversation_{session_id}")
        
        # Verify graph client was called
        episodic_service.graph_client.add_episode.assert_called_once()
        call_args = episodic_service.graph_client.add_episode.call_args[1]
        
        assert call_args['episode_id'] == episode_id
        assert "User Query:" in call_args['content']
        assert "Assistant Response:" in call_args['content']
        assert call_args['metadata']['session_id'] == session_id
        assert call_args['metadata']['user_id'] == "test_user"
    
    def test_extract_medical_entities(self, episodic_service):
        """Test medical entity extraction."""
        text = "I have severe chest pain and shortness of breath for 3 days"
        
        entities = episodic_service.extract_medical_entities(text)
        
        assert 'symptoms' in entities
        assert len(entities['symptoms']) > 0
        
        # Check for chest pain
        symptoms = entities['symptoms']
        chest_pain = next((s for s in symptoms if 'pain' in s['name']), None)
        assert chest_pain is not None
        assert chest_pain['severity'] == 'severe'
        assert chest_pain['location'] == 'chest'
    
    def test_extract_fact_triples(self, episodic_service):
        """Test fact triple extraction."""
        user_message = "I have moderate headache in my head for 2 days"
        assistant_response = "I recommend trying ibuprofen for your headache"
        
        facts = episodic_service.extract_fact_triples(user_message, assistant_response)
        
        assert len(facts) > 0
        
        # Check for symptom fact
        symptom_fact = next((f for f in facts if f[1] == "HAS_SYMPTOM"), None)
        assert symptom_fact is not None
        assert symptom_fact[0] == "Patient"
        assert "headache" in symptom_fact[2]
        
        # Check for treatment recommendation
        treatment_fact = next((f for f in facts if f[1] == "RECOMMENDS_TREATMENT"), None)
        assert treatment_fact is not None
        assert treatment_fact[0] == "Assistant"
    
    def test_calculate_memory_importance(self, episodic_service):
        """Test memory importance calculation."""
        entities = {
            "symptoms": [{"name": "chest pain"}, {"name": "shortness of breath"}],
            "conditions": [{"name": "heart attack"}]
        }
        facts = [
            ("Patient", "HAS_SYMPTOM", "chest pain"),
            ("Patient", "HAS_SYMPTOM", "shortness of breath")
        ]
        metadata = {"severity": "severe"}
        
        score = episodic_service.calculate_memory_importance(entities, facts, metadata)
        
        assert 0 <= score <= 1
        assert score > 0.5  # Should be high due to multiple symptoms and severity
    
    @pytest.mark.asyncio
    async def test_create_symptom_timeline(self, episodic_service):
        """Test symptom timeline creation."""
        episodic_service.graph_client.add_episode = AsyncMock()
        
        session_id = str(uuid4())
        symptom = "headache"
        timestamp = datetime.now(timezone.utc)
        severity = "moderate"
        
        timeline_id = await episodic_service.create_symptom_timeline(
            session_id=session_id,
            symptom=symptom,
            timestamp=timestamp,
            severity=severity
        )
        
        assert timeline_id is not None
        assert timeline_id.startswith(f"symptom_timeline_{session_id}")
        
        # Verify graph client was called
        episodic_service.graph_client.add_episode.assert_called_once()
        call_args = episodic_service.graph_client.add_episode.call_args[1]
        
        assert "Symptom Timeline Entry: headache" in call_args['content']
        assert "Severity: moderate" in call_args['content']
    
    @pytest.mark.asyncio
    async def test_search_episodic_memories(self, episodic_service):
        """Test searching episodic memories."""
        episodic_service.graph_client.search = AsyncMock(return_value=[
            {
                "fact": "Patient has headache",
                "uuid": str(uuid4()),
                "source_node_uuid": "conversation_123"
            }
        ])
        
        results = await episodic_service.search_episodic_memories(
            query="headache",
            session_id="test_session"
        )
        
        assert len(results) > 0
        episodic_service.graph_client.search.assert_called_once()


class TestMedicalEntityExtractor:
    """Test medical entity extraction."""
    
    def test_extract_symptoms(self, entity_extractor):
        """Test symptom extraction."""
        text = "I have severe chest pain and mild nausea"
        
        symptoms = entity_extractor.extract_symptoms(text)
        
        assert len(symptoms) >= 2
        
        # Check chest pain
        chest_pain = next((s for s in symptoms if s['name'] == 'pain'), None)
        assert chest_pain is not None
        assert chest_pain['severity'] == 'severe'
        assert chest_pain['location'] == 'chest'
        
        # Check nausea
        nausea = next((s for s in symptoms if s['name'] == 'nausea'), None)
        assert nausea is not None
        assert nausea['severity'] == 'mild'
    
    def test_extract_conditions(self, entity_extractor):
        """Test condition extraction."""
        text = "I was diagnosed with diabetes and hypertension"
        
        conditions = entity_extractor.extract_conditions(text)
        
        assert len(conditions) >= 2
        
        condition_names = [c['name'] for c in conditions]
        assert 'diabetes' in condition_names
        assert 'hypertension' in condition_names
    
    def test_extract_treatments(self, entity_extractor):
        """Test treatment extraction."""
        text = "I'm taking medication and doing physical therapy"
        
        treatments = entity_extractor.extract_treatments(text)
        
        assert len(treatments) >= 2
        
        treatment_names = [t['name'] for t in treatments]
        assert 'medication' in treatment_names
        assert 'therapy' in treatment_names
    
    def test_extract_body_parts(self, entity_extractor):
        """Test body part extraction."""
        text = "Pain in my head, chest, and stomach"
        
        body_parts = entity_extractor.extract_body_parts(text)
        
        assert len(body_parts) >= 3
        assert 'head' in body_parts
        assert 'chest' in body_parts
        assert 'stomach' in body_parts
    
    def test_extract_all_entities(self, entity_extractor):
        """Test comprehensive entity extraction."""
        text = "I have severe headache and was diagnosed with migraine. Taking medication helps."
        
        entities = entity_extractor.extract_all_entities(text)
        
        assert 'symptoms' in entities
        assert 'conditions' in entities
        assert 'treatments' in entities
        assert 'body_parts' in entities
        
        assert len(entities['symptoms']) > 0
        assert len(entities['conditions']) > 0
        assert len(entities['treatments']) > 0


class TestMedicalFactExtractor:
    """Test medical fact extraction."""
    
    def test_extract_pattern_based_facts(self, fact_extractor):
        """Test pattern-based fact extraction."""
        user_message = "I have severe headache"
        assistant_response = "I recommend trying ibuprofen"
        
        facts = fact_extractor._extract_pattern_based_facts(user_message, assistant_response)
        
        assert len(facts) > 0
        
        # Check for symptom fact
        symptom_fact = next((f for f in facts if f.predicate == "HAS_SYMPTOM"), None)
        assert symptom_fact is not None
        assert symptom_fact.subject == "Patient"
        assert "headache" in symptom_fact.object
        
        # Check for treatment fact
        treatment_fact = next((f for f in facts if f.predicate in ["RECOMMENDS", "SUGGESTS"]), None)
        assert treatment_fact is not None
        assert treatment_fact.subject == "Assistant"
    
    def test_extract_temporal_facts(self, fact_extractor):
        """Test temporal fact extraction."""
        text = "The pain started 3 days ago and occurs daily"
        
        facts = fact_extractor._extract_temporal_facts(text)
        
        assert len(facts) > 0
        
        # Check for onset fact
        onset_fact = next((f for f in facts if f.predicate == "ONSET"), None)
        assert onset_fact is not None
        assert "3 days ago" in onset_fact.object
        
        # Check for frequency fact
        frequency_fact = next((f for f in facts if f.predicate == "FREQUENCY"), None)
        assert frequency_fact is not None
        assert "daily" in frequency_fact.object
    
    def test_extract_causal_facts(self, fact_extractor):
        """Test causal fact extraction."""
        user_message = "Stress triggers my headache"
        assistant_response = "Rest helps relieve the pain"
        
        facts = fact_extractor._extract_causal_facts(user_message, assistant_response)
        
        assert len(facts) > 0
        
        # Check for trigger fact
        trigger_fact = next((f for f in facts if f.predicate == "CAUSES"), None)
        assert trigger_fact is not None
        
        # Check for relief fact
        relief_fact = next((f for f in facts if f.predicate == "RELIEVES"), None)
        assert relief_fact is not None
    
    def test_validate_facts(self, fact_extractor):
        """Test fact validation."""
        facts = [
            MedicalFact("symptom", "HAS_SEVERITY", "extreme", 0.8, "test"),
            MedicalFact("symptom", "HAS_SEVERITY", "invalid", 0.8, "test"),
            MedicalFact("symptom", "LOCATED_IN", "head", 0.8, "test"),
        ]
        
        validated = fact_extractor.validate_facts(facts)
        
        # Invalid severity should be filtered out
        assert len(validated) == 2
        severity_facts = [f for f in validated if f.predicate == "HAS_SEVERITY"]
        assert len(severity_facts) == 1
        assert severity_facts[0].object == "extreme"
    
    def test_consolidate_facts(self, fact_extractor):
        """Test fact consolidation."""
        facts = [
            MedicalFact("Patient", "HAS_SYMPTOM", "headache", 0.7, "source1"),
            MedicalFact("Patient", "HAS_SYMPTOM", "headache", 0.9, "source2"),
            MedicalFact("Patient", "HAS_SYMPTOM", "nausea", 0.8, "source3"),
        ]
        
        consolidated = fact_extractor.consolidate_facts(facts)
        
        assert len(consolidated) == 2
        
        # Should keep higher confidence headache fact
        headache_fact = next((f for f in consolidated if f.object == "headache"), None)
        assert headache_fact is not None
        assert headache_fact.confidence == 0.9


class TestEpisodicMemoryQueue:
    """Test episodic memory queue."""
    
    @pytest.mark.asyncio
    async def test_add_to_queue(self):
        """Test adding episodes to queue."""
        queue = EpisodicMemoryQueue(batch_size=2)
        
        episode1 = {"episode_id": "ep1", "content": "test1"}
        episode2 = {"episode_id": "ep2", "content": "test2"}
        
        await queue.add(episode1)
        assert len(queue.queue) == 1
        
        # Mock the flush to prevent actual processing
        with patch.object(queue, 'flush', new_callable=AsyncMock):
            await queue.add(episode2)
            # Should trigger flush when batch size reached
            queue.flush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_flush_queue(self):
        """Test flushing queue."""
        queue = EpisodicMemoryQueue(batch_size=5)
        
        # Mock the create_episode method
        with patch.object(queue, '_create_episode', new_callable=AsyncMock):
            queue.queue = [
                {"episode_id": "ep1"},
                {"episode_id": "ep2"}
            ]
            
            await queue.flush()
            
            assert len(queue.queue) == 0
            assert queue._create_episode.call_count == 2


@pytest.mark.asyncio
async def test_end_to_end_episodic_memory():
    """Test end-to-end episodic memory flow."""
    with patch('agent.episodic_memory.graph_client') as mock_graph:
        mock_graph.add_episode = AsyncMock()
        mock_graph.search = AsyncMock(return_value=[
            {
                "fact": "Patient has severe headache",
                "uuid": str(uuid4()),
                "valid_at": datetime.now(timezone.utc).isoformat()
            }
        ])
        
        service = EpisodicMemoryService()
        service.graph_client = mock_graph
        
        # Create an episode
        session_id = str(uuid4())
        episode_id = await service.create_conversation_episode(
            session_id=session_id,
            user_message="I have a severe headache that started yesterday",
            assistant_response="I understand. Have you taken any medication?",
            metadata={"user_id": "test_user"}
        )
        
        assert episode_id is not None
        
        # Search for the episode
        results = await service.search_episodic_memories(
            query="headache",
            session_id=session_id
        )
        
        assert len(results) > 0
        assert "headache" in results[0]["fact"].lower()
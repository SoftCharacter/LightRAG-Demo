"""
Entity Type Validator Module
Validates and filters entities based on allowed entity types from config
"""

import logging
from typing import List, Dict, Any, Set, Optional
from difflib import SequenceMatcher


class EntityTypeValidator:
    """Validates entity types against allowed types from configuration"""
    
    def __init__(self, allowed_types: List[str], logger: Optional[logging.Logger] = None):
        """
        Initialize validator with allowed entity types
        
        Args:
            allowed_types: List of allowed entity type names
            logger: Optional logger instance
        """
        self.allowed_types = set(t.lower() for t in allowed_types)
        self.allowed_types_original = {t.lower(): t for t in allowed_types}
        self.logger = logger or logging.getLogger("EntityTypeValidator")
        
        # Add "Other" as fallback type if not present
        if "other" not in self.allowed_types:
            self.allowed_types.add("other")
            self.allowed_types_original["other"] = "Other"
        
        self.logger.info(f"EntityTypeValidator initialized with types: {allowed_types}")
    
    def is_valid_type(self, entity_type: str) -> bool:
        """Check if entity type is in allowed list"""
        return entity_type.lower() in self.allowed_types
    
    def find_closest_type(self, entity_type: str, threshold: float = 0.6) -> Optional[str]:
        """
        Find the closest matching allowed type using string similarity
        
        Args:
            entity_type: The entity type to match
            threshold: Minimum similarity threshold (0.0-1.0)
        
        Returns:
            Closest matching type or None if no match above threshold
        """
        entity_type_lower = entity_type.lower()
        best_match = None
        best_score = 0.0
        
        for allowed_type in self.allowed_types:
            score = SequenceMatcher(None, entity_type_lower, allowed_type).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_match = allowed_type
        
        if best_match:
            return self.allowed_types_original[best_match]
        return None
    
    def validate_entity(self, entity: Dict[str, Any], mode: str = "reclassify") -> Optional[Dict[str, Any]]:
        """
        Validate a single entity and optionally reclassify
        
        Args:
            entity: Entity dictionary with 'entity_type' field
            mode: Validation mode - 'strict' (reject invalid), 'reclassify' (find closest), 'fallback' (use Other)
        
        Returns:
            Validated entity or None if rejected
        """
        entity_type = entity.get("entity_type", "")
        entity_name = entity.get("entity_name", entity.get("name", "Unknown"))
        
        # Already valid
        if self.is_valid_type(entity_type):
            return entity
        
        original_type = entity_type
        
        if mode == "strict":
            self.logger.warning(f"Entity '{entity_name}' rejected: invalid type '{entity_type}'")
            return None
        
        elif mode == "reclassify":
            closest = self.find_closest_type(entity_type)
            if closest:
                entity["entity_type"] = closest
                self.logger.info(f"Entity '{entity_name}' reclassified: '{original_type}' -> '{closest}'")
                return entity
            else:
                # Fall back to "Other"
                entity["entity_type"] = "Other"
                self.logger.info(f"Entity '{entity_name}' reclassified to 'Other' (was '{original_type}')")
                return entity
        
        elif mode == "fallback":
            entity["entity_type"] = "Other"
            self.logger.info(f"Entity '{entity_name}' type set to 'Other' (was '{original_type}')")
            return entity
        
        return entity
    
    def validate_entities(self, entities: List[Dict[str, Any]], mode: str = "reclassify") -> List[Dict[str, Any]]:
        """
        Validate a list of entities
        
        Args:
            entities: List of entity dictionaries
            mode: Validation mode
        
        Returns:
            List of validated entities (invalid ones filtered out in strict mode)
        """
        validated = []
        rejected_count = 0
        reclassified_count = 0
        
        for entity in entities:
            original_type = entity.get("entity_type", "")
            result = self.validate_entity(entity.copy(), mode)
            
            if result is None:
                rejected_count += 1
            else:
                if result.get("entity_type") != original_type:
                    reclassified_count += 1
                validated.append(result)
        
        self.logger.info(
            f"Entity validation complete: {len(validated)} valid, "
            f"{rejected_count} rejected, {reclassified_count} reclassified"
        )
        return validated


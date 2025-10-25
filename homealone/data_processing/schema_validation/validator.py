from pydantic import ValidationError, BaseModel
from typing import List, Type, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ValidationManager:
    """Manages schema validation across the pipeline"""
    
    def __init__(self):
        self.failed_records = []
        self.validation_stats = {
            'total_records': 0,
            'valid_records': 0,
            'invalid_records': 0
        }
    
    def validate_record(self, record: dict, schema: Type[BaseModel]) -> Tuple[bool, Optional[BaseModel], str]:
        """
        Validate a single record against schema
        
        Returns: (is_valid, validated_data, error_message)
        """
        try:
            self.validation_stats['total_records'] += 1
            validated = schema(**record)
            self.validation_stats['valid_records'] += 1
            return True, validated, ""
        except ValidationError as e:
            self.validation_stats['invalid_records'] += 1
            error_msg = str(e)
            logger.warning(f"Validation failed for {schema.__name__}: {error_msg}")
            self.failed_records.append({
                'schema': schema.__name__,
                'record': record,
                'error': error_msg
            })
            return False, None, error_msg
    
    def validate_batch(self, records: List[dict], schema: Type[BaseModel]) -> Tuple[List[BaseModel], List[dict]]:
        """
        Validate a batch of records
        
        Returns: (valid_records, failed_records)
        """
        valid = []
        failed = []
        
        for record in records:
            is_valid, validated, error = self.validate_record(record, schema)
            if is_valid:
                valid.append(validated)
            else:
                failed.append({'record': record, 'error': error})
        
        return valid, failed
    
    def get_stats(self) -> dict:
        """Get validation statistics"""
        total = self.validation_stats['total_records']
        if total > 0:
            valid_pct = (self.validation_stats['valid_records'] / total) * 100
        else:
            valid_pct = 0
        
        return {
            **self.validation_stats,
            'valid_percentage': valid_pct,
            'failed_records_count': len(self.failed_records)
        }
    
    def get_failed_records(self) -> List[dict]:
        """Get list of failed records for audit trail"""
        return self.failed_records
    
    def clear_failed_records(self):
        """Clear failed records log"""
        self.failed_records = []
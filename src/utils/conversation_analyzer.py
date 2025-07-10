"""
Conversation History Analyzer - Analyzes AI messages in conversation JSON files
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass
class MessageAnalysis:
    """Analysis result for a single message"""
    message_id: int
    role: str
    content: str
    timestamp: Optional[str]
    detections: Dict[str, Dict[str, Any]]
    risk_level: str
    recommendations: List[str]
    

@dataclass
class ConversationAnalysisResult:
    """Complete analysis result for a conversation"""
    total_messages: int
    ai_messages_analyzed: int
    user_messages: int
    high_risk_messages: int
    medium_risk_messages: int
    low_risk_messages: int
    message_analyses: List[MessageAnalysis]
    summary: Dict[str, Any]
    generated_at: str


@dataclass
class SimplifiedAnalysisResult:
    """Simplified analysis result for quick overview"""
    conversation: List[Dict[str, Any]]
    summary: Dict[str, Any]
    generated_at: str


class ConversationAnalyzer:
    """Analyzes conversation history for eldercare communication patterns"""
    
    def __init__(self, optimized_prompts_path: Optional[str] = None):
        """Initialize analyzer with detection pipeline"""
        # Lazy import to avoid circular dependency
        from ..workflow.detection_pipeline import DetectionPipeline
        
        self.pipeline = DetectionPipeline(optimized_prompts_path)
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
        
    def validate_conversation_json(self, data: Any) -> bool:
        """Validate conversation JSON structure"""
        if not isinstance(data, list):
            raise ValueError("JSON must be a list of messages")
            
        for i, msg in enumerate(data):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dictionary")
                
            if 'role' not in msg:
                raise ValueError(f"Message {i} missing 'role' field")
                
            if msg['role'] not in ['user', 'ai', 'assistant', 'system']:
                raise ValueError(f"Message {i} has invalid role: {msg['role']}")
                
            if 'content' not in msg and 'message' not in msg:
                raise ValueError(f"Message {i} missing content/message field")
                
        return True
        
    def analyze_conversation(self, conversation_data: List[Dict[str, Any]]) -> ConversationAnalysisResult:
        """Analyze a conversation history"""
        # Validate data
        self.validate_conversation_json(conversation_data)
        
        # Normalize messages
        messages = self._normalize_messages(conversation_data)
        
        # Filter AI messages
        ai_messages = [msg for msg in messages if msg['role'] in ['ai', 'assistant']]
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        
        # Analyze AI messages in parallel
        message_analyses = self._analyze_messages_parallel(ai_messages)
        
        # Count risk levels
        risk_counts = {
            'high': sum(1 for ma in message_analyses if ma.risk_level == 'high'),
            'medium': sum(1 for ma in message_analyses if ma.risk_level == 'medium'),
            'low': sum(1 for ma in message_analyses if ma.risk_level == 'low')
        }
        
        # Generate summary
        summary = self._generate_summary(message_analyses)
        
        return ConversationAnalysisResult(
            total_messages=len(messages),
            ai_messages_analyzed=len(ai_messages),
            user_messages=len(user_messages),
            high_risk_messages=risk_counts['high'],
            medium_risk_messages=risk_counts['medium'],
            low_risk_messages=risk_counts['low'],
            message_analyses=message_analyses,
            summary=summary,
            generated_at=datetime.now().isoformat()
        )
        
    def analyze_conversation_simplified(self, conversation_data: List[Dict[str, Any]]) -> SimplifiedAnalysisResult:
        """Generate simplified analysis result"""
        # Run full analysis first
        full_result = self.analyze_conversation(conversation_data)
        
        # Build simplified conversation view
        simplified_conv = []
        
        # Map message analyses by ID for quick lookup
        analysis_map = {ma.message_id: ma for ma in full_result.message_analyses}
        
        # Process all messages in order
        messages = self._normalize_messages(conversation_data)
        
        for msg in messages:
            if msg['role'] == 'user':
                simplified_conv.append({
                    'role': 'user',
                    'content': msg['content'][:100] + '...' if len(msg['content']) > 100 else msg['content']
                })
            elif msg['role'] in ['ai', 'assistant']:
                # Get analysis for this message
                analysis = analysis_map.get(msg['id'])
                
                if analysis:
                    # Extract detections as simple 0/1
                    detections = {}
                    for detector_name, detection in analysis.detections.items():
                        if 'error' not in detection:
                            detections[detector_name] = detection.get('prediction', '0')
                    
                    simplified_conv.append({
                        'role': 'ai',
                        'content': msg['content'][:100] + '...' if len(msg['content']) > 100 else msg['content'],
                        'detections': detections,
                        'risk': analysis.risk_level
                    })
        
        # Create summary
        summary = {
            'total_issues': full_result.high_risk_messages + full_result.medium_risk_messages,
            'high_risk': full_result.high_risk_messages,
            'medium_risk': full_result.medium_risk_messages,
            'most_common_issue': full_result.summary['most_common_issues'][0] if full_result.summary['most_common_issues'] else None
        }
        
        return SimplifiedAnalysisResult(
            conversation=simplified_conv,
            summary=summary,
            generated_at=datetime.now().isoformat()
        )
        
    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize message format"""
        normalized = []
        for i, msg in enumerate(messages):
            normalized.append({
                'id': i,
                'role': msg.get('role', 'unknown'),
                'content': msg.get('content') or msg.get('message', ''),
                'timestamp': msg.get('timestamp', None)
            })
        return normalized
        
    def _analyze_messages_parallel(self, messages: List[Dict[str, Any]]) -> List[MessageAnalysis]:
        """Analyze multiple messages in parallel"""
        analyses = []
        
        # Use thread pool for parallel API calls
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for msg in messages:
                future = executor.submit(self._analyze_single_message, msg)
                futures.append((msg, future))
                
            for msg, future in futures:
                try:
                    analysis = future.result(timeout=30)
                    analyses.append(analysis)
                except Exception as e:
                    self.logger.error(f"Error analyzing message {msg['id']}: {e}")
                    # Add error analysis
                    analyses.append(MessageAnalysis(
                        message_id=msg['id'],
                        role=msg['role'],
                        content=msg['content'],
                        timestamp=msg.get('timestamp'),
                        detections={'error': {'error': str(e)}},
                        risk_level='unknown',
                        recommendations=["Error occurred during analysis"]
                    ))
                    
        return analyses
        
    def _analyze_single_message(self, message: Dict[str, Any]) -> MessageAnalysis:
        """Analyze a single message"""
        # Run detection pipeline
        result = self.pipeline.analyze(message['content'])
        
        return MessageAnalysis(
            message_id=message['id'],
            role=message['role'],
            content=message['content'],
            timestamp=message.get('timestamp'),
            detections=result.detections,
            risk_level=result.risk_level,
            recommendations=result.recommendations
        )
        
    def _generate_summary(self, analyses: List[MessageAnalysis]) -> Dict[str, Any]:
        """Generate summary statistics"""
        detector_stats = {}
        
        for analysis in analyses:
            for detector_name, detection in analysis.detections.items():
                if 'error' not in detection:
                    if detector_name not in detector_stats:
                        detector_stats[detector_name] = {
                            'harmful_count': 0,
                            'safe_count': 0,
                            'total': 0
                        }
                    
                    detector_stats[detector_name]['total'] += 1
                    if detection.get('prediction') == '1':
                        detector_stats[detector_name]['harmful_count'] += 1
                    else:
                        detector_stats[detector_name]['safe_count'] += 1
                        
        return {
            'detector_statistics': detector_stats,
            'most_common_issues': self._get_most_common_issues(analyses),
            'recommendations_summary': self._get_recommendations_summary(analyses)
        }
        
    def _get_most_common_issues(self, analyses: List[MessageAnalysis]) -> List[str]:
        """Get most common detected issues"""
        issue_counts = {}
        
        for analysis in analyses:
            for detector_name, detection in analysis.detections.items():
                if 'error' not in detection and detection.get('prediction') == '1':
                    issue_counts[detector_name] = issue_counts.get(detector_name, 0) + 1
                    
        # Sort by count and return top 3
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [name for name, _ in sorted_issues[:3]]
        
    def _get_recommendations_summary(self, analyses: List[MessageAnalysis]) -> Dict[str, int]:
        """Get summary of recommendations"""
        rec_counts = {}
        
        for analysis in analyses:
            for rec in analysis.recommendations:
                rec_counts[rec] = rec_counts.get(rec, 0) + 1
                
        return rec_counts
        
    def save_analysis_result(self, result: ConversationAnalysisResult, output_path: str):
        """Save analysis result to JSON file"""
        # Convert to dict
        result_dict = asdict(result)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Analysis result saved to {output_path}")
        
    def save_simplified_result(self, result: SimplifiedAnalysisResult, output_path: str):
        """Save simplified analysis result to JSON file"""
        # Convert to dict
        result_dict = asdict(result)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Simplified analysis result saved to {output_path}")
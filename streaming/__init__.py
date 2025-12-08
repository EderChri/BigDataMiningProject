"""
Streaming algorithms and detectors for real-time message analysis.

This package provides:
- algorithms: Count-Min Sketch, DGIM, Bloom Filter
- detectors: frequency, burst, duplicate detection
- streaming_pipeline: Orchestration for processing message streams
- message_processor: Structured processing for text messages
"""

from .streaming_pipeline import StreamingPipeline
from .message_processor import MessageProcessor

__all__ = ["StreamingPipeline", "MessageProcessor"]

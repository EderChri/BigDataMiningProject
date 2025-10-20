"""
Streaming algorithms and detectors for real-time message analysis.

This package provides:
- algorithms: Count-Min Sketch, DGIM, Bloom Filter
- detectors: frequency, burst, duplicate detection
- streaming_pipeline: Orchestration for processing message streams
"""

from .streaming_pipeline import StreamingPipeline

__all__ = ["StreamingPipeline"]

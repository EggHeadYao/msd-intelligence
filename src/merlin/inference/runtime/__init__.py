"""Validated online inference assembly and execution."""

from .factory import load_inference_pipeline
from .pipeline import MerlinPipeline

__all__ = ["MerlinPipeline", "load_inference_pipeline"]

"""MRO-SI training utilities.

Heavy training dependencies such as torch, TRL, and vLLM are imported only by
the training entrypoint. This keeps data-preparation utilities lightweight.
"""

__all__ = []

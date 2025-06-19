# errors.py
class PipelineError(Exception):
    """Base class for all pipeline-related errors."""

class COLMAPError(PipelineError):
    def __init__(self, message="COLMAP Stage failed."):
        super().__init__(message)

class Open3DError(PipelineError):
    def __init__(self, message="Meshing Stage failed."):
        super().__init__(message)

class NeRFError(PipelineError):
    def __init__(self, message="Meshing Stage failed."):
        super().__init__(message)

class CPPError(PipelineError):
    def __init__(self, message="Meshing Stage failed."):
        super().__init__(message)
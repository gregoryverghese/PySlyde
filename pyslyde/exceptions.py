"""
Custom PySlyde exceptions.
"""

from typing import List


class StitchingMissingPatches(Exception):
    """Exception raised when patches are missing during stitching."""
    
    def __init__(self, patch_names: List[str], message: str = "patches missing") -> None:
        """
        Initialize the exception.
        
        Args:
            patch_names: List of missing patch names.
            message: Custom error message.
        """
        self.patch_names = patch_names
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return string representation of the exception."""
        num_missing = len(self.patch_names)
        return f'{num_missing} -> {self.message}'




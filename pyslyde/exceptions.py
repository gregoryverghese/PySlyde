"""
Custom PySlyde exceptions.
"""

from typing import List


class StitchingMissingPatches(Exception):
    """Exception raised when patches are missing during stitching."""

    def __init__(
        self, patch_names: List[str], message: str = "patches missing"
    ) -> None:
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
        return f"{num_missing} -> {self.message}"


class InvalidRoundingPolicyError(ValueError):
    """
    Raised when an invalid rounding policy is requested.

    This is used by util.utilities.round_dim(...) and any API that accepts the
    `rounding` argument.
    """

    def __init__(
        self, rounding: str, *, allowed: tuple[str, ...] = ("round", "floor", "ceil")
    ):
        self.rounding = rounding
        self.allowed = allowed
        super().__init__(f"Invalid rounding policy: {rounding!r}. Allowed: {allowed}.")

"""
First-run onboarding wizard base class.
Full implementation is net-new per product (v0.1+ for snipe, etc.)
"""
from __future__ import annotations


class BaseWizard:
    """
    Base class for CircuitForge first-run wizards.
    Subclass and implement run() in each product.
    """

    def run(self) -> None:
        """Execute the onboarding wizard flow. Must be overridden by subclass."""
        raise NotImplementedError(
            "BaseWizard.run() must be implemented by a product-specific subclass."
        )

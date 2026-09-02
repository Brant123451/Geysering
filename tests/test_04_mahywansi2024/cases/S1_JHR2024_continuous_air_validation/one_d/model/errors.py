"""Errors raised when a scientific or numerical contract is not satisfied."""


class ContractViolation(ValueError):
    """A state or flux violates a declared model contract."""


class MissingPhysicalClosure(RuntimeError):
    """A requested advance has no evidence-backed physical closure."""


class ConservationError(ContractViolation):
    """An atomic packet does not close its conservation ledger."""


class AtomicCommitError(ContractViolation):
    """An atomic packet is stale, duplicated, or otherwise uncommittable."""

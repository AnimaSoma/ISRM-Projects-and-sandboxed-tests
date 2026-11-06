"""Independent loaders for the external data referenced by frozen specs.

Each loader verifies the archive against the SHA-256 recorded in the spec before
reading it and is written from the archive's own documentation and the written
loader rules in reports/REIMPLEMENTATION_SPECS.md, never from the original
project's code (which the import guard refuses in any case).
"""

__version__ = "0.14.0"

try:
    from circuitforge_core.community import CommunityDB, CommunityPost, SharedStore
    __all__ = ["CommunityDB", "CommunityPost", "SharedStore"]
except ImportError:
    # psycopg2 not installed — install with: pip install circuitforge-core[community]
    pass

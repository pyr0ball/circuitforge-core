# circuitforge_core/community/models.py
# MIT License

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

PostType = Literal["plan", "recipe_success", "recipe_blooper"]
CreativityLevel = Literal[1, 2, 3, 4]

_VALID_POST_TYPES: frozenset[str] = frozenset(["plan", "recipe_success", "recipe_blooper"])


def _validate_score(name: str, value: float) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value!r}")
    return value


@dataclass(frozen=True)
class CommunityPost:
    """Immutable snapshot of a published community post.

    Lists (dietary_tags, allergen_flags, flavor_molecules, slots) are stored as
    tuples to enforce immutability. Pass lists -- they are converted in __post_init__.
    """

    # Identity
    slug: str
    pseudonym: str
    post_type: PostType
    published: datetime
    title: str

    # Optional content
    description: str | None
    photo_url: str | None

    # Plan slots -- list[dict] for post_type="plan"
    slots: tuple

    # Recipe result fields -- for post_type="recipe_success" | "recipe_blooper"
    recipe_id: int | None
    recipe_name: str | None
    level: CreativityLevel | None
    outcome_notes: str | None

    # Element snapshot
    seasoning_score: float
    richness_score: float
    brightness_score: float
    depth_score: float
    aroma_score: float
    structure_score: float
    texture_profile: str

    # Dietary/allergen/flavor
    dietary_tags: tuple
    allergen_flags: tuple
    flavor_molecules: tuple

    # USDA FDC macros (optional -- may not be available for all recipes)
    fat_pct: float | None
    protein_pct: float | None
    moisture_pct: float | None

    def __init__(self, **kwargs):
        # 1. Coerce list fields to tuples
        for key in ("slots", "dietary_tags", "allergen_flags", "flavor_molecules"):
            if key in kwargs and isinstance(kwargs[key], list):
                kwargs[key] = tuple(kwargs[key])

        # 2. Validate BEFORE assignment
        post_type = kwargs.get("post_type")
        if post_type not in _VALID_POST_TYPES:
            raise ValueError(
                f"post_type must be one of {sorted(_VALID_POST_TYPES)}, got {post_type!r}"
            )
        for score_name in (
            "seasoning_score", "richness_score", "brightness_score",
            "depth_score", "aroma_score", "structure_score",
        ):
            if score_name in kwargs:
                _validate_score(score_name, kwargs[score_name])

        # 3. Assign all fields
        for f in self.__dataclass_fields__:
            object.__setattr__(self, f, kwargs[f])

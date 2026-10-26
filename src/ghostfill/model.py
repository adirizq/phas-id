from typing import List

from pydantic import BaseModel, Field


class ClaimEvidence(BaseModel):
    claim: str = Field(
        description="A specific, concrete claim about what makes this author distinctive in this dimension"
    )
    evidence: str = Field(
        description="A direct quote or specific example from the author's text that supports the claim"
    )


class WritingStyleSheet(BaseModel):
    plot: List[ClaimEvidence] = Field(
        description=(
            "2-10 Claim-Evidence pairs about how the author structures their writing: "
            "how they open and close posts, introduce and develop the central argument or topic, "
            "organize the logical flow, and use transitions"
        )
    )
    creativity: List[ClaimEvidence] = Field(
        description=(
            "2-10 Claim-Evidence pairs about the author's unique creative elements: "
            "unconventional reinterpretation of topics, blending of genres or registers, "
            "surprising content choices, humor, and imaginative angles"
        )
    )
    development: List[ClaimEvidence] = Field(
        description=(
            "2-10 Claim-Evidence pairs about how the author develops and elaborates on ideas: "
            "use of examples, personal anecdotes, data, analogies, emotional arcs, "
            "and the level of detail and depth applied to each point"
        )
    )
    language_use: List[ClaimEvidence] = Field(
        description=(
            "2-10 Claim-Evidence pairs about the author's distinctive vocabulary, tone, "
            "sentence structure, rhetorical devices, figures of speech, pacing, and narrative voice"
        )
    )


class BasePost(BaseModel):
    topic_phrase: str = Field(
        description="A short 5-10 word description of what this blog post is about"
    )
    post: str = Field(description="The generated generic blog post text")


class PersonaProfile(BaseModel):
    description: str = Field(
        description=(
            "A ~300-word second-person ('You...') narrative persona description organized into "
            "four sections: Structure, Creativity, Development, and Language Use. "
            "Written as flowing prose, NOT bullet points."
        )
    )


class FillRules(BaseModel):
    plot: str = Field(
        description="Actionable second-person instructions for plot/structure and organization in this document"
    )
    creativity: str = Field(
        description="Actionable second-person instructions for creative choices in this document"
    )
    development: str = Field(
        description="Actionable second-person instructions for idea development (character, setting, elaboration) in this document"
    )
    language_use: str = Field(
        description=(
            "Actionable second-person instructions for language, tone, and style in this document, "
            "with verbatim style examples from the author's writing"
        )
    )


class FilledDocument(BaseModel):
    text: str = Field(
        description="The complete, naturally flowing blog post with all [MASK] tokens filled in."
    )


class BestFillSelection(BaseModel):
    best_index: int = Field(description="1-based index of the best fill version")

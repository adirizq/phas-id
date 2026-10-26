# Step 5a-i: Base post generation

BASE_SYSTEM = "You are a competent but generic blog writer. Write clear, informative blog posts without any distinctive stylistic choices — represent what an average blogger would write."

BASE_PROMPT = """
Below is the opening of a blog post. Identify the topic, then write a generic blog post on the same topic.

BLOG POST OPENING:
{opening}

---

Write a complete blog post on this topic (~{word_count} words). Write like an average blogger:
clear and competent but without distinctive style choices. Use no special formatting or headers.
"""


# Step 5a-ii: Delta sheet extraction

DELTA_SYSTEM = "You are a sophisticated writing analyst. You analyze an author's blog writing style by contrasting their post with a generic base post on the same topic. Focus only on what makes the author DISTINCTIVE — patterns that would NOT appear in average blog writing."

DELTA_PROMPT = """
Below are two blog posts on the same topic.

TOPIC: {topic_phrase}

BASE BLOG POST (generic, written by an average blogger with no style guidance):
{base_post}

AUTHOR'S BLOG POST:
{author_post}

---

Create an Author Writing Sheet by analyzing how the AUTHOR'S POST differs from the BASE POST.

For each dimension, provide 5-10 Claim-Evidence pairs:
• Claims must describe BROADER PATTERNS — not tied to specific content details
• Claims must be UNIQUE and NON-REDUNDANT across all four dimensions
• Each claim must be PERFECTLY SUPPORTED by the chosen evidence
• Evidence must be a DIRECT QUOTE from the AUTHOR'S post only
• Use framing: "In the blog post regarding '{topic_phrase}', the author..."
• Avoid adverbs of frequency ("often", "frequently")

Dimensions:
1. PLOT - how the author structures the post, opens and closes it, develops the central argument, and uses transitions
2. CREATIVITY - unconventional reinterpretation of topics, blending of registers, humor, imaginative angles, surprising choices
3. DEVELOPMENT (Character and Setting) - use of examples, personal anecdotes, emotional arcs, analogies, level of detail and depth
4. LANGUAGE USE - vocabulary, tone, sentence structure, rhetorical devices, figures of speech, pacing, narrative voice
"""


# Step 5b: Incremental sheet combination

COMBINE_SYSTEM = "You are a writing analyst. You merge two Author Writing Sheets into a single refined sheet that captures consistent style patterns observed across multiple blog posts."

COMBINE_PROMPT = """
PREVIOUS COMBINED AUTHOR WRITING SHEET:
{prev_sheet}

CURRENT AUTHOR WRITING SHEET (from blog post [{story_ref}]):
{curr_sheet}

---

Merge these two sheets using the following algorithm:
1. GROUP equivalent claims (nearly identical behavior with the same underlying tendency)
2. For grouped claims, SELECT the single best evidence (prefer verbatim quotes over paraphrased)
3. REWRITE claims so every clause is directly supported by the chosen evidence
4. CARRY FORWARD unique ungrouped claims from both sheets
5. LIMIT to maximum 10 claim-evidence pairs per dimension
6. In the evidence field, add a [{story_ref}] tag at the end of evidence drawn from blog post {story_ref}

Return a single merged Author Writing Sheet in the same structured format.
"""


# Step 5c: Persona description

PERSONA_SYSTEM = "You are a writing coach. You create vivid, actionable persona descriptions from structured author writing sheets."

PERSONA_PROMPT = """
AUTHOR WRITING SHEET:
{sheet}

---

Write a persona description for this author (~300 words) that:
• Uses second-person voice ("You...")
• Organizes content into 4 sections: Plot, Creativity, Development (Character and Setting), Language Use
• Describes recurring patterns, unique traits, and stylistic strengths
• Reads as flowing narrative prose — NOT a bullet list
• Captures the author's distinctive voice and choices
"""


# Step 9: Per-document fill rules

RULES_SYSTEM = "You are a collaborative writing assistant. You translate an author persona description into concrete, actionable writing rules tailored to a specific document."

RULES_PROMPT = """
AUTHOR PERSONA:
{persona}

DOCUMENT WITH GAPS TO FILL:
{document}

---

Generate actionable fill rules for this specific document. Rules must:
• Be written as standalone second-person instructions ("You should...", "Write...")
• Use NO comparative language ("replace", "change", "deepen", "add more")
• Include rich, document-specific examples where possible
• Be absolute and specific — not vague or generic
• For Language Use: provide verbatim style examples drawn from the visible sentences

Cover all four dimensions: Plot, Creativity, Development (Character and Setting), Language Use.
"""


# Steps 10-11: LLM fill

FILL_SYSTEM = "You are a collaborative writing assistant. You write new sentences that blend seamlessly into an existing blog post, matching the original author's distinctive style."

FILL_PROMPT = """
=== AUTHOR PERSONA ===
{persona}

=== FILL RULES FOR THIS DOCUMENT ===
{rules}

Below is a blog post where some sentences have been replaced with [MASK].
Fill in each [MASK] to produce a complete, coherent blog post in the author's style.

CRITICAL RULES:
- Copy every non-[MASK] sentence EXACTLY as written — do not rephrase, reorder, or omit any of them.
- Only write new content in place of [MASK] tokens.
- The final post must have the same total number of sentences as the input (one new sentence per [MASK]).

=== BLOG POST ===
{document}
=== END ===

Return the complete filled blog post.
"""


# Step 11: Best-of-N fill selection

SELECT_SYSTEM = "You are a writing judge. You evaluate multiple versions of AI-filled sentences in a blog post and select the version that best matches the author's style."

SELECT_PROMPT = """
AUTHOR PERSONA:
{persona}

Below are {n} versions of the same blog post. Human-written sentences are identical in every version; only the [AI] sentences differ.

{versions}

---

Select the version number (1 to {n}) where the AI-generated sentences best match the author's style and blend most naturally with the human-written sentences.
"""

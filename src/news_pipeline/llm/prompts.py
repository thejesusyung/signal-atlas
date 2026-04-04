from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from jinja2 import Template


@dataclass(frozen=True, slots=True)
class PromptSpec:
    name: str
    version: str
    system_prompt: str
    user_prompt_template: str

    def render(self, **kwargs: Any) -> tuple[str, str]:
        return self.system_prompt, Template(self.user_prompt_template).render(**kwargs)


ENTITY_EXTRACTION_PROMPT = PromptSpec(
    name="entity_extraction",
    version="entity_v3",
    system_prompt=(
        "You extract named entities from news articles. "
        "Return JSON only. Never include prose."
    ),
    user_prompt_template=(
        """
Extract named entities from the article below.

Return JSON with this shape:
{
  "entities": [
    {
      "name": "Elon Musk",
      "type": "person",
      "role": "mentioned",
      "confidence": 0.96
    }
  ]
}

Allowed type values: person, company, organization, location, product.
Allowed role values: subject, mentioned, quoted.
The article text may be a bounded excerpt of a longer article. Extract only what is supported by the provided text.
If a mention does not fit one of the allowed type values, omit it.

Example:
Article title: "Apple unveils new Mac lineup"
Article text: "Apple announced new Mac products in Cupertino."
Expected output:
{"entities":[{"name":"Apple","type":"company","role":"subject","confidence":0.98},{"name":"Cupertino","type":"location","role":"mentioned","confidence":0.73},{"name":"Mac","type":"product","role":"mentioned","confidence":0.82}]}

Article title: {{ title }}
Article text:
{{ article_text }}
        """.strip()
    ),
)

TOPIC_CLASSIFICATION_PROMPT = PromptSpec(
    name="topic_classification",
    version="topic_v2",
    system_prompt=(
        "You classify news articles into topics. "
        "Return JSON only. Prefer the most relevant 1 to 3 labels."
    ),
    user_prompt_template=(
        """
Classify the article into one to three topics from this fixed list:
{{ topic_labels|join(", ") }}

Return JSON with this shape:
{
  "topics": [
    {"topic_name": "technology", "confidence": 0.95},
    {"topic_name": "business", "confidence": 0.58}
  ]
}

The article text may be a bounded excerpt of a longer article. Classify only from the provided text.

Article title: {{ title }}
Article text:
{{ article_text }}
        """.strip()
    ),
)

JSON_REPAIR_PROMPT = PromptSpec(
    name="json_repair",
    version="json_fix_v1",
    system_prompt="You repair invalid JSON. Return valid JSON only.",
    user_prompt_template=(
        """
The following model output should be JSON but is invalid.
Convert it to valid JSON while preserving the intended structure.

Broken output:
{{ broken_output }}
        """.strip()
    ),
)


def parse_json_payload(raw_text: str) -> dict[str, Any]:
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    return payload

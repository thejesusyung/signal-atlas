from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from news_pipeline.contracts import EntityRecord
from news_pipeline.db.models import (
    ArticleEntity,
    Entity,
    EntityType,
    ExtractionRun,
    ExtractionRunType,
    RawArticle,
)
from news_pipeline.extraction.errors import ExtractionStepError
from news_pipeline.llm.prompts import ENTITY_EXTRACTION_PROMPT, JSON_REPAIR_PROMPT, PromptSpec, parse_json_payload
from news_pipeline.tracking.prompt_registry import get_prompt_template
from news_pipeline.llm.provider import LLMProvider, LLMTraceContext
from news_pipeline.utils import (
    DEFAULT_LLM_ARTICLE_TEXT_CHARS,
    DEFAULT_LLM_SUMMARY_TEXT_CHARS,
    choose_article_text,
    normalize_entity_name,
    text_contains_entity,
)

ENTITY_TYPE_ALIASES = {
    "corp": "company",
    "corporation": "company",
    "business": "company",
    "org": "organization",
    "institution": "organization",
    "place": "location",
    "country": "location",
    "city": "location",
    "region": "location",
    "product_name": "product",
}
MIN_ENTITY_CONFIDENCE = 0.01


class EntityExtractor:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self._prompt = PromptSpec(
            name=ENTITY_EXTRACTION_PROMPT.name,
            version=ENTITY_EXTRACTION_PROMPT.version,
            system_prompt=ENTITY_EXTRACTION_PROMPT.system_prompt,
            user_prompt_template=get_prompt_template("entity_extraction", ENTITY_EXTRACTION_PROMPT),
        )

    def extract_for_article(self, session: Session, article: RawArticle) -> list[EntityRecord]:
        article_text = choose_article_text(
            article.full_text,
            article.summary,
            article.title,
            cleaned_text=article.cleaned_text,
            max_chars=DEFAULT_LLM_ARTICLE_TEXT_CHARS,
            summary_max_chars=DEFAULT_LLM_SUMMARY_TEXT_CHARS,
        )
        system_prompt, prompt = self._prompt.render(
            title=article.title,
            article_text=article_text,
        )

        response = None
        success = False
        error_message = None
        records: list[EntityRecord] = []
        primary_trace_context = LLMTraceContext(
            operation="entity_extraction",
            article_id=str(article.id),
            prompt_version=self._prompt.version,
            article_title=article.title,
        )

        try:
            response = self.provider.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                trace_context=primary_trace_context,
                temperature=0.0,
            )
            records = self._parse_entities(response.text)
            records = self._filter_entities(records, article_text)
            self._upsert_entities(session, article, records)
            success = True
        except Exception as error:
            error_message = str(error)
            if response is not None:
                try:
                    repaired = self._repair_json(
                        response.text,
                        article_id=str(article.id),
                        article_title=article.title,
                    )
                    records = self._parse_entities(repaired.text)
                    records = self._filter_entities(records, article_text)
                    self._upsert_entities(session, article, records)
                    success = True
                    error_message = None
                    response = repaired
                except Exception as repair_error:
                    error_message = str(repair_error)

        if not success:
            raise ExtractionStepError(
                run_type=ExtractionRunType.entity,
                llm_provider=response.provider_name if response else self.provider.provider_name,
                model_name=response.model if response else getattr(self.provider, "model", "unknown"),
                prompt_version=self._prompt.version,
                tokens_used=response.tokens_used if response else 0,
                latency_ms=response.latency_ms if response else 0,
                error_message=error_message or "Entity extraction failed",
            )

        self._log_run(
            session=session,
            article=article,
            response=response,
            success=True,
            error_message=None,
            prompt_version=self._prompt.version,
        )
        return records

    def _repair_json(self, broken_output: str, article_id: str, article_title: str):
        system_prompt, prompt = JSON_REPAIR_PROMPT.render(broken_output=broken_output)
        return self.provider.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            trace_context=LLMTraceContext(
                operation="entity_json_repair",
                article_id=article_id,
                prompt_version=JSON_REPAIR_PROMPT.version,
                article_title=article_title,
            ),
        )

    @staticmethod
    def _parse_entities(raw_text: str) -> list[EntityRecord]:
        payload = parse_json_payload(raw_text)
        entities = payload.get("entities", [])
        if not isinstance(entities, list):
            raise ValueError("entities must be a list")

        parsed: list[EntityRecord] = []
        for item in entities:
            if not isinstance(item, dict):
                raise ValueError("entity item must be an object")
            entity_type = EntityExtractor._normalize_entity_type(item.get("type"))
            if entity_type is None:
                continue
            parsed.append(
                EntityRecord(
                    name=str(item["name"]).strip(),
                    entity_type=entity_type,
                    role=str(item.get("role", "mentioned")).strip(),
                    confidence=float(item.get("confidence", 0.0)),
                )
            )
        return parsed

    @staticmethod
    def _filter_entities(entities: list[EntityRecord], article_text: str) -> list[EntityRecord]:
        filtered: list[EntityRecord] = []
        for record in entities:
            if record.confidence < MIN_ENTITY_CONFIDENCE:
                continue
            if not text_contains_entity(record.name, article_text):
                continue
            filtered.append(record)
        return filtered

    @staticmethod
    def _normalize_entity_type(value: object) -> str | None:
        normalized = str(value or "").strip().lower()
        normalized = ENTITY_TYPE_ALIASES.get(normalized, normalized)
        if normalized in {member.value for member in EntityType}:
            return normalized
        return None

    @staticmethod
    def _upsert_entities(session: Session, article: RawArticle, entities: list[EntityRecord]) -> None:
        unique_entities = EntityExtractor._dedupe_entities(entities)
        existing_link_ids = {
            entity_id
            for entity_id, in session.execute(
                select(ArticleEntity.entity_id).where(ArticleEntity.article_id == article.id)
            )
        }

        desired_links: dict[UUID, EntityRecord] = {}
        current_link_ids: set[UUID] = set()

        for record in unique_entities:
            normalized_name = normalize_entity_name(record.name)
            entity_type = EntityType(record.entity_type)
            entity = EntityExtractor._get_or_create_entity(
                session=session,
                name=record.name,
                normalized_name=normalized_name,
                entity_type=entity_type,
            )

            if entity.id not in existing_link_ids and entity.id not in current_link_ids:
                entity.article_count += 1

            desired_links[entity.id] = record
            current_link_ids.add(entity.id)

        removed_link_ids = existing_link_ids - current_link_ids
        if removed_link_ids:
            session.execute(
                delete(ArticleEntity).where(
                    ArticleEntity.article_id == article.id,
                    ArticleEntity.entity_id.in_(removed_link_ids),
                )
            )
            for entity_id in removed_link_ids:
                entity = session.get(Entity, entity_id)
                if entity is not None and entity.article_count > 0:
                    entity.article_count -= 1

        for entity_id, record in desired_links.items():
            link = session.scalar(
                select(ArticleEntity).where(
                    ArticleEntity.article_id == article.id,
                    ArticleEntity.entity_id == entity_id,
                )
            )
            if link is None:
                session.add(
                    ArticleEntity(
                        article_id=article.id,
                        entity_id=entity_id,
                        role=record.role,
                        confidence=record.confidence,
                    )
                )
            else:
                link.role = record.role
                link.confidence = record.confidence

    @staticmethod
    def _dedupe_entities(entities: list[EntityRecord]) -> list[EntityRecord]:
        deduped: dict[tuple[str, str], EntityRecord] = {}
        for record in entities:
            key = (normalize_entity_name(record.name), str(record.entity_type).strip().lower())
            existing = deduped.get(key)
            if existing is None or record.confidence >= existing.confidence:
                deduped[key] = record
        return list(deduped.values())

    @staticmethod
    def _log_run(
        session: Session,
        article: RawArticle,
        response,
        success: bool,
        error_message: str | None,
        prompt_version: str = ENTITY_EXTRACTION_PROMPT.version,
    ) -> None:
        session.add(
            ExtractionRun(
                article_id=article.id,
                run_type=ExtractionRunType.entity,
                llm_provider=response.provider_name if response else "groq",
                model_name=response.model if response else "unknown",
                prompt_version=prompt_version,
                tokens_used=response.tokens_used if response else 0,
                latency_ms=response.latency_ms if response else 0,
                success=success,
                error_message=error_message,
            )
        )

    @staticmethod
    def _get_or_create_entity(
        session: Session,
        name: str,
        normalized_name: str,
        entity_type: EntityType,
    ) -> Entity:
        entity = session.scalar(
            select(Entity).where(
                Entity.normalized_name == normalized_name,
                Entity.entity_type == entity_type,
            )
        )
        if entity is not None:
            return entity

        try:
            with session.begin_nested():
                entity = Entity(
                    name=name,
                    entity_type=entity_type,
                    normalized_name=normalized_name,
                    article_count=0,
                )
                session.add(entity)
                session.flush()
                return entity
        except IntegrityError:
            entity = session.scalar(
                select(Entity).where(
                    Entity.normalized_name == normalized_name,
                    Entity.entity_type == entity_type,
                )
            )
            if entity is not None:
                return entity
            raise

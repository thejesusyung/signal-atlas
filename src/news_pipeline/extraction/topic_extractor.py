from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from news_pipeline.config import get_settings
from news_pipeline.contracts import TopicAssignment
from news_pipeline.db.models import (
    ArticleTopic,
    ExtractionRun,
    ExtractionRunType,
    RawArticle,
    Topic,
    TopicMethod,
)
from news_pipeline.extraction.errors import ExtractionStepError
from news_pipeline.llm.prompts import JSON_REPAIR_PROMPT, TOPIC_CLASSIFICATION_PROMPT, parse_json_payload
from news_pipeline.llm.provider import LLMProvider, LLMTraceContext
from news_pipeline.utils import (
    DEFAULT_LLM_ARTICLE_TEXT_CHARS,
    DEFAULT_LLM_SUMMARY_TEXT_CHARS,
    choose_article_text,
)


class TopicExtractor:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.topic_labels = get_settings().topic_names

    def extract_for_article(self, session: Session, article: RawArticle) -> list[TopicAssignment]:
        article_text = choose_article_text(
            article.full_text,
            article.summary,
            article.title,
            cleaned_text=article.cleaned_text,
            max_chars=DEFAULT_LLM_ARTICLE_TEXT_CHARS,
            summary_max_chars=DEFAULT_LLM_SUMMARY_TEXT_CHARS,
        )
        system_prompt, prompt = TOPIC_CLASSIFICATION_PROMPT.render(
            title=article.title,
            article_text=article_text,
            topic_labels=self.topic_labels,
        )

        response = None
        success = False
        error_message = None
        assignments: list[TopicAssignment] = []
        primary_trace_context = LLMTraceContext(
            operation="topic_classification",
            article_id=str(article.id),
            prompt_version=TOPIC_CLASSIFICATION_PROMPT.version,
            article_title=article.title,
        )
        try:
            response = self.provider.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                trace_context=primary_trace_context,
            )
            assignments = self._parse_topics(response.text)
            self._upsert_topics(session, article, assignments)
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
                    assignments = self._parse_topics(repaired.text)
                    self._upsert_topics(session, article, assignments)
                    success = True
                    error_message = None
                    response = repaired
                except Exception as repair_error:
                    error_message = str(repair_error)

        if not success:
            raise ExtractionStepError(
                run_type=ExtractionRunType.topic,
                llm_provider=response.provider_name if response else self.provider.provider_name,
                model_name=response.model if response else getattr(self.provider, "model", "unknown"),
                prompt_version=TOPIC_CLASSIFICATION_PROMPT.version,
                tokens_used=response.tokens_used if response else 0,
                latency_ms=response.latency_ms if response else 0,
                error_message=error_message or "Topic extraction failed",
            )

        session.add(
            ExtractionRun(
                article_id=article.id,
                run_type=ExtractionRunType.topic,
                llm_provider=response.provider_name if response else self.provider.provider_name,
                model_name=response.model if response else getattr(self.provider, "model", "unknown"),
                prompt_version=TOPIC_CLASSIFICATION_PROMPT.version,
                tokens_used=response.tokens_used if response else 0,
                latency_ms=response.latency_ms if response else 0,
                success=True,
                error_message=None,
            )
        )
        return assignments

    def _repair_json(self, broken_output: str, article_id: str, article_title: str):
        system_prompt, prompt = JSON_REPAIR_PROMPT.render(broken_output=broken_output)
        return self.provider.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            trace_context=LLMTraceContext(
                operation="topic_json_repair",
                article_id=article_id,
                prompt_version=JSON_REPAIR_PROMPT.version,
                article_title=article_title,
            ),
        )

    def _parse_topics(self, raw_text: str) -> list[TopicAssignment]:
        payload = parse_json_payload(raw_text)
        topics = payload.get("topics", [])
        if not isinstance(topics, list):
            raise ValueError("topics must be a list")

        assignments: list[TopicAssignment] = []
        for item in topics:
            if not isinstance(item, dict):
                raise ValueError("topic item must be an object")
            topic_name = str(item["topic_name"]).strip().lower()
            if topic_name not in self.topic_labels:
                raise ValueError(f"Unsupported topic label: {topic_name}")
            assignments.append(
                TopicAssignment(
                    topic_name=topic_name,
                    confidence=float(item.get("confidence", 0.0)),
                )
            )
        return assignments

    @staticmethod
    def _upsert_topics(session: Session, article: RawArticle, assignments: list[TopicAssignment]) -> None:
        unique_assignments = TopicExtractor._dedupe_topics(assignments)
        existing_topic_ids = {
            topic_id
            for topic_id, in session.execute(
                select(ArticleTopic.topic_id).where(ArticleTopic.article_id == article.id)
            )
        }

        desired_topics: dict[UUID, TopicAssignment] = {}
        current_topic_ids: set[UUID] = set()

        for assignment in unique_assignments:
            topic = TopicExtractor._get_or_create_topic(session, assignment.topic_name)
            desired_topics[topic.id] = assignment
            current_topic_ids.add(topic.id)

        removed_topic_ids = existing_topic_ids - current_topic_ids
        if removed_topic_ids:
            session.execute(
                delete(ArticleTopic).where(
                    ArticleTopic.article_id == article.id,
                    ArticleTopic.topic_id.in_(removed_topic_ids),
                )
            )

        for topic_id, assignment in desired_topics.items():
            link = session.scalar(
                select(ArticleTopic).where(
                    ArticleTopic.article_id == article.id,
                    ArticleTopic.topic_id == topic_id,
                )
            )
            if link is None:
                session.add(
                    ArticleTopic(
                        article_id=article.id,
                        topic_id=topic_id,
                        confidence=assignment.confidence,
                        method=TopicMethod.llm,
                    )
                )
            else:
                link.confidence = assignment.confidence
                link.method = TopicMethod.llm

    @staticmethod
    def _dedupe_topics(assignments: list[TopicAssignment]) -> list[TopicAssignment]:
        deduped: dict[str, TopicAssignment] = {}
        for assignment in assignments:
            key = assignment.topic_name.strip().lower()
            existing = deduped.get(key)
            if existing is None or assignment.confidence >= existing.confidence:
                deduped[key] = assignment
        return list(deduped.values())

    @staticmethod
    def _get_or_create_topic(session: Session, topic_name: str) -> Topic:
        topic = session.scalar(select(Topic).where(Topic.name == topic_name))
        if topic is not None:
            return topic

        try:
            with session.begin_nested():
                topic = Topic(name=topic_name)
                session.add(topic)
                session.flush()
                return topic
        except IntegrityError:
            topic = session.scalar(select(Topic).where(Topic.name == topic_name))
            if topic is not None:
                return topic
            raise

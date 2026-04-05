"""Simulation DAG — Phase 5 (complete).

Tasks:
  seed_db              → idempotent: load writers.yaml + personas.yaml into DB
  fetch_stories        → query top signals/articles to use as story fuel
  create_cycle         → insert a SimCycle row and return cycle metadata
  prepare_tweet_inputs → build one input dict per (writer × story) combination
  generate_tweet       → fan-out LLM task: one tweet per input dict (llm_pool)
  evaluate_tweet       → fan-out LLM task: sample N personas, record reactions (llm_pool)
  score_and_mutate     → reduce: score writers, mutate bottom performers (llm_pool)
  log_to_mlflow        → parent + nested writer runs, tweet table, prompt registry
  check_weekly_reset   → tag weekly champion, log week summary on week boundary
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from airflow.decorators import dag, task

LOGGER = logging.getLogger(__name__)

_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dag(
    dag_id="simulation_dag",
    schedule="30 10 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"retries": 1},
    tags=["simulation"],
)
def build_simulation_dag():

    @task
    def seed_db() -> None:
        """Seed writers and personas from YAML if not already present."""
        from news_pipeline.config import get_settings
        from news_pipeline.db.session import session_scope
        from news_pipeline.simulation.seeder import seed_all

        settings = get_settings()
        with session_scope() as session:
            seed_all(
                session,
                writers_path=settings.sim_writers_config_path,
                personas_path=settings.sim_personas_config_path,
            )

    @task
    def fetch_stories() -> list[dict]:
        """Return up to SIM_CYCLE_TOP_STORIES story dicts from recent signals.

        Each dict has: article_id, title, summary, entities,
                       signal_type, signal_score.
        Falls back to the most recently extracted articles if no signals exist.
        """
        from sqlalchemy import desc, select

        from news_pipeline.config import get_settings
        from news_pipeline.db.models import (
            ProcessingStatus,
            RawArticle,
            Signal,
        )
        from news_pipeline.db.session import session_scope

        settings = get_settings()
        limit = settings.sim_cycle_top_stories
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=12)

        with session_scope() as session:
            signals = session.scalars(
                select(Signal)
                .where(Signal.detected_at >= cutoff)
                .order_by(desc(Signal.score))
                .limit(limit)
            ).all()

            stories: list[dict] = []
            seen_article_ids: set[str] = set()

            for signal in signals:
                article_ids: list = signal.article_ids or []
                for aid in article_ids:
                    aid_str = str(aid)
                    if aid_str in seen_article_ids:
                        continue
                    article = session.get(RawArticle, aid)
                    if article is None:
                        continue
                    seen_article_ids.add(aid_str)
                    stories.append(
                        _article_to_story(article, signal.signal_type, signal.score)
                    )
                    break  # one article per signal is enough

            if len(stories) < limit:
                # Pad with most-recently extracted articles not already included.
                extras = session.scalars(
                    select(RawArticle)
                    .where(RawArticle.processing_status == ProcessingStatus.extracted)
                    .order_by(desc(RawArticle.ingested_at))
                    .limit(limit * 3)
                ).all()
                for article in extras:
                    if len(stories) >= limit:
                        break
                    if str(article.id) in seen_article_ids:
                        continue
                    seen_article_ids.add(str(article.id))
                    stories.append(_article_to_story(article, "recent", 0.0))

        LOGGER.info("Fetched %d stories for simulation cycle", len(stories))
        return stories

    @task
    def create_cycle(stories: list[dict]) -> dict:
        """Insert a SimCycle row. Returns cycle metadata for downstream tasks."""
        from sqlalchemy import func, select

        from news_pipeline.db.session import session_scope
        from news_pipeline.simulation.models import SimCycle

        with session_scope() as session:
            max_cycle = session.scalar(select(func.max(SimCycle.cycle_number))) or 0
            cycle_number = max_cycle + 1
            week_number = (datetime.now(tz=timezone.utc) - _EPOCH).days // 7

            cycle = SimCycle(
                cycle_number=cycle_number,
                week_number=week_number,
                story_ids=[s["article_id"] for s in stories],
            )
            session.add(cycle)
            session.flush()
            cycle_id = str(cycle.id)

        LOGGER.info(
            "Created simulation cycle %d (week %d) with %d stories",
            cycle_number,
            week_number,
            len(stories),
        )
        return {
            "cycle_id": cycle_id,
            "cycle_number": cycle_number,
            "week_number": week_number,
            "story_count": len(stories),
        }

    @task
    def prepare_tweet_inputs(cycle: dict, stories: list[dict]) -> list[dict]:
        """Return one input dict per (writer × story) pair.

        Each dict contains everything generate_tweet needs so no DB reads are
        required at fan-out time other than the single LLM call itself.
        """
        import uuid
        from sqlalchemy import select

        from news_pipeline.db.session import session_scope
        from news_pipeline.simulation.models import SimPromptVersion, SimWriter

        with session_scope() as session:
            writers = session.scalars(select(SimWriter)).all()
            inputs = []
            for writer in writers:
                if writer.current_version_id is None:
                    LOGGER.warning("Writer %r has no current version, skipping", writer.name)
                    continue
                version = session.get(SimPromptVersion, writer.current_version_id)
                if version is None:
                    LOGGER.warning(
                        "Prompt version not found for writer %r, skipping", writer.name
                    )
                    continue
                for story in stories:
                    inputs.append(
                        {
                            "cycle_id": cycle["cycle_id"],
                            "writer_id": str(writer.id),
                            "writer_name": writer.name,
                            "prompt_version_id": str(version.id),
                            "prompt_version_number": version.version_number,
                            "style_prompt": version.style_prompt,
                            "story": story,
                        }
                    )

        LOGGER.info(
            "Prepared %d tweet inputs (%d writers × %d stories)",
            len(inputs),
            len(writers),
            len(stories),
        )
        return inputs

    @task(pool="llm_pool")
    def generate_tweet(tweet_input: dict) -> dict:
        """Generate and persist one tweet. Returns a result dict for downstream tasks."""
        import uuid

        from news_pipeline.llm.groq_client import GroqProvider
        from news_pipeline.db.session import session_scope
        from news_pipeline.simulation.models import SimTweet
        from news_pipeline.simulation.writer import TweetWriter

        cycle_id = tweet_input["cycle_id"]
        writer_id = tweet_input["writer_id"]
        writer_name = tweet_input["writer_name"]
        prompt_version_id = tweet_input["prompt_version_id"]
        prompt_version_number = tweet_input["prompt_version_number"]
        style_prompt = tweet_input["style_prompt"]
        story = tweet_input["story"]

        provider = GroqProvider()
        content = TweetWriter().generate(
            story=story,
            style_prompt=style_prompt,
            writer_name=writer_name,
            prompt_version=prompt_version_number,
            provider=provider,
        )

        with session_scope() as session:
            article_uuid = uuid.UUID(story["article_id"]) if story.get("article_id") else None
            tweet = SimTweet(
                cycle_id=uuid.UUID(cycle_id),
                writer_id=uuid.UUID(writer_id),
                prompt_version_id=uuid.UUID(prompt_version_id),
                article_id=article_uuid,
                content=content,
            )
            session.add(tweet)
            session.flush()
            tweet_id = str(tweet.id)

        LOGGER.info(
            "Writer %s generated tweet (%d chars) for %r",
            writer_name,
            len(content),
            story["title"][:60],
        )
        return {
            "tweet_id": tweet_id,
            "cycle_id": cycle_id,
            "writer_id": writer_id,
            "writer_name": writer_name,
            "prompt_version_id": prompt_version_id,
            "article_id": story["article_id"],
            "content": content,
        }

    @task(pool="llm_pool")
    def evaluate_tweet(tweet: dict) -> dict:
        """Sample SIM_PERSONAS_PER_TWEET personas and record their reaction to one tweet.

        LLM calls are made outside the DB session to avoid holding a connection
        open during inference. Two short sessions: one read (personas), one write
        (engagements).

        Returns the tweet dict augmented with aggregate engagement counts, ready
        for Phase 4 scoring.
        """
        import uuid
        from sqlalchemy import select
        from sqlalchemy import func as sqlfunc

        from news_pipeline.config import get_settings
        from news_pipeline.db.session import session_scope
        from news_pipeline.llm.groq_client import GroqProvider
        from news_pipeline.simulation.models import SimEngagement, SimPersona
        from news_pipeline.simulation.reader import PersonaReader

        settings = get_settings()
        tweet_id = tweet["tweet_id"]
        writer_name = tweet["writer_name"]
        content = tweet["content"]

        # ── 1. Fetch a random sample of personas (short read session) ──────────
        with session_scope() as session:
            rows = session.scalars(
                select(SimPersona)
                .order_by(sqlfunc.random())
                .limit(settings.sim_personas_per_tweet)
            ).all()
            # Materialise into plain dicts so we can close the session safely.
            persona_data = [
                {"id": str(p.id), "name": p.name, "description": p.description}
                for p in rows
            ]

        # ── 2. LLM calls — outside any DB session ─────────────────────────────
        provider = GroqProvider()
        reader = PersonaReader()

        repost_count = like_count = comment_count = skip_count = 0
        engagement_rows: list[dict] = []

        for persona in persona_data:
            result = reader.evaluate(
                tweet_content=content,
                persona_name=persona["name"],
                persona_description=persona["description"],
                provider=provider,
            )
            action = result["action"]
            engagement_rows.append(
                {
                    "persona_id": persona["id"],
                    "persona_name": persona["name"],
                    "action": action,
                    "reason": result["reason"],
                }
            )
            if action == "repost":
                repost_count += 1
            elif action == "like":
                like_count += 1
            elif action == "comment":
                comment_count += 1
            else:
                skip_count += 1

        # ── 3. Persist engagements (short write session) ───────────────────────
        with session_scope() as session:
            for row in engagement_rows:
                session.add(
                    SimEngagement(
                        tweet_id=uuid.UUID(tweet_id),
                        persona_id=uuid.UUID(row["persona_id"]),
                        action=row["action"],
                        reason=row["reason"],
                    )
                )

        LOGGER.info(
            "Evaluated tweet by %s — repost=%d like=%d comment=%d skip=%d",
            writer_name,
            repost_count,
            like_count,
            comment_count,
            skip_count,
        )
        return {
            **tweet,
            "repost_count": repost_count,
            "like_count": like_count,
            "comment_count": comment_count,
            "skip_count": skip_count,
            "readers_sampled": len(engagement_rows),
        }

    @task(pool="llm_pool")
    def score_and_mutate(evaluations: list[dict], cycle: dict) -> dict:
        """Reduce step: aggregate scores, persist them, mutate bottom performers.

        Receives the full list of evaluate_tweet outputs (one per tweet).
        Makes up to 2 LLM mutation calls, so sits in llm_pool.

        Returns a summary dict consumed by Phase 5 (MLflow logging).
        """
        import uuid
        from sqlalchemy import desc, select

        from news_pipeline.config import get_settings
        from news_pipeline.db.session import session_scope
        from news_pipeline.llm.groq_client import GroqProvider
        from news_pipeline.simulation.models import (
            SimPromptVersion,
            SimWriter,
            SimWriterCycleScore,
        )
        from news_pipeline.simulation.mutator import PromptMutator
        from news_pipeline.simulation.scorer import (
            aggregate_writer_scores,
            select_writers_to_mutate,
        )

        settings = get_settings()
        cycle_id = cycle["cycle_id"]
        cycle_number = cycle["cycle_number"]

        # ── 1. Aggregate tweet evaluations into per-writer scores ─────────────
        writer_scores = aggregate_writer_scores(evaluations)

        if not writer_scores:
            LOGGER.warning("No evaluations received for cycle %d — nothing to score", cycle_number)
            return {"cycle_id": cycle_id, "cycle_number": cycle_number,
                    "writer_scores": [], "mutations": []}

        sorted_scores = sorted(
            writer_scores.items(), key=lambda kv: kv[1]["engagement_score"], reverse=True
        )
        top_writer_id = sorted_scores[0][0]
        top_writer_data = sorted_scores[0][1]

        # ── 2. Persist SimWriterCycleScore rows ───────────────────────────────
        with session_scope() as session:
            for writer_id, data in writer_scores.items():
                session.add(
                    SimWriterCycleScore(
                        cycle_id=uuid.UUID(cycle_id),
                        writer_id=uuid.UUID(writer_id),
                        prompt_version_id=(
                            uuid.UUID(data["prompt_version_id"])
                            if data.get("prompt_version_id")
                            else None
                        ),
                        engagement_score=data["engagement_score"],
                        repost_count=data["repost_count"],
                        like_count=data["like_count"],
                        comment_count=data["comment_count"],
                        skip_count=data["skip_count"],
                        tweet_count=data["tweet_count"],
                        reader_sample_count=data["readers_sampled"],
                    )
                )

        LOGGER.info(
            "Cycle %d scores: %s",
            cycle_number,
            " | ".join(
                f"{d['writer_name']}={d['engagement_score']:.3f}"
                for _, d in sorted_scores
            ),
        )

        # ── 3. Fetch top performer's current prompt (for mutation context) ────
        with session_scope() as session:
            top_writer = session.get(SimWriter, uuid.UUID(top_writer_id))
            top_version = (
                session.get(SimPromptVersion, top_writer.current_version_id)
                if top_writer and top_writer.current_version_id
                else None
            )
            top_prompt = top_version.style_prompt if top_version else ""
            top_name = top_writer.name if top_writer else top_writer_data["writer_name"]

        # ── 4. Identify and mutate underperformers ────────────────────────────
        mutate_ids = select_writers_to_mutate(writer_scores, bottom_n=settings.sim_bottom_n_mutate)
        provider = GroqProvider()
        mutator = PromptMutator()
        mutations: list[dict] = []

        for writer_id in mutate_ids:
            data = writer_scores[writer_id]
            writer_name = data["writer_name"]

            # Load writer + recent score history
            with session_scope() as session:
                writer = session.get(SimWriter, uuid.UUID(writer_id))
                if writer is None or writer.current_version_id is None:
                    continue

                current_version = session.get(SimPromptVersion, writer.current_version_id)
                if current_version is None:
                    continue

                # Last 3 cycle scores for this writer (newest first → reverse for oldest-first)
                recent = session.scalars(
                    select(SimWriterCycleScore)
                    .where(SimWriterCycleScore.writer_id == writer.id)
                    .order_by(desc(SimWriterCycleScore.cycle_id))
                    .limit(3)
                ).all()
                recent_scores = [r.engagement_score for r in reversed(recent)]

                writer_data = {
                    "id": str(writer.id),
                    "name": writer.name,
                    "persona": writer.persona_description,
                    "current_prompt": current_version.style_prompt,
                    "current_version_number": current_version.version_number,
                    "current_version_id": str(current_version.id),
                }

            # Mutation LLM call — outside any DB session
            new_prompt = mutator.mutate(
                writer_name=writer_data["name"],
                persona_description=writer_data["persona"],
                current_prompt=writer_data["current_prompt"],
                recent_scores=recent_scores,
                top_performer_name=top_name,
                top_performer_prompt=top_prompt,
                provider=provider,
            )

            # Persist new prompt version + update writer pointer
            with session_scope() as session:
                new_version = SimPromptVersion(
                    writer_id=uuid.UUID(writer_data["id"]),
                    version_number=writer_data["current_version_number"] + 1,
                    style_prompt=new_prompt,
                    parent_id=uuid.UUID(writer_data["current_version_id"]),
                    cycle_introduced=cycle_number,
                    triggered_by_score=data["engagement_score"],
                )
                session.add(new_version)
                session.flush()
                new_version_id = str(new_version.id)
                new_version_number = new_version.version_number

                writer = session.get(SimWriter, uuid.UUID(writer_data["id"]))
                writer.current_version_id = new_version.id

            mutations.append(
                {
                    "writer_id": writer_id,
                    "writer_name": writer_name,
                    "old_version": writer_data["current_version_number"],
                    "new_version": new_version_number,
                    "new_version_id": new_version_id,
                    "triggered_by_score": data["engagement_score"],
                    "old_prompt": writer_data["current_prompt"],
                    "new_prompt": new_prompt,
                }
            )
            LOGGER.info(
                "Mutated %s: v%d → v%d (score was %.3f)",
                writer_name,
                writer_data["current_version_number"],
                new_version_number,
                data["engagement_score"],
            )

        return {
            "cycle_id": cycle_id,
            "cycle_number": cycle_number,
            "writer_scores": [
                {"writer_id": wid, **d} for wid, d in sorted_scores
            ],
            "mutations": mutations,
        }

    @task
    def log_to_mlflow(result: dict, cycle: dict) -> None:
        """Log cycle results to MLflow and update SimCycle with run_id + completed_at.

        Steps:
        1. Query DB: writer prompt versions (for version numbers + initial prompts).
        2. Query DB: tweets generated this cycle (for tweet table artifact).
        3. Enrich writer_scores with prompt_version_number.
        4. Call tracker.log_cycle_to_mlflow() — parent + nested writer runs.
        5. Register initial or mutated prompts in MLflow Prompt Registry.
        6. Update SimCycle.mlflow_run_id and completed_at.
        """
        import uuid
        from datetime import datetime, timezone
        from sqlalchemy import select

        from news_pipeline.config import get_settings
        from news_pipeline.db.session import session_scope
        from news_pipeline.simulation.models import (
            SimCycle,
            SimPromptVersion,
            SimTweet,
            SimWriter,
        )
        from news_pipeline.simulation.tracker import (
            log_cycle_to_mlflow,
            register_prompt,
        )

        settings = get_settings()
        cycle_id = result["cycle_id"]
        cycle_number = result["cycle_number"]
        week_number = cycle["week_number"]
        story_count = cycle["story_count"]

        # ── 1. Fetch writer prompt version numbers + style_prompts ────────────
        with session_scope() as session:
            writer_rows = session.execute(
                select(
                    SimWriter.id,
                    SimWriter.name,
                    SimPromptVersion.version_number,
                    SimPromptVersion.style_prompt,
                )
                .join(
                    SimPromptVersion,
                    SimWriter.current_version_id == SimPromptVersion.id,
                )
            ).all()
            writer_prompt_map = {
                str(row.id): {
                    "version_number": row.version_number,
                    "style_prompt": row.style_prompt,
                    "name": row.name,
                }
                for row in writer_rows
            }

        # ── 2. Fetch tweet content for this cycle ─────────────────────────────
        with session_scope() as session:
            tweet_rows_db = session.execute(
                select(SimTweet.content, SimWriter.name.label("writer_name"))
                .join(SimWriter, SimTweet.writer_id == SimWriter.id)
                .where(SimTweet.cycle_id == uuid.UUID(cycle_id))
                .order_by(SimWriter.name)
            ).all()
            tweet_rows = [
                {"writer_name": r.writer_name, "content": r.content}
                for r in tweet_rows_db
            ]

        # ── 3. Enrich writer_scores with prompt_version_number ────────────────
        writer_scores = result["writer_scores"]
        for ws in writer_scores:
            pm = writer_prompt_map.get(ws["writer_id"], {})
            ws["prompt_version_number"] = pm.get("version_number", 1)

        # ── 4. Log to MLflow ──────────────────────────────────────────────────
        run_id = log_cycle_to_mlflow(
            cycle_number=cycle_number,
            week_number=week_number,
            story_count=story_count,
            personas_per_tweet=settings.sim_personas_per_tweet,
            writer_scores=writer_scores,
            mutations=result["mutations"],
            tweet_rows=tweet_rows,
        )

        # ── 5. Register prompts in MLflow Prompt Registry ─────────────────────
        # Register the initial prompt (v1) for each writer on their first cycle.
        # We detect "first ever" by version_number == 1 AND no mutation produced them.
        mutated_writer_ids = {m["writer_id"] for m in result["mutations"]}
        for ws in writer_scores:
            pm = writer_prompt_map.get(ws["writer_id"], {})
            if pm.get("version_number") == 1 and ws["writer_id"] not in mutated_writer_ids:
                register_prompt(
                    writer_name=ws["writer_name"],
                    style_prompt=pm["style_prompt"],
                    version_number=1,
                    cycle_number=0,
                    triggered_by_score=None,
                )

        # Register each mutated prompt (new version created this cycle).
        for mutation in result["mutations"]:
            register_prompt(
                writer_name=mutation["writer_name"],
                style_prompt=mutation["new_prompt"],
                version_number=mutation["new_version"],
                cycle_number=cycle_number,
                triggered_by_score=mutation["triggered_by_score"],
            )

        # ── 6. Stamp SimCycle with run_id and completed_at ────────────────────
        with session_scope() as session:
            sim_cycle = session.get(SimCycle, uuid.UUID(cycle_id))
            if sim_cycle is not None:
                sim_cycle.mlflow_run_id = run_id
                sim_cycle.completed_at = datetime.now(tz=timezone.utc)

        LOGGER.info(
            "Cycle %d logged to MLflow run %s (%d tweets, %d mutations)",
            cycle_number,
            run_id,
            len(tweet_rows),
            len(result["mutations"]),
        )

    @task
    def check_weekly_reset(result: dict, cycle: dict) -> None:
        """On a week boundary: tag the weekly champion and log a week summary run.

        A week boundary is detected when the current cycle's week_number is
        greater than the previous cycle's week_number. The champion is the
        writer with the highest average engagement_score across all cycles
        in the completed week.
        """
        import uuid
        from sqlalchemy import func, select

        from news_pipeline.db.session import session_scope
        from news_pipeline.simulation.models import SimCycle, SimWriterCycleScore, SimWriter
        from news_pipeline.simulation.tracker import (
            SIMULATION_EXPERIMENT,
            tag_weekly_champion,
        )
        from news_pipeline.tracking.experiment import configure_mlflow
        import mlflow

        cycle_number = result["cycle_number"]
        current_week = cycle["week_number"]

        # ── Detect week boundary ──────────────────────────────────────────────
        with session_scope() as session:
            prev_cycle = session.scalars(
                select(SimCycle)
                .where(SimCycle.cycle_number == cycle_number - 1)
            ).first()
            previous_week = prev_cycle.week_number if prev_cycle else current_week

        if current_week <= previous_week:
            LOGGER.debug("No week boundary at cycle %d (week %d)", cycle_number, current_week)
            return

        completed_week = previous_week
        LOGGER.info("Week boundary detected: week %d just completed", completed_week)

        # ── Find this week's champion (highest avg score across the week) ──────
        with session_scope() as session:
            rows = session.execute(
                select(
                    SimWriter.name,
                    SimWriter.id,
                    func.avg(SimWriterCycleScore.engagement_score).label("avg_score"),
                    func.count(SimWriterCycleScore.cycle_id).label("cycle_count"),
                )
                .join(SimWriterCycleScore, SimWriterCycleScore.writer_id == SimWriter.id)
                .join(SimCycle, SimCycle.id == SimWriterCycleScore.cycle_id)
                .where(SimCycle.week_number == completed_week)
                .group_by(SimWriter.id, SimWriter.name)
                .order_by(func.avg(SimWriterCycleScore.engagement_score).desc())
            ).all()

            if not rows:
                LOGGER.warning("No scores found for completed week %d", completed_week)
                return

            champion_name = rows[0].name
            champion_avg = float(rows[0].avg_score)
            cycle_count = int(rows[0].cycle_count)

            # Get the cycle numbers for this week (for tagging context)
            week_cycle_numbers = session.scalars(
                select(SimCycle.cycle_number).where(SimCycle.week_number == completed_week)
            ).all()

        # ── Tag the champion in MLflow ────────────────────────────────────────
        tag_weekly_champion(
            writer_name=champion_name,
            week_number=completed_week,
            avg_score=champion_avg,
            cycle_numbers=list(week_cycle_numbers),
        )

        # ── Log a week summary run ────────────────────────────────────────────
        configure_mlflow()
        mlflow.set_experiment(SIMULATION_EXPERIMENT)
        with mlflow.start_run(run_name=f"week_{completed_week:03d}_summary"):
            mlflow.log_params(
                {
                    "week_number": completed_week,
                    "cycles_in_week": cycle_count,
                    "champion_writer": champion_name,
                }
            )
            mlflow.set_tags({"run_type": "week_summary", "week": str(completed_week)})
            mlflow.log_metrics(
                {
                    "champion_avg_score": champion_avg,
                    "writers_ranked": float(len(rows)),
                },
                step=completed_week,
            )
            # Full week leaderboard as a table artifact
            mlflow.log_table(
                data={
                    "writer_name": [r.name for r in rows],
                    "avg_engagement_score": [float(r.avg_score) for r in rows],
                    "cycles_participated": [int(r.cycle_count) for r in rows],
                },
                artifact_file="week_leaderboard.json",
            )

        LOGGER.info(
            "Week %d summary: champion=%s avg=%.3f across %d cycles",
            completed_week,
            champion_name,
            champion_avg,
            cycle_count,
        )

    seeded = seed_db()
    stories = fetch_stories()
    cycle = create_cycle(stories)
    tweet_inputs = prepare_tweet_inputs(cycle, stories)
    tweets = generate_tweet.expand(tweet_input=tweet_inputs)
    evaluations = evaluate_tweet.expand(tweet=tweets)
    result = score_and_mutate(evaluations, cycle)
    logged = log_to_mlflow(result, cycle)
    weekly = check_weekly_reset(result, cycle)

    seeded >> stories >> cycle >> tweet_inputs >> tweets >> evaluations >> result >> [logged, weekly]


def _article_to_story(article, signal_type: str, signal_score: float) -> dict:
    entity_names = [
        ae.entity.name
        for ae in (article.entities or [])[:5]
        if ae.entity is not None
    ]
    return {
        "article_id": str(article.id),
        "title": article.title,
        "summary": (article.cleaned_text or article.summary or "")[:500],
        "entities": entity_names,
        "signal_type": signal_type,
        "signal_score": signal_score,
    }


simulation_dag = build_simulation_dag()

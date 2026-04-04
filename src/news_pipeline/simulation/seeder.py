"""Idempotent seeder: loads writers.yaml and personas.yaml into the DB on first run."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import yaml
from sqlalchemy.orm import Session

from news_pipeline.simulation.models import SimPersona, SimPromptVersion, SimWriter

LOGGER = logging.getLogger(__name__)


def seed_all(session: Session, writers_path: Path, personas_path: Path) -> None:
    """Seed writers and personas if they do not exist yet. Safe to call repeatedly."""
    _seed_writers(session, writers_path)
    _seed_personas(session, personas_path)


# ── Writers ──────────────────────────────────────────────────────────────────


def _seed_writers(session: Session, path: Path) -> None:
    raw = yaml.safe_load(path.read_text())
    entries = raw.get("writers", [])

    for entry in entries:
        name = entry["name"]
        existing = session.query(SimWriter).filter_by(name=name).first()
        if existing is not None:
            LOGGER.debug("Writer %r already exists, skipping", name)
            continue

        writer = SimWriter(
            id=uuid.uuid4(),
            name=name,
            persona_description=entry["persona"].strip(),
        )
        session.add(writer)
        session.flush()  # get writer.id before creating the version

        version = SimPromptVersion(
            id=uuid.uuid4(),
            writer_id=writer.id,
            version_number=1,
            style_prompt=entry["style_prompt"].strip(),
            parent_id=None,
            cycle_introduced=0,
        )
        session.add(version)
        session.flush()

        writer.current_version_id = version.id
        LOGGER.info("Seeded writer %r (v1)", name)

    session.commit()


# ── Personas ─────────────────────────────────────────────────────────────────


def _seed_personas(session: Session, path: Path) -> None:
    raw = yaml.safe_load(path.read_text())
    entries = raw.get("personas", [])

    inserted = 0
    for entry in entries:
        name = entry["name"]
        existing = session.query(SimPersona).filter_by(name=name).first()
        if existing is not None:
            continue

        persona = SimPersona(
            id=uuid.uuid4(),
            name=name,
            archetype_group=entry["archetype_group"],
            description=entry["description"].strip(),
        )
        session.add(persona)
        inserted += 1

    session.commit()
    if inserted:
        LOGGER.info("Seeded %d personas", inserted)
    else:
        LOGGER.debug("All personas already present, nothing to seed")

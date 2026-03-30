from __future__ import annotations

from sqlalchemy import text

from news_pipeline.db.session import engine


def main() -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    print("Database connection OK")


if __name__ == "__main__":
    main()

"""Owns: Alembic environment for registry schema migrations under this package.

Does not own: async application engines, connection pooling policies,
or multi-db sharding.
"""

from __future__ import annotations

import importlib
import os
from logging.config import fileConfig

from alembic import context
from forgeai.registry.models.base import Base
from sqlalchemy import MetaData, engine_from_config, pool


def _load_orm_models_for_metadata() -> None:
    """Import model modules so ``Base.metadata`` is fully populated."""

    importlib.import_module("forgeai.registry.models")


_load_orm_models_for_metadata()

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata: MetaData = Base.metadata


def _sync_database_url() -> str:
    """Return the synchronous Postgres URL used by offline/online migrations."""

    url = os.environ.get("ALEMBIC_DATABASE_URL")
    if url is None or url == "":
        msg = "ALEMBIC_DATABASE_URL must be set for migrations."
        raise RuntimeError(msg)
    return url


def run_migrations_offline() -> None:
    """Run migrations in offline mode."""

    context.configure(
        url=_sync_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode."""

    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _sync_database_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

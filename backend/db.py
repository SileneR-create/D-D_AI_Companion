"""Connexion a la base de donnees, session SQLModel et micro-migration."""
from collections.abc import Iterator

from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, create_engine

from backend.config import DATABASE_URL

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, echo=False, connect_args=_connect_args)


def _add_missing_columns() -> None:
    """Ajoute les colonnes presentes dans les modeles mais absentes en base.

    SQLModel.create_all() cree les tables manquantes mais ne modifie pas les
    tables existantes. Cette passe legere ajoute les nouvelles colonnes (en
    nullable) pour eviter les erreurs apres une evolution de schema, sans outil
    de migration lourd.
    """
    insp = inspect(engine)
    existing = set(insp.get_table_names())
    for table in SQLModel.metadata.sorted_tables:
        if table.name not in existing:
            continue
        cols = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name in cols:
                continue
            col_type = col.type.compile(dialect=engine.dialect)
            ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col_type}'
            try:
                with engine.begin() as conn:
                    conn.execute(text(ddl))
                print(f"Migration : colonne ajoutee {table.name}.{col.name}")
            except Exception as e:  # noqa: BLE001
                print(f"Migration ignoree {table.name}.{col.name}: {e}")


def _relax_campaign_name_unique() -> None:
    """Supprime l'ancienne contrainte d'unicite GLOBALE sur campaign.name.

    Desormais le nom est unique *par utilisateur* (verifie dans la route). On
    retire donc l'index/contrainte unique cree par les anciennes versions.
    """
    insp = inspect(engine)
    if "campaign" not in set(insp.get_table_names()):
        return
    # Index uniques sur (name)
    for ix in insp.get_indexes("campaign"):
        if ix.get("unique") and ix.get("column_names") == ["name"] and ix.get("name"):
            try:
                with engine.begin() as conn:
                    conn.execute(text(f'DROP INDEX {ix["name"]}'))
                print(f"Migration : index unique supprime ({ix['name']})")
            except Exception as e:  # noqa: BLE001
                print(f"Migration : suppression index ignoree: {e}")
    # Contraintes uniques (Postgres)
    try:
        for uc in insp.get_unique_constraints("campaign"):
            if uc.get("column_names") == ["name"] and uc.get("name"):
                with engine.begin() as conn:
                    conn.execute(text(f'ALTER TABLE campaign DROP CONSTRAINT "{uc["name"]}"'))
                print(f"Migration : contrainte unique supprimee ({uc['name']})")
    except Exception as e:  # noqa: BLE001
        print(f"Migration : contraintes uniques ignorees: {e}")


def init_db() -> None:
    """Cree les tables manquantes puis complete les colonnes manquantes."""
    import backend.models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    _add_missing_columns()
    _relax_campaign_name_unique()


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session

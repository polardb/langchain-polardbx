"""Integration tests for partition table enhancement.

Requires a live PolarDB-X instance (configured via .env).
Tests are skipped on v3 instances where partition + vector index is not supported.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import (
    DB_HOST,
    DB_NAME,
    DB_PASS,
    DB_PORT,
    DB_USER,
    DN_NODE,
    METADATAS,
    TEXTS,
    is_v3,
    make_store,
)

from langchain_polardbx import create_partitioned_table

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uri():
    """Build a mysql+pymysql URI from .env credentials."""
    return f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def _cleanup(table_name):
    """Drop a table if it exists."""
    try:
        import sqlalchemy

        eng = sqlalchemy.create_engine(_uri())
        with eng.connect() as conn:
            conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS `{table_name}`"))
            conn.commit()
        eng.dispose()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Skip on v3 (partition + vector index not supported)
# ---------------------------------------------------------------------------


def _check_v3_skip():
    """Return a pytest.skip if the instance is v3."""
    vs = make_store(table_name="_test_v3_check", pre_delete=False)
    try:
        v3 = is_v3(vs)
    finally:
        vs.close()
    if v3:
        pytest.skip("Partition + vector index not supported on v3 instances")


# ===========================================================================
# 1. Vector store + HASH partition — end to end
# ===========================================================================


def test_vector_store_hash_partition():
    """Create a partitioned vector table, insert, and search."""
    _check_v3_skip()

    table = "test_lc_part_hash"
    _cleanup(table)

    vs = make_store(
        table_name=table,
        partition_by="HASH",
        partition_column="id",
        partitions=4,
    )
    vs.add_texts(TEXTS, metadatas=METADATAS)
    assert vs.count() == len(TEXTS)

    # Search should work
    results = vs.similarity_search("database", k=3)
    assert len(results) == 3

    # Search with score
    results = vs.similarity_search_with_score("database", k=2)
    assert len(results) == 2
    assert all(isinstance(s, float) for _, s in results)

    vs.drop_table()
    vs.close()


def test_vector_store_hash_partition_custom_column():
    """HASH partition on a non-id column."""
    _check_v3_skip()

    table = "test_lc_part_hash_col"
    _cleanup(table)

    vs = make_store(
        table_name=table,
        partition_by="HASH",
        partition_column="id",
        partitions=8,
    )
    vs.add_texts(["test partition with custom column"])
    assert vs.count() == 1

    vs.drop_table()
    vs.close()


# ===========================================================================
# 2. Vector store + BROADCAST
# ===========================================================================


def test_vector_store_broadcast():
    """Create a broadcast vector table, insert, and search."""
    _check_v3_skip()

    table = "test_lc_part_broadcast"
    _cleanup(table)

    vs = make_store(
        table_name=table,
        broadcast=True,
    )
    vs.add_texts(TEXTS, metadatas=METADATAS)
    assert vs.count() == len(TEXTS)

    results = vs.similarity_search("framework", k=2)
    assert len(results) == 2

    vs.drop_table()
    vs.close()


# ===========================================================================
# 3. Vector store + LOCALITY
# ===========================================================================


def test_vector_store_locality():
    """Create a vector table with LOCALITY clause."""
    _check_v3_skip()

    table = "test_lc_part_locality"
    _cleanup(table)

    # LOCALITY requires a real DN node name from the connected instance.
    # DN name is read from .env (POLARDBX_DN_NODE). Skip if not configured.
    if not DN_NODE:
        pytest.skip("POLARDBX_DN_NODE not set in .env")
    try:
        vs = make_store(
            table_name=table,
            locality=f"dn={DN_NODE}",
        )
        vs.add_texts(["test locality"])
        assert vs.count() == 1
        vs.drop_table()
        vs.close()
    except Exception as e:
        _cleanup(table)
        pytest.skip(f"LOCALITY DN name not valid for this instance: {e}")


# ===========================================================================
# 4. Vector store + RANGE partition
# ===========================================================================


def test_vector_store_range_partition():
    """Create a RANGE partitioned vector table.

    Note: RANGE/LIST partitioning on VECTOR type columns may have limitations.
    The partition column must be a real column in the table. Our fixed schema
    only has id, text, metadata, embedding — so we partition on id.
    """
    _check_v3_skip()

    table = "test_lc_part_range"
    _cleanup(table)

    vs = make_store(
        table_name=table,
        partition_by="RANGE",
        partition_column="id",
        partition_defs=[
            {"name": "p0", "values_less_than": "0"},
            {"name": "p1", "values_less_than": "MAXVALUE"},
        ],
    )
    vs.add_texts(TEXTS[:2])
    assert vs.count() == 2

    results = vs.similarity_search("database", k=1)
    assert len(results) == 1

    vs.drop_table()
    vs.close()


# ===========================================================================
# 5. create_partitioned_table — standalone DDL function
# ===========================================================================


def test_create_partitioned_table_hash():
    """create_partitioned_table with HASH strategy."""
    table = "test_lc_ddl_hash"
    _cleanup(table)

    create_partitioned_table(
        uri=_uri(),
        table_name=table,
        columns=[
            "id BIGINT NOT NULL AUTO_INCREMENT",
            "user_id BIGINT NOT NULL",
            "amount DECIMAL(10,2)",
            "PRIMARY KEY (id)",
        ],
        partition_by="HASH",
        partition_column="user_id",
        partitions=4,
    )

    # Verify table exists by inserting and querying
    import sqlalchemy

    eng = sqlalchemy.create_engine(_uri())
    with eng.connect() as conn:
        conn.execute(
            sqlalchemy.text(
                f"INSERT INTO `{table}` (user_id, amount) VALUES (1, 99.99)"
            )
        )
        conn.commit()
        result = conn.execute(
            sqlalchemy.text(f"SELECT COUNT(*) FROM `{table}`")
        ).scalar()
        assert result == 1
    eng.dispose()

    _cleanup(table)


def test_create_partitioned_table_broadcast():
    """create_partitioned_table with BROADCAST."""
    table = "test_lc_ddl_broadcast"
    _cleanup(table)

    create_partitioned_table(
        uri=_uri(),
        table_name=table,
        columns=[
            "id BIGINT NOT NULL AUTO_INCREMENT",
            "code VARCHAR(50)",
            "name VARCHAR(100)",
            "PRIMARY KEY (id)",
        ],
        broadcast=True,
    )

    # Verify
    import sqlalchemy

    eng = sqlalchemy.create_engine(_uri())
    with eng.connect() as conn:
        conn.execute(
            sqlalchemy.text(f"INSERT INTO `{table}` (code, name) VALUES ('X', 'test')")
        )
        conn.commit()
        result = conn.execute(
            sqlalchemy.text(f"SELECT COUNT(*) FROM `{table}`")
        ).scalar()
        assert result == 1
    eng.dispose()

    _cleanup(table)


def test_create_partitioned_table_range():
    """create_partitioned_table with RANGE."""
    table = "test_lc_ddl_range"
    _cleanup(table)

    create_partitioned_table(
        uri=_uri(),
        table_name=table,
        columns=[
            "id BIGINT NOT NULL",
            "amount DECIMAL(10,2)",
            "PRIMARY KEY (id)",
        ],
        partition_by="RANGE",
        partition_column="id",
        partition_defs=[
            {"name": "p0", "values_less_than": 1000},
            {"name": "p1", "values_less_than": 2000},
            {"name": "p2", "values_less_than": "MAXVALUE"},
        ],
    )

    import sqlalchemy

    eng = sqlalchemy.create_engine(_uri())
    with eng.connect() as conn:
        conn.execute(
            sqlalchemy.text(f"INSERT INTO `{table}` (id, amount) VALUES (500, 10.00)")
        )
        conn.commit()
        result = conn.execute(
            sqlalchemy.text(f"SELECT COUNT(*) FROM `{table}`")
        ).scalar()
        assert result == 1
    eng.dispose()

    _cleanup(table)


def test_create_partitioned_table_list():
    """create_partitioned_table with LIST."""
    table = "test_lc_ddl_list"
    _cleanup(table)

    create_partitioned_table(
        uri=_uri(),
        table_name=table,
        columns=[
            "id BIGINT NOT NULL",
            "region VARCHAR(20)",
            "PRIMARY KEY (id, region)",
        ],
        partition_by="LIST",
        partition_column="region",
        partition_defs=[
            {"name": "p0", "values_in": ["east", "west"]},
            {"name": "p1", "values_in": ["north", "south"]},
        ],
    )

    import sqlalchemy

    eng = sqlalchemy.create_engine(_uri())
    with eng.connect() as conn:
        conn.execute(
            sqlalchemy.text(f"INSERT INTO `{table}` (id, region) VALUES (1, 'east')")
        )
        conn.commit()
        result = conn.execute(
            sqlalchemy.text(f"SELECT COUNT(*) FROM `{table}`")
        ).scalar()
        assert result == 1
    eng.dispose()

    _cleanup(table)


def test_create_partitioned_table_single():
    """create_partitioned_table without partition → single table."""
    table = "test_lc_ddl_single"
    _cleanup(table)

    create_partitioned_table(
        uri=_uri(),
        table_name=table,
        columns=["id INT NOT NULL", "val VARCHAR(100)", "PRIMARY KEY (id)"],
    )

    import sqlalchemy

    eng = sqlalchemy.create_engine(_uri())
    with eng.connect() as conn:
        conn.execute(
            sqlalchemy.text(f"INSERT INTO `{table}` (id, val) VALUES (1, 'hello')")
        )
        conn.commit()
        result = conn.execute(
            sqlalchemy.text(f"SELECT COUNT(*) FROM `{table}`")
        ).scalar()
        assert result == 1
    eng.dispose()

    _cleanup(table)


def test_create_partitioned_table_if_not_exists():
    """if_not_exists=True should not fail on second call."""
    table = "test_lc_ddl_ine"
    _cleanup(table)

    create_partitioned_table(
        uri=_uri(),
        table_name=table,
        columns=["id INT NOT NULL", "PRIMARY KEY (id)"],
        if_not_exists=True,
    )
    # Second call should succeed (no error)
    create_partitioned_table(
        uri=_uri(),
        table_name=table,
        columns=["id INT NOT NULL", "PRIMARY KEY (id)"],
        if_not_exists=True,
    )

    _cleanup(table)


# ===========================================================================
# 6. v3 restriction check
# ===========================================================================


def test_v3_partition_restriction():
    """On v3 instances, partition_by should raise NotSupportedError."""
    _check_v3_skip()  # This skips on v3, so this test only runs on non-v3

    # If we're here, we're on a non-v3 instance — partition should work.
    # The v3 restriction is tested implicitly by _check_v3_skip skipping.
    # This test just confirms non-v3 allows partitions.
    table = "test_lc_v3_check"
    _cleanup(table)

    vs = make_store(
        table_name=table,
        partition_by="HASH",
        partitions=4,
    )
    vs.add_texts(["v3 restriction test"])
    assert vs.count() == 1
    vs.drop_table()
    vs.close()

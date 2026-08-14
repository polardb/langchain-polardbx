"""Unit tests for partition table enhancement (no database required).

Covers:
1. PolarDBXVectorStore._build_partition_clause() — all branches (delegates to _partition)
2. _partition._build_partition_clause() — standalone helper, all branches
3. PolarDBXVectorStore constructor validation (invalid params, mutual exclusivity)
4. sql.create_partitioned_table() — DDL generation (mocked engine)
5. sql.create_partitioned_table() — parameter + identifier validation
6. Edge cases: RANGE string quoting, LIST special chars, missing keys, SQL injection
7. _partition module independence from sqlalchemy (no hidden [sql] extra dependency)
"""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from langchain_polardbx._partition import (
    _build_partition_clause as _build_from_partition,
)
from langchain_polardbx.sql import create_partitioned_table
from langchain_polardbx.vectorstores.polardbx import PolarDBXVectorStore

# Alias for backward-compatible references in test code
_build_partition_clause = _build_from_partition


# ---------------------------------------------------------------------------
# Helpers — create a bare PolarDBXVectorStore without DB connection
# ---------------------------------------------------------------------------


def _make_store(**kwargs):
    """Create a PolarDBXVectorStore instance without calling __init__."""
    vs = PolarDBXVectorStore.__new__(PolarDBXVectorStore)
    vs._partition_by = kwargs.get("partition_by")
    vs._partitions = kwargs.get("partitions", 0)
    vs._partition_column = kwargs.get("partition_column", "id")
    vs._broadcast = kwargs.get("broadcast", False)
    vs._locality = kwargs.get("locality")
    vs._partition_defs = kwargs.get("partition_defs")
    vs._hnsw_m = 6
    vs._distance_strategy = "cosine"
    vs._ef_construction = None
    vs._capabilities = {}
    vs._table_name = "test_table"
    return vs


# ===========================================================================
# 1. VectorStore._build_partition_clause — all branches (delegates to sql)
# ===========================================================================


class TestVectorStorePartitionClause:
    """Test PolarDBXVectorStore._build_partition_clause() output."""

    def test_single_table_no_clause(self):
        """Non-partitioned, non-broadcast → empty string."""
        vs = _make_store()
        assert vs._build_partition_clause() == ""

    def test_hash_partition(self):
        vs = _make_store(partition_by="HASH", partitions=8)
        clause = vs._build_partition_clause()
        assert "PARTITION BY HASH(id)" in clause
        assert "PARTITIONS 8" in clause

    def test_key_partition(self):
        vs = _make_store(partition_by="KEY", partitions=4)
        clause = vs._build_partition_clause()
        assert "PARTITION BY KEY(id)" in clause
        assert "PARTITIONS 4" in clause

    def test_hash_partition_custom_column(self):
        vs = _make_store(partition_by="HASH", partitions=8, partition_column="user_id")
        clause = vs._build_partition_clause()
        assert "PARTITION BY HASH(user_id)" in clause

    def test_range_partition(self):
        vs = _make_store(
            partition_by="RANGE",
            partition_column="amount",
            partition_defs=[
                {"name": "p0", "values_less_than": 1000},
                {"name": "p1", "values_less_than": 2000},
                {"name": "p2", "values_less_than": "MAXVALUE"},
            ],
        )
        clause = vs._build_partition_clause()
        assert "PARTITION BY RANGE(amount)" in clause
        assert "PARTITION p0 VALUES LESS THAN (1000)" in clause
        assert "PARTITION p1 VALUES LESS THAN (2000)" in clause
        assert "PARTITION p2 VALUES LESS THAN (MAXVALUE)" in clause

    def test_range_partition_with_string_values(self):
        """RANGE with date string values should be SQL-quoted."""
        vs = _make_store(
            partition_by="RANGE",
            partition_column="created_at",
            partition_defs=[
                {"name": "p0", "values_less_than": "2024-01-01"},
                {"name": "p1", "values_less_than": "MAXVALUE"},
            ],
        )
        clause = vs._build_partition_clause()
        assert "PARTITION BY RANGE(created_at)" in clause
        assert "PARTITION p0 VALUES LESS THAN ('2024-01-01')" in clause
        assert "PARTITION p1 VALUES LESS THAN (MAXVALUE)" in clause

    def test_list_partition(self):
        vs = _make_store(
            partition_by="LIST",
            partition_column="region",
            partition_defs=[
                {"name": "p0", "values_in": ["east", "west"]},
                {"name": "p1", "values_in": ["north", "south"]},
            ],
        )
        clause = vs._build_partition_clause()
        assert "PARTITION BY LIST(region)" in clause
        assert "PARTITION p0 VALUES IN ('east', 'west')" in clause
        assert "PARTITION p1 VALUES IN ('north', 'south')" in clause

    def test_broadcast(self):
        vs = _make_store(broadcast=True)
        clause = vs._build_partition_clause()
        assert "BROADCAST" in clause
        assert "PARTITION" not in clause

    def test_locality_only(self):
        """LOCALITY without partition → just LOCALITY clause."""
        vs = _make_store(locality="dn=polardbx-storage-0-master")
        clause = vs._build_partition_clause()
        assert "LOCALITY='dn=polardbx-storage-0-master'" in clause

    def test_hash_partition_with_locality(self):
        """HASH partition + LOCALITY → both clauses."""
        vs = _make_store(partition_by="HASH", partitions=4, locality="dn=node-0")
        clause = vs._build_partition_clause()
        assert "PARTITION BY HASH(id)" in clause
        assert "PARTITIONS 4" in clause
        assert "LOCALITY='dn=node-0'" in clause

    def test_broadcast_with_locality(self):
        vs = _make_store(broadcast=True, locality="dn=node-0")
        clause = vs._build_partition_clause()
        assert "BROADCAST" in clause
        assert "LOCALITY='dn=node-0'" in clause


# ===========================================================================
# 2. VectorStore constructor validation
# ===========================================================================


class TestVectorStoreValidation:
    """Test parameter validation in __init__ (without DB connection).

    We test validation by checking that invalid partition params raise the
    correct errors before any DB connection is attempted.
    """

    def test_invalid_partition_by_raises(self):
        """Invalid partition_by value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid partition_by"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="u",
                password="p",
                database="db",
                embedding=MagicMock(),
                table_name="t",
                partition_by="INVALID",
            )

    def test_hash_without_partitions_raises(self):
        """HASH partition without partitions > 0 raises ValueError."""
        with pytest.raises(ValueError, match="partitions must be > 0"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="u",
                password="p",
                database="db",
                embedding=MagicMock(),
                table_name="t",
                partition_by="HASH",
                partitions=0,
            )

    def test_range_without_defs_raises(self):
        """RANGE partition without partition_defs raises ValueError."""
        with pytest.raises(ValueError, match="partition_defs must be provided"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="u",
                password="p",
                database="db",
                embedding=MagicMock(),
                table_name="t",
                partition_by="RANGE",
            )

    def test_list_without_defs_raises(self):
        """LIST partition without partition_defs raises ValueError."""
        with pytest.raises(ValueError, match="partition_defs must be provided"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="u",
                password="p",
                database="db",
                embedding=MagicMock(),
                table_name="t",
                partition_by="LIST",
            )

    def test_broadcast_and_partition_mutually_exclusive(self):
        """broadcast=True and partition_by set together raises ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="u",
                password="p",
                database="db",
                embedding=MagicMock(),
                table_name="t",
                broadcast=True,
                partition_by="HASH",
                partitions=4,
            )


# ===========================================================================
# 2b. **kwargs validation — typo detection and invalid argument handling
# ===========================================================================


class TestKwargsValidation:
    """Test _validate_kwargs() catches typos before MySQL PoolError."""

    def test_typo_dimension_suggests_embedding_dimension(self):
        """'dimension' should suggest 'embedding_dimension'."""
        with pytest.raises(TypeError, match="Did you mean 'embedding_dimension'"):
            PolarDBXVectorStore._validate_kwargs({"dimension": 4})

    def test_unknown_kwarg_raises_typeerror(self):
        """Non-existent param like 'auto_create_table' raises TypeError."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            PolarDBXVectorStore._validate_kwargs({"auto_create_table": True})

    def test_ssl_ca_passes_through(self):
        """SSL/TLS params should pass validation without error."""
        PolarDBXVectorStore._validate_kwargs({"ssl_ca": "/path/ca.pem"})

    def test_ssl_cert_passes_through(self):
        PolarDBXVectorStore._validate_kwargs(
            {"ssl_cert": "/path/cert.pem", "ssl_key": "/path/key.pem"}
        )

    def test_known_mysql_params_pass_through(self):
        """Common MySQL connector params should pass validation."""
        PolarDBXVectorStore._validate_kwargs(
            {"client_flag": 1, "collation": "utf8mb4_bin"}
        )

    def test_empty_kwargs_passes(self):
        """No kwargs should not raise."""
        PolarDBXVectorStore._validate_kwargs({})

    def test_init_rejects_typo_before_db_connection(self):
        """__init__ raises TypeError for typo'd kwargs before DB connection."""
        with pytest.raises(TypeError, match="Did you mean 'embedding_dimension'"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="u",
                password="p",
                database="db",
                embedding=MagicMock(),
                table_name="t",
                dimension=4,
            )

    def test_init_rejects_nonexistent_param(self):
        """__init__ raises TypeError for non-existent parameter."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="u",
                password="p",
                database="db",
                embedding=MagicMock(),
                table_name="t",
                auto_create_table=True,
            )


# ===========================================================================
# 3. sql._build_partition_clause — standalone helper, all branches
# ===========================================================================


class TestSqlBuildPartitionClause:
    """Test the standalone _build_partition_clause() in _partition.py."""

    def test_no_partition_returns_empty(self):
        assert _build_from_partition() == ""

    def test_hash(self):
        clause = _build_from_partition(
            partition_by="HASH", partition_column="id", partitions=8
        )
        assert "PARTITION BY HASH(id)" in clause
        assert "PARTITIONS 8" in clause

    def test_key(self):
        clause = _build_from_partition(
            partition_by="KEY", partition_column="uid", partitions=4
        )
        assert "PARTITION BY KEY(uid)" in clause
        assert "PARTITIONS 4" in clause

    def test_range_numeric(self):
        clause = _build_from_partition(
            partition_by="RANGE",
            partition_column="amount",
            partition_defs=[
                {"name": "p0", "values_less_than": 1000},
                {"name": "p1", "values_less_than": "MAXVALUE"},
            ],
        )
        assert "PARTITION BY RANGE(amount)" in clause
        assert "PARTITION p0 VALUES LESS THAN (1000)" in clause
        assert "PARTITION p1 VALUES LESS THAN (MAXVALUE)" in clause

    def test_range_string_values_quoted(self):
        """RANGE with string values (e.g., dates) should be SQL-quoted."""
        clause = _build_from_partition(
            partition_by="RANGE",
            partition_column="ts",
            partition_defs=[
                {"name": "p0", "values_less_than": "2024-01-01"},
                {"name": "p1", "values_less_than": "2025-01-01"},
                {"name": "p2", "values_less_than": "MAXVALUE"},
            ],
        )
        assert "PARTITION p0 VALUES LESS THAN ('2024-01-01')" in clause
        assert "PARTITION p1 VALUES LESS THAN ('2025-01-01')" in clause
        assert "PARTITION p2 VALUES LESS THAN (MAXVALUE)" in clause

    def test_list(self):
        clause = _build_partition_clause(
            partition_by="LIST",
            partition_column="region",
            partition_defs=[
                {"name": "p0", "values_in": ["east", "west"]},
                {"name": "p1", "values_in": ["north", "south"]},
            ],
        )
        assert "PARTITION BY LIST(region)" in clause
        assert "PARTITION p0 VALUES IN ('east', 'west')" in clause
        assert "PARTITION p1 VALUES IN ('north', 'south')" in clause

    def test_list_with_integers(self):
        clause = _build_partition_clause(
            partition_by="LIST",
            partition_column="code",
            partition_defs=[{"name": "p0", "values_in": [1, 2, 3]}],
        )
        assert "PARTITION p0 VALUES IN (1, 2, 3)" in clause

    def test_list_with_single_quote_string(self):
        """String with single quote should use SQL-standard doubling."""
        clause = _build_partition_clause(
            partition_by="LIST",
            partition_column="name",
            partition_defs=[{"name": "p0", "values_in": ["O'Brien"]}],
        )
        assert "PARTITION p0 VALUES IN ('O''Brien')" in clause

    def test_broadcast(self):
        clause = _build_partition_clause(broadcast=True)
        assert "BROADCAST" in clause
        assert "PARTITION" not in clause

    def test_locality(self):
        clause = _build_partition_clause(locality="dn=node-0")
        assert "LOCALITY='dn=node-0'" in clause

    def test_hash_with_locality(self):
        clause = _build_partition_clause(
            partition_by="HASH", partitions=4, locality="dn=node-0"
        )
        assert "PARTITION BY HASH(id)" in clause
        assert "PARTITIONS 4" in clause
        assert "LOCALITY='dn=node-0'" in clause

    def test_broadcast_takes_precedence_over_partition(self):
        """When broadcast=True, partition_by is ignored."""
        clause = _build_partition_clause(
            broadcast=True, partition_by="HASH", partitions=4
        )
        assert "BROADCAST" in clause
        assert "PARTITION BY" not in clause

    def test_lowercase_partition_by(self):
        """partition_by is case-insensitive."""
        clause = _build_partition_clause(partition_by="hash", partitions=4)
        assert "PARTITION BY HASH(id)" in clause

    def test_locality_with_single_quote_escaped(self):
        """LOCALITY value with single quote should be escaped."""
        clause = _build_partition_clause(locality="dn=it's-node")
        assert "LOCALITY='dn=it''s-node'" in clause


# ===========================================================================
# 4. Missing partition_defs keys — KeyError → ValueError
# ===========================================================================


class TestMissingPartitionDefKeys:
    """Test that missing required keys in partition_defs raise ValueError."""

    def test_range_missing_values_less_than(self):
        with pytest.raises(ValueError, match="missing 'values_less_than'"):
            _build_partition_clause(
                partition_by="RANGE",
                partition_defs=[{"name": "p0"}],
            )

    def test_range_missing_name(self):
        with pytest.raises(ValueError, match="must have a 'name' key"):
            _build_partition_clause(
                partition_by="RANGE",
                partition_defs=[{"values_less_than": 100}],
            )

    def test_list_missing_values_in(self):
        with pytest.raises(ValueError, match="missing 'values_in'"):
            _build_partition_clause(
                partition_by="LIST",
                partition_defs=[{"name": "p0"}],
            )

    def test_list_missing_name(self):
        with pytest.raises(ValueError, match="must have a 'name' key"):
            _build_partition_clause(
                partition_by="LIST",
                partition_defs=[{"values_in": ["a"]}],
            )


# ===========================================================================
# 5. sql.create_partitioned_table — DDL generation (mocked engine)
# ===========================================================================


class TestCreatePartitionedTableDDL:
    """Test that create_partitioned_table generates correct DDL."""

    def _mock_engine(self, captured):
        def fake_create_engine(uri):
            conn = MagicMock()
            conn.execute.side_effect = lambda stmt: captured.append(stmt.text)
            conn.commit.return_value = None
            engine = MagicMock()
            engine.connect.return_value.__enter__ = lambda s: conn
            engine.connect.return_value.__exit__ = lambda *a: None
            engine.dispose.return_value = None
            return engine

        return fake_create_engine

    def test_hash_table_ddl(self):
        captured = []
        with patch("sqlalchemy.create_engine", side_effect=self._mock_engine(captured)):
            create_partitioned_table(
                uri="mysql+pymysql://u:p@host:3306/db",
                table_name="orders",
                columns=["id BIGINT", "amount DECIMAL(10,2)"],
                partition_by="HASH",
                partition_column="user_id",
                partitions=8,
            )
        ddl = captured[0]
        assert "CREATE TABLE IF NOT EXISTS `orders`" in ddl
        assert "PARTITION BY HASH(user_id)" in ddl
        assert "PARTITIONS 8" in ddl

    def test_broadcast_table_ddl(self):
        captured = []
        with patch("sqlalchemy.create_engine", side_effect=self._mock_engine(captured)):
            create_partitioned_table(
                uri="mysql+pymysql://u:p@host:3306/db",
                table_name="dim_table",
                columns=["id INT", "code VARCHAR(50)"],
                broadcast=True,
            )
        ddl = captured[0]
        assert "CREATE TABLE IF NOT EXISTS `dim_table`" in ddl
        assert "BROADCAST" in ddl

    def test_range_table_ddl(self):
        captured = []
        with patch("sqlalchemy.create_engine", side_effect=self._mock_engine(captured)):
            create_partitioned_table(
                uri="mysql+pymysql://u:p@host:3306/db",
                table_name="logs",
                columns=["id BIGINT", "ts DATETIME"],
                partition_by="RANGE",
                partition_column="id",
                partition_defs=[
                    {"name": "p0", "values_less_than": 1000},
                    {"name": "p1", "values_less_than": "MAXVALUE"},
                ],
            )
        ddl = captured[0]
        assert "PARTITION BY RANGE(id)" in ddl
        assert "PARTITION p0 VALUES LESS THAN (1000)" in ddl
        assert "PARTITION p1 VALUES LESS THAN (MAXVALUE)" in ddl

    def test_range_with_date_string_ddl(self):
        """RANGE with date string should generate quoted DDL."""
        captured = []
        with patch("sqlalchemy.create_engine", side_effect=self._mock_engine(captured)):
            create_partitioned_table(
                uri="mysql+pymysql://u:p@host:3306/db",
                table_name="events",
                columns=["id BIGINT", "ts DATETIME"],
                partition_by="RANGE",
                partition_column="ts",
                partition_defs=[
                    {"name": "p0", "values_less_than": "2024-01-01"},
                    {"name": "p1", "values_less_than": "MAXVALUE"},
                ],
            )
        ddl = captured[0]
        assert "PARTITION p0 VALUES LESS THAN ('2024-01-01')" in ddl

    def test_uri_auto_swap_mysql_pymysql(self):
        captured_uri = []
        with patch(
            "sqlalchemy.create_engine",
            side_effect=lambda uri: (captured_uri.append(uri), MagicMock())[1],
        ):
            create_partitioned_table(
                uri="mysql+pymysql://u:p@host:3306/db",
                table_name="t",
                columns=["id INT"],
            )
        assert captured_uri[0].startswith("polardbx+pymysql://")

    def test_uri_auto_swap_mysql_plain(self):
        captured_uri = []
        with patch(
            "sqlalchemy.create_engine",
            side_effect=lambda uri: (captured_uri.append(uri), MagicMock())[1],
        ):
            create_partitioned_table(
                uri="mysql://u:p@host:3306/db",
                table_name="t",
                columns=["id INT"],
            )
        assert captured_uri[0].startswith("polardbx+pymysql://")

    def test_if_not_exists_false(self):
        captured = []
        with patch("sqlalchemy.create_engine", side_effect=self._mock_engine(captured)):
            create_partitioned_table(
                uri="mysql+pymysql://u:p@host:3306/db",
                table_name="t",
                columns=["id INT"],
                if_not_exists=False,
            )
        assert "IF NOT EXISTS" not in captured[0]


# ===========================================================================
# 6. sql.create_partitioned_table — parameter + identifier validation
# ===========================================================================


class TestCreatePartitionedTableValidation:
    """Test that create_partitioned_table validates params and identifiers."""

    def test_invalid_partition_by_raises(self):
        with pytest.raises(ValueError, match="Invalid partition_by"):
            create_partitioned_table(
                uri="mysql://u:p@h:3306/db",
                table_name="t",
                columns=["id INT"],
                partition_by="INVALID",
            )

    def test_hash_without_partitions_raises(self):
        with pytest.raises(ValueError, match="partitions must be > 0"):
            create_partitioned_table(
                uri="mysql://u:p@h:3306/db",
                table_name="t",
                columns=["id INT"],
                partition_by="HASH",
                partitions=0,
            )

    def test_range_without_defs_raises(self):
        with pytest.raises(ValueError, match="partition_defs required"):
            create_partitioned_table(
                uri="mysql://u:p@h:3306/db",
                table_name="t",
                columns=["id INT"],
                partition_by="RANGE",
            )

    def test_list_without_defs_raises(self):
        with pytest.raises(ValueError, match="partition_defs required"):
            create_partitioned_table(
                uri="mysql://u:p@h:3306/db",
                table_name="t",
                columns=["id INT"],
                partition_by="LIST",
            )

    def test_broadcast_and_partition_raises(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            create_partitioned_table(
                uri="mysql://u:p@h:3306/db",
                table_name="t",
                columns=["id INT"],
                broadcast=True,
                partition_by="HASH",
                partitions=4,
            )

    def test_invalid_table_name_raises(self):
        """Table name with SQL injection attempt should be rejected."""
        with pytest.raises(ValueError, match="Invalid table name"):
            create_partitioned_table(
                uri="mysql://u:p@h:3306/db",
                table_name="t; DROP TABLE users; --",
                columns=["id INT"],
            )

    def test_invalid_partition_column_raises(self):
        """Partition column with SQL injection attempt should be rejected."""
        with pytest.raises(ValueError, match="Invalid partition column"):
            create_partitioned_table(
                uri="mysql://u:p@h:3306/db",
                table_name="t",
                columns=["id INT"],
                partition_by="HASH",
                partition_column="id) PARTITIONS 1; DROP TABLE users; --",
                partitions=4,
            )


# ---------------------------------------------------------------------------
# Test: _partition module independence from sqlalchemy
# ---------------------------------------------------------------------------


class TestPartitionModuleIndependence:
    """Verify that _partition.py can be imported without sqlalchemy.

    Regression test for a bug where VectorStore._build_partition_clause()
    delegated to sql.py, which has top-level sqlalchemy imports, breaking
    partition features for users who install without the [sql] extra.
    """

    def test_partition_module_has_no_sqlalchemy_import(self):
        """_partition module should not import sqlalchemy at module level."""
        mod = importlib.import_module("langchain_polardbx._partition")
        assert not hasattr(mod, "sqlalchemy"), (
            "_partition module must not have sqlalchemy as a dependency; "
            "VectorStore partition features must work without [sql] extra."
        )

    def test_build_from_partition_works(self):
        """_build_partition_clause imported from _partition should produce correct DDL."""
        clause = _build_from_partition(
            partition_by="HASH",
            partition_column="user_id",
            partitions=8,
        )
        assert "PARTITION BY HASH(user_id) PARTITIONS 8" in clause

    def test_vectorstore_delegates_to_partition_not_sql(self):
        """VectorStore._build_partition_clause should import from _partition, not sql."""
        import inspect

        store = _make_store(partition_by="HASH", partitions=4)
        source = inspect.getsource(store._build_partition_clause)
        assert "_partition" in source, (
            "VectorStore._build_partition_clause should import from "
            "langchain_polardbx._partition, not langchain_polardbx.sql"
        )

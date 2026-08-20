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
    # Custom column defaults
    vs._id_column = kwargs.get("id_column", "id")
    vs._content_column = kwargs.get("content_column", "text")
    vs._embedding_column = kwargs.get("embedding_column", "embedding")
    vs._metadata_json_column = kwargs.get("metadata_json_column", "metadata")
    vs._metadata_column_objs = kwargs.get("metadata_column_objs", [])
    vs._metadata_column_names = kwargs.get("metadata_column_names", [])
    # partition_column defaults to id_column (mirrors __init__ logic)
    vs._partition_column = kwargs.get("partition_column") or vs._id_column
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


# ===========================================================================
# 7. Custom column support — Column class + constructor validation
# ===========================================================================


class TestColumnDataclass:
    """Test the Column dataclass."""

    def test_basic_creation(self):
        from langchain_polardbx import Column

        col = Column("price", "DECIMAL(10,2)")
        assert col.name == "price"
        assert col.data_type == "DECIMAL(10,2)"
        assert col.nullable is True
        assert col.default is None

    def test_with_options(self):
        from langchain_polardbx import Column

        col = Column("category", "VARCHAR(100)", nullable=False, default="'misc'")
        assert col.nullable is False
        assert col.default == "'misc'"

    def test_import_from_vectorstores(self):
        from langchain_polardbx.vectorstores import Column

        assert Column.__name__ == "Column"

    def test_import_from_top_level(self):
        from langchain_polardbx import Column

        assert Column.__name__ == "Column"


class TestCustomColumnValidation:
    """Test constructor validation for custom column params."""

    def test_column_name_conflict_raises(self):
        """metadata_columns can't include core column names."""
        with pytest.raises(ValueError, match="Column name conflict"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="u",
                password="p",
                database="db",
                embedding=MagicMock(),
                table_name="t",
                id_column="product_id",
                metadata_columns=["product_id"],
            )

    def test_embedding_column_conflict_raises(self):
        """metadata_columns can't include the embedding column."""
        with pytest.raises(ValueError, match="Column name conflict"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="u",
                password="p",
                database="db",
                embedding=MagicMock(),
                table_name="t",
                embedding_column="vec",
                metadata_columns=["vec"],
            )

    def test_no_json_column_no_metadata_columns_raises(self):
        """Can't disable JSON column without metadata_columns."""
        with pytest.raises(ValueError, match="no place to store metadata"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="u",
                password="p",
                database="db",
                embedding=MagicMock(),
                table_name="t",
                metadata_json_column=None,
            )

    def test_no_json_column_with_metadata_columns_ok(self):
        """Disabling JSON column is OK if metadata_columns is provided.

        The error should be a connection error (host unreachable), not
        a validation error — proving validation passed.
        """
        with pytest.raises(Exception) as exc_info:
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="u",
                password="p",
                database="db",
                embedding=MagicMock(),
                table_name="t",
                metadata_json_column=None,
                metadata_columns=["price", "category"],
            )
        # Should NOT be a ValueError about metadata storage
        assert "no place to store metadata" not in str(exc_info.value)

    def test_invalid_type_in_metadata_columns_raises(self):
        """Non-Column, non-str items in metadata_columns raise TypeError."""
        with pytest.raises(TypeError, match="must be Column or str"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="u",
                password="p",
                database="db",
                embedding=MagicMock(),
                table_name="t",
                metadata_columns=[123],
            )

    def test_invalid_identifier_in_id_column_raises(self):
        """SQL injection attempt in id_column is rejected."""
        with pytest.raises(ValueError, match="Invalid id_column"):
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="u",
                password="p",
                database="db",
                embedding=MagicMock(),
                table_name="t",
                id_column="id; DROP TABLE",
            )

    def test_validate_kwargs_accepts_new_params(self):
        """New param names should not trigger typo detection."""
        # These should all pass without raising TypeError
        PolarDBXVectorStore._validate_kwargs(
            {"id_column": "pid", "content_column": "desc"}
        )
        PolarDBXVectorStore._validate_kwargs(
            {"embedding_column": "vec", "metadata_json_column": "extra"}
        )
        PolarDBXVectorStore._validate_kwargs({"metadata_columns": []})

    def test_default_column_names(self):
        """When not passing custom columns, defaults are used."""
        # We can't fully init without a DB, but we can verify the
        # validation logic doesn't raise for default values by catching
        # the connection error (not a ValueError from validation).
        with pytest.raises(Exception) as exc_info:
            PolarDBXVectorStore(
                host="localhost",
                port=3306,
                user="u",
                password="p",
                database="db",
                embedding=MagicMock(),
                table_name="t",
            )
        # Should fail at connection, not at validation
        assert "Column name conflict" not in str(exc_info.value)
        assert "no place to store metadata" not in str(exc_info.value)


# ===========================================================================
# 8. Custom column support — DDL generation
# ===========================================================================


class TestCreateTableSqlCustom:
    """Test _build_create_table_sql with custom columns."""

    def test_default_uses_static_template(self):
        """No custom columns → same SQL_CREATE_TABLE template."""
        vs = _make_store()
        sql = vs._build_create_table_sql(4)
        # Should match the static template output
        assert "id VARCHAR(36) PRIMARY KEY" in sql
        assert "text LONGTEXT NOT NULL" in sql
        assert "metadata JSON" in sql
        assert "embedding VECTOR(4) NOT NULL" in sql
        assert "VECTOR INDEX (embedding) M=6 DISTANCE=COSINE" in sql

    def test_custom_id_column(self):
        vs = _make_store(id_column="product_id")
        sql = vs._build_create_table_sql(4)
        assert "`product_id` VARCHAR(36) PRIMARY KEY" in sql
        assert "`product_id`" in sql

    def test_custom_content_column(self):
        vs = _make_store(content_column="description")
        sql = vs._build_create_table_sql(4)
        assert "`description` LONGTEXT NOT NULL" in sql

    def test_custom_embedding_column(self):
        vs = _make_store(embedding_column="vec")
        sql = vs._build_create_table_sql(1536)
        assert "`vec` VECTOR(1536) NOT NULL" in sql
        assert "VECTOR INDEX (`vec`)" in sql

    def test_custom_metadata_json_column(self):
        vs = _make_store(metadata_json_column="extra")
        sql = vs._build_create_table_sql(4)
        assert "`extra` JSON" in sql
        assert "`metadata`" not in sql  # default name should not appear

    def test_no_metadata_json_column(self):
        """When metadata_json_column=None, no JSON column in DDL."""
        vs = _make_store(
            metadata_json_column=None,
            metadata_column_names=["price"],
        )
        sql = vs._build_create_table_sql(4)
        assert "JSON" not in sql

    def test_metadata_columns_with_column_objects(self):
        """Column objects provide type info for DDL."""
        from langchain_polardbx import Column

        cols = [
            Column("price", "DECIMAL(10,2)"),
            Column("category", "VARCHAR(100)", nullable=False),
        ]
        vs = _make_store(
            metadata_column_objs=cols,
            metadata_column_names=["price", "category"],
        )
        sql = vs._build_create_table_sql(4)
        assert "`price` DECIMAL(10,2)" in sql
        assert "`category` VARCHAR(100) NOT NULL" in sql

    def test_metadata_column_with_default(self):
        from langchain_polardbx import Column

        col = Column("status", "VARCHAR(20)", default="'active'")
        vs = _make_store(
            metadata_column_objs=[col],
            metadata_column_names=["status"],
        )
        sql = vs._build_create_table_sql(4)
        assert "`status` VARCHAR(20) DEFAULT 'active'" in sql

    def test_custom_columns_with_partition(self):
        """Custom columns + partition clause works together."""
        from langchain_polardbx import Column

        vs = _make_store(
            id_column="product_id",
            partition_by="HASH",
            partitions=4,
            metadata_column_objs=[Column("price", "INT")],
            metadata_column_names=["price"],
        )
        sql = vs._build_create_table_sql(4)
        assert "`product_id` VARCHAR(36) PRIMARY KEY" in sql
        assert "PARTITION BY HASH" in sql
        assert "PARTITIONS 4" in sql
        assert "`price` INT" in sql

    def test_custom_columns_distance_strategy(self):
        """Distance strategy appears in VECTOR INDEX clause."""
        vs = _make_store(
            embedding_column="vec",
            distance_strategy="euclidean",
        )
        # Need to set distance_strategy properly
        vs._distance_strategy = "euclidean"
        sql = vs._build_create_table_sql(4)
        assert "DISTANCE=EUCLIDEAN" in sql

    def test_has_custom_columns_flag(self):
        """_has_custom_columns property detects custom config."""
        vs = _make_store()
        assert not vs._has_custom_columns

        vs2 = _make_store(id_column="pid")
        assert vs2._has_custom_columns

        vs3 = _make_store(metadata_column_names=["price"])
        assert vs3._has_custom_columns

        vs4 = _make_store(metadata_json_column="extra")
        assert vs4._has_custom_columns


# ===========================================================================
# 9. Custom column support — UPSERT SQL and values builder
# ===========================================================================


class TestUpsertSqlBuilder:
    """Test _build_upsert_sql and _build_upsert_values."""

    def test_default_upsert_sql(self):
        """No custom columns → same as SQL_UPSERT template."""
        vs = _make_store()
        sql = vs._build_upsert_sql()
        assert "INSERT INTO `test_table`" in sql
        assert "id, text, metadata, embedding" in sql
        assert "VEC_FROMTEXT(%s)" in sql
        assert "ON DUPLICATE KEY UPDATE" in sql

    def test_custom_upsert_sql_column_names(self):
        vs = _make_store(
            id_column="product_id",
            content_column="description",
            embedding_column="vec",
            metadata_json_column="extra",
        )
        sql = vs._build_upsert_sql()
        assert "`product_id`, `description`, `extra`, `vec`" in sql
        assert "ON DUPLICATE KEY UPDATE" in sql
        assert "`description` = VALUES(`description`)" in sql
        assert "`vec` = VALUES(`vec`)" in sql
        # id column should NOT be in the UPDATE clause
        assert "`product_id` = VALUES" not in sql

    def test_upsert_sql_with_metadata_columns(self):
        from langchain_polardbx import Column

        vs = _make_store(
            metadata_column_objs=[
                Column("price", "DECIMAL(10,2)"),
                Column("category", "VARCHAR(100)"),
            ],
            metadata_column_names=["price", "category"],
        )
        sql = vs._build_upsert_sql()
        assert "`price`, `category`" in sql
        assert "`price` = VALUES(`price`)" in sql
        assert "`category` = VALUES(`category`)" in sql

    def test_upsert_sql_no_json_column(self):
        vs = _make_store(
            metadata_json_column=None,
            metadata_column_names=["price"],
        )
        sql = vs._build_upsert_sql()
        # No JSON column in INSERT
        assert "metadata" not in sql.lower()
        assert "`price`" in sql

    def test_default_upsert_values(self):
        vs = _make_store()
        vals = vs._build_upsert_values("id1", "hello", {"k": "v"}, "vec_str")
        assert vals == ("id1", "hello", '{"k": "v"}', "vec_str")

    def test_custom_upsert_values_with_metadata_split(self):
        """Metadata keys mapped to columns are extracted; rest goes to JSON."""
        from langchain_polardbx import Column

        vs = _make_store(
            metadata_column_objs=[Column("price", "INT")],
            metadata_column_names=["price"],
        )
        metadata = {"price": 100, "category": "electronics", "tags": "new"}
        vals = vs._build_upsert_values("id1", "text", metadata, "vec_str")
        # Order: id, text, json(remaining), vec, price
        assert vals[0] == "id1"
        assert vals[1] == "text"
        # JSON should contain only unmapped keys
        import json as _json

        remaining = _json.loads(vals[2])
        assert "category" in remaining
        assert "tags" in remaining
        assert "price" not in remaining
        assert vals[3] == "vec_str"
        assert vals[4] == 100

    def test_upsert_values_no_json_column(self):
        """When JSON column is disabled, mapped values still extracted."""
        vs = _make_store(
            metadata_json_column=None,
            metadata_column_names=["price"],
        )
        metadata = {"price": 50, "other": "data"}
        vals = vs._build_upsert_values("id1", "text", metadata, "vec_str")
        # Order: id, text, vec, price (no json column)
        assert vals == ("id1", "text", "vec_str", 50)

    def test_upsert_values_missing_metadata_key(self):
        """Missing mapped key yields None for that column."""
        vs = _make_store(
            metadata_column_names=["price"],
        )
        vals = vs._build_upsert_values("id1", "text", {}, "vec_str")
        # price should be None (metadata.get returns None)
        assert vals[-1] is None

    def test_upsert_sql_and_values_order_match(self):
        """Column order in SQL must match values tuple order."""
        from langchain_polardbx import Column

        vs = _make_store(
            id_column="pid",
            content_column="desc",
            embedding_column="emb",
            metadata_json_column="meta",
            metadata_column_objs=[
                Column("price", "INT"),
                Column("brand", "VARCHAR(50)"),
            ],
            metadata_column_names=["price", "brand"],
        )
        sql = vs._build_upsert_sql()
        vals = vs._build_upsert_values(
            "x1", "hello", {"price": 10, "brand": "Nike", "extra": "data"}, "v"
        )

        # SQL column order: pid, desc, meta, emb, price, brand
        assert "`pid`, `desc`, `meta`, `emb`, `price`, `brand`" in sql

        # Values order: id, text, json, vec, price, brand
        assert vals[0] == "x1"       # pid
        assert vals[1] == "hello"   # desc
        # vals[2] is JSON of remaining metadata
        assert vals[3] == "v"       # emb
        assert vals[4] == 10        # price
        assert vals[5] == "Nike"    # brand


# ===========================================================================
# 10. Custom column support — SELECT SQL and result mapping
# ===========================================================================


class TestSelectSqlAndMapping:
    """Test _build_select_columns, _build_search_sql, _build_get_by_ids_sql,
    and _record_to_metadata."""

    def test_default_select_columns(self):
        vs = _make_store()
        assert vs._build_select_columns() == "id, text, metadata"

    def test_custom_select_columns(self):
        vs = _make_store(
            id_column="product_id",
            content_column="description",
            metadata_json_column="extra",
        )
        cols = vs._build_select_columns()
        assert "`product_id` AS `id`" in cols
        assert "`description` AS `text`" in cols
        assert "`extra` AS `metadata`" in cols

    def test_select_columns_no_json(self):
        vs = _make_store(
            metadata_json_column=None,
            metadata_column_names=["price"],
        )
        cols = vs._build_select_columns()
        assert "AS `metadata`" not in cols
        assert "`price`" in cols

    def test_select_columns_with_metadata_cols(self):
        vs = _make_store(
            metadata_column_names=["price", "category"],
        )
        cols = vs._build_select_columns()
        assert "`price`" in cols
        assert "`category`" in cols

    def test_default_search_sql(self):
        vs = _make_store()
        sql = vs._build_search_sql(
            distance_func="VEC_DISTANCE",
            index_hint="",
            where_clause="",
        )
        assert "SELECT id, text, metadata" in sql
        assert "VEC_DISTANCE(`embedding`, VEC_FROMTEXT(%s)) AS distance" in sql
        assert "ORDER BY distance" in sql
        assert "LIMIT %s" in sql

    def test_custom_search_sql(self):
        vs = _make_store(
            id_column="pid",
            content_column="desc",
            embedding_column="emb",
            metadata_json_column="meta",
        )
        sql = vs._build_search_sql(
            distance_func="VEC_DISTANCE",
            index_hint=" /*+ FORCE_INDEX */",
            where_clause="WHERE 1=1",
        )
        assert "`pid` AS `id`" in sql
        assert "`desc` AS `text`" in sql
        assert "`meta` AS `metadata`" in sql
        assert "VEC_DISTANCE(`emb`, VEC_FROMTEXT(%s))" in sql
        assert "/*+ FORCE_INDEX */" in sql
        assert "WHERE 1=1" in sql

    def test_default_get_by_ids_sql(self):
        vs = _make_store()
        sql = vs._build_get_by_ids_sql("%s,%s")
        assert "SELECT id, text, metadata" in sql
        assert "WHERE `id` IN (%s,%s)" in sql

    def test_custom_get_by_ids_sql(self):
        vs = _make_store(id_column="product_id")
        sql = vs._build_get_by_ids_sql("%s,%s,%s")
        assert "`product_id` AS `id`" in sql
        assert "WHERE `product_id` IN (%s,%s,%s)" in sql

    def test_default_record_to_metadata(self):
        vs = _make_store()
        record = {"metadata": '{"k": "v"}'}
        meta = vs._record_to_metadata(record)
        assert meta == {"k": "v"}

    def test_default_record_to_metadata_dict(self):
        """If driver returns dict (not str), pass through."""
        vs = _make_store()
        record = {"metadata": {"k": "v"}}
        meta = vs._record_to_metadata(record)
        assert meta == {"k": "v"}

    def test_default_record_to_metadata_empty(self):
        vs = _make_store()
        record = {"metadata": None}
        meta = vs._record_to_metadata(record)
        assert meta == {}

    def test_custom_record_to_metadata_merge(self):
        """JSON column + mapped columns are merged."""
        vs = _make_store(
            metadata_column_names=["price", "category"],
        )
        record = {
            "metadata": '{"tags": "new"}',  # JSON remainder
            "price": 100,
            "category": "electronics",
        }
        meta = vs._record_to_metadata(record)
        assert meta["tags"] == "new"
        assert meta["price"] == 100
        assert meta["category"] == "electronics"

    def test_custom_record_to_metadata_no_json(self):
        """When JSON column is None, only mapped columns are used."""
        vs = _make_store(
            metadata_json_column=None,
            metadata_column_names=["price"],
        )
        record = {"price": 50}
        meta = vs._record_to_metadata(record)
        assert meta == {"price": 50}

    def test_custom_record_to_metadata_null_mapped(self):
        """None values from mapped columns are skipped."""
        vs = _make_store(
            metadata_column_names=["price", "category"],
        )
        record = {
            "metadata": '{"tags": "new"}',
            "price": 100,
            "category": None,
        }
        meta = vs._record_to_metadata(record)
        assert "price" in meta
        assert "category" not in meta  # None values skipped

    def test_custom_record_to_metadata_mapped_overrides_json(self):
        """Mapped column values take precedence over JSON."""
        vs = _make_store(
            metadata_column_names=["price"],
        )
        record = {
            "metadata": '{"price": 50, "tags": "new"}',
            "price": 100,  # mapped column value overrides JSON
        }
        meta = vs._record_to_metadata(record)
        assert meta["price"] == 100  # mapped column wins
        assert meta["tags"] == "new"  # JSON-only key preserved


# ===========================================================================
# 11. Custom column support — Filter logic (mapped vs JSON_EXTRACT)
# ===========================================================================


class TestFilterClauseCustom:
    """Test _build_filter_clause with custom columns."""

    def test_default_filter_uses_json_extract(self):
        """Default schema: filter uses JSON_EXTRACT(metadata, ...)."""
        vs = _make_store()
        where, params = vs._build_filter_clause({"category": "phone"})
        assert "JSON_EXTRACT(metadata, '$.category')" in where
        assert params == ["phone"]

    def test_mapped_column_filter_uses_direct_reference(self):
        """Mapped column: filter uses direct column reference."""
        vs = _make_store(
            metadata_column_names=["category", "price"],
        )
        where, params = vs._build_filter_clause({"category": "phone"})
        assert "`category`" in where
        assert "JSON_EXTRACT" not in where
        assert params == ["phone"]

    def test_mapped_column_numeric_filter(self):
        vs = _make_store(
            metadata_column_names=["price"],
        )
        where, params = vs._build_filter_clause({"price": 100})
        assert "`price` = %s" in where
        assert "JSON_EXTRACT" not in where
        assert params == [100]

    def test_mapped_column_operator_filter(self):
        vs = _make_store(
            metadata_column_names=["price"],
        )
        where, params = vs._build_filter_clause(
            {"price": {"$gt": 50, "$lt": 200}}
        )
        assert "`price` > %s" in where
        assert "`price` < %s" in where
        assert "JSON_EXTRACT" not in where
        assert params == [50, 200]

    def test_mapped_column_in_filter(self):
        vs = _make_store(
            metadata_column_names=["category"],
        )
        where, params = vs._build_filter_clause(
            {"category": {"$in": ["phone", "tablet"]}}
        )
        assert "`category` IN" in where
        assert "JSON_EXTRACT" not in where
        assert params == ["phone", "tablet"]

    def test_unmapped_key_uses_json_extract(self):
        """Unmapped key with JSON column: falls back to JSON_EXTRACT."""
        vs = _make_store(
            metadata_column_names=["price"],
            metadata_json_column="meta",
        )
        where, params = vs._build_filter_clause({"tags": "new"})
        assert "JSON_EXTRACT(`meta`, '$.tags')" in where
        assert params == ["new"]

    def test_no_json_column_unmapped_key_raises(self):
        """No JSON column + unmapped key: raises ValueError."""
        vs = _make_store(
            metadata_json_column=None,
            metadata_column_names=["price"],
        )
        with pytest.raises(ValueError, match="Cannot filter on 'tags'"):
            vs._build_filter_clause({"tags": "new"})

    def test_no_json_column_mapped_key_works(self):
        """No JSON column + mapped key: works fine."""
        vs = _make_store(
            metadata_json_column=None,
            metadata_column_names=["price"],
        )
        where, params = vs._build_filter_clause({"price": 100})
        assert "`price` = %s" in where
        assert params == [100]

    def test_mixed_mapped_and_unmapped_filter(self):
        """Mix of mapped and unmapped keys in same filter."""
        vs = _make_store(
            metadata_column_names=["price", "category"],
            metadata_json_column="meta",
        )
        where, params = vs._build_filter_clause({
            "price": {"$gt": 50},
            "tags": "new",
        })
        # price is mapped → direct column
        assert "`price` > %s" in where
        # tags is unmapped → JSON_EXTRACT
        assert "JSON_EXTRACT(`meta`, '$.tags')" in where
        assert params == [50, "new"]

    def test_default_filter_preserves_backward_compat(self):
        """Default schema filter SQL should be identical to pre-custom-column."""
        vs = _make_store()
        where, params = vs._build_filter_clause({
            "category": "phone",
            "price": {"$gt": 100, "$lt": 1000},
        })
        # Must use metadata (no backticks) for backward compat
        assert "JSON_EXTRACT(metadata, '$.category')" in where
        assert "JSON_EXTRACT(metadata, '$.price')" in where
        assert "`metadata`" not in where  # no backticks in default path
        assert len(params) == 3  # phone, 100, 1000


# ===========================================================================
# 12. Custom column support — DELETE SQL + end-to-end integration
# ===========================================================================


class TestDeleteSqlCustom:
    """Test _build_delete_by_ids_sql with custom columns."""

    def test_default_delete_sql(self):
        vs = _make_store()
        sql = vs._build_delete_by_ids_sql("%s,%s")
        assert "DELETE FROM `test_table`" in sql
        assert "WHERE `id` IN (%s,%s)" in sql

    def test_custom_delete_sql(self):
        vs = _make_store(id_column="product_id")
        sql = vs._build_delete_by_ids_sql("%s,%s,%s")
        assert "WHERE `product_id` IN" in sql

    def test_default_partition_column_follows_id_column(self):
        """_partition_column should default to _id_column when not set."""
        vs = _make_store(id_column="product_id")
        assert vs._partition_column == "product_id"


class TestEndToEndCustomColumns:
    """Integration-level tests: SQL consistency across all builders."""

    def test_all_builders_use_consistent_column_names(self):
        """DDL, UPSERT, SELECT, DELETE all use the same column names."""
        from langchain_polardbx import Column

        vs = _make_store(
            id_column="product_id",
            content_column="description",
            embedding_column="embed",
            metadata_json_column="meta",
            metadata_column_objs=[
                Column("price", "DECIMAL(10,2)"),
                Column("category", "VARCHAR(50)"),
            ],
            metadata_column_names=["price", "category"],
        )

        ddl = vs._build_create_table_sql(4)
        upsert_sql = vs._build_upsert_sql()
        search_sql = vs._build_search_sql("VEC_DISTANCE", "", "")
        get_sql = vs._build_get_by_ids_sql("%s")
        delete_sql = vs._build_delete_by_ids_sql("%s")

        # DDL uses custom column names
        assert "`product_id` VARCHAR(36) PRIMARY KEY" in ddl
        assert "`description` LONGTEXT NOT NULL" in ddl
        assert "`meta` JSON" in ddl
        assert "`embed` VECTOR(4) NOT NULL" in ddl
        assert "VECTOR INDEX (`embed`)" in ddl

        # UPSERT uses same column names
        assert "`product_id`" in upsert_sql
        assert "`description`" in upsert_sql
        assert "`meta`" in upsert_sql
        assert "`embed`" in upsert_sql
        assert "`price`" in upsert_sql
        assert "`category`" in upsert_sql

        # SELECT uses aliases for stable record access
        assert "`product_id` AS `id`" in search_sql
        assert "`description` AS `text`" in search_sql
        assert "`meta` AS `metadata`" in search_sql
        assert "VEC_DISTANCE(`embed`" in search_sql

        # GET uses same aliases
        assert "`product_id` AS `id`" in get_sql

        # DELETE uses custom id column
        assert "WHERE `product_id` IN" in delete_sql

    def test_values_tuple_matches_upsert_column_order(self):
        """The _build_upsert_values output matches _build_upsert_sql columns."""
        from langchain_polardbx import Column

        vs = _make_store(
            id_column="pid",
            content_column="desc",
            embedding_column="emb",
            metadata_json_column="meta",
            metadata_column_objs=[
                Column("price", "INT"),
                Column("brand", "VARCHAR(50)"),
            ],
            metadata_column_names=["price", "brand"],
        )

        sql = vs._build_upsert_sql()
        vals = vs._build_upsert_values(
            "x1", "hello", {"price": 10, "brand": "Nike", "extra": "data"}, "v"
        )

        # SQL INSERT columns: pid, desc, meta, emb, price, brand (6 cols)
        assert "`pid`, `desc`, `meta`, `emb`, `price`, `brand`" in sql
        # VALUES placeholders: %s, %s, %s, VEC_FROMTEXT(%s), %s, %s (6 vals)
        assert vals[0] == "x1"  # pid
        assert vals[1] == "hello"  # desc
        # vals[2] is json of remaining
        assert vals[3] == "v"  # emb
        assert vals[4] == 10  # price
        assert vals[5] == "Nike"  # brand
        assert len(vals) == 6


# ===========================================================================
# 14. Audit fix tests — P1-1, P1-2, P2
# ===========================================================================


class TestColumnValidation:
    """Test Column.__post_init__ validation for data_type and default (P2)."""

    def test_valid_data_types_pass(self):
        from langchain_polardbx import Column

        for dt in [
            "VARCHAR(255)", "INT", "BIGINT", "DECIMAL(10,2)",
            "TEXT", "JSON", "LONGTEXT", "DATETIME", "BOOLEAN",
            "CHAR(36)", "FLOAT", "DOUBLE",
        ]:
            col = Column("test_col", dt)
            assert col.data_type == dt

    def test_data_type_with_semicolon_raises(self):
        from langchain_polardbx import Column

        with pytest.raises(ValueError, match="forbidden sequence"):
            Column("x", "INT; DROP TABLE users")

    def test_data_type_with_comment_raises(self):
        from langchain_polardbx import Column

        with pytest.raises(ValueError, match="forbidden sequence"):
            Column("x", "INT-- comment")

    def test_data_type_with_block_comment_raises(self):
        from langchain_polardbx import Column

        with pytest.raises(ValueError, match="forbidden sequence"):
            Column("x", "INT /* comment */")

    def test_data_type_with_newline_raises(self):
        from langchain_polardbx import Column

        with pytest.raises(ValueError, match="forbidden sequence"):
            Column("x", "INT\nDROP TABLE")

    def test_empty_data_type_raises(self):
        from langchain_polardbx import Column

        with pytest.raises(ValueError, match="non-empty"):
            Column("x", "")

    def test_default_with_semicolon_raises(self):
        from langchain_polardbx import Column

        with pytest.raises(ValueError, match="forbidden sequence"):
            Column("x", "INT", default="0; DROP TABLE users")

    def test_default_with_comment_raises(self):
        from langchain_polardbx import Column

        with pytest.raises(ValueError, match="forbidden sequence"):
            Column("x", "INT", default="0 -- comment")

    def test_default_none_passes(self):
        from langchain_polardbx import Column

        col = Column("x", "INT", default=None)
        assert col.default is None

    def test_valid_default_passes(self):
        from langchain_polardbx import Column

        col = Column("x", "VARCHAR(20)", default="'active'")
        assert col.default == "'active'"


class TestStringMetadataColumnsDDL:
    """Test DDL generation for string-only metadata_columns (P1-1)."""

    def test_string_columns_get_text_type(self):
        """String metadata_columns get TEXT type in DDL."""
        vs = _make_store(
            metadata_column_names=["category", "price"],
        )
        sql = vs._build_create_table_sql(4)
        assert "`category` TEXT" in sql
        assert "`price` TEXT" in sql

    def test_mixed_column_and_string(self):
        """Column objects use their type; strings get TEXT."""
        from langchain_polardbx import Column

        vs = _make_store(
            metadata_column_objs=[Column("price", "DECIMAL(10,2)")],
            metadata_column_names=["price", "category"],
        )
        sql = vs._build_create_table_sql(4)
        assert "`price` DECIMAL(10,2)" in sql
        assert "`category` TEXT" in sql

    def test_string_only_all_get_text(self):
        """All string columns get TEXT."""
        vs = _make_store(
            metadata_column_names=["a", "b", "c"],
        )
        sql = vs._build_create_table_sql(4)
        assert "`a` TEXT" in sql
        assert "`b` TEXT" in sql
        assert "`c` TEXT" in sql

    def test_ddl_and_dml_consistent_for_strings(self):
        """DDL and UPSERT SQL reference same column names for strings."""
        vs = _make_store(
            metadata_column_names=["category", "price"],
        )
        ddl = vs._build_create_table_sql(4)
        upsert_sql = vs._build_upsert_sql()

        # DDL creates these columns
        assert "`category`" in ddl
        assert "`price`" in ddl

        # UPSERT references same columns
        assert "`category`" in upsert_sql
        assert "`price`" in upsert_sql


class TestNotNullValidation:
    """Test NOT NULL validation in _build_upsert_values (P1-2)."""

    def test_not_null_missing_value_raises(self):
        from langchain_polardbx import Column

        vs = _make_store(
            metadata_column_objs=[Column("status", "VARCHAR(20)", nullable=False)],
            metadata_column_names=["status"],
        )
        with pytest.raises(ValueError, match="NOT NULL"):
            vs._build_upsert_values("id1", "text", {}, "vec_str")

    def test_nullable_missing_value_passes(self):
        from langchain_polardbx import Column

        vs = _make_store(
            metadata_column_objs=[Column("status", "VARCHAR(20)", nullable=True)],
            metadata_column_names=["status"],
        )
        # Should not raise; None is inserted
        vals = vs._build_upsert_values("id1", "text", {}, "vec_str")
        assert "status" not in {}  # sanity check
        # The last element should be None
        assert vals[-1] is None

    def test_not_null_with_value_passes(self):
        from langchain_polardbx import Column

        vs = _make_store(
            metadata_column_objs=[Column("status", "VARCHAR(20)", nullable=False)],
            metadata_column_names=["status"],
        )
        vals = vs._build_upsert_values(
            "id1", "text", {"status": "active"}, "vec_str"
        )
        assert vals[-1] == "active"

    def test_string_column_missing_no_validation(self):
        """String-only columns (no Column obj) can't check nullable."""
        vs = _make_store(
            metadata_column_names=["category"],
        )
        # Should not raise — we can't know nullable for string-only
        vals = vs._build_upsert_values("id1", "text", {}, "vec_str")
        assert vals[-1] is None

    def test_not_null_missing_with_default_still_raises(self):
        """Even with a DEFAULT, NOT NULL + missing value raises ValueError."""
        from langchain_polardbx import Column

        vs = _make_store(
            metadata_column_objs=[
                Column("status", "VARCHAR(20)", nullable=False, default="'active'")
            ],
            metadata_column_names=["status"],
        )
        with pytest.raises(ValueError, match="NOT NULL"):
            vs._build_upsert_values("id1", "text", {}, "vec_str")

    def test_not_null_explicit_none_raises(self):
        """Explicit None value for NOT NULL column also raises."""
        from langchain_polardbx import Column

        vs = _make_store(
            metadata_column_objs=[Column("status", "VARCHAR(20)", nullable=False)],
            metadata_column_names=["status"],
        )
        with pytest.raises(ValueError, match="NOT NULL"):
            vs._build_upsert_values(
                "id1", "text", {"status": None}, "vec_str"
            )

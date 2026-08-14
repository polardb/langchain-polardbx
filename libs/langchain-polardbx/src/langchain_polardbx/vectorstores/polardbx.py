"""PolarDB-X Vector Store.

This module provides a LangChain VectorStore implementation using
PolarDB-X with native vector search capabilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from mysql.connector.pooling import PooledMySQLConnection

logger = logging.getLogger(__name__)


class NotSupportedError(NotImplementedError):
    """Raised when a feature is not supported by the current PolarDB-X version."""

    pass


# Supported filter operators for dict-style filters
FILTER_OPERATORS = {
    "$eq": "=",
    "$ne": "!=",
    "$gt": ">",
    "$gte": ">=",
    "$lt": "<",
    "$lte": "<=",
    "$in": "IN",
    "$nin": "NOT IN",
    "$like": "LIKE",
}

# Default SQL templates
SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS `{table_name}` (
    id VARCHAR(36) PRIMARY KEY,
    text LONGTEXT NOT NULL,
    metadata JSON,
    embedding VECTOR({dimension}) NOT NULL,
    VECTOR INDEX (embedding) M={hnsw_m}{index_extra} DISTANCE={distance_function}
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

# Partition clause is built dynamically by _build_partition_clause()

SQL_INSERT = """
INSERT INTO `{table_name}` (id, text, metadata, embedding)
VALUES (%s, %s, %s, VEC_FROMTEXT(%s))
"""

SQL_UPSERT = """
INSERT INTO `{table_name}` (id, text, metadata, embedding)
VALUES (%s, %s, %s, VEC_FROMTEXT(%s))
ON DUPLICATE KEY UPDATE
    text = VALUES(text),
    metadata = VALUES(metadata),
    embedding = VALUES(embedding)
"""

SQL_SEARCH = """
SELECT id, text, metadata,
       {distance_func}(embedding, VEC_FROMTEXT(%s)) AS distance
FROM `{table_name}`{index_hint}
{where_clause}
ORDER BY distance
LIMIT %s
"""

SQL_DELETE_BY_IDS = """
DELETE FROM `{table_name}` WHERE id IN ({placeholders})
"""

SQL_GET_BY_IDS = """
SELECT id, text, metadata FROM `{table_name}` WHERE id IN ({placeholders})
"""


class PolarDBXVectorStore(VectorStore):
    """PolarDB-X Vector Store.

    This class provides a vector store implementation using PolarDB-X
    with native vector search capabilities (VECTOR data type and VECTOR INDEX).

    Requirements:
        - PolarDB-X with vector index enabled (vidx_disabled = OFF)
        - DN version >= 20260605

    Example:
        .. code-block:: python

            from langchain_polardbx import PolarDBXVectorStore
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()

            vectorstore = PolarDBXVectorStore(
                host="your-rds-host.mysql.rds.aliyuncs.com",
                port=3306,
                user="your-user",
                password="your-password",
                database="your-database",
                embedding=embeddings,
                table_name="polardbx_vectors",
            )

            # Add documents
            vectorstore.add_texts(["Hello world", "LangChain is great"])

            # Search for similar documents
            results = vectorstore.similarity_search("Hello", k=2)
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        embedding: Embeddings,
        table_name: str = "polardbx_vectors",
        distance_strategy: Literal["cosine", "euclidean", "inner_product"] = "cosine",
        hnsw_m: int = 6,
        pool_size: int = 5,
        pre_delete_table: bool = False,
        *,
        embedding_dimension: Optional[int] = None,
        ef_construction: Optional[int] = None,
        connection_retries: int = 3,
        retry_delay: float = 1.0,
        vector_index_name: Optional[str] = None,
        partition_by: Optional[str] = None,
        partitions: int = 0,
        partition_column: Optional[str] = None,
        broadcast: bool = False,
        locality: Optional[str] = None,
        partition_defs: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the PolarDB-X vector store.

        Args:
            host: The MySQL host address.
            port: The MySQL port number.
            user: The MySQL username.
            password: The MySQL password.
            database: The MySQL database name.
            embedding: The embedding model to use.
            table_name: The name of the table to store vectors. Defaults to
                "polardbx_vectors".
            distance_strategy: Distance function for vector search. One of
                "cosine", "euclidean", or "inner_product" (v3 only).
                Defaults to "cosine".
            hnsw_m: M parameter for HNSW index (3-200). Higher values = more accurate
                but slower indexing. Defaults to 6.
            pool_size: Connection pool size. Defaults to 5.
            pre_delete_table: If True, delete the table before creating.
                Defaults to False.
            embedding_dimension: Embedding dimension. If not provided, will be
                inferred from the embedding model.
            ef_construction: HNSW build-time candidate list size (5-1000, v3 only).
                Larger values improve index quality at the cost of slower builds.
                Ignored on old versions. Defaults to None.
            connection_retries: Number of connection retry attempts. Defaults to 3.
            retry_delay: Delay between retry attempts in seconds. Defaults to 1.0.
            vector_index_name: Name of the vector index for FORCE INDEX hints.
                If None, auto-detected on first use. Defaults to None.
            partition_by: Partition strategy for the table. One of "HASH",
                "KEY", "RANGE", "LIST", or None. If None and broadcast is
                False, creates a single (non-partitioned) table.
                Defaults to None.
            partitions: Number of partitions. Required when partition_by
                is "HASH" or "KEY". Defaults to 0.
            partition_column: Column to partition on. Defaults to "id".
            broadcast: If True, creates a broadcast table (full copy on
                every DN node). Mutually exclusive with partition_by.
                Defaults to False.
            locality: Storage node specification, e.g. "dn=xxx".
                Appended to DDL as LOCALITY clause. Defaults to None.
            partition_defs: Partition definitions for RANGE/LIST
                strategies. Each dict has a "name" key and either
                "values_less_than" (RANGE) or "values_in" (LIST).
                Defaults to None.
            **kwargs: Additional connection arguments passed to both sync and
                async connection pools (e.g. ssl_ca, ssl_cert, ssl_key,
                ssl_disabled for SSL/TLS encryption).
        """
        try:
            import mysql.connector.pooling
        except ImportError as e:
            raise ImportError(
                "Could not import mysql-connector-python. "
                "Please install it with `pip install mysql-connector-python`."
            ) from e

        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = self._validate_identifier(database, "database name")
        self._embedding = embedding
        self._table_name = self._validate_table_name(table_name)
        self._distance_strategy = distance_strategy.lower()
        self._hnsw_m = hnsw_m
        self._pool_size = pool_size
        self._embedding_dimension = embedding_dimension
        self._connection_retries = connection_retries
        self._retry_delay = retry_delay
        self._vector_index_name = vector_index_name
        self._partition_by = partition_by.upper() if partition_by else None
        self._partitions = partitions
        self._partition_column = partition_column or "id"
        self._broadcast = broadcast
        self._locality = locality
        self._partition_defs = partition_defs
        self._ef_construction = ef_construction
        self._conn_kwargs = kwargs  # Extra connection args (e.g. SSL params)

        # Capabilities detected at init time (filled by _detect_capabilities)
        self._capabilities: Dict[str, bool] = {}

        # Validate distance strategy
        if self._distance_strategy not in (
            "cosine", "euclidean", "inner_product"
        ):
            raise ValueError(
                f"Invalid distance_strategy: {distance_strategy}. "
                "Must be 'cosine', 'euclidean', or 'inner_product'."
            )

        # Validate inner_product requires v3 capability
        # (checked after _detect_capabilities, just store for now)

        # Validate partition params
        if self._partition_by and self._partition_by not in (
            "HASH", "KEY", "RANGE", "LIST"
        ):
            raise ValueError(
                f"Invalid partition_by: {partition_by}. "
                "Must be 'HASH', 'KEY', 'RANGE', or 'LIST'."
            )
        if self._partition_by in ("HASH", "KEY") and self._partitions <= 0:
            raise ValueError(
                "partitions must be > 0 when partition_by is "
                "'HASH' or 'KEY'."
            )
        if self._partition_by in ("RANGE", "LIST") and not self._partition_defs:
            raise ValueError(
                "partition_defs must be provided when partition_by is "
                "'RANGE' or 'LIST'."
            )
        if self._broadcast and self._partition_by:
            raise ValueError(
                "broadcast and partition_by are mutually exclusive. "
                "Use one or the other."
            )
        if self._partition_by and self._partition_column != "id":
            self._validate_identifier(
                self._partition_column, "partition column"
            )

        # Create connection pool
        pool_config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "charset": "utf8mb4",
            "autocommit": True,
            "pool_name": f"polardbx_{table_name}_{uuid.uuid4().hex[:8]}",
            "pool_size": pool_size,
            "pool_reset_session": True,
            "connect_timeout": 30,  # Connection timeout in seconds
            "connection_timeout": 30,  # Alias for connect_timeout
        }
        # Merge extra connection args (e.g. SSL/TLS params) from kwargs
        pool_config.update(self._conn_kwargs)
        self._pool = mysql.connector.pooling.MySQLConnectionPool(**pool_config)  # type: ignore[arg-type]

        # Async pool will be lazily initialized
        self._async_pool: Optional[Any] = None
        self._async_pool_lock = asyncio.Lock()

        # Detect capabilities and check vector support
        # — close pool on failure to avoid resource leak
        try:
            self._detect_capabilities()
        except Exception:
            self.close()
            raise

        # Validate inner_product requires v3 distance functions
        if (
            self._distance_strategy == "inner_product"
            and not self._capabilities.get("vec_distance", False)
        ):
            self.close()
            raise NotSupportedError(
                "distance_strategy='inner_product' requires PolarDB-X v3 "
                "with VEC_DISTANCE_INNER_PRODUCT support. "
                "Use 'cosine' or 'euclidean' for old versions."
            )

        # Validate partition/broadcast is not supported on v3 instances
        # (v3 does not support vector indexes on partitioned/broadcast tables)
        if (
            (self._partition_by or self._broadcast)
            and self._capabilities.get("vec_distance", False)
        ):
            self.close()
            raise NotSupportedError(
                "partition_by and broadcast are not supported on "
                "PolarDB-X v3 instances because v3 does not support "
                "vector indexes on partitioned/broadcast tables. "
                "Omit the partition_by/broadcast parameter to create "
                "a non-partitioned table."
            )

        # Handle table creation
        if pre_delete_table:
            self._drop_table()

    @property
    def embeddings(self) -> Embeddings:
        """Return the embedding model."""
        return self._embedding

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Select the relevance score function based on distance strategy."""
        if self._distance_strategy == "cosine":
            return self._cosine_relevance_score_fn
        elif self._distance_strategy == "euclidean":
            return self._euclidean_relevance_score_fn
        elif self._distance_strategy == "inner_product":
            return self._inner_product_relevance_score_fn
        else:
            raise ValueError(
                f"Unsupported distance strategy: {self._distance_strategy}. "
                "Must be 'cosine', 'euclidean', or 'inner_product'."
            )

    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        """Convert cosine distance to relevance score (0-1)."""
        return max(0.0, 1.0 - distance / 2.0)

    @staticmethod
    def _euclidean_relevance_score_fn(distance: float) -> float:
        """Convert euclidean distance to relevance score (0-1)."""
        return 1.0 / (1.0 + distance)

    @staticmethod
    def _inner_product_relevance_score_fn(distance: float) -> float:
        """Convert inner-product distance to relevance score (0-1).

        VEC_DISTANCE_INNER_PRODUCT returns the *negative* dot product
        (smaller distance = larger dot product = more similar).
        """
        if distance < 0:
            return 1.0
        return max(0.0, 1.0 - distance)

    @staticmethod
    def _validate_table_name(table_name: str) -> str:
        """Validate table name to prevent SQL injection.

        Args:
            table_name: The table name to validate.

        Returns:
            The validated table name.

        Raises:
            ValueError: If the table name is invalid.
        """
        # Only allow alphanumeric characters and underscores
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", table_name):
            raise ValueError(
                f"Invalid table name: {table_name}. "
                "Table name must start with a letter or underscore, "
                "and contain only alphanumeric characters and underscores."
            )
        if len(table_name) > 64:
            raise ValueError(
                f"Table name too long: {table_name}. Maximum length is 64 characters."
            )
        return table_name

    @staticmethod
    def _validate_identifier(name: str, label: str = "identifier") -> str:
        """Validate a SQL identifier (index name, etc.) to prevent SQL injection.

        Args:
            name: The identifier to validate.
            label: Human-readable label for error messages.

        Returns:
            The validated identifier.

        Raises:
            ValueError: If the identifier is invalid.
        """
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(
                f"Invalid {label}: {name}. "
                "Must start with a letter or underscore, "
                "and contain only alphanumeric characters and underscores."
            )
        if len(name) > 64:
            raise ValueError(
                f"{label.capitalize()} too long: {name}. Maximum length is 64 characters."
            )
        return name

    def close(self) -> None:
        """Close the connection pool.

        Call this method when you're done using the vector store
        to release database connections.
        """
        if hasattr(self, "_pool") and self._pool:
            try:
                pool = self._pool
                self._pool = None  # type: ignore

                # Access the internal queue directly to avoid blocking.
                # MySQL Connector's pool stores connections in _cnx_queue.
                # This is a private API and may break in future versions of
                # mysql-connector-python. If it does, fall back to pool.closeall().
                if hasattr(pool, "_cnx_queue"):
                    while not pool._cnx_queue.empty():
                        try:
                            cnx = pool._cnx_queue.get_nowait()
                            if cnx and hasattr(cnx, "disconnect"):
                                # Use disconnect() instead of close()
                                # to avoid reset_session()
                                cnx.disconnect()
                        except Exception:
                            break
            except Exception as e:
                logger.debug("Error while closing pool connections: %s", e)

            logger.info("Connection pool closed for table %s", self._table_name)

    async def aclose(self) -> None:
        """Async close the connection pools.

        Call this method when you're done using the vector store
        to release database connections.
        """
        self.close()
        if hasattr(self, "_async_pool") and self._async_pool:
            self._async_pool.close()
            await self._async_pool.wait_closed()
            self._async_pool = None
            logger.info("Async connection pool closed for table %s", self._table_name)

    def __del__(self) -> None:
        """Destructor to clean up resources."""
        # Only close sync connections in __del__
        # Async connections should be closed via aclose() before event loop closes
        # Closing async connections here can cause "Event loop is closed" errors
        try:
            self.close()
            # Clear async pool reference to prevent __del__ from trying to close it
            # after event loop is closed
            if hasattr(self, "_async_pool"):
                self._async_pool = None
        except Exception:
            # Ignore errors during cleanup in __del__
            pass

    def _get_connection_with_retry(self) -> "PooledMySQLConnection":
        """Get a database connection with retry logic.

        Returns:
            A pooled MySQL connection.

        Raises:
            Exception: If all retry attempts fail.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self._connection_retries):
            try:
                return self._pool.get_connection()
            except Exception as e:
                last_exception = e
                if attempt < self._connection_retries - 1:
                    logger.warning(
                        "Connection attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt + 1,
                        self._connection_retries,
                        str(e),
                        self._retry_delay,
                    )
                    time.sleep(self._retry_delay)
                else:
                    logger.error(
                        "All %d connection attempts failed. Last error: %s",
                        self._connection_retries,
                        str(e),
                    )

        raise last_exception  # type: ignore

    @contextmanager
    def _get_cursor(
        self,
    ) -> Generator[Any, None, None]:
        """Get a database cursor from the connection pool with retry."""
        conn: PooledMySQLConnection = self._get_connection_with_retry()
        cursor = conn.cursor(dictionary=True)
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    async def _get_async_pool(self) -> Any:
        """Get or create the async connection pool (lazy initialization) with retry."""
        if self._async_pool is not None:
            return self._async_pool

        async with self._async_pool_lock:
            # Double-check after acquiring lock
            if self._async_pool is not None:
                return self._async_pool

            try:
                import aiomysql  # type: ignore[import-untyped]
            except ImportError as e:
                raise ImportError(
                    "Could not import aiomysql. "
                    "Please install it with `pip install aiomysql` "
                    "to use async methods."
                ) from e

            last_exception: Optional[Exception] = None

            for attempt in range(self._connection_retries):
                try:
                    async_config = {
                        "host": self._host,
                        "port": self._port,
                        "user": self._user,
                        "password": self._password,
                        "db": self._database,
                        "charset": "utf8mb4",
                        "autocommit": True,
                        "minsize": 1,
                        "maxsize": self._pool_size,
                        "connect_timeout": 30,
                    }
                    async_config.update(self._conn_kwargs)
                    self._async_pool = await aiomysql.create_pool(**async_config)
                    return self._async_pool
                except Exception as e:
                    last_exception = e
                    if attempt < self._connection_retries - 1:
                        logger.warning(
                            "Async pool creation attempt %d/%d failed: %s. "
                            "Retrying in %.1fs...",
                            attempt + 1,
                            self._connection_retries,
                            str(e),
                            self._retry_delay,
                        )
                        await asyncio.sleep(self._retry_delay)
                    else:
                        logger.error(
                            "All %d async pool creation attempts failed. "
                            "Last error: %s",
                            self._connection_retries,
                            str(e),
                        )

            raise last_exception  # type: ignore

    async def _aget_connection_with_retry(self, pool: Any) -> Any:
        """Get an async connection with retry logic.

        Args:
            pool: The async connection pool.

        Returns:
            An async MySQL connection.

        Raises:
            Exception: If all retry attempts fail.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self._connection_retries):
            try:
                return await pool.acquire()
            except Exception as e:
                last_exception = e
                if attempt < self._connection_retries - 1:
                    logger.warning(
                        "Async connection attempt %d/%d failed: %s. "
                        "Retrying in %.1fs...",
                        attempt + 1,
                        self._connection_retries,
                        str(e),
                        self._retry_delay,
                    )
                    await asyncio.sleep(self._retry_delay)
                else:
                    logger.error(
                        "All %d async connection attempts failed. Last error: %s",
                        self._connection_retries,
                        str(e),
                    )

        raise last_exception  # type: ignore

    @asynccontextmanager
    async def _aget_cursor(self) -> AsyncGenerator[Any, None]:
        """Get an async database cursor from the connection pool with retry."""
        import aiomysql

        pool = await self._get_async_pool()
        conn = await self._aget_connection_with_retry(pool)
        try:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                try:
                    yield cursor
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise
        finally:
            pool.release(conn)

    def _detect_capabilities(self) -> None:
        """Detect PolarDB-X vector capabilities and check vector support.

        Probes the database for vector feature availability and caches
        results in ``self._capabilities`` for later use by conditional
        code paths.

        Detected capabilities:
            - vec_distance: VEC_DISTANCE() function (DN pushdown with HNSW)
            - vec_totext: VEC_TOTEXT() function
            - vec_dim: VECTOR_DIM() function
            - vector_indexes_view: information_schema.VECTOR_INDEXES view

        Raises:
            ValueError: If vector index is disabled or vector functions
                are not available.
        """
        with self._get_cursor() as cursor:
            try:
                # Check if vector index is disabled via system variable
                cursor.execute("SHOW GLOBAL VARIABLES LIKE 'vidx_disabled'")
                result = cursor.fetchone()
                if result and result.get("Value", "").upper() == "ON":
                    raise ValueError(
                        "PolarDB-X vector index is disabled. "
                        "Please execute SET GLOBAL vidx_disabled = OFF and reconnect."
                    )

                # Verify vector functions are available
                cursor.execute(
                    "SELECT VEC_FROMTEXT('[1,2,3]') IS NOT NULL as vector_support"
                )
                result = cursor.fetchone()
                if not result or not result.get("vector_support"):
                    raise ValueError(
                        "PolarDB-X vector functions are not available. "
                        "Please verify the DN version is >= 20260605 "
                        "and vector index support is enabled."
                    )

            except ValueError:
                raise
            except Exception as e:
                if "FUNCTION" in str(e) and "VEC_FROMTEXT" in str(e):
                    raise ValueError(
                        "PolarDB-X vector functions are not available. "
                        "Please verify the DN version is >= 20260605 "
                        "and vector index support is enabled."
                    ) from e
                raise

        # Probe extended capabilities (non-fatal — default to False)
        # VEC_DISTANCE needs special handling: on DN the function exists
        # but errors without a vector-index context, so we inspect the
        # error message to distinguish "exists" from "not found".
        caps = {
            "vec_distance": self._probe_vec_distance(),
            "vec_totext": self._probe_function(
                None,
                "SELECT VEC_TOTEXT(VEC_FROMTEXT('[1,2,3]'))"
                " IS NOT NULL",
            ),
            "vec_dim": self._probe_function(
                None,
                "SELECT VECTOR_DIM(VEC_FROMTEXT('[1,2,3]'))"
                " IS NOT NULL",
            ),
        }
        caps["vector_indexes_view"] = self._probe_table_exists(
            "information_schema", "VECTOR_INDEXES"
        )
        self._capabilities = caps
        logger.info("Detected capabilities: %s", self._capabilities)

    def _probe_vec_distance(self) -> bool:
        """Probe whether VEC_DISTANCE() is available.

        VEC_DISTANCE requires a vector-index context to infer the distance
        metric, so a standalone ``SELECT VEC_DISTANCE(...)`` fails on DN
        even though the function exists.  We inspect the error message to
        tell apart "function exists but needs index" from "function not
        found".

        Returns:
            True if the function is available, False otherwise.
        """
        sql = (
            "SELECT VEC_DISTANCE(VEC_FROMTEXT('[1,2,3]'),"
            " VEC_FROMTEXT('[1,2,3]')) IS NOT NULL"
        )
        try:
            with self._get_cursor() as cursor:
                cursor.execute(sql)
                return cursor.fetchone() is not None
        except Exception as e:
            err_msg = str(e).upper()
            # Function exists but needs a vector index to determine distance type
            if "NO VECTOR INDEX" in err_msg or "CANNOT DETERMINE" in err_msg:
                logger.debug(
                    "VEC_DISTANCE exists but needs index context: %s", e
                )
                return True
            logger.debug("VEC_DISTANCE probe failed: %s", e)
            return False

    def _probe_function(self, cursor_fn: Any, sql: str) -> bool:
        """Probe whether a SQL function is available.

        Args:
            cursor_fn: Unused (kept for API compatibility).
            sql: The SQL statement to test.

        Returns:
            True if the function is available, False otherwise.
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute(sql)
                return cursor.fetchone() is not None
        except Exception as e:
            logger.debug("Function probe failed [%s]: %s", sql, e)
            return False

    def _probe_table_exists(self, schema: str, table: str) -> bool:
        """Check if a table/view exists in information_schema.

        Args:
            schema: The schema name.
            table: The table/view name.

        Returns:
            True if the table/view exists, False otherwise.
        """
        try:
            with self._get_cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) AS cnt FROM information_schema.tables "
                    "WHERE table_schema = %s AND table_name = %s",
                    (schema, table),
                )
                result = cursor.fetchone()
                return result["cnt"] > 0 if result else False
        except Exception as e:
            logger.debug("Table probe failed [%s.%s]: %s", schema, table, e)
            return False

    def _get_distance_func(self) -> str:
        """Return the optimal distance function for the current instance.

        Prefers ``VEC_DISTANCE`` (DN pushdown, HNSW accelerated) when
        available; falls back to explicit ``VEC_DISTANCE_COSINE`` or
        ``VEC_DISTANCE_EUCLIDEAN`` otherwise.

        Returns:
            The distance function name to use in SQL.
        """
        if self._capabilities.get("vec_distance", False):
            return "VEC_DISTANCE"
        if self._distance_strategy == "cosine":
            return "VEC_DISTANCE_COSINE"
        if self._distance_strategy == "inner_product":
            return "VEC_DISTANCE_INNER_PRODUCT"
        return "VEC_DISTANCE_EUCLIDEAN"

    def _build_partition_clause(self) -> str:
        """Build the PARTITION/BROADCAST/LOCALITY clause for CREATE TABLE.

        Returns an empty string for a single (non-partitioned) table.
        """
        parts: List[str] = []

        if self._broadcast:
            parts.append("BROADCAST")
        elif self._partition_by:
            col = self._partition_column
            if self._partition_by in ("HASH", "KEY"):
                parts.append(
                    f"PARTITION BY {self._partition_by}({col}) "
                    f"PARTITIONS {self._partitions}"
                )
            elif self._partition_by == "RANGE":
                defs = self._build_range_partition_defs()
                parts.append(f"PARTITION BY RANGE({col}) {defs}")
            elif self._partition_by == "LIST":
                defs = self._build_list_partition_defs()
                parts.append(f"PARTITION BY LIST({col}) {defs}")

        if self._locality:
            parts.append(f"LOCALITY='{self._locality}'")

        return "".join(f" {p}" for p in parts) if parts else ""

    def _build_range_partition_defs(self) -> str:
        """Build the RANGE partition definitions list."""
        if not self._partition_defs:
            raise ValueError(
                "partition_defs required for RANGE partitioning"
            )
        items = []
        for d in self._partition_defs:
            name = d["name"]
            vlt = d["values_less_than"]
            if isinstance(vlt, str) and vlt.upper() == "MAXVALUE":
                items.append(
                    f"PARTITION {name} VALUES LESS THAN (MAXVALUE)"
                )
            else:
                items.append(
                    f"PARTITION {name} VALUES LESS THAN ({vlt})"
                )
        return "(" + ", ".join(items) + ")"

    def _build_list_partition_defs(self) -> str:
        """Build the LIST partition definitions list."""
        if not self._partition_defs:
            raise ValueError(
                "partition_defs required for LIST partitioning"
            )
        items = []
        for d in self._partition_defs:
            name = d["name"]
            vals = d["values_in"]
            val_str = ", ".join(
                repr(v) if isinstance(v, str) else str(v)
                for v in vals
            )
            items.append(
                f"PARTITION {name} VALUES IN ({val_str})"
            )
        return "(" + ", ".join(items) + ")"

    def _build_create_table_sql(self, dimension: int) -> str:
        """Build the CREATE TABLE SQL with optional partition clause.

        Args:
            dimension: The vector embedding dimension.

        Returns:
            The complete CREATE TABLE SQL statement.
        """
        # Build optional EF_CONSTRUCTION clause (v3 only)
        index_extra = ""
        if (
            self._ef_construction is not None
            and self._capabilities.get("vec_distance", False)
        ):
            index_extra = f" EF_CONSTRUCTION={self._ef_construction}"

        base_sql = SQL_CREATE_TABLE.format(
            table_name=self._table_name,
            dimension=dimension,
            hnsw_m=self._hnsw_m,
            distance_function=self._distance_strategy.upper(),
            index_extra=index_extra,
        )
        partition_clause = self._build_partition_clause()
        if partition_clause:
            return base_sql.rstrip() + partition_clause
        return base_sql

    def _detect_vector_index_name(self) -> Optional[str]:
        """Auto-detect the vector index name.

        On v3 instances, queries ``information_schema.VECTOR_INDEXES``
        for the index name.  Falls back to ``SHOW CREATE TABLE`` +
        regex parsing on older versions.
        """
        if self._vector_index_name is not None:
            return self._vector_index_name

        # v3: use information_schema.VECTOR_INDEXES view (preferred)
        if self._capabilities.get("vector_indexes_view", False):
            try:
                with self._get_cursor() as cursor:
                    cursor.execute(
                        "SELECT INDEX_NAME FROM information_schema.VECTOR_INDEXES "
                        "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s "
                        "LIMIT 1",
                        (self._database, self._table_name),
                    )
                    row = cursor.fetchone()
                    if row:
                        name = row.get("INDEX_NAME", row.get("index_name", ""))
                        if name:
                            self._vector_index_name = name
                            return self._vector_index_name
            except Exception as e:
                logger.debug("VECTOR_INDEXES query failed: %s", e)

        # Fallback: parse SHOW CREATE TABLE with regex
        try:
            with self._get_cursor() as cursor:
                cursor.execute(f"SHOW CREATE TABLE `{self._table_name}`")
                row = cursor.fetchone()
                if row:
                    create_sql = (
                        row.get("Create Table")
                        or row.get("CREATE TABLE")
                        or ""
                    )
                    m = re.search(
                        r"VECTOR INDEX `([^`]+)`", create_sql, re.IGNORECASE
                    )
                    if m:
                        self._vector_index_name = m.group(1)
                        return self._vector_index_name
        except Exception as e:
            logger.debug("Failed to detect vector index name: %s", e)
        return None

    def _build_index_hint(self, search_type: Optional[str]) -> str:
        """Build index hint string for search_type.

        - knn: FORCE INDEX(PRIMARY) to force full table scan (brute force)
        - ann: FORCE INDEX(vector_index) to force vector index usage
        - auto/None: no hint, let optimizer decide
        """
        if search_type is None or search_type == "auto":
            return ""
        if search_type == "knn":
            # FORCE INDEX(PRIMARY) works on both old and new versions
            # to bypass the vector index and force a full scan
            return " FORCE INDEX(PRIMARY)"
        if search_type == "ann":
            idx_name = self._detect_vector_index_name()
            if idx_name:
                return f" FORCE INDEX(`{idx_name}`)"
            return ""
        if search_type == "ignore":
            # IGNORE INDEX forces the optimizer to skip the vector index
            idx_name = self._detect_vector_index_name()
            if idx_name:
                return f" IGNORE INDEX(`{idx_name}`)"
            return ""
        return ""

    def _set_ef_search(self, cursor: Any, ef_search: Optional[int]) -> None:
        """Set ef_search session variable if specified."""
        if ef_search is not None:
            cursor.execute(f"SET SESSION vidx_hnsw_ef_search = {int(ef_search)}")

    # ---- Phase 2: Dynamic vector index management ----

    def apply_vector_index(
        self,
        index_name: str = "vi",
        m: Optional[int] = None,
        distance: Optional[str] = None,
        ef_construction: Optional[int] = None,
    ) -> None:
        """Create a vector index on the embedding column dynamically.

        Args:
            index_name: Name for the vector index. Defaults to "vi".
            m: HNSW M parameter (3-200). Defaults to the store's hnsw_m.
            distance: Distance function ("COSINE", "EUCLIDEAN", or
                "INNER_PRODUCT"). Defaults to the store's distance_strategy.
            ef_construction: HNSW build-time candidate list size (5-1000).
                v3 only; silently ignored on old versions.
                Defaults to the store's ef_construction if set.
        """
        self._validate_identifier(index_name, "index name")
        m_val = m or self._hnsw_m
        dist_val = (distance or self._distance_strategy).upper()
        if dist_val not in ("COSINE", "EUCLIDEAN", "INNER_PRODUCT"):
            raise ValueError(
                f"Invalid distance function: {dist_val}. "
                "Must be 'COSINE', 'EUCLIDEAN', or 'INNER_PRODUCT'."
            )
        ef_val = ef_construction or self._ef_construction
        # Build optional EF_CONSTRUCTION clause (v3 only)
        ef_clause = ""
        if ef_val is not None and self._capabilities.get("vec_distance", False):
            ef_clause = f" EF_CONSTRUCTION={ef_val}"
        with self._get_cursor() as cursor:
            cursor.execute(
                f"ALTER TABLE `{self._table_name}` "
                f"ADD VECTOR INDEX `{index_name}` (embedding) "
                f"M={m_val}{ef_clause} DISTANCE={dist_val}"
            )
        self._vector_index_name = index_name
        logger.info("Vector index '%s' created on table %s", index_name, self._table_name)

    def drop_vector_index(self, index_name: Optional[str] = None) -> None:
        """Drop a vector index.

        Args:
            index_name: Name of the index to drop. If None, uses the detected/stored name.
        """
        name = index_name or self._detect_vector_index_name()
        if not name:
            raise ValueError("No vector index name specified or detectable.")
        with self._get_cursor() as cursor:
            cursor.execute(f"ALTER TABLE `{self._table_name}` DROP INDEX `{name}`")
        self._vector_index_name = None
        logger.info("Vector index '%s' dropped from table %s", name, self._table_name)

    # ---- Phase 2: Monitoring & maintenance ----

    def get_stats(self) -> Dict[str, Any]:
        """Get vector index runtime statistics from PolarDB-X.

        Returns:
            Dictionary of Vidx* status variables.
        """
        with self._get_cursor() as cursor:
            cursor.execute("SHOW GLOBAL STATUS LIKE 'Vidx%'")
            return {
                row.get("Variable_name", row.get("variable_name", "")): (
                    row.get("Value", row.get("value", ""))
                )
                for row in cursor
            }

    def optimize(self) -> None:
        """Rebuild the vector index to reclaim space and improve recall."""
        with self._get_cursor() as cursor:
            cursor.execute(f"OPTIMIZE TABLE `{self._table_name}`")
            cursor.fetchall()
        logger.info("OPTIMIZE TABLE executed on %s", self._table_name)

    async def aoptimize(self) -> None:
        """Async rebuild the vector index."""
        async with self._aget_cursor() as cursor:
            await cursor.execute(f"OPTIMIZE TABLE `{self._table_name}`")
            await cursor.fetchall()
        logger.info("OPTIMIZE TABLE executed on %s", self._table_name)

    # ---- P2: v3 Enhanced features (raise NotSupportedError on old versions) ----

    def _require_v3(self, feature: str) -> None:
        """Raise NotSupportedError if v3 capabilities are not available."""
        if not self._capabilities.get("vec_distance", False):
            raise NotSupportedError(
                f"{feature} requires PolarDB-X v3 with vector index support. "
                "Current instance does not support v3 vector features."
            )

    def preload_index(self) -> None:
        """Preload the HNSW vector index into memory cache (v3 only).

        Loads the entire HNSW auxiliary table graph into the shared cache
        to eliminate cold-start latency on the first query.

        Raises:
            NotSupportedError: If the instance does not support v3 features.
        """
        self._require_v3("preload_index()")
        with self._get_cursor() as cursor:
            cursor.execute(
                f"CALL dbms_vidx.preload('{self._database}', "
                f"'{self._table_name}', 'embedding')"
            )
            cursor.fetchall()
        logger.info("Preloaded vector index for table %s", self._table_name)

    def preload_check(self) -> Dict[str, Any]:
        """Check if preloading the vector index would fit in cache (v3 only).

        Estimates the memory required and compares it with
        ``vidx_hnsw_cache_size`` without actually loading.

        Returns:
            Dictionary with check results.

        Raises:
            NotSupportedError: If the instance does not support v3 features.
        """
        self._require_v3("preload_check()")
        with self._get_cursor() as cursor:
            cursor.execute(
                f"CALL dbms_vidx.preload_check('{self._database}', "
                f"'{self._table_name}', 'embedding')"
            )
            rows = cursor.fetchall()
            return {
                row.get("Message", row.get("message", str(row))): row
                for row in rows
            } if rows else {}

    def explain_index_health(self) -> Dict[str, Any]:
        """Check vector index health and return diagnostics (v3 only).

        Combines ``information_schema.VECTOR_INDEXES`` metadata with
        ``EXPLAIN`` output to provide a comprehensive health report.

        Returns:
            Dictionary with index metadata and query plan info.

        Raises:
            NotSupportedError: If the instance does not support v3 features.
        """
        self._require_v3("explain_index_health()")
        result: Dict[str, Any] = {}

        # 1. Query VECTOR_INDEXES view for index metadata
        with self._get_cursor() as cursor:
            cursor.execute(
                "SELECT INDEX_NAME, ALGORITHM, METRIC_TYPE, "
                "DIMENSION, M, EF_CONSTRUCTION, QUANTIZE_TYPE "
                "FROM information_schema.VECTOR_INDEXES "
                "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s",
                (self._database, self._table_name),
            )
            rows = cursor.fetchall()
            if rows:
                result["index_info"] = rows[0]
                # Use actual dimension from VECTOR_INDEXES view
                dim = rows[0].get("DIMENSION", self._embedding_dimension or 4)
            else:
                result["index_info"] = None
                dim = self._embedding_dimension or 4

            # 2. Run EXPLAIN to check if vector index is used
            dist_func = self._get_distance_func()
            sample_vec = self._vector_to_string([0.0] * dim)
            cursor.execute(
                f"EXPLAIN SELECT id FROM `{self._table_name}` "
                f"ORDER BY {dist_func}(embedding, "
                f"VEC_FROMTEXT('{sample_vec}')) LIMIT 10"
            )
            explain_rows = cursor.fetchall()
            result["explain"] = explain_rows

        return result

    async def apreload_index(self) -> None:
        """Async preload the HNSW vector index into memory cache (v3 only)."""
        self._require_v3("apreload_index()")
        async with self._aget_cursor() as cursor:
            await cursor.execute(
                f"CALL dbms_vidx.preload('{self._database}', "
                f"'{self._table_name}', 'embedding')"
            )
            await cursor.fetchall()
        logger.info("Preloaded vector index for table %s", self._table_name)

    async def apreload_check(self) -> Dict[str, Any]:
        """Async check if preloading would fit in cache (v3 only)."""
        self._require_v3("apreload_check()")
        async with self._aget_cursor() as cursor:
            await cursor.execute(
                f"CALL dbms_vidx.preload_check('{self._database}', "
                f"'{self._table_name}', 'embedding')"
            )
            rows = await cursor.fetchall()
            return {
                row.get("Message", row.get("message", str(row))): row
                for row in rows
            } if rows else {}

    async def aexplain_index_health(self) -> Dict[str, Any]:
        """Async check vector index health (v3 only)."""
        self._require_v3("aexplain_index_health()")
        result: Dict[str, Any] = {}
        async with self._aget_cursor() as cursor:
            await cursor.execute(
                "SELECT INDEX_NAME, ALGORITHM, METRIC_TYPE, "
                "DIMENSION, M, EF_CONSTRUCTION, QUANTIZE_TYPE "
                "FROM information_schema.VECTOR_INDEXES "
                "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s",
                (self._database, self._table_name),
            )
            rows = await cursor.fetchall()
            result["index_info"] = rows[0] if rows else None

            # Use actual dimension from VECTOR_INDEXES view
            if rows:
                dim = rows[0].get("DIMENSION", self._embedding_dimension or 4)
            else:
                dim = self._embedding_dimension or 4

            dist_func = self._get_distance_func()
            sample_vec = self._vector_to_string([0.0] * dim)
            await cursor.execute(
                f"EXPLAIN SELECT id FROM `{self._table_name}` "
                f"ORDER BY {dist_func}(embedding, "
                f"VEC_FROMTEXT('{sample_vec}')) LIMIT 10"
            )
            result["explain"] = await cursor.fetchall()
        return result

    async def aapply_vector_index(
        self,
        index_name: str = "vi",
        m: Optional[int] = None,
        distance: Optional[str] = None,
        ef_construction: Optional[int] = None,
    ) -> None:
        """Async create a vector index on the embedding column."""
        self._validate_identifier(index_name, "index name")
        m_val = m or self._hnsw_m
        dist_val = (distance or self._distance_strategy).upper()
        if dist_val not in ("COSINE", "EUCLIDEAN", "INNER_PRODUCT"):
            raise ValueError(
                f"Invalid distance function: {dist_val}. "
                "Must be 'COSINE', 'EUCLIDEAN', or 'INNER_PRODUCT'."
            )
        ef_val = ef_construction or self._ef_construction
        ef_clause = ""
        if ef_val is not None and self._capabilities.get("vec_distance", False):
            ef_clause = f" EF_CONSTRUCTION={ef_val}"
        async with self._aget_cursor() as cursor:
            await cursor.execute(
                f"ALTER TABLE `{self._table_name}` "
                f"ADD VECTOR INDEX `{index_name}` (embedding) "
                f"M={m_val}{ef_clause} DISTANCE={dist_val}"
            )
        self._vector_index_name = index_name
        logger.info("Vector index '%s' created on table %s", index_name, self._table_name)

    async def adrop_vector_index(self, index_name: Optional[str] = None) -> None:
        """Async drop a vector index."""
        name = index_name or self._detect_vector_index_name()
        if not name:
            raise ValueError("No vector index name specified or detectable.")
        async with self._aget_cursor() as cursor:
            await cursor.execute(f"ALTER TABLE `{self._table_name}` DROP INDEX `{name}`")
        self._vector_index_name = None
        logger.info("Vector index '%s' dropped from table %s", name, self._table_name)

    async def aget_stats(self) -> Dict[str, Any]:
        """Async get vector index runtime statistics."""
        async with self._aget_cursor() as cursor:
            await cursor.execute("SHOW GLOBAL STATUS LIKE 'Vidx%'")
            rows = await cursor.fetchall()
            return {
                row.get("Variable_name", row.get("variable_name", "")): (
                    row.get("Value", row.get("value", ""))
                )
                for row in rows
            }

    def _table_exists(self) -> bool:
        """Check if the table exists."""
        with self._get_cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) AS cnt FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name = %s",
                (self._database, self._table_name),
            )
            result = cursor.fetchone()
            return result["cnt"] > 0 if result else False

    async def _atable_exists(self) -> bool:
        """Async check if the table exists."""
        async with self._aget_cursor() as cursor:
            await cursor.execute(
                "SELECT COUNT(*) AS cnt FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name = %s",
                (self._database, self._table_name),
            )
            result = await cursor.fetchone()
            # aiomysql DictCursor returns dict
            return result["cnt"] > 0 if result else False

    def _create_table_if_not_exists(self, dimension: int) -> None:
        """Create the vector table if it doesn't exist."""
        self._embedding_dimension = dimension
        if self._table_exists():
            return
        with self._get_cursor() as cursor:
            sql = self._build_create_table_sql(dimension)
            cursor.execute(sql)
            logger.info(
                "Created table %s with vector dimension %d", self._table_name, dimension
            )

    async def _acreate_table_if_not_exists(self, dimension: int) -> None:
        """Async create the vector table if it doesn't exist."""
        self._embedding_dimension = dimension
        if await self._atable_exists():
            return
        async with self._aget_cursor() as cursor:
            sql = self._build_create_table_sql(dimension)
            await cursor.execute(sql)
            logger.info(
                "Created table %s with vector dimension %d", self._table_name, dimension
            )

    def _drop_table(self) -> None:
        """Drop the vector table."""
        with self._get_cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS `{self._table_name}`")
            logger.info("Dropped table %s", self._table_name)

    async def _adrop_table(self) -> None:
        """Async drop the vector table."""
        async with self._aget_cursor() as cursor:
            await cursor.execute(f"DROP TABLE IF EXISTS `{self._table_name}`")
            logger.info("Dropped table %s", self._table_name)

    def _vector_to_string(self, vector: List[float]) -> str:
        """Convert a vector to PolarDB-X vector string format."""
        return "[" + ",".join(map(str, vector)) + "]"

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert distance to similarity score."""
        if self._distance_strategy == "cosine":
            # For cosine distance [0, 2]: similarity = 1 - distance/2
            return max(0.0, 1.0 - distance / 2.0)
        elif self._distance_strategy == "inner_product":
            # VEC_DISTANCE_INNER_PRODUCT returns negative dot product
            # (smaller distance = larger dot product = more similar)
            if distance < 0:
                return 1.0
            return max(0.0, 1.0 - distance)
        else:
            # For euclidean distance [0, inf): similarity = 1 / (1 + distance)
            return 1.0 / (1.0 + distance)

    def _validate_embedding_dimensions(
        self, embeddings: List[List[float]], expected: int
    ) -> None:
        """Validate that all embeddings have the expected dimension.

        Uses ``VECTOR_DIM`` on a sample vector to cross-check that the
        DN's notion of dimension matches the client's.  This catches
        mismatches early before batch insert.

        Args:
            embeddings: List of embedding vectors.
            expected: Expected dimension.

        Raises:
            ValueError: If any embedding has a different dimension.
        """
        for i, emb in enumerate(embeddings):
            if len(emb) != expected:
                raise ValueError(
                    f"Embedding at index {i} has dimension {len(emb)}, "
                    f"expected {expected}."
                )
        # Cross-check with DN's VECTOR_DIM if available
        try:
            sample_vec = self._vector_to_string(embeddings[0])
            with self._get_cursor() as cursor:
                cursor.execute(
                    "SELECT VECTOR_DIM(VEC_FROMTEXT(%s)) AS dim",
                    (sample_vec,)
                )
                row = cursor.fetchone()
                if row and row.get("dim") != expected:
                    raise ValueError(
                        f"DN VECTOR_DIM reports {row.get('dim')}, "
                        f"but client expected {expected}."
                    )
        except ValueError:
            raise
        except Exception as e:
            logger.debug("VECTOR_DIM cross-check failed: %s", e)

    def _fetch_embeddings_by_ids(self, ids: List[str]) -> List[List[float]]:
        """Fetch embedding vectors from the database by document IDs.

        Args:
            ids: List of document IDs.

        Returns:
            List of embedding vectors in the same order as input ids.
            Empty list for any ID not found.
        """
        if not ids:
            return []
        placeholders = ", ".join(["%s"] * len(ids))
        # Use VEC_TOTEXT if available (v3), otherwise CAST(embedding AS CHAR)
        emb_expr = (
            "VEC_TOTEXT(embedding)"
            if self._capabilities.get("vec_totext", False)
            else "CAST(embedding AS CHAR)"
        )
        sql = f"""
        SELECT id, {emb_expr} as emb_str
        FROM `{self._table_name}`
        WHERE id IN ({placeholders})
        """
        id_to_emb: Dict[str, List[float]] = {}
        try:
            with self._get_cursor() as cursor:
                cursor.execute(sql, ids)
                for row in cursor:
                    emb_str = row["emb_str"]
                    if isinstance(emb_str, str) and emb_str:
                        try:
                            id_to_emb[row["id"]] = json.loads(emb_str)
                        except (json.JSONDecodeError, ValueError):
                            pass
        except Exception as e:
            if "1146" in str(e) or "doesn't exist" in str(e).lower():
                return []
            raise
        return [id_to_emb.get(id_, []) for id_ in ids]

    async def _afetch_embeddings_by_ids(self, ids: List[str]) -> List[List[float]]:
        """Async fetch embedding vectors from the database by document IDs.

        Args:
            ids: List of document IDs.

        Returns:
            List of embedding vectors in the same order as input ids.
            Empty list for any ID not found.
        """
        if not ids:
            return []
        placeholders = ", ".join(["%s"] * len(ids))
        # Use VEC_TOTEXT if available (v3), otherwise CAST(embedding AS CHAR)
        emb_expr = (
            "VEC_TOTEXT(embedding)"
            if self._capabilities.get("vec_totext", False)
            else "CAST(embedding AS CHAR)"
        )
        sql = f"""
        SELECT id, {emb_expr} as emb_str
        FROM `{self._table_name}`
        WHERE id IN ({placeholders})
        """
        id_to_emb: Dict[str, List[float]] = {}
        try:
            async with self._aget_cursor() as cursor:
                await cursor.execute(sql, ids)
                async for row in cursor:
                    emb_str = row["emb_str"]
                    if isinstance(emb_str, str) and emb_str:
                        try:
                            id_to_emb[row["id"]] = json.loads(emb_str)
                        except (json.JSONDecodeError, ValueError):
                            pass
        except Exception as e:
            if "1146" in str(e) or "doesn't exist" in str(e).lower():
                return []
            raise
        return [id_to_emb.get(id_, []) for id_ in ids]

    def _build_filter_clause(
        self,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Any]]:
        """Build SQL WHERE clause from filter dictionary.

        Args:
            filter: Optional filter dictionary. Supports:
                - Simple: {"key": "value"} -> key = 'value'
                - Operators: {"key": {"$gt": 10}} -> key > 10
                - Supported operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $like

        Returns:
            Tuple of (where_clause, params) for SQL query.

        Example:
            .. code-block:: python

                # Simple equality
                filter = {"category": "phone"}

                # With operators
                filter = {
                    "category": {"$in": ["phone", "tablet"]},
                    "price": {"$gt": 100, "$lt": 1000},
                    "status": {"$ne": "deleted"}
                }
        """
        if not filter:
            return "", []

        conditions: List[str] = []
        params: List[Any] = []

        for key, value in filter.items():
            # Validate key to prevent SQL injection in JSON path
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", str(key)):
                raise ValueError(
                    f"Invalid filter key: {key}. "
                    "Keys must start with a letter or underscore, "
                    "and contain only alphanumeric characters and underscores."
                )
            # Use JSON_EXTRACT for numeric comparisons, JSON_UNQUOTE for string
            json_path_str = f"JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.{key}'))"
            json_path_num = f"JSON_EXTRACT(metadata, '$.{key}')"

            if isinstance(value, dict):
                # Operator-based filter: {"price": {"$gt": 100}}
                for op, op_value in value.items():
                    if op not in FILTER_OPERATORS:
                        raise ValueError(
                            f"Unsupported filter operator: {op}. "
                            f"Supported: {list(FILTER_OPERATORS.keys())}"
                        )

                    sql_op = FILTER_OPERATORS[op]

                    if op in ("$in", "$nin"):
                        # Handle IN/NOT IN with list values
                        if not isinstance(op_value, (list, tuple)):
                            raise ValueError(f"Operator {op} requires a list value")
                        placeholders = ",".join(["%s"] * len(op_value))
                        conditions.append(f"{json_path_str} {sql_op} ({placeholders})")
                        params.extend([str(v) for v in op_value])
                    elif isinstance(op_value, (int, float)):
                        # Numeric comparison: use JSON_EXTRACT (preserves type)
                        conditions.append(f"{json_path_num} {sql_op} %s")
                        params.append(op_value)
                    else:
                        # String comparison
                        conditions.append(f"{json_path_str} {sql_op} %s")
                        params.append(str(op_value))
            else:
                # Simple equality filter: {"category": "phone"}
                if isinstance(value, (int, float)):
                    conditions.append(f"{json_path_num} = %s")
                    params.append(value)
                else:
                    conditions.append(f"{json_path_str} = %s")
                    params.append(str(value))

        if conditions:
            return "WHERE " + " AND ".join(conditions), params
        return "", []

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        *,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store.

        Args:
            texts: Iterable of strings to add to the vector store.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of IDs. If not provided, UUIDs will be generated.
            batch_size: Number of texts to insert per batch. Defaults to 500.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the added texts.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        # Generate embeddings
        embeddings = self._embedding.embed_documents(texts_list)

        # Ensure table exists with correct dimension
        dimension = len(embeddings[0])
        self._create_table_if_not_exists(dimension)

        # Validate vector dimensions using VECTOR_DIM if available (v3)
        if self._capabilities.get("vec_dim", False):
            self._validate_embedding_dimensions(embeddings, dimension)

        # Prepare IDs
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]

        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in texts_list]

        # Validate lengths
        if len(texts_list) != len(metadatas):
            raise ValueError(
                f"Number of texts ({len(texts_list)}) must match "
                f"number of metadatas ({len(metadatas)})"
            )
        if len(texts_list) != len(ids):
            raise ValueError(
                f"Number of texts ({len(texts_list)}) must match "
                f"number of ids ({len(ids)})"
            )

        # Insert in batches (executemany uses prepared statements internally)
        with self._get_cursor() as cursor:
            sql = SQL_UPSERT.format(table_name=self._table_name)

            for i in range(0, len(texts_list), batch_size):
                batch_end = min(i + batch_size, len(texts_list))
                batch_values = []

                for j in range(i, batch_end):
                    vector_str = self._vector_to_string(embeddings[j])
                    batch_values.append(
                        (
                            ids[j],
                            texts_list[j],
                            json.dumps(metadatas[j]),
                            vector_str,
                        )
                    )

                cursor.executemany(sql, batch_values)

        logger.info("Added %d texts to table %s", len(texts_list), self._table_name)
        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        *,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        """Async add texts to the vector store.

        Args:
            texts: Iterable of strings to add to the vector store.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of IDs. If not provided, UUIDs will be generated.
            batch_size: Number of texts to insert per batch. Defaults to 500.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the added texts.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        # Generate embeddings (use async if available)
        if hasattr(self._embedding, "aembed_documents"):
            embeddings = await self._embedding.aembed_documents(texts_list)
        else:
            embeddings = await run_in_executor(
                None, self._embedding.embed_documents, texts_list
            )

        # Ensure table exists with correct dimension
        dimension = len(embeddings[0])
        await self._acreate_table_if_not_exists(dimension)

        # Prepare IDs
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts_list]

        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in texts_list]

        # Insert in batches
        async with self._aget_cursor() as cursor:
            sql = SQL_UPSERT.format(table_name=self._table_name)

            for i in range(0, len(texts_list), batch_size):
                batch_end = min(i + batch_size, len(texts_list))
                batch_values = []

                for j in range(i, batch_end):
                    vector_str = self._vector_to_string(embeddings[j])
                    batch_values.append(
                        (
                            ids[j],
                            texts_list[j],
                            json.dumps(metadatas[j]),
                            vector_str,
                        )
                    )

                await cursor.executemany(sql, batch_values)

        logger.info("Added %d texts to table %s", len(texts_list), self._table_name)
        return ids

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        *,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of Document objects to add.
            ids: Optional list of IDs. If not provided, UUIDs will be generated.
                If provided in document.id, those will be used preferentially.
            batch_size: Number of documents to insert per batch. Defaults to 500.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the added documents.

        Example:
            .. code-block:: python

                from langchain_core.documents import Document

                documents = [
                    Document(
                        page_content="Hello world", metadata={"source": "web"}
                    ),
                    Document(
                        page_content="LangChain is great",
                        metadata={"source": "doc"},
                    ),
                ]
                ids = vectorstore.add_documents(documents)
        """
        if not documents:
            return []

        # Extract texts and metadatas from documents
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Determine IDs: use provided ids, or document.id, or generate UUIDs
        if ids is None:
            ids = []
            for doc in documents:
                if doc.id is not None:
                    ids.append(doc.id)
                else:
                    ids.append(str(uuid.uuid4()))

        return self.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=batch_size,
            **kwargs,
        )

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        *,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        """Async add documents to the vector store.

        Args:
            documents: List of Document objects to add.
            ids: Optional list of IDs. If not provided, UUIDs will be generated.
                If provided in document.id, those will be used preferentially.
            batch_size: Number of documents to insert per batch. Defaults to 500.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the added documents.
        """
        if not documents:
            return []

        # Extract texts and metadatas from documents
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Determine IDs: use provided ids, or document.id, or generate UUIDs
        if ids is None:
            ids = []
            for doc in documents:
                if doc.id is not None:
                    ids.append(doc.id)
                else:
                    ids.append(str(uuid.uuid4()))

        return await self.aadd_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            batch_size=batch_size,
            **kwargs,
        )

    def add_embeddings(
        self,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        *,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        """Add pre-computed embeddings to the vector store.

        This method allows you to add texts with pre-computed embeddings,
        avoiding the need to re-compute embeddings if you already have them.

        Args:
            text_embeddings: Iterable of (text, embedding) tuples.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of IDs. If not provided, UUIDs will be generated.
            batch_size: Number of records to insert per batch. Defaults to 500.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the added embeddings.

        Example:
            .. code-block:: python

                # With pre-computed embeddings
                texts = ["Hello world", "LangChain is great"]
                embeddings = embedding_model.embed_documents(texts)
                text_embeddings = list(zip(texts, embeddings))

                ids = vectorstore.add_embeddings(
                    text_embeddings=text_embeddings,
                    metadatas=[{"source": "web"}, {"source": "doc"}]
                )
        """
        # Convert to list for multiple iterations
        text_embeddings_list = list(text_embeddings)
        if not text_embeddings_list:
            return []

        # Extract texts and embeddings
        texts = [te[0] for te in text_embeddings_list]
        embeddings = [te[1] for te in text_embeddings_list]

        # Validate vector dimensions using VECTOR_DIM if available (v3)
        dimension = len(embeddings[0])
        if self._capabilities.get("vec_dim", False):
            self._validate_embedding_dimensions(embeddings, dimension)

        # Prepare IDs
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in text_embeddings_list]

        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in text_embeddings_list]

        # Validate lengths before creating table
        if len(texts) != len(metadatas):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match "
                f"number of metadatas ({len(metadatas)})"
            )
        if len(texts) != len(ids):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match number of ids ({len(ids)})"
            )

        # Ensure table exists with correct dimension
        dimension = len(embeddings[0])
        self._create_table_if_not_exists(dimension)

        # Insert in batches
        with self._get_cursor() as cursor:
            sql = SQL_UPSERT.format(table_name=self._table_name)

            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                batch_values = []

                for j in range(i, batch_end):
                    vector_str = self._vector_to_string(embeddings[j])
                    batch_values.append(
                        (
                            ids[j],
                            texts[j],
                            json.dumps(metadatas[j]),
                            vector_str,
                        )
                    )

                cursor.executemany(sql, batch_values)

        logger.info("Added %d embeddings to table %s", len(texts), self._table_name)
        return ids

    async def aadd_embeddings(
        self,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        *,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        """Async add pre-computed embeddings to the vector store.

        This method allows you to add texts with pre-computed embeddings,
        avoiding the need to re-compute embeddings if you already have them.

        Args:
            text_embeddings: Iterable of (text, embedding) tuples.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of IDs. If not provided, UUIDs will be generated.
            batch_size: Number of records to insert per batch. Defaults to 500.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the added embeddings.
        """
        # Convert to list for multiple iterations
        text_embeddings_list = list(text_embeddings)
        if not text_embeddings_list:
            return []

        # Extract texts and embeddings
        texts = [te[0] for te in text_embeddings_list]
        embeddings = [te[1] for te in text_embeddings_list]

        # Validate vector dimensions using VECTOR_DIM if available (v3)
        dimension = len(embeddings[0])
        if self._capabilities.get("vec_dim", False):
            self._validate_embedding_dimensions(embeddings, dimension)

        # Prepare IDs
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in text_embeddings_list]

        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in text_embeddings_list]

        # Validate lengths before creating table
        if len(texts) != len(metadatas):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match "
                f"number of metadatas ({len(metadatas)})"
            )
        if len(texts) != len(ids):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match number of ids ({len(ids)})"
            )

        # Ensure table exists with correct dimension
        dimension = len(embeddings[0])
        await self._acreate_table_if_not_exists(dimension)

        # Insert in batches
        async with self._aget_cursor() as cursor:
            sql = SQL_UPSERT.format(table_name=self._table_name)

            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                batch_values = []

                for j in range(i, batch_end):
                    vector_str = self._vector_to_string(embeddings[j])
                    batch_values.append(
                        (
                            ids[j],
                            texts[j],
                            json.dumps(metadatas[j]),
                            vector_str,
                        )
                    )

                await cursor.executemany(sql, batch_values)

        logger.info("Added %d embeddings to table %s", len(texts), self._table_name)
        return ids

    def upsert(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        *,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        """Update or insert documents to the vector store.

        If a document with the same ID exists, it will be updated.
        Otherwise, a new document will be inserted.

        Args:
            documents: List of Document objects to upsert.
            ids: Optional list of IDs. If not provided, will use document.id
                or generate new UUIDs.
            batch_size: Number of documents to process per batch. Defaults to 500.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the upserted documents.

        Example:
            .. code-block:: python

                from langchain_core.documents import Document

                # Insert new documents
                docs = [
                    Document(id="doc1", page_content="Hello", metadata={"k": "v1"}),
                    Document(id="doc2", page_content="World", metadata={"k": "v2"}),
                ]
                vectorstore.upsert(docs)

                # Update existing documents (same IDs, new content)
                updated_docs = [
                    Document(
                        id="doc1",
                        page_content="Updated Hello",
                        metadata={"k": "v1_new"},
                    ),
                ]
                vectorstore.upsert(updated_docs)
        """
        if not documents:
            logger.debug("No documents to upsert.")
            return []

        # Determine IDs: use provided ids, or document.id, or generate UUIDs
        if ids is None:
            ids = []
            for doc in documents:
                if doc.id is not None:
                    ids.append(doc.id)
                else:
                    ids.append(str(uuid.uuid4()))
        else:
            if len(ids) != len(documents):
                raise ValueError(
                    f"Number of ids ({len(ids)}) must match "
                    f"number of documents ({len(documents)})"
                )

        # Extract texts and metadatas
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings
        embeddings = self._embedding.embed_documents(texts)

        # Ensure table exists with correct dimension
        dimension = len(embeddings[0])
        self._create_table_if_not_exists(dimension)

        # Upsert in batches (using SQL_UPSERT which does ON DUPLICATE KEY UPDATE)
        with self._get_cursor() as cursor:
            sql = SQL_UPSERT.format(table_name=self._table_name)

            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                batch_values = []

                for j in range(i, batch_end):
                    vector_str = self._vector_to_string(embeddings[j])
                    batch_values.append(
                        (
                            ids[j],
                            texts[j],
                            json.dumps(metadatas[j]),
                            vector_str,
                        )
                    )

                cursor.executemany(sql, batch_values)

        logger.info(
            "Upserted %d documents to table %s", len(documents), self._table_name
        )
        return ids

    async def aupsert(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        *,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        """Async update or insert documents to the vector store.

        If a document with the same ID exists, it will be updated.
        Otherwise, a new document will be inserted.

        Args:
            documents: List of Document objects to upsert.
            ids: Optional list of IDs. If not provided, will use document.id
                or generate new UUIDs.
            batch_size: Number of documents to process per batch. Defaults to 500.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs of the upserted documents.
        """
        if not documents:
            logger.debug("No documents to upsert.")
            return []

        # Determine IDs: use provided ids, or document.id, or generate UUIDs
        if ids is None:
            ids = []
            for doc in documents:
                if doc.id is not None:
                    ids.append(doc.id)
                else:
                    ids.append(str(uuid.uuid4()))
        else:
            if len(ids) != len(documents):
                raise ValueError(
                    f"Number of ids ({len(ids)}) must match "
                    f"number of documents ({len(documents)})"
                )

        # Extract texts and metadatas
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings (use async if available)
        if hasattr(self._embedding, "aembed_documents"):
            embeddings = await self._embedding.aembed_documents(texts)
        else:
            embeddings = await run_in_executor(
                None, self._embedding.embed_documents, texts
            )

        # Ensure table exists with correct dimension
        dimension = len(embeddings[0])
        await self._acreate_table_if_not_exists(dimension)

        # Upsert in batches (using SQL_UPSERT which does ON DUPLICATE KEY UPDATE)
        async with self._aget_cursor() as cursor:
            sql = SQL_UPSERT.format(table_name=self._table_name)

            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                batch_values = []

                for j in range(i, batch_end):
                    vector_str = self._vector_to_string(embeddings[j])
                    batch_values.append(
                        (
                            ids[j],
                            texts[j],
                            json.dumps(metadatas[j]),
                            vector_str,
                        )
                    )

                await cursor.executemany(sql, batch_values)

        logger.info(
            "Upserted %d documents to table %s", len(documents), self._table_name
        )
        return ids

    def bulk_upsert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
        *,
        batch_size: int = 500,
    ) -> List[str]:
        """Bulk upsert texts with pre-computed embeddings.

        Unlike ``upsert``, this method skips the embedding step entirely,
        making it suitable for large-scale data imports where embeddings
        are computed externally.

        Args:
            texts: List of text strings.
            embeddings: List of pre-computed embedding vectors.
            ids: Optional list of IDs. If not provided, UUIDs are generated.
            metadatas: Optional list of metadata dictionaries.
            batch_size: Number of records per batch. Defaults to 500.

        Returns:
            List of IDs of the upserted records.
        """
        if not texts:
            return []
        if len(texts) != len(embeddings):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        dimension = len(embeddings[0])
        self._create_table_if_not_exists(dimension)

        if self._capabilities.get("vec_dim", False):
            self._validate_embedding_dimensions(embeddings, dimension)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        with self._get_cursor() as cursor:
            sql = SQL_UPSERT.format(table_name=self._table_name)
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                batch_values = [
                    (
                        ids[j],
                        texts[j],
                        json.dumps(metadatas[j]),
                        self._vector_to_string(embeddings[j]),
                    )
                    for j in range(i, batch_end)
                ]
                cursor.executemany(sql, batch_values)

        logger.info(
            "Bulk upserted %d records to table %s", len(texts), self._table_name
        )
        return ids

    async def abulk_upsert(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
        *,
        batch_size: int = 500,
    ) -> List[str]:
        """Async bulk upsert texts with pre-computed embeddings."""
        if not texts:
            return []
        if len(texts) != len(embeddings):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        dimension = len(embeddings[0])
        await self._acreate_table_if_not_exists(dimension)

        if self._capabilities.get("vec_dim", False):
            self._validate_embedding_dimensions(embeddings, dimension)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        async with self._aget_cursor() as cursor:
            sql = SQL_UPSERT.format(table_name=self._table_name)
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                batch_values = [
                    (
                        ids[j],
                        texts[j],
                        json.dumps(metadatas[j]),
                        self._vector_to_string(embeddings[j]),
                    )
                    for j in range(i, batch_end)
                ]
                await cursor.executemany(sql, batch_values)

        logger.info(
            "Bulk upserted %d records to table %s", len(texts), self._table_name
        )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to the query.

        Args:
            query: Query string to search for.
            k: Number of documents to return. Defaults to 4.
            filter: Optional metadata filter dictionary. Supports:
                - Simple: {"key": "value"}
                - Operators: {"key": {"$gt": 10, "$lt": 100}}
                - Supported operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $like
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query.
        """
        docs_with_scores = self.similarity_search_with_score(
            query=query, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_with_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return documents and similarity scores most similar to the query.

        Args:
            query: Query string to search for.
            k: Number of documents to return. Defaults to 4.
            filter: Optional metadata filter dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            List of tuples of (Document, similarity_score).
        """
        # Generate query embedding
        query_embedding = self._embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding=query_embedding, k=k, filter=filter, **kwargs
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to the embedding vector.

        Args:
            embedding: Embedding vector to search for.
            k: Number of documents to return. Defaults to 4.
            filter: Optional metadata filter dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_with_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_with_scores]

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return documents and scores most similar to the embedding vector.

        Args:
            embedding: Embedding vector to search for.
            k: Number of documents to return. Defaults to 4.
            filter: Optional metadata filter dictionary. Supports:
                - Simple: {"key": "value"}
                - Operators: {"key": {"$gt": 10}}
            score_threshold: Optional maximum distance threshold. Only results
                with distance <= score_threshold are returned. Lower distance
                means more similar. Defaults to None (no filtering).
            **kwargs: Additional keyword arguments:
                - ef_search: int — HNSW ef_search parameter [1, 10000].
                  Higher = more accurate but slower. Default: None (server default 20).
                - search_type: str — "ann" (force vector index), "knn" (force full scan),
                  or "auto" (let optimizer decide). Default: "auto".

        Returns:
            List of tuples of (Document, distance_score). Lower distance means more similar.
        """
        # Extract Phase 2 enhancement params
        ef_search = kwargs.pop("ef_search", None)
        search_type = kwargs.pop("search_type", None)

        # Build WHERE clause using the filter builder
        where_clause, filter_params = self._build_filter_clause(filter)

        # Choose distance function (VEC_DISTANCE if available for HNSW acceleration)
        distance_func = self._get_distance_func()

        # Build index hint for ANN/KNN mode
        index_hint = self._build_index_hint(search_type)

        # Build and execute query
        query_vector_str = self._vector_to_string(embedding)

        sql = SQL_SEARCH.format(
            table_name=self._table_name,
            distance_func=distance_func,
            index_hint=index_hint,
            where_clause=where_clause,
        )

        # When score_threshold is set, fetch more candidates to compensate for filtering
        fetch_k = k * 3 if score_threshold is not None else k
        query_params = [query_vector_str] + filter_params + [fetch_k]

        results: List[Tuple[Document, float]] = []

        try:
            with self._get_cursor() as cursor:
                self._set_ef_search(cursor, ef_search)
                cursor.execute(sql, query_params)

                for record in cursor:
                    distance = float(record["distance"])

                    # Apply distance threshold (filter out results that are too far)
                    if score_threshold is not None and distance > score_threshold:
                        continue

                    metadata = record["metadata"]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)

                    doc = Document(
                        id=record["id"],
                        page_content=record["text"],
                        metadata=metadata or {},
                    )
                    results.append((doc, distance))
        except Exception as e:
            # If table doesn't exist (error 1146), return empty list
            error_msg = str(e)
            if "1146" in error_msg or "doesn't exist" in error_msg.lower():
                logger.debug(
                    "Table %s does not exist, returning empty results", self._table_name
                )
                return []
            raise

        return results[:k]

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async return documents most similar to the query.

        Args:
            query: Query string to search for.
            k: Number of documents to return. Defaults to 4.
            filter: Optional metadata filter dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query.
        """
        docs_with_scores = await self.asimilarity_search_with_score(
            query=query, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_with_scores]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Async return documents and similarity scores most similar to the query.

        Args:
            query: Query string to search for.
            k: Number of documents to return. Defaults to 4.
            filter: Optional metadata filter dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            List of tuples of (Document, similarity_score).
        """
        # Generate query embedding (use async if available)
        if hasattr(self._embedding, "aembed_query"):
            query_embedding = await self._embedding.aembed_query(query)
        else:
            query_embedding = await run_in_executor(
                None, self._embedding.embed_query, query
            )
        return await self.asimilarity_search_with_score_by_vector(
            embedding=query_embedding, k=k, filter=filter, **kwargs
        )

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async return documents most similar to the embedding vector.

        Args:
            embedding: Embedding vector to search for.
            k: Number of documents to return. Defaults to 4.
            filter: Optional metadata filter dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_with_scores = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_with_scores]

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Async return documents and scores most similar to the embedding vector.

        Args:
            embedding: Embedding vector to search for.
            k: Number of documents to return. Defaults to 4.
            filter: Optional metadata filter dictionary.
            score_threshold: Optional maximum distance threshold. Only results
                with distance <= score_threshold are returned. Lower distance
                means more similar. Defaults to None (no filtering).
            **kwargs: Additional keyword arguments:
                - ef_search: int — HNSW ef_search parameter [1, 10000].
                - search_type: str — "ann", "knn", or "auto". Default: "auto".

        Returns:
            List of tuples of (Document, distance_score). Lower distance means more similar.
        """
        # Extract Phase 2 enhancement params
        ef_search = kwargs.pop("ef_search", None)
        search_type = kwargs.pop("search_type", None)

        # Build WHERE clause using the filter builder
        where_clause, filter_params = self._build_filter_clause(filter)

        # Choose distance function (VEC_DISTANCE if available for HNSW acceleration)
        distance_func = self._get_distance_func()

        # Build index hint for ANN/KNN mode
        index_hint = self._build_index_hint(search_type)

        # Build and execute query
        query_vector_str = self._vector_to_string(embedding)

        sql = SQL_SEARCH.format(
            table_name=self._table_name,
            distance_func=distance_func,
            index_hint=index_hint,
            where_clause=where_clause,
        )

        # When score_threshold is set, fetch more candidates to compensate for filtering
        fetch_k = k * 3 if score_threshold is not None else k
        query_params = [query_vector_str] + filter_params + [fetch_k]

        results: List[Tuple[Document, float]] = []

        try:
            async with self._aget_cursor() as cursor:
                if ef_search is not None:
                    await cursor.execute(
                        f"SET SESSION vidx_hnsw_ef_search = {int(ef_search)}"
                    )
                await cursor.execute(sql, query_params)

                async for record in cursor:
                    distance = float(record["distance"])

                    # Apply distance threshold (filter out results that are too far)
                    if score_threshold is not None and distance > score_threshold:
                        continue

                    metadata = record["metadata"]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)

                    doc = Document(
                        id=record["id"],
                        page_content=record["text"],
                        metadata=metadata or {},
                    )
                    results.append((doc, distance))
        except Exception as e:
            # If table doesn't exist (error 1146), return empty list
            error_msg = str(e)
            if "1146" in error_msg or "doesn't exist" in error_msg.lower():
                logger.debug(
                    "Table %s does not exist, returning empty results", self._table_name
                )
                return []
            raise

        return results[:k]

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Delete vectors by IDs.

        Args:
            ids: List of IDs to delete.
            **kwargs: Additional keyword arguments.

        Returns:
            True if deletion was successful, None if no IDs provided.
        """
        if not ids:
            return None

        try:
            with self._get_cursor() as cursor:
                placeholders = ",".join(["%s"] * len(ids))
                sql = SQL_DELETE_BY_IDS.format(
                    table_name=self._table_name, placeholders=placeholders
                )
                cursor.execute(sql, ids)

            logger.info("Deleted %d vectors from table %s", len(ids), self._table_name)
            return True
        except Exception as e:
            # If table doesn't exist (error 1146), deletion is considered successful
            error_msg = str(e)
            if "1146" in error_msg or "doesn't exist" in error_msg.lower():
                logger.debug(
                    "Table %s does not exist, skipping delete operation",
                    self._table_name,
                )
                return True
            raise

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Async delete vectors by IDs.

        Args:
            ids: List of IDs to delete.
            **kwargs: Additional keyword arguments.

        Returns:
            True if deletion was successful, None if no IDs provided.
        """
        if not ids:
            return None

        try:
            async with self._aget_cursor() as cursor:
                placeholders = ",".join(["%s"] * len(ids))
                sql = SQL_DELETE_BY_IDS.format(
                    table_name=self._table_name, placeholders=placeholders
                )
                await cursor.execute(sql, ids)

            logger.info("Deleted %d vectors from table %s", len(ids), self._table_name)
            return True
        except Exception as e:
            # If table doesn't exist (error 1146), deletion is considered successful
            error_msg = str(e)
            if "1146" in error_msg or "doesn't exist" in error_msg.lower():
                logger.debug(
                    "Table %s does not exist, skipping delete operation",
                    self._table_name,
                )
                return True
            raise

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Get documents by their IDs.

        Args:
            ids: List of IDs to retrieve.

        Returns:
            List of Document objects.
        """
        if not ids:
            return []

        try:
            with self._get_cursor() as cursor:
                placeholders = ",".join(["%s"] * len(ids))
                sql = SQL_GET_BY_IDS.format(
                    table_name=self._table_name, placeholders=placeholders
                )
                cursor.execute(sql, list(ids))

                documents = []
                for record in cursor:
                    metadata = record["metadata"]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)

                    documents.append(
                        Document(
                            id=record["id"],
                            page_content=record["text"],
                            metadata=metadata or {},
                        )
                    )

            return documents
        except Exception as e:
            # If table doesn't exist (error 1146), return empty list
            error_msg = str(e)
            if "1146" in error_msg or "doesn't exist" in error_msg.lower():
                logger.debug(
                    "Table %s does not exist, returning empty results", self._table_name
                )
                return []
            raise

    async def aget_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Async get documents by their IDs.

        Args:
            ids: List of IDs to retrieve.

        Returns:
            List of Document objects.
        """
        if not ids:
            return []

        try:
            async with self._aget_cursor() as cursor:
                placeholders = ",".join(["%s"] * len(ids))
                sql = SQL_GET_BY_IDS.format(
                    table_name=self._table_name, placeholders=placeholders
                )
                await cursor.execute(sql, list(ids))

                documents = []
                async for record in cursor:
                    metadata = record["metadata"]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)

                    documents.append(
                        Document(
                            id=record["id"],
                            page_content=record["text"],
                            metadata=metadata or {},
                        )
                    )

            return documents
        except Exception as e:
            # If table doesn't exist (error 1146), return empty list
            error_msg = str(e)
            if "1146" in error_msg or "doesn't exist" in error_msg.lower():
                logger.debug(
                    "Table %s does not exist, returning empty results", self._table_name
                )
                return []
            raise

    @classmethod
    def from_texts(
        cls: Type["PolarDBXVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        host: str,
        port: int = 3306,
        user: str,
        password: str,
        database: str,
        table_name: str = "polardbx_vectors",
        distance_strategy: Literal["cosine", "euclidean", "inner_product"] = "cosine",
        hnsw_m: int = 6,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "PolarDBXVectorStore":
        """Create an PolarDBXVectorStore vector store from a list of texts.

        Args:
            texts: List of texts to add to the vector store.
            embedding: Embedding model to use.
            metadatas: Optional list of metadata dictionaries.
            host: MySQL host (required).
            port: MySQL port. Defaults to 3306.
            user: MySQL user (required).
            password: MySQL password (required).
            database: MySQL database (required).
            table_name: Table name. Defaults to "polardbx_vectors".
            distance_strategy: Distance strategy. Defaults to "cosine".
            hnsw_m: HNSW M parameter. Defaults to 6.
            ids: Optional list of IDs.
            **kwargs: Additional keyword arguments.

        Returns:
            PolarDBXVectorStore vector store instance.
        """
        store = cls(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            embedding=embedding,
            table_name=table_name,
            distance_strategy=distance_strategy,
            hnsw_m=hnsw_m,
            **kwargs,
        )
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls: Type["PolarDBXVectorStore"],
        documents: List[Document],
        embedding: Embeddings,
        *,
        host: str,
        port: int = 3306,
        user: str,
        password: str,
        database: str,
        table_name: str = "polardbx_vectors",
        distance_strategy: Literal["cosine", "euclidean", "inner_product"] = "cosine",
        hnsw_m: int = 6,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "PolarDBXVectorStore":
        """Create an PolarDBXVectorStore vector store from documents.

        Args:
            documents: List of documents to add.
            embedding: Embedding model to use.
            host: MySQL host (required).
            port: MySQL port. Defaults to 3306.
            user: MySQL user (required).
            password: MySQL password (required).
            database: MySQL database (required).
            table_name: Table name. Defaults to "polardbx_vectors".
            distance_strategy: Distance strategy. Defaults to "cosine".
            hnsw_m: HNSW M parameter. Defaults to 6.
            ids: Optional list of IDs.
            **kwargs: Additional keyword arguments.

        Returns:
            PolarDBXVectorStore vector store instance.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        if ids is None:
            ids = []
            for doc in documents:
                if doc.id is not None:
                    ids.append(doc.id)
                else:
                    ids.append(str(uuid.uuid4()))

        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            table_name=table_name,
            distance_strategy=distance_strategy,
            hnsw_m=hnsw_m,
            ids=ids,
            **kwargs,
        )

    @classmethod
    async def afrom_texts(
        cls: Type["PolarDBXVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        host: str,
        port: int = 3306,
        user: str,
        password: str,
        database: str,
        table_name: str = "polardbx_vectors",
        distance_strategy: Literal["cosine", "euclidean", "inner_product"] = "cosine",
        hnsw_m: int = 6,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "PolarDBXVectorStore":
        """Async create an PolarDBXVectorStore vector store from a list of texts.

        Args:
            texts: List of texts to add to the vector store.
            embedding: Embedding model to use.
            metadatas: Optional list of metadata dictionaries.
            host: MySQL host (required).
            port: MySQL port. Defaults to 3306.
            user: MySQL user (required).
            password: MySQL password (required).
            database: MySQL database (required).
            table_name: Table name. Defaults to "polardbx_vectors".
            distance_strategy: Distance strategy. Defaults to "cosine".
            hnsw_m: HNSW M parameter. Defaults to 6.
            ids: Optional list of IDs.
            **kwargs: Additional keyword arguments.

        Returns:
            PolarDBXVectorStore vector store instance.
        """
        store = cls(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            embedding=embedding,
            table_name=table_name,
            distance_strategy=distance_strategy,
            hnsw_m=hnsw_m,
            **kwargs,
        )
        await store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    async def afrom_documents(
        cls: Type["PolarDBXVectorStore"],
        documents: List[Document],
        embedding: Embeddings,
        *,
        host: str,
        port: int = 3306,
        user: str,
        password: str,
        database: str,
        table_name: str = "polardbx_vectors",
        distance_strategy: Literal["cosine", "euclidean", "inner_product"] = "cosine",
        hnsw_m: int = 6,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "PolarDBXVectorStore":
        """Async create an PolarDBXVectorStore vector store from documents.

        Args:
            documents: List of documents to add.
            embedding: Embedding model to use.
            host: MySQL host (required).
            port: MySQL port. Defaults to 3306.
            user: MySQL user (required).
            password: MySQL password (required).
            database: MySQL database (required).
            table_name: Table name. Defaults to "polardbx_vectors".
            distance_strategy: Distance strategy. Defaults to "cosine".
            hnsw_m: HNSW M parameter. Defaults to 6.
            ids: Optional list of IDs.
            **kwargs: Additional keyword arguments.

        Returns:
            PolarDBXVectorStore vector store instance.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        if ids is None:
            ids = []
            for doc in documents:
                if doc.id is not None:
                    ids.append(doc.id)
                else:
                    ids.append(str(uuid.uuid4()))

        return await cls.afrom_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            table_name=table_name,
            distance_strategy=distance_strategy,
            hnsw_m=hnsw_m,
            ids=ids,
            **kwargs,
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance (MMR).

        MMR balances similarity to the query with diversity among selected documents.

        Args:
            query: Query string to search for.
            k: Number of documents to return. Defaults to 4.
            fetch_k: Number of documents to fetch before filtering. Defaults to 20.
            lambda_mult: Diversity factor (0 = max diversity, 1 = min diversity).
                Defaults to 0.5.
            filter: Optional metadata filter dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents selected by MMR.
        """
        query_embedding = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding=query_embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using MMR by embedding vector.

        Args:
            embedding: Embedding vector to search for.
            k: Number of documents to return. Defaults to 4.
            fetch_k: Number of documents to fetch before filtering. Defaults to 20.
            lambda_mult: Diversity factor. Defaults to 0.5.
            filter: Optional metadata filter dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents selected by MMR.
        """
        # Fetch more documents than needed for MMR
        docs_with_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=fetch_k,
            filter=filter,
            **kwargs,
        )

        if not docs_with_scores:
            return []

        # Fetch stored embeddings from the database
        doc_ids = [doc.id for doc, _ in docs_with_scores]
        doc_embeddings = self._fetch_embeddings_by_ids(doc_ids)

        # Fallback to re-embedding if database fetch fails
        if not doc_embeddings or any(len(e) == 0 for e in doc_embeddings):
            doc_texts = [doc.page_content for doc, _ in docs_with_scores]
            doc_embeddings = self._embedding.embed_documents(doc_texts)

        # Apply MMR
        selected_indices = self._maximal_marginal_relevance(
            query_embedding=embedding,
            embedding_list=doc_embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        return [docs_with_scores[i][0] for i in selected_indices]

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async return docs selected using the maximal marginal relevance (MMR).

        MMR balances similarity to the query with diversity among selected documents.

        Args:
            query: Query string to search for.
            k: Number of documents to return. Defaults to 4.
            fetch_k: Number of documents to fetch before filtering. Defaults to 20.
            lambda_mult: Diversity factor (0 = max diversity, 1 = min diversity).
                Defaults to 0.5.
            filter: Optional metadata filter dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents selected by MMR.
        """
        # Generate query embedding (use async if available)
        if hasattr(self._embedding, "aembed_query"):
            query_embedding = await self._embedding.aembed_query(query)
        else:
            query_embedding = await run_in_executor(
                None, self._embedding.embed_query, query
            )
        return await self.amax_marginal_relevance_search_by_vector(
            embedding=query_embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async return docs selected using MMR by embedding vector.

        Args:
            embedding: Embedding vector to search for.
            k: Number of documents to return. Defaults to 4.
            fetch_k: Number of documents to fetch before filtering. Defaults to 20.
            lambda_mult: Diversity factor. Defaults to 0.5.
            filter: Optional metadata filter dictionary.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents selected by MMR.
        """
        # Fetch more documents than needed for MMR
        docs_with_scores = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding,
            k=fetch_k,
            filter=filter,
            **kwargs,
        )

        if not docs_with_scores:
            return []

        # Fetch stored embeddings from the database
        doc_ids = [doc.id for doc, _ in docs_with_scores]
        doc_embeddings = await self._afetch_embeddings_by_ids(doc_ids)

        # Fallback to re-embedding if database fetch fails
        if not doc_embeddings or any(len(e) == 0 for e in doc_embeddings):
            doc_texts = [doc.page_content for doc, _ in docs_with_scores]
            if hasattr(self._embedding, "aembed_documents"):
                doc_embeddings = await self._embedding.aembed_documents(doc_texts)
            else:
                doc_embeddings = await run_in_executor(
                    None, self._embedding.embed_documents, doc_texts
                )

        # Apply MMR
        selected_indices = self._maximal_marginal_relevance(
            query_embedding=embedding,
            embedding_list=doc_embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        return [docs_with_scores[i][0] for i in selected_indices]

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs and scores selected using MMR by embedding vector.

        Args:
            embedding: Embedding vector to search for.
            k: Number of documents to return. Defaults to 4.
            fetch_k: Number of documents to fetch before filtering. Defaults to 20.
            lambda_mult: Diversity factor. Defaults to 0.5.
            filter: Optional metadata filter dictionary.
            **kwargs: Additional keyword arguments (ef_search, search_type, etc.).

        Returns:
            List of tuples of (Document, similarity_score) selected by MMR.
        """
        docs_with_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=fetch_k,
            filter=filter,
            **kwargs,
        )

        if not docs_with_scores:
            return []

        doc_ids = [doc.id for doc, _ in docs_with_scores]
        doc_embeddings = self._fetch_embeddings_by_ids(doc_ids)

        if not doc_embeddings or any(len(e) == 0 for e in doc_embeddings):
            doc_texts = [doc.page_content for doc, _ in docs_with_scores]
            doc_embeddings = self._embedding.embed_documents(doc_texts)

        selected_indices = self._maximal_marginal_relevance(
            query_embedding=embedding,
            embedding_list=doc_embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        return [docs_with_scores[i] for i in selected_indices]

    async def amax_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Async return docs and scores selected using MMR by embedding vector.

        Args:
            embedding: Embedding vector to search for.
            k: Number of documents to return. Defaults to 4.
            fetch_k: Number of documents to fetch before filtering. Defaults to 20.
            lambda_mult: Diversity factor. Defaults to 0.5.
            filter: Optional metadata filter dictionary.
            **kwargs: Additional keyword arguments (ef_search, search_type, etc.).

        Returns:
            List of tuples of (Document, similarity_score) selected by MMR.
        """
        docs_with_scores = await self.asimilarity_search_with_score_by_vector(
            embedding=embedding,
            k=fetch_k,
            filter=filter,
            **kwargs,
        )

        if not docs_with_scores:
            return []

        doc_ids = [doc.id for doc, _ in docs_with_scores]
        doc_embeddings = await self._afetch_embeddings_by_ids(doc_ids)

        if not doc_embeddings or any(len(e) == 0 for e in doc_embeddings):
            doc_texts = [doc.page_content for doc, _ in docs_with_scores]
            if hasattr(self._embedding, "aembed_documents"):
                doc_embeddings = await self._embedding.aembed_documents(doc_texts)
            else:
                doc_embeddings = await run_in_executor(
                    None, self._embedding.embed_documents, doc_texts
                )

        selected_indices = self._maximal_marginal_relevance(
            query_embedding=embedding,
            embedding_list=doc_embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        return [docs_with_scores[i] for i in selected_indices]

    @staticmethod
    def _maximal_marginal_relevance(
        query_embedding: List[float],
        embedding_list: List[List[float]],
        k: int = 4,
        lambda_mult: float = 0.5,
    ) -> List[int]:
        """Calculate maximal marginal relevance.

        Args:
            query_embedding: Query embedding.
            embedding_list: List of document embeddings.
            k: Number of documents to select.
            lambda_mult: Diversity factor.

        Returns:
            List of selected indices.
        """
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "numpy is required for MMR search. "
                "Please install it with `pip install langchain-polardbx[mmr]`."
            ) from e

        if not embedding_list:
            return []

        query_vec = np.array(query_embedding)
        doc_vecs = np.array(embedding_list)

        # Calculate similarity to query
        query_doc_similarity = np.dot(doc_vecs, query_vec) / (
            np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec)
        )

        # Start with the most similar document
        selected = [int(np.argmax(query_doc_similarity))]
        candidates = list(range(len(embedding_list)))
        candidates.remove(selected[0])

        while len(selected) < min(k, len(embedding_list)) and candidates:
            best_score = -float("inf")
            best_idx = -1

            for idx in candidates:
                # Calculate similarity to already selected documents
                max_sim_to_selected = max(
                    np.dot(doc_vecs[idx], doc_vecs[sel_idx])
                    / (
                        np.linalg.norm(doc_vecs[idx])
                        * np.linalg.norm(doc_vecs[sel_idx])
                    )
                    for sel_idx in selected
                )

                # MMR score
                mmr_score = (
                    lambda_mult * query_doc_similarity[idx]
                    - (1 - lambda_mult) * max_sim_to_selected
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx != -1:
                selected.append(best_idx)
                candidates.remove(best_idx)

        return selected

    def drop_table(self) -> None:
        """Drop the vector table."""
        self._drop_table()

    def clear(self) -> None:
        """Clear all data from the table."""
        try:
            with self._get_cursor() as cursor:
                cursor.execute(f"TRUNCATE TABLE `{self._table_name}`")
            logger.info("Cleared all data from table %s", self._table_name)
        except Exception as e:
            if "1146" in str(e) or "doesn't exist" in str(e).lower():
                logger.debug("Table %s does not exist, skipping clear", self._table_name)
            else:
                raise

    async def aclear(self) -> None:
        """Async clear all data from the table."""
        try:
            async with self._aget_cursor() as cursor:
                await cursor.execute(f"TRUNCATE TABLE `{self._table_name}`")
            logger.info("Cleared all data from table %s", self._table_name)
        except Exception as e:
            if "1146" in str(e) or "doesn't exist" in str(e).lower():
                logger.debug("Table %s does not exist, skipping clear", self._table_name)
            else:
                raise

    def count(self) -> int:
        """Get the number of vectors in the table."""
        try:
            with self._get_cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) as count FROM `{self._table_name}`")
                result = cursor.fetchone()
                return result["count"] if result else 0
        except Exception as e:
            if "1146" in str(e) or "doesn't exist" in str(e).lower():
                return 0
            raise

    async def acount(self) -> int:
        """Async get the number of vectors in the table."""
        try:
            async with self._aget_cursor() as cursor:
                await cursor.execute(f"SELECT COUNT(*) as count FROM `{self._table_name}`")
                result = await cursor.fetchone()
                return result["count"] if result else 0
        except Exception as e:
            if "1146" in str(e) or "doesn't exist" in str(e).lower():
                return 0
            raise

    def search_by_metadata(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Document]:
        """Search documents by metadata conditions only (no vector similarity).

        This method performs a metadata-based query without vector similarity search.

        Args:
            filter: Filter dictionary with optional operators.
                Example: {"category": "phone", "price": {"$gt": 1000}}
            limit: Maximum number of results to return. Defaults to 10.

        Returns:
            List of Documents matching the metadata filter.

        Example:
            .. code-block:: python

                # Simple equality
                docs = vectorstore.search_by_metadata(
                    filter={"category": "phone"},
                    limit=10
                )

                # With operators
                docs = vectorstore.search_by_metadata(
                    filter={"category": "phone", "price": {"$gt": 1000}},
                    limit=10
                )
        """
        where_clause, params = self._build_filter_clause(filter)

        sql = f"""
        SELECT id, text, metadata
        FROM `{self._table_name}`
        {where_clause}
        LIMIT %s
        """

        params.append(limit)

        try:
            with self._get_cursor() as cursor:
                cursor.execute(sql, params)

                documents = []
                for record in cursor:
                    metadata = record["metadata"]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)

                    documents.append(
                        Document(
                            id=record["id"],
                            page_content=record["text"],
                            metadata=metadata or {},
                        )
                    )
        except Exception as e:
            if "1146" in str(e) or "doesn't exist" in str(e).lower():
                return []
            raise

        return documents

    def delete_by_metadata(
        self,
        filter: Dict[str, Any],
    ) -> int:
        """Delete documents matching metadata conditions.

        Args:
            filter: Filter dictionary (required).
                Example: {"category": "phone", "status": {"$eq": "deleted"}}

        Returns:
            Number of deleted documents.

        Example:
            .. code-block:: python

                # Delete all documents with category 'phone'
                deleted = vectorstore.delete_by_metadata(
                    filter={"category": "phone"}
                )

                # Delete with operator
                deleted = vectorstore.delete_by_metadata(
                    filter={"status": "deleted", "price": {"$lt": 100}}
                )
        """
        if not filter:
            raise ValueError("'filter' must be provided for delete_by_metadata")

        where_clause, params = self._build_filter_clause(filter)

        if not where_clause:
            raise ValueError("Filter resulted in empty condition")

        sql = f"DELETE FROM `{self._table_name}` {where_clause}"

        try:
            with self._get_cursor() as cursor:
                cursor.execute(sql, params)
                deleted_count = cursor.rowcount
        except Exception as e:
            if "1146" in str(e) or "doesn't exist" in str(e).lower():
                return 0
            raise

        logger.info(
            "Deleted %d documents from table %s by metadata",
            deleted_count,
            self._table_name,
        )
        return deleted_count

    def exists(self, id: str) -> bool:
        """Check if a document with the given ID exists.

        Args:
            id: The document ID to check.

        Returns:
            True if the document exists, False otherwise.
        """
        with self._get_cursor() as cursor:
            cursor.execute(
                f"SELECT 1 FROM `{self._table_name}` WHERE id = %s LIMIT 1",
                (id,),
            )
            return cursor.fetchone() is not None

    async def asearch_by_metadata(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Document]:
        """Async search documents by metadata conditions only (no vector similarity).

        Args:
            filter: Filter dictionary with optional operators.
            limit: Maximum number of results to return. Defaults to 10.

        Returns:
            List of Documents matching the metadata filter.
        """
        where_clause, params = self._build_filter_clause(filter)

        sql = f"""
        SELECT id, text, metadata
        FROM `{self._table_name}`
        {where_clause}
        LIMIT %s
        """

        params.append(limit)

        try:
            async with self._aget_cursor() as cursor:
                await cursor.execute(sql, params)

                documents = []
                async for record in cursor:
                    metadata = record["metadata"]
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)

                    documents.append(
                        Document(
                            id=record["id"],
                            page_content=record["text"],
                            metadata=metadata or {},
                        )
                    )
        except Exception as e:
            if "1146" in str(e) or "doesn't exist" in str(e).lower():
                return []
            raise

        return documents

    async def adelete_by_metadata(
        self,
        filter: Dict[str, Any],
    ) -> int:
        """Async delete documents matching metadata conditions.

        Args:
            filter: Filter dictionary (required).

        Returns:
            Number of deleted documents.
        """
        if not filter:
            raise ValueError("'filter' must be provided for adelete_by_metadata")

        where_clause, params = self._build_filter_clause(filter)

        if not where_clause:
            raise ValueError("Filter resulted in empty condition")

        sql = f"DELETE FROM `{self._table_name}` {where_clause}"

        try:
            async with self._aget_cursor() as cursor:
                await cursor.execute(sql, params)
                deleted_count = cursor.rowcount
        except Exception as e:
            if "1146" in str(e) or "doesn't exist" in str(e).lower():
                return 0
            raise

        logger.info(
            "Deleted %d documents from table %s by metadata (async)",
            deleted_count,
            self._table_name,
        )
        return deleted_count

    async def aexists(self, id: str) -> bool:
        """Async check if a document with the given ID exists.

        Args:
            id: The document ID to check.

        Returns:
            True if the document exists, False otherwise.
        """
        async with self._aget_cursor() as cursor:
            await cursor.execute(
                f"SELECT 1 FROM `{self._table_name}` WHERE id = %s LIMIT 1",
                (id,),
            )
            return await cursor.fetchone() is not None

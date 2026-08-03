from abc import abstractmethod
from typing import Any

from uilts.configs import SQLDBConfigs, resolve_secret
from uilts.logger import logger


class BaseSQLDB(SQLDBConfigs):
    """Common interface for tabular/SQL database connectors (PostgreSQL, MySQL, BigQuery, SQLite, ...).

    Subclasses implement `_initialize_connection(**kwargs)`: build the driver's connection object and
    assign it to `self.connection`. Any kwarg matching an existing class attribute (e.g. `sslmode`,
    `charset`) is applied via `setattr` first, so driver-specific options can be set at construction
    time and listed in the class-level `arguments` list, which `_connect_kwargs` forwards to the driver.

    `query` and `execute` are implemented here against the DB-API 2.0 cursor protocol, which every
    driver below follows except BigQuery — that one overrides both.

    Driver packages are imported lazily inside `_initialize_connection` rather than at module import,
    so using one database does not require installing every other database's driver.

    Args:
        name: Db name as referenced by agents, e.g. "ds_knowledge_db".
        db: Engine to connect with, e.g. "postgresql".
        host/port/database/user/password: Standard connection fields.
        connection_string: Full DSN; when set it takes precedence over the individual fields.
        path: File path for file-backed engines (SQLite, DuckDB).
        project/credentials: Cloud project id and service-account JSON path, used by BigQuery.
        **kwargs: Driver-specific overrides applied to matching class attributes on the subclass.
    """

    connection: Any = None
    arguments: list[str] = []  # names of the driver-specific params a subclass forwards to its driver

    def __init__(
        self,
        name: str,
        db: str,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        connection_string: str | None = None,
        path: str | None = None,
        project: str | None = None,
        credentials: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            db=db,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            connection_string=connection_string,
            path=path,
            project=project,
            credentials=credentials,
        )
        self.secret_checker()
        self._initialize_connection(**kwargs)

    def secret_checker(self) -> None:
        """Resolve `password` and `credentials` as env var names or literal values; either may be unset."""
        for field in ('password', 'credentials'):
            value = resolve_secret(getattr(self, field, None), self.name, field=field, required=False)
            setattr(self, field, value)

    def _connect_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Apply construction-time overrides, then collect this driver's optional params."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        connect_kwargs: dict[str, Any] = {}
        for argument in self.arguments:
            if kwargs.get(argument) or getattr(self, argument, None) is not None:
                connect_kwargs[argument] = kwargs.get(argument, getattr(self, argument, None))
        return connect_kwargs

    @abstractmethod
    def _initialize_connection(self, **kwargs: Any) -> None:
        """Build the driver connection and assign it to `self.connection`."""

    def query(self, sql: str, params: tuple | dict | None = None) -> list[dict[str, Any]]:
        """Run a read query and return its rows as dicts keyed by column name."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql, params or ())
            columns = [column[0] for column in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            logger.info(f"{self.name} returned {len(rows)} rows.")
            return rows
        finally:
            cursor.close()

    def execute(self, sql: str, params: tuple | dict | None = None) -> int:
        """Run a write statement, commit it, and return the affected row count."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql, params or ())
            self.connection.commit()
            logger.info(f"{self.name} affected {cursor.rowcount} rows.")
            return cursor.rowcount
        finally:
            cursor.close()

    def close(self) -> None:
        """Close the underlying connection."""
        if self.connection:
            self.connection.close()
            self.connection = None


class PostgreSQLDB(BaseSQLDB):
    """PostgreSQL connector, via `psycopg2`.

    Extra params: `sslmode`, `connect_timeout`, `application_name`, `options`.
    """

    sslmode: str | None = None  # e.g. "require", "verify-full"
    connect_timeout: int | None = None  # seconds to wait before giving up on connecting
    application_name: str | None = None  # label shown in pg_stat_activity
    options: str | None = None  # command-line options passed to the server, e.g. "-c search_path=..."
    arguments: list[str] = [
        'sslmode',
        'connect_timeout',
        'application_name',
        'options',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        import psycopg2

        connect_kwargs = self._connect_kwargs(**kwargs)
        if self.connection_string:
            self.connection = psycopg2.connect(self.connection_string, **connect_kwargs)
        else:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port or 5432,
                dbname=self.database,
                user=self.user,
                password=self.password,
                **connect_kwargs,
            )


class MySQLDB(BaseSQLDB):
    """MySQL connector, via `pymysql`.

    Extra params: `charset`, `connect_timeout`, `ssl_ca`, `autocommit`.
    """

    charset: str | None = None  # e.g. "utf8mb4"
    connect_timeout: int | None = None  # seconds to wait before giving up on connecting
    ssl_ca: str | None = None  # path to the CA bundle used to verify the server certificate
    autocommit: bool | None = None  # commit each statement instead of opening a transaction
    arguments: list[str] = [
        'charset',
        'connect_timeout',
        'ssl_ca',
        'autocommit',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        import pymysql

        self.connection = pymysql.connect(
            host=self.host,
            port=self.port or 3306,
            database=self.database,
            user=self.user,
            password=self.password,
            **self._connect_kwargs(**kwargs),
        )


class SQLiteDB(BaseSQLDB):
    """Local SQLite connector, via the stdlib `sqlite3` module.

    Uses `path` as the database file; falls back to an in-memory database when unset.
    Extra params: `timeout`, `isolation_level`, `check_same_thread`.
    """

    timeout: float | None = None  # seconds to wait for a lock before raising
    isolation_level: str | None = None  # "" for autocommit, or "DEFERRED"/"IMMEDIATE"/"EXCLUSIVE"
    check_same_thread: bool | None = None  # allow the connection to be used across threads when False
    arguments: list[str] = [
        'timeout',
        'isolation_level',
        'check_same_thread',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        import sqlite3

        self.connection = sqlite3.connect(
            self.path or ":memory:",
            **self._connect_kwargs(**kwargs),
        )


class DuckDBDB(BaseSQLDB):
    """Embedded DuckDB connector, via `duckdb`.

    Uses `path` as the database file; falls back to an in-memory database when unset.
    Extra params: `read_only`, `config`.
    """

    read_only: bool | None = None  # open the file without allowing writes
    config: dict[str, Any] | None = None  # engine settings, e.g. {"threads": 4, "memory_limit": "4GB"}
    arguments: list[str] = [
        'read_only',
        'config',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        import duckdb

        self.connection = duckdb.connect(
            database=self.path or ":memory:",
            **self._connect_kwargs(**kwargs),
        )


class SnowflakeDB(BaseSQLDB):
    """Snowflake connector, via `snowflake-connector-python`.

    Extra params: `account`, `warehouse`, `schema`, `role`, `authenticator`.
    """

    account: str | None = None  # Snowflake account identifier, e.g. "xy12345.eu-central-1"
    warehouse: str | None = None  # compute warehouse used to run queries
    schema: str | None = None  # default schema within the database
    role: str | None = None  # role to assume for the session
    authenticator: str | None = None  # e.g. "externalbrowser" for SSO
    arguments: list[str] = [
        'account',
        'warehouse',
        'schema',
        'role',
        'authenticator',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        import snowflake.connector

        self.connection = snowflake.connector.connect(
            user=self.user,
            password=self.password,
            database=self.database,
            **self._connect_kwargs(**kwargs),
        )


class RedshiftDB(PostgreSQLDB):
    """Amazon Redshift connector.

    Redshift speaks the PostgreSQL wire protocol, so this reuses `PostgreSQLDB`'s connection handling
    and only changes the default port.
    """

    def _initialize_connection(self, **kwargs: Any) -> None:
        self.port = self.port or 5439
        super()._initialize_connection(**kwargs)


class BigQueryDB(BaseSQLDB):
    """Google BigQuery connector, via `google-cloud-bigquery`.

    BigQuery's client is not DB-API 2.0, so `query` and `execute` are overridden to go through
    `client.query(...)` instead of a cursor. `database` is treated as the default dataset.
    Extra params: `location`, `maximum_bytes_billed`, `use_query_cache`.
    """

    location: str | None = None  # dataset region, e.g. "EU" or "us-central1"
    maximum_bytes_billed: int | None = None  # cap a query's cost; it fails rather than exceeding this
    use_query_cache: bool | None = None  # reuse cached results for identical queries
    arguments: list[str] = [
        'location',
        'maximum_bytes_billed',
        'use_query_cache',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        from google.cloud import bigquery

        self._connect_kwargs(**kwargs)  # applies construction-time overrides onto self
        if self.credentials:
            self.connection = bigquery.Client.from_service_account_json(
                self.credentials,
                project=self.project,
                location=self.location,
            )
        else:
            self.connection = bigquery.Client(project=self.project, location=self.location)

    def _job_config(self) -> Any:
        """Build a QueryJobConfig from the cost/cache params that were set."""
        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig()
        if self.maximum_bytes_billed is not None:
            job_config.maximum_bytes_billed = self.maximum_bytes_billed
        if self.use_query_cache is not None:
            job_config.use_query_cache = self.use_query_cache
        if self.database:
            job_config.default_dataset = f"{self.project}.{self.database}"
        return job_config

    def query(self, sql: str, params: tuple | dict | None = None) -> list[dict[str, Any]]:
        """Run a read query and return its rows as dicts keyed by column name."""
        job = self.connection.query(sql, job_config=self._job_config())
        rows = [dict(row) for row in job.result()]
        logger.info(f"{self.name} returned {len(rows)} rows.")
        return rows

    def execute(self, sql: str, params: tuple | dict | None = None) -> int:
        """Run a write statement and return the affected row count."""
        job = self.connection.query(sql, job_config=self._job_config())
        job.result()
        affected = job.num_dml_affected_rows or 0
        logger.info(f"{self.name} affected {affected} rows.")
        return affected

    def close(self) -> None:
        """Close the underlying client."""
        if self.connection:
            self.connection.close()
            self.connection = None

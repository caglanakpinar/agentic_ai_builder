from abc import abstractmethod
from typing import Any

from uilts.configs import TextDBConfigs, resolve_secret
from uilts.logger import logger


DOCUMENT_FIELD = "document"  # field every connector stores a record's text under


class BaseTextDB(TextDBConfigs):
    """Common interface for text database connectors (Elasticsearch, OpenSearch, Meilisearch, ...).

    The read side of a text db is a lookup rather than a search: `query` takes the indexes to fetch and
    returns the documents stored against them. That is the second half of a split retrieval flow — a
    vector db runs the similarity search and hands back ids, and this store turns those ids into the
    text an agent puts in its prompt.

    Subclasses implement `_initialize_connection(**kwargs)`: build the driver's client and assign it to
    `self.client`. Any kwarg matching an existing class attribute (e.g. `verify_certs`, `num_typos`) is
    applied via `setattr` first, so driver-specific options can be set at construction time and listed
    in the class-level `arguments` list, which `_connect_kwargs` forwards to the driver.

    Every connector exposes the same four operations, so an agent can swap engines without changing
    call sites:
      - `upsert(ids, documents, metadatas)`: insert or replace documents by id.
      - `query(indexes, top_k, filter)`: fetch the records stored against `indexes`, returning
        `[{"id": ..., "metadata": {...}, "document": ...}]` in the order the indexes were asked for.
        An index that isn't stored, or whose metadata fails `filter`, is left out — so the result can
        be shorter than the request and is never padded. There is no `score`: this is a lookup rather
        than a ranking, and the order carries no relevance.
      - `delete(ids)`: remove documents by id.
      - `count()`: number of stored documents.

    Driver packages are imported lazily inside `_initialize_connection` rather than at module import,
    so using one engine does not require installing every other engine's driver.

    Args:
        name: Db name as referenced by agents, e.g. "ds_knowledge_db_text".
        db: Engine to connect with, e.g. "elasticsearch".
        type: Db category, "text" for every connector here.
        host/port/url: Server location; `url` wins when both are set.
        api_key: Literal key, or the name of an env var holding it.
        user/password: Basic-auth credentials for engines that take them instead of a key.
        path: Storage directory for embedded engines.
        collection_name: Index/collection the connector reads and writes.
        analyzer: Text analyzer the engine tokenises with, e.g. "standard", "english".
        **kwargs: Driver-specific overrides applied to matching class attributes on the subclass.
    """

    client: Any = None
    arguments: list[str] = []  # names of the driver-specific params a subclass forwards to its driver

    def __init__(
        self,
        name: str,
        db: str,
        type: str = "text",
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        user: str | None = None,
        password: str | None = None,
        path: str | None = None,
        url: str | None = None,
        collection_name: str = "default",
        analyzer: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            db=db,
            type=type,
            host=host,
            port=port,
            api_key=api_key,
            user=user,
            password=password,
            path=path,
            url=url,
            collection_name=collection_name,
            analyzer=analyzer,
        )
        self.secret_checker()
        self._initialize_connection(**kwargs)

    def secret_checker(self) -> None:
        """Resolve `api_key` and `password` as env var names or literal values; either may be unset."""
        self.api_key = resolve_secret(self.api_key, self.name, required=False)
        self.password = resolve_secret(self.password, self.name, field="password", required=False)

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

    def _server_url(self, default_port: int) -> str:
        """Build the server URL from `url`, or from `host`/`port` falling back to `default_port`."""
        return self.url or f"http://{self.host or 'localhost'}:{self.port or default_port}"

    def _record(self, id: str, source: dict[str, Any]) -> dict[str, Any]:
        """Split one stored record into the `{id, metadata, document}` shape every engine returns.

        `source` is whatever the engine round-tripped from `upsert`: the metadata fields plus the text
        under `DOCUMENT_FIELD`. The text is lifted out so `metadata` holds only what the caller stored
        as metadata.
        """
        metadata = {key: value for key, value in source.items() if key not in (DOCUMENT_FIELD, 'id')}
        return {"id": id, "metadata": metadata, "document": source.get(DOCUMENT_FIELD)}

    def _matches(self, metadata: dict[str, Any], filter: dict[str, Any] | None) -> bool:
        """Whether `metadata` satisfies `filter`, for engines whose fetch takes no filter of its own."""
        return not filter or all(metadata.get(key) == value for key, value in filter.items())

    def _ordered_results(
        self,
        results: list[dict[str, Any]],
        indexes: list[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Put fetched records back into the order their indexes were asked for, capped at `top_k`.

        Engines return a fetch in whatever order suits their storage, so every connector runs its
        results through here to give callers the one order they can rely on: the caller's own.
        """
        order = {str(index): position for position, index in enumerate(indexes)}
        results.sort(key=lambda result: order.get(str(result["id"]), len(order)))
        return results[:top_k] if top_k else results

    def _documents_and_metadatas(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Merge each document and its metadata into the one record body engines are written."""
        if len(documents) != len(ids):
            raise ValueError(
                f"{self.name}: got {len(ids)} ids but {len(documents)} documents; they must line up."
            )

        bodies = []
        for position in range(len(ids)):
            body = dict((metadatas or [{}] * len(ids))[position])
            body[DOCUMENT_FIELD] = documents[position]
            bodies.append(body)
        return bodies

    @abstractmethod
    def _initialize_connection(self, **kwargs: Any) -> None:
        """Build the driver client and assign it to `self.client`."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or replace documents by id."""

    @abstractmethod
    def query(
        self,
        indexes: list[str],
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the records stored against `indexes`, in the order they were asked for."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove documents by id."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored documents."""

    def close(self) -> None:
        """Release the client. Overridden by engines that hold an open connection."""
        self.client = None


class ChromaTextDB(BaseTextDB):
    """Chroma as a document store, via `chromadb` — the text half of a split retrieval flow.

    Chroma is a vector database, but nothing here searches it: a text db is only ever asked to turn the
    ids a vector search returned into the documents behind them, and that is `get(ids=...)`, a lookup
    that touches no index. So the collection is opened with `embedding_function=None` — no model is
    downloaded, no text is embedded on write, and the vectors stay in the vector db where the search
    happens. It is the engine to reach for when the knowledge base should live in a directory rather
    than behind a search server.

    Runs embedded against `path`, or against a Chroma server when `host` is set. Chroma has no fetch-time
    filter that matches the rest of this interface, so `filter` is applied to the returned metadata here.
    Extra params: `tenant`, `database`.
    """

    tenant: str | None = None  # multi-tenant deployments only
    database: str | None = None  # database within the tenant
    arguments: list[str] = [
        'tenant',
        'database',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        import chromadb

        connect_kwargs = self._connect_kwargs(**kwargs)
        if self.host:
            client = chromadb.HttpClient(host=self.host, port=self.port or 8000, **connect_kwargs)
        else:
            client = chromadb.PersistentClient(path=self.path or "./chroma", **connect_kwargs)

        self.client = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,  # this store is a lookup; embedding here would be a second, unused copy
        )
        logger.info(f"{self.name} opened collection {self.collection_name} holding {self.count()} documents.")

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        # Validates that the documents line up with the ids. The merged bodies it returns are not what
        # Chroma is written, though: it stores the text and the metadata in separate fields of its own.
        self._documents_and_metadatas(ids, documents, metadatas)

        records = [dict(metadata) for metadata in (metadatas or [])]
        self.client.upsert(
            ids=[str(id) for id in ids],
            documents=list(documents),
            # Chroma refuses an empty metadata dict, so metadata is sent only when every record carries some.
            metadatas=records if len(records) == len(ids) and all(records) else None,
        )
        logger.info(f"{self.name} upserted {len(ids)} documents.")

    def query(
        self,
        indexes: list[str],
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        response = self.client.get(ids=[str(index) for index in indexes])

        documents = response.get("documents") or []
        metadatas = response.get("metadatas") or []
        results = []
        for position, id in enumerate(response.get("ids") or []):
            metadata = metadatas[position] if position < len(metadatas) else {}
            if not self._matches(metadata or {}, filter):
                continue
            results.append({
                "id": id,
                "metadata": metadata or {},
                "document": documents[position] if position < len(documents) else None,
            })

        logger.info(f"{self.name} fetched {len(results)} of {len(indexes)} requested documents.")
        return self._ordered_results(results, indexes, top_k)

    def delete(self, ids: list[str]) -> None:
        self.client.delete(ids=[str(id) for id in ids])

    def count(self) -> int:
        return int(self.client.count())


class ElasticsearchTextDB(BaseTextDB):
    """Elasticsearch connector, via the `elasticsearch` client.

    Authenticates with `api_key` when set, otherwise with `user`/`password`. The index named by
    `collection_name` is created when missing, with `document` mapped as text under `analyzer`.
    Fetches use `mget`, which takes ids but no filter, so `filter` is applied to the returned metadata
    here rather than by the engine.
    Extra params: `request_timeout`, `verify_certs`, `ca_certs`.
    """

    request_timeout: int | None = None  # seconds to wait on a request before giving up
    verify_certs: bool | None = None  # check the server's TLS certificate
    ca_certs: str | None = None  # path to a CA bundle used to verify that certificate
    arguments: list[str] = [
        'request_timeout',
        'verify_certs',
        'ca_certs',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        from elasticsearch import Elasticsearch

        connect_kwargs = self._connect_kwargs(**kwargs)
        if self.api_key:
            connect_kwargs["api_key"] = self.api_key
        elif self.user:
            connect_kwargs["basic_auth"] = (self.user, self.password)

        self.client = Elasticsearch(self._server_url(9200), **connect_kwargs)
        if not self.client.indices.exists(index=self.collection_name):
            self.client.indices.create(
                index=self.collection_name,
                mappings={
                    "properties": {
                        DOCUMENT_FIELD: {"type": "text", "analyzer": self.analyzer or "standard"},
                    },
                },
            )
            logger.info(f"{self.name} created index {self.collection_name}.")

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        operations: list[dict[str, Any]] = []
        for position, body in enumerate(self._documents_and_metadatas(ids, documents, metadatas)):
            operations.append({"index": {"_index": self.collection_name, "_id": ids[position]}})
            operations.append(body)

        # refresh so the documents are searchable — and fetchable — as soon as this call returns
        self.client.bulk(operations=operations, refresh=True)
        logger.info(f"{self.name} upserted {len(ids)} documents.")

    def query(
        self,
        indexes: list[str],
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        response = self.client.mget(
            index=self.collection_name,
            ids=[str(index) for index in indexes],
        )

        results = []
        for document in response["docs"]:
            if not document.get("found"):
                continue
            record = self._record(document["_id"], document.get("_source") or {})
            if self._matches(record["metadata"], filter):
                results.append(record)

        logger.info(f"{self.name} fetched {len(results)} of {len(indexes)} requested documents.")
        return self._ordered_results(results, indexes, top_k)

    def delete(self, ids: list[str]) -> None:
        operations = [{"delete": {"_index": self.collection_name, "_id": id}} for id in ids]
        self.client.bulk(operations=operations, refresh=True)

    def count(self) -> int:
        return int(self.client.count(index=self.collection_name)["count"])

    def close(self) -> None:
        if self.client:
            self.client.close()
            self.client = None


class OpenSearchTextDB(BaseTextDB):
    """OpenSearch connector, via `opensearch-py`.

    The same index/mget model as Elasticsearch, but the client is built from a node dict and its calls
    take a `body` rather than keyword payloads, so it is spelled out separately here.
    Extra params: `timeout`, `verify_certs`, `ca_certs`, `use_ssl`.
    """

    timeout: int | None = None  # seconds to wait on a request before giving up
    verify_certs: bool | None = None  # check the server's TLS certificate
    ca_certs: str | None = None  # path to a CA bundle used to verify that certificate
    use_ssl: bool | None = None  # connect over HTTPS
    arguments: list[str] = [
        'timeout',
        'verify_certs',
        'ca_certs',
        'use_ssl',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        from opensearchpy import OpenSearch

        connect_kwargs = self._connect_kwargs(**kwargs)
        if self.user:
            connect_kwargs["http_auth"] = (self.user, self.password)

        self.client = OpenSearch(
            hosts=[{"host": self.host or "localhost", "port": self.port or 9200}]
            if not self.url
            else [self.url],
            **connect_kwargs,
        )

        if not self.client.indices.exists(index=self.collection_name):
            self.client.indices.create(
                index=self.collection_name,
                body={
                    "mappings": {
                        "properties": {
                            DOCUMENT_FIELD: {"type": "text", "analyzer": self.analyzer or "standard"},
                        },
                    },
                },
            )
            logger.info(f"{self.name} created index {self.collection_name}.")

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        operations: list[dict[str, Any]] = []
        for position, body in enumerate(self._documents_and_metadatas(ids, documents, metadatas)):
            operations.append({"index": {"_index": self.collection_name, "_id": ids[position]}})
            operations.append(body)

        self.client.bulk(body=operations, refresh=True)
        logger.info(f"{self.name} upserted {len(ids)} documents.")

    def query(
        self,
        indexes: list[str],
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        response = self.client.mget(
            index=self.collection_name,
            body={"ids": [str(index) for index in indexes]},
        )

        results = []
        for document in response["docs"]:
            if not document.get("found"):
                continue
            record = self._record(document["_id"], document.get("_source") or {})
            if self._matches(record["metadata"], filter):
                results.append(record)

        logger.info(f"{self.name} fetched {len(results)} of {len(indexes)} requested documents.")
        return self._ordered_results(results, indexes, top_k)

    def delete(self, ids: list[str]) -> None:
        operations = [{"delete": {"_index": self.collection_name, "_id": id}} for id in ids]
        self.client.bulk(body=operations, refresh=True)

    def count(self) -> int:
        return int(self.client.count(index=self.collection_name)["count"])

    def close(self) -> None:
        if self.client:
            self.client.close()
            self.client = None


class MeilisearchTextDB(BaseTextDB):
    """Meilisearch connector, via the `meilisearch` client.

    The index named by `collection_name` is created when missing, keyed on `id`. Writes are queued as
    tasks server-side, so `upsert` waits for its task to finish — otherwise a fetch straight after a
    write can miss documents that are still being indexed. Documents are fetched one id at a time,
    since Meilisearch filters the documents endpoint only on attributes declared filterable.
    Extra params: `request_timeout`.
    """

    request_timeout: int | None = None  # seconds to wait on a request before giving up
    arguments: list[str] = [
        'request_timeout',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        import meilisearch
        from meilisearch.errors import MeilisearchApiError

        connect_kwargs = self._connect_kwargs(**kwargs)
        connection = meilisearch.Client(self._server_url(7700), self.api_key, **connect_kwargs)

        try:
            connection.get_index(self.collection_name)
        except MeilisearchApiError:
            connection.wait_for_task(
                connection.create_index(self.collection_name, {"primaryKey": "id"}).task_uid
            )
            logger.info(f"{self.name} created index {self.collection_name}.")

        self._connection = connection  # kept so `count` can read the index's stats
        self.client = connection.index(self.collection_name)

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        records = []
        for position, body in enumerate(self._documents_and_metadatas(ids, documents, metadatas)):
            records.append({"id": ids[position], **body})

        self._connection.wait_for_task(self.client.add_documents(records).task_uid)
        logger.info(f"{self.name} upserted {len(ids)} documents.")

    def query(
        self,
        indexes: list[str],
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from meilisearch.errors import MeilisearchApiError

        results = []
        for index in indexes:
            try:
                document = self.client.get_document(str(index))
            except MeilisearchApiError:  # not stored under that id
                continue

            # the client hands back a Document object whose attributes are the stored fields
            record = self._record(str(index), vars(document))
            if self._matches(record["metadata"], filter):
                results.append(record)

        logger.info(f"{self.name} fetched {len(results)} of {len(indexes)} requested documents.")
        return self._ordered_results(results, indexes, top_k)

    def delete(self, ids: list[str]) -> None:
        self._connection.wait_for_task(
            self.client.delete_documents([str(id) for id in ids]).task_uid
        )

    def count(self) -> int:
        return int(self.client.get_stats().number_of_documents)

    def close(self) -> None:
        self._connection = None
        self.client = None


class TypesenseTextDB(BaseTextDB):
    """Typesense connector, via the `typesense` client.

    `api_key` is required. The collection named by `collection_name` is created when missing, with
    `document` as a string field and everything else auto-typed, so metadata keys need no schema up
    front. Documents are fetched one id at a time, which is Typesense's own retrieve-by-id call.
    Extra params: `protocol`, `connection_timeout_seconds`.
    """

    protocol: str = "http"  # "http" or "https"; only used when `url` is not set
    connection_timeout_seconds: int | None = None  # seconds to wait on a request before giving up
    arguments: list[str] = [
        'connection_timeout_seconds',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        import typesense
        from typesense.exceptions import ObjectNotFound

        connect_kwargs = self._connect_kwargs(**kwargs)
        connection = typesense.Client({
            "nodes": [{
                "host": self.host or "localhost",
                "port": self.port or 8108,
                "protocol": self.protocol,
            }],
            "api_key": self.api_key,
            **connect_kwargs,
        })

        try:
            connection.collections[self.collection_name].retrieve()
        except ObjectNotFound:
            connection.collections.create({
                "name": self.collection_name,
                "fields": [
                    {"name": DOCUMENT_FIELD, "type": "string"},
                    {"name": ".*", "type": "auto"},  # metadata keys are typed as they arrive
                ],
            })
            logger.info(f"{self.name} created collection {self.collection_name}.")

        self._connection = connection  # kept so `count` can read the collection's stats
        self.client = connection.collections[self.collection_name]

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        records = []
        for position, body in enumerate(self._documents_and_metadatas(ids, documents, metadatas)):
            records.append({"id": str(ids[position]), **body})

        self.client.documents.import_(records, {"action": "upsert"})
        logger.info(f"{self.name} upserted {len(ids)} documents.")

    def query(
        self,
        indexes: list[str],
        top_k: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from typesense.exceptions import ObjectNotFound

        results = []
        for index in indexes:
            try:
                document = self.client.documents[str(index)].retrieve()
            except ObjectNotFound:  # not stored under that id
                continue

            record = self._record(str(index), dict(document))
            if self._matches(record["metadata"], filter):
                results.append(record)

        logger.info(f"{self.name} fetched {len(results)} of {len(indexes)} requested documents.")
        return self._ordered_results(results, indexes, top_k)

    def delete(self, ids: list[str]) -> None:
        for id in ids:
            self.client.documents[str(id)].delete()

    def count(self) -> int:
        return int(self.client.retrieve()["num_documents"])

    def close(self) -> None:
        self._connection = None
        self.client = None

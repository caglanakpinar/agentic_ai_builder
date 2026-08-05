from abc import abstractmethod
from pathlib import Path
from typing import Any

from utils.configs import VectorDBConfigs, resolve_secret
from utils.logger import logger


class BaseVectorDB(VectorDBConfigs):
    """Common interface for vector database connectors (FAISS, Chroma, Qdrant, Pinecone, ...).

    Subclasses implement `_initialize_connection(**kwargs)`: build the driver's client and assign it to
    `self.client`. Any kwarg matching an existing class attribute (e.g. `nlist`, `namespace`) is applied
    via `setattr` first, so driver-specific options can be set at construction time and listed in the
    class-level `arguments` list, which `_connect_kwargs` forwards to the driver.

    Every connector exposes the same four operations, so an agent can swap engines without changing
    call sites:
      - `upsert(ids, vectors, metadatas, documents)`: insert or replace vectors by id.
      - `query(vector, top_k, filter)`: nearest-neighbour search, returning
        `[{"id": ..., "score": ..., "metadata": {...}, "document": ...}]` ordered best-first.
        Results are always best-first, but `score` is whatever the engine natively reports and its
        direction differs between them: FAISS returns a similarity (higher is closer), while Chroma,
        Weaviate, Milvus and LanceDB return a distance (lower is closer). Rank on list order rather
        than comparing raw scores across engines.
      - `delete(ids)`: remove vectors by id.
      - `count()`: number of stored vectors.

    Driver packages are imported lazily inside `_initialize_connection` rather than at module import,
    so using one engine does not require installing every other engine's driver.

    Args:
        name: Db name as referenced by agents, e.g. "ds_knowledge_db".
        db: Engine to connect with, e.g. "chroma".
        host/port/url: Server location for client-server engines.
        api_key: Literal key, or the name of an env var holding it.
        path: Storage directory/file for embedded engines (FAISS, Chroma, Qdrant, LanceDB).
        collection_name: Collection/index/class the connector reads and writes.
        dimension: Embedding width; required by engines that build the index up front.
        metric: Similarity metric — "cosine", "l2", or "ip".
        **kwargs: Driver-specific overrides applied to matching class attributes on the subclass.
    """

    client: Any = None
    arguments: list[str] = []  # names of the driver-specific params a subclass forwards to its driver

    def __init__(
        self,
        name: str,
        db: str,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        path: str | None = None,
        url: str | None = None,
        collection_name: str = "default",
        dimension: int | None = None,
        metric: str = "cosine",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            db=db,
            host=host,
            port=port,
            api_key=api_key,
            path=path,
            url=url,
            collection_name=collection_name,
            dimension=dimension,
            metric=metric,
        )
        self.secret_checker()
        self._initialize_connection(**kwargs)

    def secret_checker(self) -> None:
        """Resolve `api_key` as either an env var name or a literal value; embedded engines need none."""
        self.api_key = resolve_secret(self.api_key, self.name, required=False)

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
        """Build the driver client and assign it to `self.client`."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """Insert or replace vectors by id."""

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the `top_k` nearest neighbours, best match first."""

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove vectors by id."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored vectors."""

    def close(self) -> None:
        """Release the client. Overridden by engines that hold an open connection."""
        self.client = None


class FAISSDB(BaseVectorDB):
    """In-process FAISS index, via `faiss-cpu`.

    FAISS is a similarity-search library rather than a database: it stores vectors against integer ids
    and holds no metadata. This connector keeps the string-id and metadata bookkeeping alongside the
    index so it satisfies the same interface as the server-backed engines. `dimension` is required.

    `path` points at an index file: it is loaded when it already exists, and `save()` writes back to it.
    Because the index itself cannot hold ids or metadata, `save()` also writes a `<path>.meta.json`
    sidecar with that bookkeeping, and `_initialize_connection` reads it back.
    Extra params: `nlist`, `nprobe`.
    """

    nlist: int | None = None  # IVF partition count; set it to build an IVF index instead of a flat one
    nprobe: int | None = None  # IVF partitions searched per query — higher is more accurate but slower
    arguments: list[str] = [
        'nlist',
        'nprobe',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        import faiss

        self._connect_kwargs(**kwargs)  # applies construction-time overrides onto self
        if not self.dimension:
            raise ValueError(f"{self.name}: FAISS needs `dimension` to build its index.")

        self._id_map: dict[int, str] = {}  # faiss integer id -> caller's string id
        self._metadata: dict[str, dict[str, Any]] = {}
        self._documents: dict[str, str] = {}
        self._next_id: int = 0

        if self.path and Path(self.path).exists():
            self.client = faiss.read_index(str(self.path))
            self._load_metadata()
            logger.info(f"{self.name} loaded FAISS index from {self.path}.")
            return

        # inner product over L2-normalised vectors is equivalent to cosine similarity
        if self.metric in ('cosine', 'ip'):
            index = faiss.IndexFlatIP(self.dimension)
        else:
            index = faiss.IndexFlatL2(self.dimension)

        if self.nlist:
            index = faiss.IndexIVFFlat(index, self.dimension, self.nlist)
        self.client = faiss.IndexIDMap(index)

    def _metadata_path(self) -> Path:
        """Path of the JSON sidecar holding the ids and metadata the index cannot store itself."""
        return Path(f"{self.path}.meta.json")

    def _load_metadata(self) -> None:
        """Restore the id/metadata bookkeeping written alongside a saved index."""
        import json

        sidecar = self._metadata_path()
        if not sidecar.exists():
            logger.warning(f"{self.name} found no metadata sidecar at {sidecar}; ids will be missing.")
            return

        saved = json.loads(sidecar.read_text())
        self._id_map = {int(key): value for key, value in saved["id_map"].items()}
        self._metadata = saved["metadata"]
        self._documents = saved["documents"]
        self._next_id = saved["next_id"]

    def _as_array(self, vectors: list[list[float]]) -> Any:
        """Convert vectors to the float32 array FAISS expects, normalising for cosine."""
        import faiss
        import numpy as np

        array = np.asarray(vectors, dtype="float32")
        if self.metric == 'cosine':
            faiss.normalize_L2(array)
        return array

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        import numpy as np

        self.delete([id for id in ids if id in self._metadata])  # replace rather than duplicate

        integer_ids = []
        for position, id in enumerate(ids):
            self._id_map[self._next_id] = id
            integer_ids.append(self._next_id)
            self._next_id += 1
            self._metadata[id] = (metadatas or [{}] * len(ids))[position]
            if documents:
                self._documents[id] = documents[position]

        array = self._as_array(vectors)
        if not self.client.index.is_trained:  # an IVF index must be trained before it accepts vectors
            self.client.index.train(array)

        self.client.add_with_ids(array, np.asarray(integer_ids, dtype="int64"))
        logger.info(f"{self.name} upserted {len(ids)} vectors.")

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.nprobe:
            self.client.nprobe = self.nprobe

        scores, indices = self.client.search(self._as_array([vector]), top_k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:  # FAISS pads with -1 when fewer than top_k vectors exist
                continue
            id = self._id_map.get(int(index))
            metadata = self._metadata.get(id, {})
            if filter and not all(metadata.get(key) == value for key, value in filter.items()):
                continue
            results.append({
                "id": id,
                "score": float(score),
                "metadata": metadata,
                "document": self._documents.get(id),
            })

        logger.info(f"{self.name} returned {len(results)} matches.")
        return results

    def delete(self, ids: list[str]) -> None:
        import numpy as np

        integer_ids = [key for key, value in self._id_map.items() if value in ids]
        if not integer_ids:
            return

        self.client.remove_ids(np.asarray(integer_ids, dtype="int64"))
        for key in integer_ids:
            self._id_map.pop(key, None)
        for id in ids:
            self._metadata.pop(id, None)
            self._documents.pop(id, None)

    def count(self) -> int:
        return int(self.client.ntotal)

    def save(self) -> None:
        """Write the index back to `path`, plus the id/metadata sidecar beside it."""
        import faiss
        import json

        if not self.path:
            raise ValueError(f"{self.name}: no `path` configured to save the FAISS index to.")

        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.client, str(self.path))
        self._metadata_path().write_text(json.dumps({
            "id_map": self._id_map,
            "metadata": self._metadata,
            "documents": self._documents,
            "next_id": self._next_id,
        }))
        logger.info(f"{self.name} saved FAISS index to {self.path}.")


class ChromaDB(BaseVectorDB):
    """Chroma connector, via `chromadb`.

    Runs embedded against `path`, or against a Chroma server when `host` is set.
    Extra params: `hnsw_space`, `tenant`, `database`.
    """

    hnsw_space: str | None = None  # index distance function: "cosine", "l2", or "ip"
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
            metadata={"hnsw:space": self.hnsw_space or self.metric},
        )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        self.client.upsert(
            ids=ids,
            embeddings=vectors,
            metadatas=metadatas,
            documents=documents,
        )
        logger.info(f"{self.name} upserted {len(ids)} vectors.")

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        response = self.client.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filter,
        )

        results = [
            {
                "id": id,
                "score": response["distances"][0][position],
                "metadata": (response.get("metadatas") or [[]])[0][position],
                "document": (response.get("documents") or [[]])[0][position],
            }
            for position, id in enumerate(response["ids"][0])
        ]
        logger.info(f"{self.name} returned {len(results)} matches.")
        return results

    def delete(self, ids: list[str]) -> None:
        self.client.delete(ids=ids)

    def count(self) -> int:
        return self.client.count()


class QdrantDB(BaseVectorDB):
    """Qdrant connector, via `qdrant-client`.

    Runs embedded against `path`, or against a server when `url`/`host` is set. Note that embedded mode
    file-locks its directory, so only one process can open it at a time.
    Extra params: `https`, `prefer_grpc`, `timeout`.
    """

    https: bool | None = None  # use TLS when connecting to the server
    prefer_grpc: bool | None = None  # use the gRPC interface, which is faster than REST
    timeout: int | None = None  # seconds to wait on a request before giving up
    arguments: list[str] = [
        'https',
        'prefer_grpc',
        'timeout',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        connect_kwargs = self._connect_kwargs(**kwargs)
        if self.url or self.host:
            self.client = QdrantClient(
                url=self.url,
                host=None if self.url else self.host,
                port=self.port or 6333,
                api_key=self.api_key,
                **connect_kwargs,
            )
        else:
            self.client = QdrantClient(path=self.path or "./qdrant", **connect_kwargs)

        distances = {"cosine": Distance.COSINE, "l2": Distance.EUCLID, "ip": Distance.DOT}
        if not self.client.collection_exists(self.collection_name):
            if not self.dimension:
                raise ValueError(f"{self.name}: Qdrant needs `dimension` to create a collection.")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=distances.get(self.metric, Distance.COSINE),
                ),
            )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        from qdrant_client.models import PointStruct

        points = []
        for position, id in enumerate(ids):
            payload = dict((metadatas or [{}] * len(ids))[position])
            if documents:
                payload["document"] = documents[position]
            points.append(PointStruct(id=id, vector=vectors[position], payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"{self.name} upserted {len(ids)} vectors.")

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if filter:
            query_filter = Filter(must=[
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter.items()
            ])

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
        )

        results = [
            {
                "id": point.id,
                "score": point.score,
                "metadata": point.payload or {},
                "document": (point.payload or {}).get("document"),
            }
            for point in response.points
        ]
        logger.info(f"{self.name} returned {len(results)} matches.")
        return results

    def delete(self, ids: list[str]) -> None:
        self.client.delete(collection_name=self.collection_name, points_selector=ids)

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count

    def close(self) -> None:
        if self.client:
            self.client.close()
            self.client = None


class PineconeDB(BaseVectorDB):
    """Pinecone connector, via `pinecone`.

    Fully managed, so there is no embedded mode — `api_key` is required. The index named by
    `collection_name` is created when missing, which needs `dimension`.
    Extra params: `namespace`, `cloud`, `region`.
    """

    namespace: str | None = None  # partition within the index
    cloud: str | None = None  # serverless cloud provider, e.g. "aws"
    region: str | None = None  # serverless region, e.g. "us-east-1"
    arguments: list[str] = [
        'cloud',
        'region',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        from pinecone import Pinecone, ServerlessSpec

        self._connect_kwargs(**kwargs)  # applies construction-time overrides onto self
        pinecone = Pinecone(api_key=self.api_key)

        if self.collection_name not in [index.name for index in pinecone.list_indexes()]:
            if not self.dimension:
                raise ValueError(f"{self.name}: Pinecone needs `dimension` to create an index.")
            pinecone.create_index(
                name=self.collection_name,
                dimension=self.dimension,
                metric="euclidean" if self.metric == 'l2' else self.metric,
                spec=ServerlessSpec(cloud=self.cloud or "aws", region=self.region or "us-east-1"),
            )

        self.client = pinecone.Index(self.collection_name)

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        records = []
        for position, id in enumerate(ids):
            metadata = dict((metadatas or [{}] * len(ids))[position])
            if documents:
                metadata["document"] = documents[position]
            records.append({"id": id, "values": vectors[position], "metadata": metadata})

        self.client.upsert(vectors=records, namespace=self.namespace)
        logger.info(f"{self.name} upserted {len(ids)} vectors.")

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        response = self.client.query(
            vector=vector,
            top_k=top_k,
            filter=filter,
            namespace=self.namespace,
            include_metadata=True,
        )

        results = [
            {
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata") or {},
                "document": (match.get("metadata") or {}).get("document"),
            }
            for match in response["matches"]
        ]
        logger.info(f"{self.name} returned {len(results)} matches.")
        return results

    def delete(self, ids: list[str]) -> None:
        self.client.delete(ids=ids, namespace=self.namespace)

    def count(self) -> int:
        return self.client.describe_index_stats()["total_vector_count"]


class WeaviateDB(BaseVectorDB):
    """Weaviate connector, via `weaviate-client`.

    Connects to Weaviate Cloud when `url` is set, otherwise to a local instance on `host`.
    `collection_name` is capitalised because Weaviate requires collection names to start upper-case.
    Extra params: `grpc_port`, `secure`.
    """

    grpc_port: int | None = None  # gRPC port, separate from the HTTP port
    secure: bool | None = None  # use TLS for a local connection
    arguments: list[str] = [
        'grpc_port',
        'secure',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        import weaviate
        from weaviate.classes.config import Configure, VectorDistances

        self._connect_kwargs(**kwargs)  # applies construction-time overrides onto self
        if self.url:
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.url,
                auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
            )
        else:
            client = weaviate.connect_to_local(
                host=self.host or "localhost",
                port=self.port or 8080,
                grpc_port=self.grpc_port or 50051,
            )

        self.collection_name = self.collection_name.capitalize()
        distances = {
            "cosine": VectorDistances.COSINE,
            "l2": VectorDistances.L2_SQUARED,
            "ip": VectorDistances.DOT,
        }
        if not client.collections.exists(self.collection_name):
            client.collections.create(
                name=self.collection_name,
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=distances.get(self.metric, VectorDistances.COSINE),
                ),
            )

        self._connection = client  # kept so `close` can shut the underlying client down
        self.client = client.collections.get(self.collection_name)

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        with self.client.batch.dynamic() as batch:
            for position, id in enumerate(ids):
                properties = dict((metadatas or [{}] * len(ids))[position])
                if documents:
                    properties["document"] = documents[position]
                batch.add_object(properties=properties, uuid=id, vector=vectors[position])

        logger.info(f"{self.name} upserted {len(ids)} vectors.")

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from weaviate.classes.query import Filter

        filters = None
        if filter:
            filters = Filter.all_of([
                Filter.by_property(key).equal(value) for key, value in filter.items()
            ])

        response = self.client.query.near_vector(
            near_vector=vector,
            limit=top_k,
            filters=filters,
            return_metadata=["distance"],
        )

        results = [
            {
                "id": str(item.uuid),
                "score": item.metadata.distance,
                "metadata": item.properties or {},
                "document": (item.properties or {}).get("document"),
            }
            for item in response.objects
        ]
        logger.info(f"{self.name} returned {len(results)} matches.")
        return results

    def delete(self, ids: list[str]) -> None:
        for id in ids:
            self.client.data.delete_by_id(id)

    def count(self) -> int:
        return self.client.aggregate.over_all(total_count=True).total_count

    def close(self) -> None:
        if getattr(self, '_connection', None):
            self._connection.close()
            self._connection = None
        self.client = None


class MilvusDB(BaseVectorDB):
    """Milvus connector, via `pymilvus`.

    Connects to a Milvus server on `url`/`host`, or to embedded Milvus Lite when only `path` is set.
    `dimension` is required to create the collection.
    Extra params: `index_type`, `nlist`, `nprobe`.
    """

    index_type: str | None = None  # e.g. "IVF_FLAT", "HNSW"; defaults to a flat index
    nlist: int | None = None  # IVF partition count used when building the index
    nprobe: int | None = None  # IVF partitions searched per query
    arguments: list[str] = [
        'index_type',
        'nlist',
        'nprobe',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        from pymilvus import MilvusClient

        self._connect_kwargs(**kwargs)  # applies construction-time overrides onto self
        uri = self.url or (f"http://{self.host}:{self.port or 19530}" if self.host else self.path)
        self.client = MilvusClient(uri=uri or "./milvus.db", token=self.api_key)

        metrics = {"cosine": "COSINE", "l2": "L2", "ip": "IP"}
        if not self.client.has_collection(self.collection_name):
            if not self.dimension:
                raise ValueError(f"{self.name}: Milvus needs `dimension` to create a collection.")
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                metric_type=metrics.get(self.metric, "COSINE"),
            )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        rows = []
        for position, id in enumerate(ids):
            row = {"id": id, "vector": vectors[position]}
            row.update((metadatas or [{}] * len(ids))[position])
            if documents:
                row["document"] = documents[position]
            rows.append(row)

        self.client.upsert(collection_name=self.collection_name, data=rows)
        logger.info(f"{self.name} upserted {len(ids)} vectors.")

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        expression = " and ".join(f'{key} == "{value}"' for key, value in (filter or {}).items())
        response = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=top_k,
            filter=expression or "",
            output_fields=["*"],
            search_params={"params": {"nprobe": self.nprobe}} if self.nprobe else None,
        )

        results = [
            {
                "id": match["id"],
                "score": match["distance"],
                "metadata": match.get("entity") or {},
                "document": (match.get("entity") or {}).get("document"),
            }
            for match in response[0]
        ]
        logger.info(f"{self.name} returned {len(results)} matches.")
        return results

    def delete(self, ids: list[str]) -> None:
        self.client.delete(collection_name=self.collection_name, ids=ids)

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats["row_count"])

    def close(self) -> None:
        if self.client:
            self.client.close()
            self.client = None


class LanceDB(BaseVectorDB):
    """LanceDB connector, via `lancedb`.

    Embedded and file-backed like FAISS, but stores metadata alongside vectors, so it needs no
    external bookkeeping. `path` is the database directory.
    Extra params: `region`, `nprobes`, `refine_factor`.
    """

    region: str | None = None  # LanceDB Cloud region; leave unset for local storage
    nprobes: int | None = None  # partitions searched per query
    refine_factor: int | None = None  # re-rank this multiple of top_k for better recall
    arguments: list[str] = [
        'region',
    ]

    def _initialize_connection(self, **kwargs: Any) -> None:
        import lancedb

        connect_kwargs = self._connect_kwargs(**kwargs)
        self._connection = lancedb.connect(
            self.url or self.path or "./lancedb",
            api_key=self.api_key,
            **connect_kwargs,
        )

        self.client = None
        if self.collection_name in self._connection.table_names():
            self.client = self._connection.open_table(self.collection_name)

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        rows = []
        for position, id in enumerate(ids):
            row = {"id": id, "vector": vectors[position]}
            row.update((metadatas or [{}] * len(ids))[position])
            if documents:
                row["document"] = documents[position]
            rows.append(row)

        if self.client is None:  # the table's schema is inferred from the first batch written
            self.client = self._connection.create_table(self.collection_name, data=rows)
        else:
            self.delete(ids)  # replace rather than duplicate
            self.client.add(rows)

        logger.info(f"{self.name} upserted {len(ids)} vectors.")

    def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        search = self.client.search(vector).limit(top_k)
        if filter:
            search = search.where(" AND ".join(
                f"{key} = '{value}'" for key, value in filter.items()
            ))
        if self.nprobes:
            search = search.nprobes(self.nprobes)
        if self.refine_factor:
            search = search.refine_factor(self.refine_factor)

        results = [
            {
                "id": row["id"],
                "score": row["_distance"],
                "metadata": {
                    key: value for key, value in row.items()
                    if key not in ('id', 'vector', '_distance')
                },
                "document": row.get("document"),
            }
            for row in search.to_list()
        ]
        logger.info(f"{self.name} returned {len(results)} matches.")
        return results

    def delete(self, ids: list[str]) -> None:
        quoted = ", ".join(f"'{id}'" for id in ids)
        self.client.delete(f"id IN ({quoted})")

    def count(self) -> int:
        return self.client.count_rows()

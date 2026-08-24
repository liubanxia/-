from __future__ import annotations

import gc
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .config import WorkbenchPaths, resolve_model_dir
from .db import KnowledgeDB


_QUERY_SPLIT_RE = re.compile(
    r"[\s，。；、,;:：/\\()（）\[\]【】]+|"
    r"(?:请|帮我|把|将|整理|汇总|总结|列出|查找|检索|关于|相关|"
    r"全部|所有|目前|当前|内容|资料|里面|其中|以及|并且|还有|和|与|的|中)+"
)


@dataclass(frozen=True)
class Evidence:
    chunk_id: int
    source_key: str
    title: str
    path: str
    page: int
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"[S{self.chunk_id}]"

    @property
    def source_label(self) -> str:
        return f"{self.title} · 第{self.page}页"


def query_variants(query: str, limit: int = 8) -> list[str]:
    """Create conservative lexical fallbacks for Chinese natural-language queries."""

    query = (query or "").strip()
    if not query:
        return []

    variants = [query]
    pieces = [
        piece.strip(" -_./")
        for piece in _QUERY_SPLIT_RE.split(query)
        if piece.strip(" -_./")
    ]
    pieces.sort(key=len, reverse=True)

    for piece in pieces:
        if len(piece) < 2:
            continue
        if piece not in variants:
            variants.append(piece)
        if len(variants) >= limit:
            break

    return variants


class EmbeddingEngine:
    model_name = "Qwen3-Embedding-0.6B"

    def __init__(self, db: KnowledgeDB, paths: WorkbenchPaths):
        self.db = db
        self.paths = paths
        self.model_path = resolve_model_dir(
            paths.model_root,
            self.model_name,
        )
        self._model = None
        self._device = None
        self._index_ids = None
        self._index_matrix = None
        self._query_cache: dict[str, object] = {}

    def available(self) -> bool:
        return self.model_path.exists() and any(self.model_path.iterdir())

    @staticmethod
    def _select_device() -> str:
        """Use modern CUDA GPUs but keep legacy hospital GPUs on CPU."""
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
            major, _minor = torch.cuda.get_device_capability(0)
            if int(major) < 5:
                return "cpu"
            return "cuda:0"
        except Exception:
            return "cpu"

    @property
    def device(self) -> str:
        return self._device or self._select_device()

    def _load(self):
        if self._model is not None:
            return self._model
        if not self.available():
            raise RuntimeError(f"Embedding模型未下载: {self.model_path}")

        from sentence_transformers import SentenceTransformer

        self._device = self._select_device()
        try:
            self._model = SentenceTransformer(
                str(self.model_path),
                device=self._device,
            )
        except Exception:
            if self._device == "cpu":
                raise
            self._device = "cpu"
            self._model = SentenceTransformer(
                str(self.model_path),
                device="cpu",
            )
        return self._model

    def unload_model(self) -> None:
        """Release inference weights while retaining the small RAM vector index."""
        self._model = None
        self._device = None
        self._query_cache.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _invalidate_vector_index(self) -> None:
        self._index_ids = None
        self._index_matrix = None
        self._query_cache.clear()

    def _load_vector_index(self):
        """Load all saved vectors once into one contiguous NumPy matrix.

        Previously every question iterated 5k+ SQLite rows and rebuilt each
        vector with np.frombuffer. Keeping a ~20 MB matrix for a 5k x 1k index
        turns repeated similarity search into one BLAS matrix-vector product.
        """
        import numpy as np

        if self._index_ids is not None and self._index_matrix is not None:
            return self._index_ids, self._index_matrix

        ids: list[int] = []
        vectors: list[object] = []
        expected_dim: int | None = None

        for row in self.db.iter_embeddings(self.model_name):
            dim = int(row["dim"])
            vector = np.frombuffer(row["vector"], dtype=np.float32, count=dim)
            if expected_dim is None:
                expected_dim = int(vector.size)
            if vector.size != expected_dim:
                continue
            ids.append(int(row["chunk_id"]))
            vectors.append(vector)

        if not vectors:
            self._index_ids = np.empty((0,), dtype=np.int64)
            self._index_matrix = np.empty((0, 0), dtype=np.float32)
            return self._index_ids, self._index_matrix

        self._index_ids = np.asarray(ids, dtype=np.int64)
        self._index_matrix = np.ascontiguousarray(
            np.vstack(vectors),
            dtype=np.float32,
        )
        return self._index_ids, self._index_matrix

    def _adaptive_batch_size(self) -> int:
        if self.device != "cuda:0":
            return 8
        try:
            import torch

            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_gb >= 12:
                return 48
            if total_gb >= 7:
                return 32
            return 16
        except Exception:
            return 16

    def build_missing(
        self,
        batch_size: int | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> int:
        """Build only missing vectors and checkpoint every mini-batch."""
        import numpy as np

        model = self._load()
        batch_size = self._adaptive_batch_size() if batch_size is None else max(1, int(batch_size))
        done = 0

        if progress:
            progress(
                0,
                0,
                f"Embedding模型已加载 | 设备={self.device} | batch={batch_size}",
            )

        while True:
            rows = self.db.missing_embedding_chunks(
                self.model_name,
                limit=max(batch_size * 4, 64),
            )
            if not rows:
                break

            for offset in range(0, len(rows), batch_size):
                batch_rows = rows[offset : offset + batch_size]
                texts = [str(row["text"]) for row in batch_rows]
                vectors = model.encode(
                    texts,
                    batch_size=len(batch_rows),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                items = []
                for row, vector in zip(batch_rows, vectors):
                    array = np.asarray(vector, dtype=np.float32)
                    items.append(
                        (
                            int(row["id"]),
                            int(array.size),
                            array.tobytes(),
                        )
                    )

                self.db.store_embeddings(self.model_name, items)
                self._invalidate_vector_index()
                done += len(items)
                if progress:
                    progress(
                        done,
                        0,
                        (
                            f"已生成 {done} 个新向量"
                            f" | 设备={self.device}"
                            f" | batch={len(items)}"
                        ),
                    )

        return done

    def _encode_query(self, query: str):
        import numpy as np

        cached = self._query_cache.get(query)
        if cached is not None:
            return cached

        model = self._load()
        vector = np.asarray(
            model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0],
            dtype=np.float32,
        )
        if len(self._query_cache) >= 32:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[query] = vector
        return vector

    def search(self, query: str, limit: int = 20) -> list[tuple[int, float]]:
        import numpy as np

        query_vector = self._encode_query(query)
        ids, matrix = self._load_vector_index()
        if matrix.size == 0 or matrix.shape[1] != query_vector.size:
            return []

        scores = matrix @ query_vector
        finite = np.isfinite(scores)
        if not finite.any():
            return []
        scores = np.where(finite, scores, -np.inf)

        k = min(max(1, int(limit)), int(scores.size))
        if k == scores.size:
            order = np.argsort(scores)[::-1]
        else:
            candidate = np.argpartition(scores, -k)[-k:]
            order = candidate[np.argsort(scores[candidate])[::-1]]

        return [
            (int(ids[index]), float(scores[index]))
            for index in order[:k]
            if math.isfinite(float(scores[index]))
        ]


class Retriever:
    def __init__(self, db: KnowledgeDB, paths: WorkbenchPaths):
        self.db = db
        self.paths = paths
        self.embeddings = EmbeddingEngine(db, paths)

    @staticmethod
    def _row_to_evidence(row, score: float) -> Evidence:
        return Evidence(
            chunk_id=int(row["id"]),
            source_key=str(row["source_key"]),
            title=str(row["title"]),
            path=str(row["path"]),
            page=int(row["page"]),
            text=str(row["text"]),
            score=float(score),
        )

    def search(
        self,
        query: str,
        limit: int = 20,
        use_embeddings: bool = True,
    ) -> list[Evidence]:
        merged: dict[int, tuple[object, float]] = {}

        variants = query_variants(query)
        lexical_limit = max(limit * 3, 30)
        for variant_index, variant in enumerate(variants):
            rows = self.db.search_lexical(
                variant,
                limit=lexical_limit,
            )
            variant_weight = 1.0 if variant_index == 0 else 0.72
            for rank, row in enumerate(rows):
                chunk_id = int(row["id"])
                increment = variant_weight / (1.0 + rank)
                if chunk_id in merged:
                    old_row, old_score = merged[chunk_id]
                    merged[chunk_id] = (
                        old_row,
                        old_score + increment,
                    )
                else:
                    merged[chunk_id] = (row, increment)

        if use_embeddings and self.embeddings.available():
            try:
                vector_hits = self.embeddings.search(
                    query, limit=max(limit * 3, 30)
                )
                vector_rows = self.db.fetch_chunks(
                    [chunk_id for chunk_id, _ in vector_hits]
                )
                score_map = {chunk_id: score for chunk_id, score in vector_hits}
                for row in vector_rows:
                    chunk_id = int(row["id"])
                    vector_score = max(0.0, score_map.get(chunk_id, 0.0))
                    if chunk_id in merged:
                        old_row, lexical_score = merged[chunk_id]
                        merged[chunk_id] = (
                            old_row,
                            lexical_score + vector_score,
                        )
                    else:
                        merged[chunk_id] = (row, vector_score)
            except Exception:
                # Semantic retrieval is an enhancement. Lexical evidence remains
                # available even if a model/runtime is temporarily unavailable.
                pass

        ordered = sorted(
            merged.values(), key=lambda item: item[1], reverse=True
        )[:limit]
        return [
            self._row_to_evidence(row, score)
            for row, score in ordered
        ]

    def search_diverse(
        self,
        query: str,
        limit: int = 200,
        use_embeddings: bool = True,
    ) -> list[Evidence]:
        """Retrieve a broad evidence pool without one book monopolizing it."""

        limit = max(1, int(limit))
        pool = self.search(
            query,
            limit=max(limit * 3, limit),
            use_embeddings=use_embeddings,
        )
        if len(pool) <= limit:
            return pool

        document_count = max(
            1,
            len({item.path for item in pool}),
        )
        soft_cap = max(
            16,
            int(math.ceil(limit / min(document_count, 10))),
        )
        counts: Counter[str] = Counter()
        selected: list[Evidence] = []
        deferred: list[Evidence] = []

        for item in pool:
            if counts[item.path] < soft_cap:
                selected.append(item)
                counts[item.path] += 1
            else:
                deferred.append(item)
            if len(selected) >= limit:
                return selected

        for item in deferred:
            selected.append(item)
            if len(selected) >= limit:
                break

        return selected

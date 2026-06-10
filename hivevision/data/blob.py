"""Azure Blob backend: the canonical store when you label from multiple machines.

The source of truth is a single blob container (default ``hive`` in the ``gamevision``
account). ``LabelStore`` keeps a read-through copy of every image under its local root
so repeated reads (and training) stay on local disk, but the blob is authoritative —
``labels.jsonl`` is always re-read from the blob before a save, so two machines can't
silently clobber each other.

Auth is the account connection string in ``.env`` (``STORAGE_CONNECTION_STRING``); the
entry points (``hivevision.capture``, the scripts) call ``load_dotenv()`` so the variable
is present in ``os.environ`` by the time :meth:`AzureBlobBackend.from_env` runs. The
``azure.storage.blob`` import is **lazy** (inside the methods) so the geometry/test paths
never need the SDK installed or a network round-trip.

Blob names are flat POSIX keys, no container-side folders required::

    inbox/<src>            raw phone photo (the stable id is the inbox-relative <src>)
    normalized/<src>.jpg   EXIF-baked JPEG written on label
    labels.jsonl           the label artifact (single blob, last write wins)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    from azure.storage.blob import ContainerClient

DEFAULT_CONTAINER = "hive"
CONNECTION_STRING_ENV = "STORAGE_CONNECTION_STRING"


class StorageBackend(Protocol):
    """The handful of object-store primitives ``LabelStore`` needs."""

    def read(self, name: str) -> bytes | None: ...
    def write(self, name: str, data: bytes) -> None: ...
    def exists(self, name: str) -> bool: ...
    def list(self, prefix: str) -> list[tuple[str, float]]: ...
    def delete(self, name: str) -> None: ...


@dataclass
class AzureBlobBackend:
    """Object-store primitives over an Azure ``ContainerClient`` (keys = blob names)."""

    container: ContainerClient

    @classmethod
    def from_env(
        cls,
        container: str = DEFAULT_CONTAINER,
        *,
        connection_string: str | None = None,
    ) -> AzureBlobBackend:
        """Build from ``STORAGE_CONNECTION_STRING`` (or an explicit one). Lazy SDK import."""
        from azure.storage.blob import BlobServiceClient

        conn = connection_string or os.environ.get(CONNECTION_STRING_ENV)
        if not conn:
            raise RuntimeError(
                f"{CONNECTION_STRING_ENV} is not set — put the storage account connection "
                "string in .env (the capture app and scripts load it via load_dotenv())."
            )
        svc = BlobServiceClient.from_connection_string(conn)
        return cls(svc.get_container_client(container))

    def read(self, name: str) -> bytes | None:
        """Blob bytes, or ``None`` if the blob does not exist."""
        from azure.core.exceptions import ResourceNotFoundError

        try:
            return self.container.download_blob(name).readall()
        except ResourceNotFoundError:
            return None

    def write(self, name: str, data: bytes) -> None:
        self.container.upload_blob(name, data, overwrite=True)

    def exists(self, name: str) -> bool:
        return self.container.get_blob_client(name).exists()

    def list(self, prefix: str) -> list[tuple[str, float]]:
        """``(name, mtime)`` for every blob under ``prefix`` (mtime = epoch seconds)."""
        return [
            (b.name, b.last_modified.timestamp())
            for b in self.container.list_blobs(name_starts_with=prefix)
        ]

    def delete(self, name: str) -> None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            self.container.delete_blob(name)
        except ResourceNotFoundError:
            pass

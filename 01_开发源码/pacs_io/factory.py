from .folder_adapter import FolderPacsAdapter
from .orthanc_adapter import OrthancAdapter


def create_pacs_adapter(source, **kwargs):

    if source == "folder":
        return FolderPacsAdapter()

    if source == "orthanc":
        return OrthancAdapter(
            kwargs.get(
                "url",
                "http://127.0.0.1:8042",
            )
        )

    raise ValueError(
        f"未知影像来源: {source}"
    )

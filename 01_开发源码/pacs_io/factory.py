from .folder_adapter import FolderPacsAdapter
from .orthanc_adapter import OrthancAdapter
from .yunpacs_adapter import YUNPACSPacsAdapter


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

    if source == "yunpacs":
        return YUNPACSPacsAdapter(
            root=kwargs.get(
                "root",
                None,
            )
            or "D:/YUNPACS/放射诊断/ImageDir_r"
        )

    raise ValueError(
        f"未知影像来源: {source}"
    )

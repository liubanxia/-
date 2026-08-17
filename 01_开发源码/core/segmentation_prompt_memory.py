import threading


class SegmentationPromptMemory:
    def __init__(self):
        self._lock = threading.RLock()
        self._items = []

    def update_from_findings(self, findings):
        items = []

        for f in findings:
            g = getattr(f, "geometry", None)

            if not isinstance(g, dict):
                continue

            bbox = g.get("bbox")
            point = g.get("point") or g.get("center") or g.get("tip")
            slice_index = g.get("slice_index")

            if bbox is None and point is None:
                continue

            items.append({
                "bbox": bbox,
                "point": point,
                "slice_index": slice_index,
                "source_expert": getattr(f, "expert_id", ""),
            })

        with self._lock:
            self._items = items

        return items

    def all(self):
        with self._lock:
            return list(self._items)

    def clear(self):
        with self._lock:
            self._items.clear()


SEGMENTATION_PROMPT_MEMORY = SegmentationPromptMemory()

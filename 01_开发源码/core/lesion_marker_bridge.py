class LesionMarkerBridge:
    """
    只生成指向标记数据。
    不画矩形框，不修改PACS原图。
    """

    def _point_from_geometry(self, geometry):
        if not isinstance(geometry, dict):
            return None

        for key in ("tip", "point", "center"):
            p = geometry.get(key)
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                return list(p)

        bbox = geometry.get("bbox")

        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            return [
                (x1 + x2) / 2,
                (y1 + y2) / 2,
            ]

        return None

    def build(self, fused_result):
        markers = []

        for finding in fused_result.findings:
            geometry = finding.geometry

            if not geometry:
                continue

            tip = self._point_from_geometry(
                geometry
            )

            if tip is None:
                continue

            markers.append({
                "slice_index": geometry.get(
                    "slice_index"
                ) if isinstance(geometry, dict) else None,

                "tip": tip,
                "style": "thin_pointer",

                "label": (
                    finding.location
                    or finding.finding
                    or "病灶"
                ),

                # 临床显示层不带score和模型名
                "clinical": True,
            })

        return markers


LESION_MARKER_BRIDGE = LesionMarkerBridge()

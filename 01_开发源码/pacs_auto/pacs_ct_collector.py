import hashlib


class PacsCTCollector:
    def __init__(self, agent):
        self.agent = agent

    @staticmethod
    def _hash(image):
        return hashlib.sha256(
            image[::8, ::8].tobytes()
        ).hexdigest()

    def seek_end(self, direction=-1, max_steps=650):
        self.agent._guard()

        unchanged = 0

        for _ in range(max_steps):
            r = self.agent.safe_scroll(direction)

            if r["changed"]:
                unchanged = 0
            else:
                unchanged += 1

            if unchanged >= 3:
                return True

        return False

    def collect_series(self, direction=1, max_slices=650):
        self.agent._guard()

        frames = []
        seen = set()
        unchanged = 0

        for _ in range(max_slices):
            image = self.agent.capture_viewport()
            key = self._hash(image)

            if key not in seen:
                seen.add(key)
                frames.append(image)

            r = self.agent.safe_scroll(direction)

            if r["changed"]:
                unchanged = 0
            else:
                unchanged += 1

            if unchanged >= 3:
                break

        return frames

    def collect_full_ct(self):
        # 先自动找到一端
        if not self.seek_end(-1):
            raise RuntimeError(
                "无法安全确认CT序列端点"
            )

        # 再从端点向另一端完整采集
        return self.collect_series(1)

    def scan_full_ct(self, callback=None):
        self.agent._guard()

        if not self.seek_end(-1):
            raise RuntimeError("无法确认CT序列端点")

        count = 0
        seen = set()
        unchanged = 0

        for _ in range(650):
            image = self.agent.capture_viewport()
            key = self._hash(image)

            if key not in seen:
                seen.add(key)
                count += 1

                if callback:
                    callback(image, count)

            image = None

            r = self.agent.safe_scroll(1)

            if r["changed"]:
                unchanged = 0
            else:
                unchanged += 1

            if unchanged >= 3:
                break

        return count

import sys
import time
import numpy as np

if sys.platform == "win32":
    sys.coinit_flags = 2

from PIL import ImageGrab
from pywinauto import Desktop, mouse


class PacsAutoAgent:

    def __init__(self):
        self.window = None
        self.window_rect = None
        self.viewport = None

    def discover(self):
        candidates = []

        for w in Desktop(backend="uia").windows():
            try:
                title = w.window_text().strip()
                r = w.rectangle()

                if not title or r.width() < 600:
                    continue

                low = title.lower()

                try:
                    cls = str(w.element_info.class_name).lower()
                except Exception:
                    cls = ""

                blocked_classes = (
                    "mintty",
                    "consolewindowclass",
                    "chrome_widget",
                    "progman",
                    "shell_traywnd",
                )

                if any(x in cls for x in blocked_classes):
                    continue

                blocked = (
                    "program manager", "任务栏",
                    "qq", "git", "嘟嘟牛",
                    "project phoenix"
                )

                if any(x in low for x in blocked):
                    continue

                score = r.width() * r.height()

                if any(x in title.lower() for x in (
                    "pacs", "dicom", "影像",
                    "阅片", "radiology"
                )):
                    score += 10000000

                candidates.append(
                    (score, title, w, r)
                )
            except Exception:
                pass

        candidates.sort(reverse=True, key=lambda x: x[0])

        return candidates

    def bind_best(self):
        c = self.discover()

        if not c:
            raise RuntimeError("未发现PACS候选")

        score, title, self.window, self.window_rect = c[0]

        # 当前阶段只有高置信度PACS才允许自动操控。
        # 后续再加入屏幕视觉识别，让未知品牌PACS也能自动通过。
        visual_score = self.visual_medical_score(
            self.window,
            self.window_rect
        )

        if score >= 10000000:
            self.pacs_confirmed = True

        elif visual_score >= 85:
            # 未知品牌PACS：
            # 自动识别为候选，但第一次必须人工确认
            self.pacs_confirmed = False

        else:
            self.window = None
            self.window_rect = None
            raise RuntimeError(
                "没有发现高置信度PACS，禁止自动鼠标操作"
            )

        r = self.window_rect

        # 第一版自动取窗口中央主要阅片区域
        self.viewport = (
            int(r.width() * 0.10),
            int(r.height() * 0.10),
            int(r.width() * 0.90),
            int(r.height() * 0.90),
        )

        return title

    def capture_viewport(self):
        r = self.window_rect
        x1, y1, x2, y2 = self.viewport

        return np.asarray(
            ImageGrab.grab(
                bbox=(
                    r.left + x1,
                    r.top + y1,
                    r.left + x2,
                    r.top + y2,
                )
            )
        )

    def verify_scroll(self):
        before = self.capture_viewport()

        x1, y1, x2, y2 = self.viewport

        center = (
            self.window_rect.left + (x1 + x2)//2,
            self.window_rect.top + (y1 + y2)//2,
        )

        mouse.scroll(
            coords=center,
            wheel_dist=-1
        )

        time.sleep(0.5)

        after = self.capture_viewport()

        diff = np.mean(
            np.abs(
                before.astype(float)
                - after.astype(float)
            )
        )

        return {
            "changed": bool(diff > 2.0),
            "difference": round(float(diff), 3),
        }

    def find_tools(self):
        found = []

        words = (
            "窗宽", "窗位", "测量",
            "window", "level",
            "measure", "zoom", "pan"
        )

        for c in self.window.descendants():
            try:
                text = c.window_text().strip()

                if any(
                    x in text.lower()
                    for x in words
                ):
                    found.append(text)

            except Exception:
                pass

        return found

    def visual_medical_score(self, window, rect):
        image = ImageGrab.grab(
            bbox=(rect.left, rect.top, rect.right, rect.bottom)
        ).convert("L")

        arr = np.asarray(image)

        if arr.size == 0:
            return 0

        mean = float(arr.mean())
        std = float(arr.std())

        # 医学影像阅片窗口通常：
        # 暗背景 + 丰富灰阶 + 大面积图像区域
        score = 0

        if mean < 150:
            score += 30

        if std > 35:
            score += 30

        dark_ratio = float(
            np.mean(arr < 70)
        )

        if dark_ratio > 0.30:
            score += 25

        if rect.width() >= 1000:
            score += 15

        return score

    def confirm_pacs(self):
        if self.window is None:
            raise RuntimeError("尚未绑定PACS候选")

        self.pacs_confirmed = True

    def can_control(self):
        return bool(
            getattr(self, "pacs_confirmed", False)
        )

    def _guard(self):
        if not self.can_control():
            raise RuntimeError("PACS未确认，禁止鼠标操作")

    def center(self):
        x1,y1,x2,y2 = self.viewport
        r = self.window_rect
        return (
            r.left + (x1+x2)//2,
            r.top + (y1+y2)//2
        )

    def safe_scroll(self, direction=-1):
        self._guard()
        before = self.capture_viewport()

        mouse.scroll(
            coords=self.center(),
            wheel_dist=direction
        )
        time.sleep(0.4)

        after = self.capture_viewport()
        diff = float(np.mean(
            np.abs(before.astype(float)-after.astype(float))
        ))

        return {"changed": diff > 2.0, "diff": round(diff,3)}

    def safe_drag(self, button="left", dx=30, dy=0):
        self._guard()

        before = self.capture_viewport()
        x,y = self.center()

        mouse.move(coords=(x,y))
        mouse.press(button=button, coords=(x,y))
        mouse.move(coords=(x+dx,y+dy))
        mouse.release(button=button, coords=(x+dx,y+dy))

        time.sleep(0.4)
        after = self.capture_viewport()

        diff = float(np.mean(
            np.abs(before.astype(float)-after.astype(float))
        ))

        return {"changed": diff > 2.0, "diff": round(diff,3)}

    def _classify_effect(self, before, after):
        a = before.mean(axis=2) if before.ndim == 3 else before
        b = after.mean(axis=2) if after.ndim == 3 else after

        mean_delta = abs(float(a.mean() - b.mean()))
        std_delta = abs(float(a.std() - b.std()))

        x = a.ravel()[::20]
        y = b.ravel()[::20]

        corr = float(np.corrcoef(x, y)[0,1]) if len(x) > 10 else 0.0

        if corr > 0.94 and (mean_delta > 1 or std_delta > 1):
            kind = "显示参数变化候选"
        elif corr < 0.94:
            kind = "视图几何变化候选"
        else:
            kind = "无明显效果"

        return {
            "kind": kind,
            "corr": round(corr,3),
            "mean_delta": round(mean_delta,3),
            "std_delta": round(std_delta,3),
        }

    def probe_drag(self, button="right", dx=25, dy=0):
        self._guard()

        before = self.capture_viewport()
        x,y = self.center()

        mouse.move(coords=(x,y))
        mouse.press(button=button, coords=(x,y))
        mouse.move(coords=(x+dx,y+dy))
        mouse.release(button=button, coords=(x+dx,y+dy))

        time.sleep(0.4)
        after = self.capture_viewport()

        result = self._classify_effect(before, after)

        # 尝试反向恢复
        mouse.move(coords=(x+dx,y+dy))
        mouse.press(button=button, coords=(x+dx,y+dy))
        mouse.move(coords=(x,y))
        mouse.release(button=button, coords=(x,y))

        return result

from ai.fracture_candidate import FractureCandidate


class FractureCandidateStore:
    """
    Project Phoenix 视觉B骨折候选容器。

    当前职责：
    - 保存视觉B候选；
    - 保留原始候选顺序；
    - 按CT层查询；
    - 清空旧病例候选。

    当前不负责：
    - 候选合并；
    - 置信度过滤；
    - 自动去重；
    - 医学诊断判断。
    """

    def __init__(self):
        self._candidates = []

    def __len__(self):
        return len(self._candidates)

    def add(self, candidate):
        """
        添加一个骨折候选。
        """

        if not isinstance(
            candidate,
            FractureCandidate,
        ):
            raise TypeError(
                "视觉B候选容器仅允许FractureCandidate"
            )

        self._candidates.append(
            candidate
        )

        return len(self._candidates) - 1

    def extend(self, candidates):
        """
        批量添加候选。

        保持输入顺序，不自动排序或合并。
        """

        candidates = list(candidates)

        for candidate in candidates:
            if not isinstance(
                candidate,
                FractureCandidate,
            ):
                raise TypeError(
                    "视觉B候选容器仅允许FractureCandidate"
                )

        start_index = len(
            self._candidates
        )

        self._candidates.extend(
            candidates
        )

        return start_index

    def get_all(self):
        """
        返回当前全部候选的只读副本。
        """

        return tuple(
            self._candidates
        )

    def get_by_slice_index(
        self,
        slice_index,
    ):
        """
        查询指定Phoenix CT层号的全部候选。
        """

        if (
            not isinstance(slice_index, int)
            or isinstance(slice_index, bool)
        ):
            raise TypeError(
                "slice_index必须是整数"
            )

        if slice_index < 0:
            raise ValueError(
                "slice_index不能小于0"
            )

        return tuple(
            candidate
            for candidate in self._candidates
            if candidate.slice_index
            == slice_index
        )

    def clear(self):
        """
        清空候选。

        病例、Study或Series变化时，
        上层必须调用本方法清除旧候选。
        """

        count = len(
            self._candidates
        )

        self._candidates.clear()

        return count

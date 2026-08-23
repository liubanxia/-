from __future__ import annotations

import re
from collections import Counter, defaultdict

_INSTALLED = False
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])([+\-\u2212]?\d+(?:\.\d+)?(?:%)?)")
_METRIC_EN = {
    "sensitivity": r"\bsensitiv(?:ity|ities)\b",
    "specificity": r"\bspecificit(?:y|ies)\b",
    "accuracy": r"\baccuracy\b",
    "precision": r"\bprecision\b",
    "recall": r"\brecall\b",
    "ppv": r"\b(?:ppv|positive predictive value)\b",
    "npv": r"\b(?:npv|negative predictive value)\b",
    "auc": r"\bauc\b|\barea under (?:the )?(?:roc )?curve\b",
}
_METRIC_ZH = {
    "sensitivity": r"敏感度|敏感性|灵敏度|sensitivity",
    "specificity": r"特异度|特异性|specificity",
    "accuracy": r"准确率|准确度|accuracy",
    "precision": r"精确率|查准率|precision",
    "recall": r"召回率|查全率|recall",
    "ppv": r"阳性预测值|ppv",
    "npv": r"阴性预测值|npv",
    "auc": r"\bauc\b|曲线下面积",
}


def _chinese_target(target: str) -> bool:
    raw = (target or "").strip()
    low = raw.lower()
    return "中文" in raw or low.startswith("zh") or low == "chinese"


def _bindings(text: str, patterns: dict[str, str], chinese: bool) -> dict[str, list[str]]:
    chunks = re.split(r"[。；;\n]+" if chinese else r"[.;\n]+", text)
    result: defaultdict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        metrics: list[tuple[int, str]] = []
        for name, pattern in patterns.items():
            metrics.extend((m.start(), name) for m in re.finditer(pattern, chunk, re.I))
        metrics.sort()
        nums = [(m.start(), m.group(1).replace("\u2212", "-")) for m in _NUMBER_RE.finditer(chunk)]
        if metrics and len(metrics) == len(nums):
            for (_, name), (_, value) in zip(metrics, nums):
                result[name].append(value)
    return dict(result)


def validate_medical_semantics(source: str, translated: str, target_language: str = "中文") -> tuple[str, ...]:
    if not _chinese_target(target_language):
        return ()
    src = (source or "").lower()
    zh = re.sub(r"\s+", "", translated or "")
    issues: list[str] = []

    anatomy = (
        r"(?:upper|middle|lower|frontal|parietal|temporal|occipital|lung|lobe|kidney|renal|"
        r"adrenal|ventricle|atrium|hemithorax|breast|ovary|testis|hip|knee|ankle|shoulder|"
        r"arm|leg|hand|foot|eye|ear|hemisphere|carotid|coronary|ureter|pelvis|colon|chest|"
        r"abdomen|flank|side)"
    )
    right = re.search(rf"\bright(?:-sided)?\b(?=\s+(?:\w+\s+){{0,2}}{anatomy}\b)|\bon the right\b|\bright side\b", src)
    left = re.search(rf"\bleft(?:-sided)?\b(?=\s+(?:\w+\s+){{0,2}}{anatomy}\b)|\bon the left\b|\bleft side\b", src)
    if right and not left and ("右" not in zh or "左" in zh):
        issues.append("左右侧关系疑似改变：原文为右侧")
    if left and not right and ("左" not in zh or "右" in zh):
        issues.append("左右侧关系疑似改变：原文为左侧")
    if re.search(r"\bbilateral\b", src) and not any(x in zh for x in ("双侧", "两侧", "双边")):
        issues.append("侧别关系疑似改变：原文为双侧")

    cannot_exclude = re.search(
        r"\b(?:cannot|can't|could not|couldn't)\s+(?:be\s+)?(?:exclude|rule out)\b|\bcannot be excluded\b",
        src,
    )
    if cannot_exclude:
        accepted = ("不能排除", "无法排除", "不除外", "尚不能排除", "难以排除", "不能除外", "无法除外")
        if not any(x in zh for x in accepted):
            issues.append("诊断排除关系疑似改变：原文为不能排除")
        if any(x in zh for x in ("明确诊断", "确诊", "证实为", "诊断为")):
            issues.append("诊断确定性疑似被提高：不能排除被译为明确诊断")

    has_negation = re.search(
        r"\bno\s+(?:evidence\s+of\s+)?[a-z]|\bnot\b|\bwithout\s+(?:evidence|enhancement|restriction|"
        r"effusion|pneumothorax|hemorrhage|bleeding|edema|mass|lesion|fracture|metastasis|progression|"
        r"complication|adenopathy|lymphadenopathy|symptom|sign)\b",
        src,
    )
    if has_negation and not any(x in zh for x in ("未见", "无", "没有", "未发现", "不存在", "未显示", "未检出", "不能", "无法", "不", "未", "阴性")):
        issues.append("否定关系疑似丢失或反转")

    uncertain = re.search(
        r"\b(?:may|might|possibly|possible|probably|probable|likely)\b|\bsuggestive of\b|\bsuspicious for\b|\bfavor(?:s|ed)?\b",
        src,
    )
    if uncertain and not any(x in zh for x in ("可能", "提示", "考虑", "疑似", "倾向", "可疑", "怀疑", "支持", "较可能", "或为")):
        issues.append("诊断确定性疑似被提高：原文含可能/提示/倾向")

    polarity = (
        (r"\bbenign\b", r"\bmalignan(?:t|cy)\b", ("良性",), ("恶性",), "良恶性"),
        (r"\bmalignan(?:t|cy)\b", r"\bbenign\b", ("恶性",), ("良性",), "良恶性"),
        (r"\bacute\b", r"\bchronic\b", ("急性",), ("慢性",), "急慢性"),
        (r"\bchronic\b", r"\bacute\b", ("慢性",), ("急性",), "急慢性"),
        (r"\bmale\b|\bman\b", r"\bfemale\b|\bwoman\b", ("男性", "男"), ("女性", "女"), "性别"),
        (r"\bfemale\b|\bwoman\b", r"\bmale\b|\bman\b", ("女性", "女"), ("男性", "男"), "性别"),
        (r"\bunilateral\b", r"\bbilateral\b", ("单侧", "一侧"), ("双侧", "两侧"), "单侧/双侧"),
    )
    for pattern, opposite_src, expected, opposite, label in polarity:
        if re.search(pattern, src) and not re.search(opposite_src, src):
            if any(x in zh for x in opposite) and not any(x in zh for x in expected):
                issues.append(f"{label}概念疑似反转")
            elif not any(x in zh for x in expected):
                issues.append(f"{label}概念疑似丢失")

    if re.search(r"\bnegative for\b", src):
        if "阳性" in zh and "阴性" not in zh:
            issues.append("阳性/阴性关系疑似反转")
        elif not any(x in zh for x in ("阴性", "未见", "无", "没有")):
            issues.append("阴性关系疑似丢失")
    if re.search(r"\bpositive for\b", src):
        if "阴性" in zh and "阳性" not in zh:
            issues.append("阳性/阴性关系疑似反转")
        elif not any(x in zh for x in ("阳性", "存在", "检出", "可见")):
            issues.append("阳性关系疑似丢失")
    if re.search(r"\b(?:nonenhancing|non-enhancing|unenhanced)\b", src):
        if any(x in zh for x in ("明显强化", "强化明显")) and not any(x in zh for x in ("无强化", "不强化", "未强化", "平扫")):
            issues.append("强化/无强化关系疑似反转")
        elif not any(x in zh for x in ("无强化", "不强化", "未强化", "平扫")):
            issues.append("无强化关系疑似丢失")

    directions = (
        (r"\b(?:increased|increasing)\s+(?:t2\s+|t1\s+)?signal\b|\bhyperintense\b", ("信号增高", "高信号", "信号增强"), ("信号减低", "低信号", "信号降低"), "信号高低"),
        (r"\b(?:decreased|decreasing)\s+(?:t2\s+|t1\s+)?signal\b|\bhypointense\b", ("信号减低", "低信号", "信号降低"), ("信号增高", "高信号", "信号增强"), "信号高低"),
        (r"\b(?:increased|increasing)\s+(?:attenuation|density)\b|\bhyperattenuat", ("密度增高", "衰减增高", "高密度", "高衰减"), ("密度减低", "衰减减低", "低密度", "低衰减"), "密度/衰减高低"),
        (r"\b(?:decreased|decreasing)\s+(?:attenuation|density)\b|\bhypoattenuat", ("密度减低", "衰减减低", "低密度", "低衰减"), ("密度增高", "衰减增高", "高密度", "高衰减"), "密度/衰减高低"),
        (r"\b(?:increased|increase|increasing)\s+in\s+size\b|\benlarg(?:ed|ing)\b|\bgrew\b|\b(?:interval|tumou?r|lesion|nodule)\s+growth\b|\b(?:demonstrated|showed)\s+growth\b", ("增大", "增长", "生长", "扩大", "进展"), ("缩小", "减小", "消退"), "大小变化"),
        (r"\b(?:decreased|decrease|decreasing)\s+in\s+size\b|\bshrank\b|\bshrunk\b|\bregress(?:ed|ion)?\b", ("缩小", "减小", "消退", "回缩", "缓解"), ("增大", "增长", "扩大", "进展"), "大小变化"),
        (r"\b(?:stable|unchanged)\b", ("稳定", "无明显变化", "未见明显变化", "无变化", "未变化"), ("进展", "恶化", "增大"), "稳定/进展"),
        (r"\bprogress(?:ion|ed|ing)?\b|\bworsen(?:ed|ing)?\b", ("进展", "恶化", "加重"), ("稳定", "好转", "缓解", "缩小"), "稳定/进展"),
        (r"\bhigher than\b|\bgreater than\b", ("高于", "大于", "超过"), ("低于", "小于"), "比较方向"),
        (r"\blower than\b|\bless than\b", ("低于", "小于", "少于"), ("高于", "大于", "超过"), "比较方向"),
    )
    for pattern, expected, opposite, label in directions:
        if re.search(pattern, src):
            if any(x in zh for x in opposite) and not any(x in zh for x in expected):
                issues.append(f"{label}关系疑似反转")
            elif not any(x in zh for x in expected):
                issues.append(f"{label}关系疑似丢失")

    src_bind = _bindings(source or "", _METRIC_EN, False)
    zh_bind = _bindings(zh, _METRIC_ZH, True)
    for metric, values in src_bind.items():
        if metric in zh_bind and Counter(values) != Counter(zh_bind[metric]):
            issues.append(f"统计指标数值绑定疑似改变：{metric}")

    return tuple(dict.fromkeys(issues))


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from . import translation_models as tm

    original_validate = tm.TranslationValidator.validate

    def validate(self, source: str, translated: str, target_language: str = "中文"):
        base = original_validate(self, source, translated, target_language)
        issues = validate_medical_semantics(source, translated, target_language)
        if not issues:
            return base
        reasons = tuple(dict.fromkeys((*base.reasons, *issues)))
        penalty = min(0.15 * len(issues), 0.45)
        return tm.QualityReport(False, max(0.0, min(base.score - penalty, 0.55)), reasons)

    def translate(self, text: str, target_language: str = "中文", *, smart_level: str = "smart1") -> str:
        level = tm._normalize_smart_level(smart_level)
        profile = "translation" if level == "smart2" else "fast"
        max_tokens = tm.translation_output_budget(text, level)
        glossary_builder = getattr(self, "glossary_prompt", None)
        glossary = glossary_builder(text) if callable(glossary_builder) else ""
        glossary_section = f"\n{glossary}\n" if glossary else ""
        prompt = f"""你是 Phoenix 医学教材精译器。把下面英文医学原文完整、准确地翻译成{target_language}。

这是医学教材正文，不是摘要任务。必须做到：
- 逐段完整翻译，不总结、不删减、不扩写，不加入原文没有的医学知识。
- 疾病、解剖、影像学征象、检查技术、药物、分级和病理术语使用规范医学中文。
- 所有数字、单位、百分比、HU、分级、剂量、图号、表号、公式、参考文献编号必须保留。
- 否定/肯定、不能排除/可以排除、可能/提示/倾向/明确诊断等诊断确定性必须逐项保持。
- 左/右/双侧、增高/降低、增大/缩小、稳定/进展、高于/低于等方向关系必须与原文一致。
- sensitivity、specificity、AUC、PPV、NPV 等统计指标必须和各自数值绑定，严禁互换。
- 医学缩写必须保留；含义严格采用固定缩写表。原文若只有一个缩写，输出“规范中文（原缩写）”。
- 正文中的缩写保持紧凑，不强制重复英文全称，避免课件文本膨胀。
- 句子损坏、OCR错误或语义无法确定时标记“[原文不清]”，不得猜测。
- 只输出译文，不输出解释、评语、翻译过程或模型信息。
{glossary_section}

原文：
{text}
"""
        return self.llm.generate(prompt, max_new_tokens=max_tokens, profile=profile).strip()

    tm.TranslationValidator.validate = validate
    tm.QwenMedicalTranslationBackend.translate = translate
    tm.TranslationValidator._phoenix_semantic_safety = True
    tm.QwenMedicalTranslationBackend._phoenix_semantic_prompt = True

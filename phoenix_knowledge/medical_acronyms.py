from __future__ import annotations

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ACRONYM_CONTRACT_VERSION = 1

_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:SUV(?:max|mean|peak)|MinIP|eGFR|mRNA|pH|"
    r"HFrEF|HFpEF|SARS-CoV-2|"
    r"[A-Z][a-z][A-Z][A-Za-z0-9]{0,9}|"
    r"[A-Z][A-Z0-9]{1,11}(?:[/+.-][A-Z0-9]{1,8})*|"
    r"[A-Z][0-9](?:WI)?)(?![A-Za-z0-9])"
)
_IGNORE = {
    "AND", "OR", "THE", "OF", "IN", "TO", "FOR", "WITH", "WITHOUT",
    "FROM", "INTO", "ON", "AT", "BY", "AS", "IS", "ARE", "WAS", "WERE",
    "THIS", "THAT", "THESE", "THOSE", "VS", "VIA", "ET", "AL",
    "IMAGE", "IMAGES", "IMAGING", "FIGURE", "TABLE", "CASE", "CASES",
    "PATIENT", "PATIENTS", "NORMAL", "ABNORMAL", "FINDING", "FINDINGS",
    "LEFT", "RIGHT", "BILATERAL", "ANTERIOR", "POSTERIOR", "SUPERIOR",
    "INFERIOR", "ACUTE", "CHRONIC", "POSITIVE", "NEGATIVE", "PRESENT",
    "ABSENT", "HIGH", "LOW", "EARLY", "LATE", "BEFORE", "AFTER",
    "SHOW", "SHOWS", "SHOWN", "FOLLOW", "UP", "KEY", "NOTE",
    "I", "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII",
}


# High-value radiology vocabulary is available immediately and offline.  Values
# are candidate senses, not unconditional substitutions: ambiguous entries are
# resolved from the whole deck by Smart2 before translation begins.
RADIOLOGY_SEED: dict[str, tuple[tuple[str, str], ...]] = {
    "CT": (("computed tomography", "计算机断层成像"),),
    "NCCT": (("non-contrast computed tomography", "非增强CT"),),
    "CECT": (("contrast-enhanced computed tomography", "增强CT"),),
    "CTA": (("computed tomography angiography", "CT血管成像"),),
    "CTV": (("computed tomography venography", "CT静脉成像"),),
    "CTP": (("computed tomography perfusion", "CT灌注成像"),),
    "HRCT": (("high-resolution computed tomography", "高分辨率CT"),),
    "LDCT": (("low-dose computed tomography", "低剂量CT"),),
    "DECT": (("dual-energy computed tomography", "双能量CT"),),
    "MDCT": (("multidetector computed tomography", "多排探测器CT"),),
    "CBCT": (("cone-beam computed tomography", "锥形束CT"),),
    "MRI": (("magnetic resonance imaging", "磁共振成像"),),
    "MR": (
        ("magnetic resonance", "磁共振"),
        ("mitral regurgitation", "二尖瓣反流"),
    ),
    "MRA": (("magnetic resonance angiography", "磁共振血管成像"),),
    "MRV": (("magnetic resonance venography", "磁共振静脉成像"),),
    "MRCP": (("magnetic resonance cholangiopancreatography", "磁共振胰胆管成像"),),
    "MRS": (("magnetic resonance spectroscopy", "磁共振波谱"),),
    "DWI": (("diffusion-weighted imaging", "弥散加权成像"),),
    "ADC": (("apparent diffusion coefficient", "表观弥散系数"),),
    "DTI": (("diffusion tensor imaging", "弥散张量成像"),),
    "DKI": (("diffusion kurtosis imaging", "弥散峰度成像"),),
    "SWI": (("susceptibility-weighted imaging", "磁敏感加权成像"),),
    "T1WI": (("T1-weighted imaging", "T1加权成像"),),
    "T2WI": (("T2-weighted imaging", "T2加权成像"),),
    "FLAIR": (("fluid-attenuated inversion recovery", "液体衰减反转恢复序列"),),
    "STIR": (("short tau inversion recovery", "短时间反转恢复序列"),),
    "GRE": (("gradient-recalled echo", "梯度回波序列"),),
    "FSE": (("fast spin echo", "快速自旋回波序列"),),
    "TSE": (("turbo spin echo", "快速自旋回波序列"),),
    "EPI": (("echo-planar imaging", "回波平面成像"),),
    "TOF": (("time of flight", "时间飞跃法"),),
    "ASL": (("arterial spin labeling", "动脉自旋标记"),),
    "DSC": (("dynamic susceptibility contrast", "动态磁敏感对比增强"),),
    "DCE": (("dynamic contrast-enhanced", "动态对比增强"),),
    "IVIM": (("intravoxel incoherent motion", "体素内不相干运动"),),
    "BOLD": (("blood oxygen level dependent", "血氧水平依赖成像"),),
    "CISS": (("constructive interference in steady state", "稳态构成干扰序列"),),
    "FIESTA": (("fast imaging employing steady-state acquisition", "稳态采集快速成像"),),
    "MIP": (("maximum intensity projection", "最大密度投影"),),
    "MinIP": (("minimum intensity projection", "最小密度投影"),),
    "MPR": (("multiplanar reconstruction", "多平面重建"),),
    "CPR": (
        ("curved planar reformation", "曲面重建"),
        ("cardiopulmonary resuscitation", "心肺复苏"),
    ),
    "VR": (("volume rendering", "容积再现"),),
    "ROI": (("region of interest", "感兴趣区"),),
    "VOI": (("volume of interest", "感兴趣容积"),),
    "FOV": (("field of view", "视野"),),
    "SNR": (("signal-to-noise ratio", "信噪比"),),
    "CNR": (("contrast-to-noise ratio", "对比噪声比"),),
    "SI": (
        ("signal intensity", "信号强度"),
        ("sacroiliac", "骶髂"),
    ),
    "PACS": (("picture archiving and communication system", "影像归档和通信系统"),),
    "DICOM": (("Digital Imaging and Communications in Medicine", "医学数字成像和通信标准"),),
    "PET": (("positron emission tomography", "正电子发射断层成像"),),
    "SPECT": (("single-photon emission computed tomography", "单光子发射计算机断层成像"),),
    "FDG": (("fluorodeoxyglucose", "氟代脱氧葡萄糖"),),
    "PSMA": (("prostate-specific membrane antigen", "前列腺特异性膜抗原"),),
    "SUV": (("standardized uptake value", "标准化摄取值"),),
    "SUVmax": (("maximum standardized uptake value", "最大标准化摄取值"),),
    "SUVmean": (("mean standardized uptake value", "平均标准化摄取值"),),
    "GGO": (("ground-glass opacity", "磨玻璃密度影"),),
    "GGN": (("ground-glass nodule", "磨玻璃结节"),),
    "SPN": (("solitary pulmonary nodule", "孤立性肺结节"),),
    "ILD": (("interstitial lung disease", "间质性肺疾病"),),
    "UIP": (("usual interstitial pneumonia", "普通型间质性肺炎"),),
    "NSIP": (("nonspecific interstitial pneumonia", "非特异性间质性肺炎"),),
    "IPF": (("idiopathic pulmonary fibrosis", "特发性肺纤维化"),),
    "COPD": (("chronic obstructive pulmonary disease", "慢性阻塞性肺疾病"),),
    "PE": (
        ("pulmonary embolism", "肺栓塞"),
        ("pleural effusion", "胸腔积液"),
        ("physical examination", "体格检查"),
    ),
    "PTE": (("pulmonary thromboembolism", "肺血栓栓塞症"),),
    "DVT": (("deep vein thrombosis", "深静脉血栓形成"),),
    "CTEPH": (("chronic thromboembolic pulmonary hypertension", "慢性血栓栓塞性肺动脉高压"),),
    "PA": (
        ("posteroanterior", "后前位"),
        ("pulmonary artery", "肺动脉"),
    ),
    "AP": (
        ("anteroposterior", "前后位"),
        ("alkaline phosphatase", "碱性磷酸酶"),
    ),
    "CSF": (("cerebrospinal fluid", "脑脊液"),),
    "SAH": (("subarachnoid hemorrhage", "蛛网膜下腔出血"),),
    "ICH": (("intracerebral hemorrhage", "脑内出血"),),
    "IVH": (("intraventricular hemorrhage", "脑室内出血"),),
    "SDH": (("subdural hematoma", "硬膜下血肿"),),
    "EDH": (("epidural hematoma", "硬膜外血肿"),),
    "AIS": (("acute ischemic stroke", "急性缺血性卒中"),),
    "TIA": (("transient ischemic attack", "短暂性脑缺血发作"),),
    "AVM": (("arteriovenous malformation", "动静脉畸形"),),
    "DAVF": (("dural arteriovenous fistula", "硬脑膜动静脉瘘"),),
    "CVST": (("cerebral venous sinus thrombosis", "脑静脉窦血栓形成"),),
    "PRES": (("posterior reversible encephalopathy syndrome", "可逆性后部脑病综合征"),),
    "MS": (
        ("multiple sclerosis", "多发性硬化"),
        ("mitral stenosis", "二尖瓣狭窄"),
    ),
    "HCC": (("hepatocellular carcinoma", "肝细胞癌"),),
    "RCC": (("renal cell carcinoma", "肾细胞癌"),),
    "CRC": (("colorectal cancer", "结直肠癌"),),
    "GBM": (("glioblastoma", "胶质母细胞瘤"),),
    "ACL": (("anterior cruciate ligament", "前交叉韧带"),),
    "PCL": (("posterior cruciate ligament", "后交叉韧带"),),
    "MCL": (("medial collateral ligament", "内侧副韧带"),),
    "LCL": (("lateral collateral ligament", "外侧副韧带"),),
    "TFCC": (("triangular fibrocartilage complex", "三角纤维软骨复合体"),),
    "BMD": (("bone mineral density", "骨密度"),),
    "PPV": (("positive predictive value", "阳性预测值"),),
    "NPV": (("negative predictive value", "阴性预测值"),),
    "ROC": (("receiver operating characteristic", "受试者工作特征"),),
    "AUC": (("area under the curve", "曲线下面积"),),
    "CI": (
        ("confidence interval", "置信区间"),
        ("cardiac index", "心脏指数"),
    ),
    "OR": (("odds ratio", "比值比"),),
    "HR": (
        ("hazard ratio", "风险比"),
        ("heart rate", "心率"),
    ),
    "SD": (
        ("standard deviation", "标准差"),
        ("stable disease", "疾病稳定"),
    ),
    "IQR": (("interquartile range", "四分位距"),),
    "ECG": (("electrocardiography", "心电图"),),
    "EKG": (("electrocardiography", "心电图"),),
    "EEG": (("electroencephalography", "脑电图"),),
    "EMG": (("electromyography", "肌电图"),),
    "TTE": (("transthoracic echocardiography", "经胸超声心动图"),),
    "TEE": (("transesophageal echocardiography", "经食管超声心动图"),),
    "CMR": (("cardiovascular magnetic resonance", "心血管磁共振"),),
    "CAD": (
        ("coronary artery disease", "冠状动脉疾病"),
        ("computer-aided detection", "计算机辅助检测"),
    ),
    "ACS": (("acute coronary syndrome", "急性冠状动脉综合征"),),
    "STEMI": (("ST-elevation myocardial infarction", "ST段抬高型心肌梗死"),),
    "NSTEMI": (("non-ST-elevation myocardial infarction", "非ST段抬高型心肌梗死"),),
    "MI": (
        ("myocardial infarction", "心肌梗死"),
        ("mechanical index", "机械指数"),
    ),
    "HF": (("heart failure", "心力衰竭"),),
    "CHF": (("congestive heart failure", "充血性心力衰竭"),),
    "HFREF": (("heart failure with reduced ejection fraction", "射血分数降低型心力衰竭"),),
    "HFPEF": (("heart failure with preserved ejection fraction", "射血分数保留型心力衰竭"),),
    "AF": (("atrial fibrillation", "心房颤动"),),
    "AFL": (("atrial flutter", "心房扑动"),),
    "SVT": (("supraventricular tachycardia", "室上性心动过速"),),
    "VT": (("ventricular tachycardia", "室性心动过速"),),
    "VF": (("ventricular fibrillation", "心室颤动"),),
    "PVC": (("premature ventricular contraction", "室性期前收缩"),),
    "LV": (("left ventricle", "左心室"),),
    "RV": (
        ("right ventricle", "右心室"),
        ("residual volume", "残气量"),
    ),
    "LA": (
        ("left atrium", "左心房"),
        ("local anesthesia", "局部麻醉"),
    ),
    "RA": (
        ("right atrium", "右心房"),
        ("rheumatoid arthritis", "类风湿关节炎"),
    ),
    "LVH": (("left ventricular hypertrophy", "左心室肥厚"),),
    "LVEF": (("left ventricular ejection fraction", "左心室射血分数"),),
    "EF": (("ejection fraction", "射血分数"),),
    "BP": (("blood pressure", "血压"),),
    "ARDS": (("acute respiratory distress syndrome", "急性呼吸窘迫综合征"),),
    "OSA": (("obstructive sleep apnea", "阻塞性睡眠呼吸暂停"),),
    "PFT": (("pulmonary function test", "肺功能检查"),),
    "FEV1": (("forced expiratory volume in one second", "第一秒用力呼气容积"),),
    "FVC": (("forced vital capacity", "用力肺活量"),),
    "TLC": (("total lung capacity", "肺总量"),),
    "DLCO": (("diffusing capacity of the lung for carbon monoxide", "肺一氧化碳弥散量"),),
    "CXR": (("chest radiograph", "胸部X线片"),),
    "AXR": (("abdominal radiograph", "腹部X线片"),),
    "KUB": (("kidneys, ureters and bladder radiograph", "泌尿系平片"),),
    "US": (
        ("ultrasonography", "超声检查"),
        ("United States", "美国"),
    ),
    "USG": (("ultrasonography", "超声检查"),),
    "CBC": (("complete blood count", "全血细胞计数"),),
    "WBC": (("white blood cell", "白细胞"),),
    "RBC": (("red blood cell", "红细胞"),),
    "HGB": (("hemoglobin", "血红蛋白"),),
    "HCT": (("hematocrit", "血细胞比容"),),
    "PLT": (("platelet", "血小板"),),
    "CRP": (("C-reactive protein", "C反应蛋白"),),
    "ESR": (("erythrocyte sedimentation rate", "红细胞沉降率"),),
    "ALT": (("alanine aminotransferase", "丙氨酸氨基转移酶"),),
    "AST": (("aspartate aminotransferase", "天冬氨酸氨基转移酶"),),
    "ALP": (("alkaline phosphatase", "碱性磷酸酶"),),
    "GGT": (("gamma-glutamyl transferase", "γ-谷氨酰转移酶"),),
    "BUN": (("blood urea nitrogen", "血尿素氮"),),
    "EGFR": (("estimated glomerular filtration rate", "估算肾小球滤过率"),),
    "INR": (("international normalized ratio", "国际标准化比值"),),
    "PT": (
        ("prothrombin time", "凝血酶原时间"),
        ("physical therapy", "物理治疗"),
    ),
    "APTT": (("activated partial thromboplastin time", "活化部分凝血活酶时间"),),
    "HBA1C": (("glycated hemoglobin A1c", "糖化血红蛋白A1c"),),
    "HIV": (("human immunodeficiency virus", "人类免疫缺陷病毒"),),
    "HBV": (("hepatitis B virus", "乙型肝炎病毒"),),
    "HCV": (("hepatitis C virus", "丙型肝炎病毒"),),
    "TB": (
        ("tuberculosis", "结核病"),
        ("total bilirubin", "总胆红素"),
    ),
    "PCR": (("polymerase chain reaction", "聚合酶链式反应"),),
    "RTPCR": (("reverse-transcription polymerase chain reaction", "逆转录聚合酶链式反应"),),
    "CNS": (("central nervous system", "中枢神经系统"),),
    "PNS": (
        ("peripheral nervous system", "周围神经系统"),
        ("paranasal sinuses", "鼻旁窦"),
    ),
    "BBB": (("blood-brain barrier", "血脑屏障"),),
    "ICP": (("intracranial pressure", "颅内压"),),
    "DNA": (("deoxyribonucleic acid", "脱氧核糖核酸"),),
    "RNA": (("ribonucleic acid", "核糖核酸"),),
    "MRNA": (("messenger RNA", "信使RNA"),),
    "AI": (
        ("artificial intelligence", "人工智能"),
        ("aortic insufficiency", "主动脉瓣关闭不全"),
    ),
    "ML": (
        ("machine learning", "机器学习"),
        ("maximum likelihood", "最大似然法"),
    ),
    "DL": (("deep learning", "深度学习"),),
    "CNN": (("convolutional neural network", "卷积神经网络"),),
    "RCT": (("randomized controlled trial", "随机对照试验"),),
    "RR": (
        ("relative risk", "相对危险度"),
        ("respiratory rate", "呼吸频率"),
    ),
    "NNT": (("number needed to treat", "需要治疗人数"),),
    "WHO": (("World Health Organization", "世界卫生组织"),),
    "TNM": (("tumor-node-metastasis", "肿瘤-淋巴结-转移分期"),),
    "RECIST": (("Response Evaluation Criteria in Solid Tumors", "实体瘤疗效评价标准"),),
    "CR": (
        ("complete response", "完全缓解"),
        ("conventional radiography", "常规X线摄影"),
    ),
    "PR": (
        ("partial response", "部分缓解"),
        ("pulse rate", "脉率"),
    ),
    "PD": (
        ("progressive disease", "疾病进展"),
        ("Parkinson disease", "帕金森病"),
        ("peritoneal dialysis", "腹膜透析"),
    ),
    "PH": (("potential of hydrogen", "酸碱度"),),
    "SPO2": (("peripheral oxygen saturation", "外周血氧饱和度"),),
    "PAO2": (("arterial oxygen partial pressure", "动脉血氧分压"),),
    "PACO2": (("arterial carbon dioxide partial pressure", "动脉血二氧化碳分压"),),
    "FIO2": (("fraction of inspired oxygen", "吸入氧浓度"),),
    "SAO2": (("arterial oxygen saturation", "动脉血氧饱和度"),),
}


def normalize_acronym(value: str) -> str:
    raw = str(value or "").strip()
    return re.sub(r"[^A-Za-z0-9]", "", raw).upper()


def _iter_acronyms(text: str):
    seen: set[str] = set()
    for match in _TOKEN_RE.finditer(str(text or "")):
        token = match.group(0)
        key = normalize_acronym(token)
        # Very long all-capital words are commonly slide headings rather than
        # abbreviations. Digit-bearing identifiers remain eligible.
        if (
            not key
            or key in _IGNORE
            or key in seen
            or (len(key) > 8 and not any(char.isdigit() for char in key))
        ):
            continue
        seen.add(key)
        yield token, match.start(), match.end()


def extract_acronyms(text: str) -> tuple[str, ...]:
    return tuple(token for token, _start, _end in _iter_acronyms(text))


def _context_window(text: str, start: int, end: int, limit: int = 220) -> str:
    half = max(48, int(limit) // 2)
    left = max(0, int(start) - half)
    right = min(len(text), int(end) + half)
    snippet = text[left:right].strip()
    if left:
        snippet = "…" + snippet
    if right < len(text):
        snippet += "…"
    return snippet


def preferred_translation(acronym: str, glossary: dict[str, dict]) -> str | None:
    key = normalize_acronym(acronym)
    row = glossary.get(key) or glossary.get(str(acronym))
    if not isinstance(row, dict):
        return None
    chinese = str(row.get("chinese", "") or "").strip()
    display = str(row.get("acronym", acronym) or acronym).strip()
    return f"{chinese}（{display}）" if chinese else None


class MedicalAcronymResolver:
    """Resolve one deck's abbreviations once, then reuse the glossary per slide."""

    def __init__(self, paths, llm):
        self.paths = paths
        self.llm = llm

    def _inventory_paths(self) -> tuple[Path, ...]:
        configured = os.environ.get("PHOENIX_MEDICAL_ACRONYM_CSV", "").strip()
        values = []
        if configured:
            values.append(Path(configured))
        model_root = getattr(self.paths, "model_root", None)
        runtime_root = getattr(self.paths, "runtime_root", None)
        if model_root is not None:
            values.append(
                Path(model_root) / "医学术语" / "Metainventory_Version1.0.0.csv"
            )
        if runtime_root is not None:
            values.append(
                Path(runtime_root) / "terminology" / "Metainventory_Version1.0.0.csv"
            )
        return tuple(dict.fromkeys(path for path in values if path.is_file()))

    def _llm_available(self) -> bool:
        if self.llm is None:
            return False
        try:
            return bool(self.llm.available("translation"))
        except Exception:
            return False

    def _external_candidates(
        self,
        wanted: set[str],
        *,
        max_senses: int = 8,
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        if not wanted:
            return result
        for path in self._inventory_paths():
            try:
                with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
                    for row in csv.DictReader(handle):
                        key = normalize_acronym(row.get("SF") or row.get("NormSF") or "")
                        if key not in wanted or len(result[key]) >= max_senses:
                            continue
                        long_form = str(row.get("LF") or row.get("NormLF") or "").strip()
                        if long_form and long_form.casefold() not in {
                            item.casefold() for item in result[key]
                        }:
                            result[key].append(long_form)
            except OSError:
                continue
        return result

    @staticmethod
    def _parse_json(raw: str) -> list[dict]:
        text = str(raw or "").strip()
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end < start:
            return []
        try:
            payload = json.loads(text[start:end + 1])
        except Exception:
            return []
        return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []

    def resolve(
        self,
        unit_texts: Iterable[tuple[int, str]],
        target_language: str = "中文",
    ) -> dict[str, dict]:
        contexts: dict[str, list[str]] = defaultdict(list)
        display: dict[str, str] = {}
        for unit_number, text in unit_texts:
            compact = re.sub(r"\s+", " ", str(text or "")).strip()
            for acronym, start, end in _iter_acronyms(compact):
                key = normalize_acronym(acronym)
                display.setdefault(key, acronym)
                context = (
                    f"第{int(unit_number)}页："
                    f"{_context_window(compact, start, end)}"
                )
                if context not in contexts[key] and len(contexts[key]) < 2:
                    contexts[key].append(context)

        wanted = set(contexts)
        external = self._external_candidates(wanted)
        glossary: dict[str, dict] = {}
        unresolved: list[dict] = []
        for key in sorted(wanted):
            seed = list(RADIOLOGY_SEED.get(display[key], ())) or list(RADIOLOGY_SEED.get(key, ()))
            english_candidates = [english for english, _chinese in seed]
            english_candidates.extend(external.get(key, ()))
            english_candidates = list(dict.fromkeys(english_candidates))[:8]
            if len(seed) == 1:
                english, chinese = seed[0]
                glossary[key] = {
                    "acronym": display[key],
                    "english": english,
                    "chinese": chinese,
                    "source": "phoenix_radiology_seed",
                }
                continue
            unresolved.append({
                "acronym": display[key],
                "candidates": english_candidates,
                "context": contexts[key],
            })

        # Resolve ambiguous/unknown terms in a bounded Smart2 call (or a few
        # bounded calls only for unusually vocabulary-heavy decks). The 104k
        # inventory remains local, so prompt size never grows with the database.
        if unresolved and self._llm_available():
            # Most decks fit in one call. Very large decks are split into
            # bounded 36-term calls so a vocabulary-heavy lecture cannot
            # overflow context or produce a truncated JSON response.
            for offset in range(0, len(unresolved), 36):
                chunk = unresolved[offset:offset + 36]
                payload = json.dumps(
                    chunk,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                prompt = f"""你是医学影像课件缩写消歧器。根据整份课件的页码上下文，为每个真正的医学缩写选择正确英文全称并给出规范{target_language}名称。

规则：
- candidates 非空时必须优先从候选中选择；只有全部明显不符时才自行补充。
- 同一缩写在本课件中保持一个固定含义。
- 如果条目其实是普通英文单词而非缩写，不要输出该条目。
- 不解释推理过程，不编造课件无关含义。
- 只输出JSON数组：[{{"acronym":"DWI","english":"diffusion-weighted imaging","chinese":"弥散加权成像"}}]。

课件缩写：
{payload}
"""
                try:
                    raw = self.llm.generate(
                        prompt,
                        max_new_tokens=max(560, min(3000, 78 * len(chunk) + 240)),
                        profile="translation",
                    )
                    for row in self._parse_json(raw):
                        acronym = str(row.get("acronym", "") or "").strip()
                        key = normalize_acronym(acronym)
                        english = str(row.get("english", "") or "").strip()
                        chinese = str(row.get("chinese", "") or "").strip()
                        if key in wanted and english and chinese:
                            glossary[key] = {
                                "acronym": display[key],
                                "english": english,
                                "chinese": chinese,
                                "source": "smart2_context_disambiguation",
                            }
                except Exception:
                    continue

        # Ambiguous seed entries still have a deterministic radiology-first
        # fallback if the API is temporarily unavailable. Unknown terms remain
        # for the normal slide translation prompt rather than being guessed.
        for key in wanted - set(glossary):
            seed = list(RADIOLOGY_SEED.get(display[key], ())) or list(RADIOLOGY_SEED.get(key, ()))
            if seed:
                english, chinese = seed[0]
                glossary[key] = {
                    "acronym": display[key],
                    "english": english,
                    "chinese": chinese,
                    "source": "radiology_fallback",
                }
        return glossary

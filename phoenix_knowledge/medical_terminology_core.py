from __future__ import annotations

"""Phoenix offline medical terminology core.

This is a compact, deterministic seed for translation context. It does not copy
external definitions. Preferred English/Chinese terms were curated for Phoenix
and cross-checked against official terminology resources such as RSNA RadLex,
NLM MeSH, NCI dictionaries, AHA terminology, and NIH/NIEHS abbreviation lists.

Only terms actually present in the current source segment are injected into a
model prompt, so the large core does not create a large token cost.
"""

import re
from dataclasses import dataclass

REFERENCE_SOURCES = {
    "RSNA RadLex": "https://www.rsna.org/practice-tools/data-tools-and-standards/radlex-radiology-lexicon",
    "NLM MeSH": "https://meshb.nlm.nih.gov/",
    "NCI Dictionary": "https://www.cancer.gov/publications/dictionaries/cancer-terms",
    "AHA Glossary": "https://newsroom.heart.org/policies/glossary-of-scientific-terms",
    "NIEHS NTP Acronyms": "https://ntp.niehs.nih.gov/support/acronyms",
}

_ABBREVIATION_DATA = r"""
CT|computed tomography|计算机断层成像|radiology
NCCT|non-contrast computed tomography|非增强CT|radiology
CECT|contrast-enhanced computed tomography|增强CT|radiology
CTA|computed tomography angiography|CT血管成像|radiology
CTV|computed tomography venography|CT静脉成像|radiology
CTP|computed tomography perfusion|CT灌注成像|radiology
HRCT|high-resolution computed tomography|高分辨率CT|radiology
LDCT|low-dose computed tomography|低剂量CT|radiology
DECT|dual-energy computed tomography|双能量CT|radiology
MDCT|multidetector computed tomography|多排探测器CT|radiology
CBCT|cone-beam computed tomography|锥形束CT|radiology
MRI|magnetic resonance imaging|磁共振成像|radiology
MR|magnetic resonance|磁共振|radiology
MR|mitral regurgitation|二尖瓣反流|cardiology
MRA|magnetic resonance angiography|磁共振血管成像|radiology
MRV|magnetic resonance venography|磁共振静脉成像|radiology
MRCP|magnetic resonance cholangiopancreatography|磁共振胰胆管成像|radiology
MRS|magnetic resonance spectroscopy|磁共振波谱|radiology
DWI|diffusion-weighted imaging|弥散加权成像|radiology
ADC|apparent diffusion coefficient|表观弥散系数|radiology
DTI|diffusion tensor imaging|弥散张量成像|radiology
DKI|diffusion kurtosis imaging|弥散峰度成像|radiology
SWI|susceptibility-weighted imaging|磁敏感加权成像|radiology
T1WI|T1-weighted imaging|T1加权成像|radiology
T2WI|T2-weighted imaging|T2加权成像|radiology
FLAIR|fluid-attenuated inversion recovery|液体衰减反转恢复序列|radiology
STIR|short tau inversion recovery|短时间反转恢复序列|radiology
GRE|gradient-recalled echo|梯度回波序列|radiology
FSE|fast spin echo|快速自旋回波序列|radiology
TSE|turbo spin echo|快速自旋回波序列|radiology
EPI|echo-planar imaging|回波平面成像|radiology
TOF|time of flight|时间飞跃法|radiology
ASL|arterial spin labeling|动脉自旋标记|radiology
DSC|dynamic susceptibility contrast|动态磁敏感对比增强|radiology
DCE|dynamic contrast-enhanced imaging|动态对比增强成像|radiology
IVIM|intravoxel incoherent motion|体素内不相干运动|radiology
BOLD|blood oxygen level dependent|血氧水平依赖成像|radiology
CISS|constructive interference in steady state|稳态构成干扰序列|radiology
FIESTA|fast imaging employing steady-state acquisition|稳态采集快速成像|radiology
MIP|maximum intensity projection|最大密度投影|radiology
MinIP|minimum intensity projection|最小密度投影|radiology
MPR|multiplanar reconstruction|多平面重建|radiology
CPR|curved planar reformation|曲面重建|radiology
CPR|cardiopulmonary resuscitation|心肺复苏|emergency
VR|volume rendering|容积再现|radiology
ROI|region of interest|感兴趣区|radiology
VOI|volume of interest|感兴趣容积|radiology
FOV|field of view|视野|radiology
SNR|signal-to-noise ratio|信噪比|radiology
CNR|contrast-to-noise ratio|对比噪声比|radiology
PACS|picture archiving and communication system|影像归档和通信系统|radiology
DICOM|Digital Imaging and Communications in Medicine|医学数字成像和通信标准|radiology
PET|positron emission tomography|正电子发射断层成像|nuclear_medicine
SPECT|single-photon emission computed tomography|单光子发射计算机断层成像|nuclear_medicine
FDG|fluorodeoxyglucose|氟代脱氧葡萄糖|nuclear_medicine
PSMA|prostate-specific membrane antigen|前列腺特异性膜抗原|nuclear_medicine
SUV|standardized uptake value|标准化摄取值|nuclear_medicine
SUVmax|maximum standardized uptake value|最大标准化摄取值|nuclear_medicine
SUVmean|mean standardized uptake value|平均标准化摄取值|nuclear_medicine
GGO|ground-glass opacity|磨玻璃密度影|thoracic
GGN|ground-glass nodule|磨玻璃结节|thoracic
SPN|solitary pulmonary nodule|孤立性肺结节|thoracic
ILD|interstitial lung disease|间质性肺疾病|thoracic
UIP|usual interstitial pneumonia|普通型间质性肺炎|thoracic
NSIP|nonspecific interstitial pneumonia|非特异性间质性肺炎|thoracic
IPF|idiopathic pulmonary fibrosis|特发性肺纤维化|thoracic
COPD|chronic obstructive pulmonary disease|慢性阻塞性肺疾病|respiratory
AECOPD|acute exacerbation of chronic obstructive pulmonary disease|慢性阻塞性肺疾病急性加重|respiratory
PE|pulmonary embolism|肺栓塞|thoracic
PE|pleural effusion|胸腔积液|thoracic
PTE|pulmonary thromboembolism|肺血栓栓塞症|thoracic
DVT|deep vein thrombosis|深静脉血栓形成|vascular
CTEPH|chronic thromboembolic pulmonary hypertension|慢性血栓栓塞性肺动脉高压|cardiology
PA|pulmonary artery|肺动脉|cardiology
PA|posteroanterior|后前位|radiology
AP|anteroposterior|前后位|radiology
AP|alkaline phosphatase|碱性磷酸酶|laboratory
CSF|cerebrospinal fluid|脑脊液|neurology
SAH|subarachnoid hemorrhage|蛛网膜下腔出血|neurology
ICH|intracerebral hemorrhage|脑内出血|neurology
IVH|intraventricular hemorrhage|脑室内出血|neurology
SDH|subdural hematoma|硬膜下血肿|neurology
EDH|epidural hematoma|硬膜外血肿|neurology
AIS|acute ischemic stroke|急性缺血性卒中|neurology
TIA|transient ischemic attack|短暂性脑缺血发作|neurology
CVA|cerebrovascular accident|脑血管意外|neurology
LVO|large vessel occlusion|大血管闭塞|neurology
AVM|arteriovenous malformation|动静脉畸形|vascular
DAVF|dural arteriovenous fistula|硬脑膜动静脉瘘|neurology
CVST|cerebral venous sinus thrombosis|脑静脉窦血栓形成|neurology
PRES|posterior reversible encephalopathy syndrome|可逆性后部脑病综合征|neurology
RCVS|reversible cerebral vasoconstriction syndrome|可逆性脑血管收缩综合征|neurology
NPH|normal pressure hydrocephalus|正常压力脑积水|neurology
MS|multiple sclerosis|多发性硬化|neurology
MS|mitral stenosis|二尖瓣狭窄|cardiology
NMOSD|neuromyelitis optica spectrum disorder|视神经脊髓炎谱系疾病|neurology
ADEM|acute disseminated encephalomyelitis|急性播散性脑脊髓炎|neurology
ALS|amyotrophic lateral sclerosis|肌萎缩侧索硬化|neurology
MG|myasthenia gravis|重症肌无力|neurology
GBS|Guillain-Barré syndrome|吉兰-巴雷综合征|neurology
CIDP|chronic inflammatory demyelinating polyneuropathy|慢性炎性脱髓鞘性多发性神经病|neurology
PD|Parkinson disease|帕金森病|neurology
PD|progressive disease|疾病进展|oncology
AD|Alzheimer disease|阿尔茨海默病|neurology
MCI|mild cognitive impairment|轻度认知障碍|neurology
MCA|middle cerebral artery|大脑中动脉|neurology
ACA|anterior cerebral artery|大脑前动脉|neurology
PCA|posterior cerebral artery|大脑后动脉|neurology
ICA|internal carotid artery|颈内动脉|vascular
ECA|external carotid artery|颈外动脉|vascular
VA|vertebral artery|椎动脉|vascular
BA|basilar artery|基底动脉|vascular
ACom|anterior communicating artery|前交通动脉|neurology
PCom|posterior communicating artery|后交通动脉|neurology
DSA|digital subtraction angiography|数字减影血管造影|radiology
TCD|transcranial Doppler|经颅多普勒|neurology
GCS|Glasgow Coma Scale|格拉斯哥昏迷量表|emergency
NIHSS|National Institutes of Health Stroke Scale|美国国立卫生研究院卒中量表|neurology
mRS|modified Rankin Scale|改良Rankin量表|neurology
ICP|intracranial pressure|颅内压|neurology
CPP|cerebral perfusion pressure|脑灌注压|neurology
HCC|hepatocellular carcinoma|肝细胞癌|oncology
ICC|intrahepatic cholangiocarcinoma|肝内胆管癌|oncology
CCA|cholangiocarcinoma|胆管癌|oncology
RCC|renal cell carcinoma|肾细胞癌|oncology
CRC|colorectal cancer|结直肠癌|oncology
GBM|glioblastoma|胶质母细胞瘤|oncology
NSCLC|non-small cell lung cancer|非小细胞肺癌|oncology
SCLC|small cell lung cancer|小细胞肺癌|oncology
HNSCC|head and neck squamous cell carcinoma|头颈部鳞状细胞癌|oncology
SCC|squamous cell carcinoma|鳞状细胞癌|oncology
ADC|adenocarcinoma|腺癌|oncology
AML|acute myeloid leukemia|急性髓系白血病|hematology
ALL|acute lymphoblastic leukemia|急性淋巴细胞白血病|hematology
CML|chronic myeloid leukemia|慢性髓系白血病|hematology
CLL|chronic lymphocytic leukemia|慢性淋巴细胞白血病|hematology
MDS|myelodysplastic syndrome|骨髓增生异常综合征|hematology
MPN|myeloproliferative neoplasm|骨髓增殖性肿瘤|hematology
MM|multiple myeloma|多发性骨髓瘤|hematology
NHL|non-Hodgkin lymphoma|非霍奇金淋巴瘤|hematology
HL|Hodgkin lymphoma|霍奇金淋巴瘤|hematology
DLBCL|diffuse large B-cell lymphoma|弥漫大B细胞淋巴瘤|hematology
FL|follicular lymphoma|滤泡性淋巴瘤|hematology
MCL|mantle cell lymphoma|套细胞淋巴瘤|hematology
MCL|medial collateral ligament|内侧副韧带|musculoskeletal
ALCL|anaplastic large cell lymphoma|间变性大细胞淋巴瘤|hematology
CAR-T|chimeric antigen receptor T-cell therapy|嵌合抗原受体T细胞治疗|oncology
HSCT|hematopoietic stem cell transplantation|造血干细胞移植|hematology
BMT|bone marrow transplantation|骨髓移植|hematology
GVHD|graft-versus-host disease|移植物抗宿主病|hematology
CR|complete response|完全缓解|oncology
PR|partial response|部分缓解|oncology
SD|stable disease|疾病稳定|oncology
SD|standard deviation|标准差|statistics
ORR|objective response rate|客观缓解率|oncology
DCR|disease control rate|疾病控制率|oncology
PFS|progression-free survival|无进展生存期|oncology
OS|overall survival|总生存期|oncology
DFS|disease-free survival|无病生存期|oncology
RFS|recurrence-free survival|无复发生存期|oncology
EFS|event-free survival|无事件生存期|oncology
pCR|pathologic complete response|病理完全缓解|oncology
MRD|minimal residual disease|微小残留病|hematology
TNM|tumor-node-metastasis staging|TNM分期|oncology
EGFR|epidermal growth factor receptor|表皮生长因子受体|oncology
ALK|anaplastic lymphoma kinase|间变性淋巴瘤激酶|oncology
ROS1|ROS proto-oncogene 1|ROS1原癌基因|oncology
KRAS|KRAS proto-oncogene|KRAS原癌基因|oncology
BRAF|B-Raf proto-oncogene|BRAF原癌基因|oncology
HER2|human epidermal growth factor receptor 2|人表皮生长因子受体2|oncology
ER|estrogen receptor|雌激素受体|oncology
PR|progesterone receptor|孕激素受体|oncology
PD-L1|programmed death-ligand 1|程序性死亡配体1|oncology
MSI|microsatellite instability|微卫星不稳定|oncology
MSS|microsatellite stable|微卫星稳定|oncology
TMB|tumor mutational burden|肿瘤突变负荷|oncology
NGS|next-generation sequencing|二代测序|genetics
ctDNA|circulating tumor DNA|循环肿瘤DNA|oncology
cfDNA|cell-free DNA|游离DNA|genetics
IHC|immunohistochemistry|免疫组织化学|pathology
FISH|fluorescence in situ hybridization|荧光原位杂交|pathology
ISH|in situ hybridization|原位杂交|pathology
PCR|polymerase chain reaction|聚合酶链式反应|laboratory
RT-PCR|reverse transcription polymerase chain reaction|逆转录聚合酶链式反应|laboratory
qPCR|quantitative polymerase chain reaction|实时定量聚合酶链式反应|laboratory
FFPE|formalin-fixed paraffin-embedded|福尔马林固定石蜡包埋|pathology
FNA|fine-needle aspiration|细针穿刺|pathology
FNAC|fine-needle aspiration cytology|细针穿刺细胞学|pathology
CNB|core needle biopsy|粗针穿刺活检|pathology
H&E|hematoxylin and eosin|苏木精-伊红染色|pathology
Ki-67|Ki-67 proliferation index|Ki-67增殖指数|pathology
HF|heart failure|心力衰竭|cardiology
HFrEF|heart failure with reduced ejection fraction|射血分数降低型心力衰竭|cardiology
HFmrEF|heart failure with mildly reduced ejection fraction|射血分数轻度降低型心力衰竭|cardiology
HFpEF|heart failure with preserved ejection fraction|射血分数保留型心力衰竭|cardiology
LVEF|left ventricular ejection fraction|左心室射血分数|cardiology
RVEF|right ventricular ejection fraction|右心室射血分数|cardiology
LV|left ventricle|左心室|cardiology
RV|right ventricle|右心室|cardiology
LA|left atrium|左心房|cardiology
RA|right atrium|右心房|cardiology
LVH|left ventricular hypertrophy|左心室肥厚|cardiology
RVH|right ventricular hypertrophy|右心室肥厚|cardiology
LVEDV|left ventricular end-diastolic volume|左心室舒张末期容积|cardiology
LVESV|left ventricular end-systolic volume|左心室收缩末期容积|cardiology
SV|stroke volume|每搏量|cardiology
CO|cardiac output|心输出量|cardiology
CI|cardiac index|心脏指数|cardiology
CI|confidence interval|置信区间|statistics
MAP|mean arterial pressure|平均动脉压|critical_care
SBP|systolic blood pressure|收缩压|cardiology
DBP|diastolic blood pressure|舒张压|cardiology
BP|blood pressure|血压|cardiology
HTN|hypertension|高血压|cardiology
CAD|coronary artery disease|冠状动脉疾病|cardiology
CAD|computer-aided detection|计算机辅助检测|radiology
CHD|coronary heart disease|冠心病|cardiology
IHD|ischemic heart disease|缺血性心脏病|cardiology
ACS|acute coronary syndrome|急性冠状动脉综合征|cardiology
STEMI|ST-elevation myocardial infarction|ST段抬高型心肌梗死|cardiology
NSTEMI|non-ST-elevation myocardial infarction|非ST段抬高型心肌梗死|cardiology
MI|myocardial infarction|心肌梗死|cardiology
AMI|acute myocardial infarction|急性心肌梗死|cardiology
PCI|percutaneous coronary intervention|经皮冠状动脉介入治疗|cardiology
CABG|coronary artery bypass grafting|冠状动脉旁路移植术|cardiology
DAPT|dual antiplatelet therapy|双联抗血小板治疗|cardiology
OAC|oral anticoagulation|口服抗凝治疗|cardiology
DOAC|direct oral anticoagulant|直接口服抗凝药|cardiology
VKA|vitamin K antagonist|维生素K拮抗剂|cardiology
AF|atrial fibrillation|心房颤动|cardiology
AFL|atrial flutter|心房扑动|cardiology
SVT|supraventricular tachycardia|室上性心动过速|cardiology
VT|ventricular tachycardia|室性心动过速|cardiology
VF|ventricular fibrillation|心室颤动|cardiology
PVC|premature ventricular contraction|室性期前收缩|cardiology
PAC|premature atrial contraction|房性期前收缩|cardiology
AVB|atrioventricular block|房室传导阻滞|cardiology
LBBB|left bundle branch block|左束支传导阻滞|cardiology
RBBB|right bundle branch block|右束支传导阻滞|cardiology
WPW|Wolff-Parkinson-White syndrome|预激综合征|cardiology
QTc|corrected QT interval|校正QT间期|cardiology
BNP|B-type natriuretic peptide|B型利钠肽|cardiology
NT-proBNP|N-terminal pro-B-type natriuretic peptide|N末端B型利钠肽原|cardiology
hs-cTnI|high-sensitivity cardiac troponin I|高敏心肌肌钙蛋白I|cardiology
hs-cTnT|high-sensitivity cardiac troponin T|高敏心肌肌钙蛋白T|cardiology
CK-MB|creatine kinase-MB|肌酸激酶同工酶MB|cardiology
CCTA|coronary CT angiography|冠状动脉CT血管成像|cardiology
CAC|coronary artery calcium|冠状动脉钙化|cardiology
FFR|fractional flow reserve|血流储备分数|cardiology
FFRCT|CT-derived fractional flow reserve|CT血流储备分数|cardiology
iFR|instantaneous wave-free ratio|瞬时无波形比值|cardiology
GLS|global longitudinal strain|整体纵向应变|cardiology
TAPSE|tricuspid annular plane systolic excursion|三尖瓣环平面收缩期位移|cardiology
LVOT|left ventricular outflow tract|左心室流出道|cardiology
RVOT|right ventricular outflow tract|右心室流出道|cardiology
ASD|atrial septal defect|房间隔缺损|cardiology
VSD|ventricular septal defect|室间隔缺损|cardiology
PDA|patent ductus arteriosus|动脉导管未闭|cardiology
PFO|patent foramen ovale|卵圆孔未闭|cardiology
TAVR|transcatheter aortic valve replacement|经导管主动脉瓣置换术|cardiology
TAVI|transcatheter aortic valve implantation|经导管主动脉瓣植入术|cardiology
SAVR|surgical aortic valve replacement|外科主动脉瓣置换术|cardiology
AS|aortic stenosis|主动脉瓣狭窄|cardiology
AR|aortic regurgitation|主动脉瓣反流|cardiology
TR|tricuspid regurgitation|三尖瓣反流|cardiology
PH|pulmonary hypertension|肺动脉高压|cardiology
PAH|pulmonary arterial hypertension|肺动脉性高压|cardiology
PVR|pulmonary vascular resistance|肺血管阻力|cardiology
ARDS|acute respiratory distress syndrome|急性呼吸窘迫综合征|critical_care
CAP|community-acquired pneumonia|社区获得性肺炎|respiratory
HAP|hospital-acquired pneumonia|医院获得性肺炎|respiratory
VAP|ventilator-associated pneumonia|呼吸机相关性肺炎|critical_care
TB|tuberculosis|结核病|infectious
MDR-TB|multidrug-resistant tuberculosis|耐多药结核病|infectious
XDR-TB|extensively drug-resistant tuberculosis|广泛耐药结核病|infectious
OSA|obstructive sleep apnea|阻塞性睡眠呼吸暂停|respiratory
CPAP|continuous positive airway pressure|持续气道正压通气|respiratory
BiPAP|bilevel positive airway pressure|双水平气道正压通气|respiratory
NIV|noninvasive ventilation|无创通气|critical_care
HFNC|high-flow nasal cannula|高流量鼻导管氧疗|critical_care
FiO2|fraction of inspired oxygen|吸入氧浓度|critical_care
SpO2|peripheral oxygen saturation|外周血氧饱和度|critical_care
PaO2|arterial oxygen partial pressure|动脉血氧分压|critical_care
PaCO2|arterial carbon dioxide partial pressure|动脉血二氧化碳分压|critical_care
SaO2|arterial oxygen saturation|动脉血氧饱和度|critical_care
ABG|arterial blood gas|动脉血气分析|critical_care
VBG|venous blood gas|静脉血气分析|critical_care
PEEP|positive end-expiratory pressure|呼气末正压|critical_care
Vt|tidal volume|潮气量|critical_care
RR|respiratory rate|呼吸频率|critical_care
DLCO|diffusing capacity for carbon monoxide|一氧化碳弥散量|respiratory
FEV1|forced expiratory volume in one second|第一秒用力呼气容积|respiratory
FVC|forced vital capacity|用力肺活量|respiratory
TLC|total lung capacity|肺总量|respiratory
GERD|gastroesophageal reflux disease|胃食管反流病|gastroenterology
PUD|peptic ulcer disease|消化性溃疡病|gastroenterology
IBD|inflammatory bowel disease|炎症性肠病|gastroenterology
UC|ulcerative colitis|溃疡性结肠炎|gastroenterology
CD|Crohn disease|克罗恩病|gastroenterology
IBS|irritable bowel syndrome|肠易激综合征|gastroenterology
GIB|gastrointestinal bleeding|消化道出血|gastroenterology
UGIB|upper gastrointestinal bleeding|上消化道出血|gastroenterology
LGIB|lower gastrointestinal bleeding|下消化道出血|gastroenterology
EGD|esophagogastroduodenoscopy|食管胃十二指肠镜检查|gastroenterology
ERCP|endoscopic retrograde cholangiopancreatography|内镜逆行胰胆管造影|gastroenterology
EUS|endoscopic ultrasonography|超声内镜|gastroenterology
CBD|common bile duct|胆总管|gastroenterology
IHBD|intrahepatic bile duct|肝内胆管|gastroenterology
EHBD|extrahepatic bile duct|肝外胆管|gastroenterology
NAFLD|nonalcoholic fatty liver disease|非酒精性脂肪性肝病|hepatology
MASLD|metabolic dysfunction-associated steatotic liver disease|代谢功能障碍相关脂肪性肝病|hepatology
MASH|metabolic dysfunction-associated steatohepatitis|代谢功能障碍相关脂肪性肝炎|hepatology
ALD|alcohol-associated liver disease|酒精相关性肝病|hepatology
HBV|hepatitis B virus|乙型肝炎病毒|infectious
HCV|hepatitis C virus|丙型肝炎病毒|infectious
AFP|alpha-fetoprotein|甲胎蛋白|oncology
CEA|carcinoembryonic antigen|癌胚抗原|oncology
CA19-9|carbohydrate antigen 19-9|糖类抗原19-9|oncology
ALT|alanine aminotransferase|丙氨酸氨基转移酶|laboratory
AST|aspartate aminotransferase|天冬氨酸氨基转移酶|laboratory
ALP|alkaline phosphatase|碱性磷酸酶|laboratory
GGT|gamma-glutamyl transferase|γ-谷氨酰转移酶|laboratory
TBil|total bilirubin|总胆红素|laboratory
DBil|direct bilirubin|直接胆红素|laboratory
MELD|Model for End-Stage Liver Disease|终末期肝病模型评分|hepatology
PVT|portal vein thrombosis|门静脉血栓形成|hepatology
TIPS|transjugular intrahepatic portosystemic shunt|经颈静脉肝内门体分流术|hepatology
PSC|primary sclerosing cholangitis|原发性硬化性胆管炎|hepatology
PBC|primary biliary cholangitis|原发性胆汁性胆管炎|hepatology
AKI|acute kidney injury|急性肾损伤|nephrology
CKD|chronic kidney disease|慢性肾脏病|nephrology
ESRD|end-stage renal disease|终末期肾病|nephrology
eGFR|estimated glomerular filtration rate|估算肾小球滤过率|nephrology
Cr|creatinine|肌酐|laboratory
BUN|blood urea nitrogen|血尿素氮|laboratory
UACR|urine albumin-to-creatinine ratio|尿白蛋白/肌酐比值|nephrology
UPCR|urine protein-to-creatinine ratio|尿蛋白/肌酐比值|nephrology
UTI|urinary tract infection|尿路感染|urology
LUTS|lower urinary tract symptoms|下尿路症状|urology
BPH|benign prostatic hyperplasia|良性前列腺增生|urology
PSA|prostate-specific antigen|前列腺特异性抗原|urology
PCa|prostate cancer|前列腺癌|oncology
VUR|vesicoureteral reflux|膀胱输尿管反流|urology
UPJ|ureteropelvic junction|输尿管肾盂连接部|urology
UVJ|ureterovesical junction|输尿管膀胱连接部|urology
PCNL|percutaneous nephrolithotomy|经皮肾镜取石术|urology
ESWL|extracorporeal shock wave lithotripsy|体外冲击波碎石术|urology
TURP|transurethral resection of the prostate|经尿道前列腺切除术|urology
TURBT|transurethral resection of bladder tumor|经尿道膀胱肿瘤切除术|urology
RRT|renal replacement therapy|肾脏替代治疗|nephrology
RRT|rapid response team|快速反应团队|critical_care
HD|hemodialysis|血液透析|nephrology
PD|peritoneal dialysis|腹膜透析|nephrology
DM|diabetes mellitus|糖尿病|endocrinology
T1DM|type 1 diabetes mellitus|1型糖尿病|endocrinology
T2DM|type 2 diabetes mellitus|2型糖尿病|endocrinology
DKA|diabetic ketoacidosis|糖尿病酮症酸中毒|endocrinology
HHS|hyperosmolar hyperglycemic state|高渗高血糖状态|endocrinology
HbA1c|glycated hemoglobin A1c|糖化血红蛋白A1c|endocrinology
FPG|fasting plasma glucose|空腹血糖|endocrinology
OGTT|oral glucose tolerance test|口服葡萄糖耐量试验|endocrinology
TSH|thyroid-stimulating hormone|促甲状腺激素|endocrinology
FT4|free thyroxine|游离甲状腺素|endocrinology
FT3|free triiodothyronine|游离三碘甲状腺原氨酸|endocrinology
PTH|parathyroid hormone|甲状旁腺激素|endocrinology
BMI|body mass index|体质指数|general
BMD|bone mineral density|骨密度|musculoskeletal
DXA|dual-energy X-ray absorptiometry|双能X线吸收法|musculoskeletal
DEXA|dual-energy X-ray absorptiometry|双能X线吸收法|musculoskeletal
PCOS|polycystic ovary syndrome|多囊卵巢综合征|gynecology
CBC|complete blood count|全血细胞计数|laboratory
WBC|white blood cell count|白细胞计数|laboratory
RBC|red blood cell count|红细胞计数|laboratory
Hb|hemoglobin|血红蛋白|laboratory
Hgb|hemoglobin|血红蛋白|laboratory
Hct|hematocrit|血细胞比容|laboratory
Plt|platelet count|血小板计数|laboratory
ANC|absolute neutrophil count|中性粒细胞绝对计数|laboratory
ALC|absolute lymphocyte count|淋巴细胞绝对计数|laboratory
MCV|mean corpuscular volume|平均红细胞体积|laboratory
MCH|mean corpuscular hemoglobin|平均红细胞血红蛋白量|laboratory
MCHC|mean corpuscular hemoglobin concentration|平均红细胞血红蛋白浓度|laboratory
RDW|red cell distribution width|红细胞分布宽度|laboratory
PT|prothrombin time|凝血酶原时间|laboratory
aPTT|activated partial thromboplastin time|活化部分凝血活酶时间|laboratory
INR|international normalized ratio|国际标准化比值|laboratory
D-dimer|D-dimer|D-二聚体|laboratory
FDP|fibrin degradation products|纤维蛋白降解产物|laboratory
Fbg|fibrinogen|纤维蛋白原|laboratory
VTE|venous thromboembolism|静脉血栓栓塞症|vascular
DIC|disseminated intravascular coagulation|弥散性血管内凝血|hematology
HIT|heparin-induced thrombocytopenia|肝素诱导的血小板减少症|hematology
ITP|immune thrombocytopenia|免疫性血小板减少症|hematology
TTP|thrombotic thrombocytopenic purpura|血栓性血小板减少性紫癜|hematology
HUS|hemolytic uremic syndrome|溶血尿毒综合征|hematology
CRP|C-reactive protein|C反应蛋白|laboratory
PCT|procalcitonin|降钙素原|laboratory
ESR|erythrocyte sedimentation rate|红细胞沉降率|laboratory
LDH|lactate dehydrogenase|乳酸脱氢酶|laboratory
CK|creatine kinase|肌酸激酶|laboratory
HIV|human immunodeficiency virus|人类免疫缺陷病毒|infectious
AIDS|acquired immunodeficiency syndrome|获得性免疫缺陷综合征|infectious
HSV|herpes simplex virus|单纯疱疹病毒|infectious
VZV|varicella-zoster virus|水痘-带状疱疹病毒|infectious
CMV|cytomegalovirus|巨细胞病毒|infectious
EBV|Epstein-Barr virus|EB病毒|infectious
HPV|human papillomavirus|人乳头瘤病毒|infectious
SARS-CoV-2|severe acute respiratory syndrome coronavirus 2|严重急性呼吸综合征冠状病毒2|infectious
COVID-19|coronavirus disease 2019|2019冠状病毒病|infectious
RSV|respiratory syncytial virus|呼吸道合胞病毒|infectious
MRSA|methicillin-resistant Staphylococcus aureus|耐甲氧西林金黄色葡萄球菌|infectious
VRE|vancomycin-resistant enterococci|耐万古霉素肠球菌|infectious
CRE|carbapenem-resistant Enterobacterales|耐碳青霉烯肠杆菌目细菌|infectious
ESBL|extended-spectrum beta-lactamase|超广谱β-内酰胺酶|infectious
MDR|multidrug resistant|多重耐药|infectious
XDR|extensively drug resistant|广泛耐药|infectious
PDR|pandrug resistant|全耐药|infectious
NAAT|nucleic acid amplification test|核酸扩增检测|laboratory
OB|obstetrics|产科|obstetrics
GYN|gynecology|妇科|gynecology
GA|gestational age|孕周|obstetrics
EDD|estimated date of delivery|预产期|obstetrics
LMP|last menstrual period|末次月经|obstetrics
hCG|human chorionic gonadotropin|人绒毛膜促性腺激素|obstetrics
β-hCG|beta human chorionic gonadotropin|β-人绒毛膜促性腺激素|obstetrics
FHR|fetal heart rate|胎心率|obstetrics
NST|nonstress test|无应激试验|obstetrics
BPP|biophysical profile|胎儿生物物理评分|obstetrics
FGR|fetal growth restriction|胎儿生长受限|obstetrics
IUGR|intrauterine growth restriction|宫内生长受限|obstetrics
SGA|small for gestational age|小于胎龄儿|obstetrics
LGA|large for gestational age|大于胎龄儿|obstetrics
PROM|prelabor rupture of membranes|临产前胎膜破裂|obstetrics
PPROM|preterm prelabor rupture of membranes|未足月临产前胎膜破裂|obstetrics
GDM|gestational diabetes mellitus|妊娠期糖尿病|obstetrics
HELLP|hemolysis elevated liver enzymes low platelet count syndrome|HELLP综合征|obstetrics
PID|pelvic inflammatory disease|盆腔炎性疾病|gynecology
IUD|intrauterine device|宫内节育器|gynecology
IVF|in vitro fertilization|体外受精|reproductive
ICSI|intracytoplasmic sperm injection|卵胞浆内单精子注射|reproductive
AMH|anti-Müllerian hormone|抗缪勒管激素|reproductive
AFC|antral follicle count|窦卵泡计数|reproductive
TVUS|transvaginal ultrasonography|经阴道超声|gynecology
TAUS|transabdominal ultrasonography|经腹超声|gynecology
ACL|anterior cruciate ligament|前交叉韧带|musculoskeletal
PCL|posterior cruciate ligament|后交叉韧带|musculoskeletal
LCL|lateral collateral ligament|外侧副韧带|musculoskeletal
TFCC|triangular fibrocartilage complex|三角纤维软骨复合体|musculoskeletal
OA|osteoarthritis|骨关节炎|rheumatology
RA|rheumatoid arthritis|类风湿关节炎|rheumatology
AS|ankylosing spondylitis|强直性脊柱炎|rheumatology
PsA|psoriatic arthritis|银屑病关节炎|rheumatology
SLE|systemic lupus erythematosus|系统性红斑狼疮|rheumatology
JIA|juvenile idiopathic arthritis|幼年特发性关节炎|rheumatology
CPPD|calcium pyrophosphate deposition disease|焦磷酸钙沉积病|rheumatology
ECMO|extracorporeal membrane oxygenation|体外膜肺氧合|critical_care
VA-ECMO|venoarterial extracorporeal membrane oxygenation|静脉-动脉体外膜肺氧合|critical_care
VV-ECMO|venovenous extracorporeal membrane oxygenation|静脉-静脉体外膜肺氧合|critical_care
IABP|intra-aortic balloon pump|主动脉内球囊反搏|critical_care
CRRT|continuous renal replacement therapy|连续性肾脏替代治疗|critical_care
CVVH|continuous venovenous hemofiltration|连续性静脉-静脉血液滤过|critical_care
CVVHD|continuous venovenous hemodialysis|连续性静脉-静脉血液透析|critical_care
CVVHDF|continuous venovenous hemodiafiltration|连续性静脉-静脉血液透析滤过|critical_care
SOFA|Sequential Organ Failure Assessment|序贯器官衰竭评估|critical_care
qSOFA|quick Sequential Organ Failure Assessment|快速序贯器官衰竭评估|critical_care
APACHE II|Acute Physiology and Chronic Health Evaluation II|急性生理与慢性健康状况评分II|critical_care
NEWS2|National Early Warning Score 2|国家早期预警评分2|critical_care
SIRS|systemic inflammatory response syndrome|全身炎症反应综合征|critical_care
MODS|multiple organ dysfunction syndrome|多器官功能障碍综合征|critical_care
ROSC|return of spontaneous circulation|自主循环恢复|emergency
ACLS|advanced cardiovascular life support|高级心血管生命支持|emergency
BLS|basic life support|基础生命支持|emergency
PALS|pediatric advanced life support|儿童高级生命支持|emergency
RT|radiation therapy|放射治疗|radiotherapy
EBRT|external beam radiation therapy|外照射放射治疗|radiotherapy
IMRT|intensity-modulated radiation therapy|调强放射治疗|radiotherapy
VMAT|volumetric modulated arc therapy|容积调强旋转放射治疗|radiotherapy
SBRT|stereotactic body radiation therapy|立体定向体部放射治疗|radiotherapy
SRS|stereotactic radiosurgery|立体定向放射外科|radiotherapy
HDR|high-dose-rate brachytherapy|高剂量率近距离治疗|radiotherapy
LDR|low-dose-rate brachytherapy|低剂量率近距离治疗|radiotherapy
GTV|gross tumor volume|大体肿瘤体积|radiotherapy
CTV|clinical target volume|临床靶体积|radiotherapy
PTV|planning target volume|计划靶体积|radiotherapy
OAR|organ at risk|危及器官|radiotherapy
BED|biologically effective dose|生物有效剂量|radiotherapy
EQD2|equivalent dose in 2 Gy fractions|2 Gy分次等效剂量|radiotherapy
RCT|randomized controlled trial|随机对照试验|research
RR|relative risk|相对危险度|statistics
OR|odds ratio|比值比|statistics
OR|operating room|手术室|surgery
HR|hazard ratio|风险比|statistics
HR|heart rate|心率|cardiology
SE|standard error|标准误|statistics
SEM|standard error of the mean|均值标准误|statistics
IQR|interquartile range|四分位距|statistics
ROC|receiver operating characteristic|受试者工作特征曲线|statistics
AUC|area under the curve|曲线下面积|statistics
PPV|positive predictive value|阳性预测值|statistics
NPV|negative predictive value|阴性预测值|statistics
NNT|number needed to treat|需治疗人数|statistics
NNH|number needed to harm|需伤害人数|statistics
ITT|intention-to-treat|意向性治疗分析|research
PRO|patient-reported outcome|患者报告结局|research
PROM|patient-reported outcome measure|患者报告结局测量|research
MCID|minimal clinically important difference|最小临床重要差异|research
LGE|late gadolinium enhancement|钆延迟强化|cardiac_mri
ECV|extracellular volume fraction|细胞外容积分数|cardiac_mri
CMR|cardiovascular magnetic resonance|心血管磁共振|cardiology
ECG|electrocardiography|心电图|cardiology
EKG|electrocardiography|心电图|cardiology
EEG|electroencephalography|脑电图|neurology
EMG|electromyography|肌电图|neurology
TTE|transthoracic echocardiography|经胸超声心动图|cardiology
TEE|transesophageal echocardiography|经食管超声心动图|cardiology
pH|potential of hydrogen|pH值|laboratory
HCO3|bicarbonate|碳酸氢根|laboratory
BE|base excess|碱剩余|laboratory
AG|anion gap|阴离子间隙|laboratory
Na|sodium|钠|laboratory
K|potassium|钾|laboratory
Cl|chloride|氯|laboratory
Ca|calcium|钙|laboratory
Mg|magnesium|镁|laboratory
Phos|phosphate|磷酸盐|laboratory
ECG|electrocardiogram|心电图|cardiology
""".strip()

_PHRASE_DATA = r"""
ground-glass opacity|磨玻璃密度影|thoracic
ground glass opacity|磨玻璃密度影|thoracic
ground-glass nodule|磨玻璃结节|thoracic
tree-in-bud|树芽征|thoracic
tree in bud|树芽征|thoracic
air bronchogram|空气支气管征|thoracic
crazy paving|铺路石征|thoracic
honeycombing|蜂窝影|thoracic
traction bronchiectasis|牵拉性支气管扩张|thoracic
bronchial wall thickening|支气管壁增厚|thoracic
mosaic attenuation|马赛克衰减|thoracic
air trapping|空气潴留|thoracic
halo sign|晕征|thoracic
reverse halo sign|反晕征|thoracic
pleural effusion|胸腔积液|thoracic
pleural thickening|胸膜增厚|thoracic
pleural plaque|胸膜斑|thoracic
pneumothorax|气胸|thoracic
hydropneumothorax|液气胸|thoracic
consolidation|实变|thoracic
atelectasis|肺不张|thoracic
pulmonary edema|肺水肿|thoracic
pulmonary embolism|肺栓塞|thoracic
pulmonary infarction|肺梗死|thoracic
pulmonary nodule|肺结节|thoracic
solitary pulmonary nodule|孤立性肺结节|thoracic
lung mass|肺肿块|thoracic
cavitary lesion|空洞性病变|thoracic
centrilobular nodule|小叶中心结节|thoracic
interlobular septal thickening|小叶间隔增厚|thoracic
intralobular lines|小叶内线影|thoracic
peribronchovascular thickening|支气管血管束周围增厚|thoracic
restricted diffusion|弥散受限|neuroradiology
diffusion restriction|弥散受限|neuroradiology
vasogenic edema|血管源性水肿|neuroradiology
cytotoxic edema|细胞毒性水肿|neuroradiology
mass effect|占位效应|neuroradiology
midline shift|中线移位|neuroradiology
brain herniation|脑疝|neuroradiology
ring enhancement|环形强化|neuroradiology
nodular enhancement|结节样强化|neuroradiology
leptomeningeal enhancement|软脑膜强化|neuroradiology
pachymeningeal enhancement|硬脑膜强化|neuroradiology
ependymal enhancement|室管膜强化|neuroradiology
flow void|流空效应|neuroradiology
susceptibility artifact|磁敏感伪影|neuroradiology
microbleed|微出血|neuroradiology
cerebral microbleed|脑微出血|neuroradiology
white matter hyperintensity|白质高信号|neuroradiology
lacunar infarct|腔隙性梗死|neuroradiology
acute infarction|急性梗死|neuroradiology
chronic infarction|慢性梗死|neuroradiology
hemorrhagic transformation|出血性转化|neuroradiology
subarachnoid hemorrhage|蛛网膜下腔出血|neuroradiology
intraparenchymal hemorrhage|脑实质内出血|neuroradiology
intraventricular hemorrhage|脑室内出血|neuroradiology
subdural hematoma|硬膜下血肿|neuroradiology
epidural hematoma|硬膜外血肿|neuroradiology
cerebral venous sinus thrombosis|脑静脉窦血栓形成|neuroradiology
large vessel occlusion|大血管闭塞|neuroradiology
perfusion deficit|灌注缺损|neuroradiology
penumbra|缺血半暗带|neuroradiology
core infarct|梗死核心|neuroradiology
bone marrow edema|骨髓水肿|musculoskeletal
joint effusion|关节积液|musculoskeletal
soft tissue swelling|软组织肿胀|musculoskeletal
soft-tissue swelling|软组织肿胀|musculoskeletal
periosteal reaction|骨膜反应|musculoskeletal
cortical destruction|骨皮质破坏|musculoskeletal
cortical breakthrough|骨皮质突破|musculoskeletal
lytic lesion|溶骨性病变|musculoskeletal
osteolytic lesion|溶骨性病变|musculoskeletal
sclerotic lesion|硬化性病变|musculoskeletal
osteoblastic lesion|成骨性病变|musculoskeletal
pathologic fracture|病理性骨折|musculoskeletal
stress fracture|应力性骨折|musculoskeletal
insufficiency fracture|骨质疏松性骨折|musculoskeletal
compression fracture|压缩性骨折|musculoskeletal
avulsion fracture|撕脱骨折|musculoskeletal
occult fracture|隐匿性骨折|musculoskeletal
dislocation|脱位|musculoskeletal
subluxation|半脱位|musculoskeletal
degenerative change|退行性改变|musculoskeletal
degenerative changes|退行性改变|musculoskeletal
disc bulge|椎间盘膨出|spine
disc herniation|椎间盘突出|spine
disc extrusion|椎间盘脱出|spine
disc sequestration|游离型椎间盘突出|spine
spinal canal stenosis|椎管狭窄|spine
neural foraminal stenosis|神经孔狭窄|spine
lateral recess stenosis|侧隐窝狭窄|spine
cord compression|脊髓受压|spine
myelomalacia|脊髓软化|spine
spondylolisthesis|脊椎滑脱|spine
spondylosis|脊椎退行性变|spine
facet arthropathy|小关节病变|spine
endplate change|终板改变|spine
Modic change|Modic改变|spine
fat stranding|脂肪间隙条索影|abdominal
perinephric stranding|肾周脂肪间隙条索影|abdominal
peripancreatic stranding|胰周脂肪间隙条索影|abdominal
hydronephrosis|肾积水|urology
hydroureter|输尿管积水|urology
urolithiasis|尿路结石|urology
nephrolithiasis|肾结石|urology
ureterolithiasis|输尿管结石|urology
cholelithiasis|胆囊结石|abdominal
choledocholithiasis|胆总管结石|abdominal
cholecystitis|胆囊炎|abdominal
appendicitis|阑尾炎|abdominal
diverticulitis|憩室炎|abdominal
bowel obstruction|肠梗阻|abdominal
small bowel obstruction|小肠梗阻|abdominal
large bowel obstruction|大肠梗阻|abdominal
free air|游离气体|abdominal
free intraperitoneal air|腹腔游离气体|abdominal
pneumoperitoneum|气腹|abdominal
ascites|腹水|abdominal
lymphadenopathy|淋巴结肿大|abdominal
hepatomegaly|肝大|abdominal
splenomegaly|脾大|abdominal
portal hypertension|门静脉高压|hepatology
portal vein thrombosis|门静脉血栓形成|hepatology
biliary dilatation|胆管扩张|abdominal
bile duct dilatation|胆管扩张|abdominal
pancreatic duct dilatation|胰管扩张|abdominal
wall thickening|壁增厚|general_imaging
mural thickening|壁增厚|general_imaging
mucosal enhancement|黏膜强化|abdominal
active extravasation|活动性造影剂外渗|vascular
contrast extravasation|造影剂外渗|vascular
arterial stenosis|动脉狭窄|vascular
arterial occlusion|动脉闭塞|vascular
venous thrombosis|静脉血栓形成|vascular
aneurysm|动脉瘤|vascular
pseudoaneurysm|假性动脉瘤|vascular
dissection|夹层|vascular
intramural hematoma|壁内血肿|vascular
penetrating atherosclerotic ulcer|穿透性动脉粥样硬化性溃疡|vascular
endoleak|内漏|vascular
stenosis|狭窄|general
occlusion|闭塞|general
thrombosis|血栓形成|general
embolism|栓塞|general
infarction|梗死|general
ischemia|缺血|general
hemorrhage|出血|general
edema|水肿|general
effusion|积液|general
lesion|病变|general
nodule|结节|general
mass|肿块|general
cyst|囊肿|general
abscess|脓肿|general
necrosis|坏死|pathology
fibrosis|纤维化|pathology
calcification|钙化|general_imaging
enhancement|强化|general_imaging
nonenhancing|无强化|general_imaging
hypoenhancing|低强化|general_imaging
hyperenhancing|高强化|general_imaging
washout|廓清|general_imaging
capsule appearance|包膜样强化|general_imaging
restricted diffusion|弥散受限|general_imaging
signal intensity|信号强度|radiology
high signal intensity|高信号|radiology
low signal intensity|低信号|radiology
isointense|等信号|radiology
hyperintense|高信号|radiology
hypointense|低信号|radiology
hyperdense|高密度|radiology
hypodense|低密度|radiology
isodense|等密度|radiology
hyperattenuating|高衰减|radiology
hypoattenuating|低衰减|radiology
arterial phase|动脉期|radiology
portal venous phase|门静脉期|radiology
venous phase|静脉期|radiology
delayed phase|延迟期|radiology
noncontrast phase|平扫期|radiology
equilibrium phase|平衡期|radiology
late gadolinium enhancement|钆延迟强化|cardiac_mri
myocardial edema|心肌水肿|cardiac_mri
myocardial fibrosis|心肌纤维化|cardiac_mri
wall motion abnormality|室壁运动异常|cardiology
regional wall motion abnormality|节段性室壁运动异常|cardiology
ejection fraction|射血分数|cardiology
stroke volume|每搏量|cardiology
cardiac output|心输出量|cardiology
aortic stenosis|主动脉瓣狭窄|cardiology
aortic regurgitation|主动脉瓣反流|cardiology
mitral regurgitation|二尖瓣反流|cardiology
mitral stenosis|二尖瓣狭窄|cardiology
tricuspid regurgitation|三尖瓣反流|cardiology
pericardial effusion|心包积液|cardiology
pericardial thickening|心包增厚|cardiology
myocardial infarction|心肌梗死|cardiology
myocardial ischemia|心肌缺血|cardiology
coronary artery disease|冠状动脉疾病|cardiology
coronary artery calcification|冠状动脉钙化|cardiology
fractional flow reserve|血流储备分数|cardiology
heart failure|心力衰竭|cardiology
acute kidney injury|急性肾损伤|nephrology
chronic kidney disease|慢性肾脏病|nephrology
glomerular filtration rate|肾小球滤过率|nephrology
urinary tract infection|尿路感染|urology
benign prostatic hyperplasia|良性前列腺增生|urology
diabetes mellitus|糖尿病|endocrinology
diabetic ketoacidosis|糖尿病酮症酸中毒|endocrinology
glycated hemoglobin|糖化血红蛋白|endocrinology
thyroid nodule|甲状腺结节|endocrinology
thyroid carcinoma|甲状腺癌|oncology
lymph node|淋巴结|anatomy
lymph node metastasis|淋巴结转移|oncology
distant metastasis|远处转移|oncology
bone metastasis|骨转移|oncology
liver metastasis|肝转移|oncology
brain metastasis|脑转移|oncology
complete response|完全缓解|oncology
partial response|部分缓解|oncology
stable disease|疾病稳定|oncology
progressive disease|疾病进展|oncology
progression-free survival|无进展生存期|oncology
overall survival|总生存期|oncology
pathologic complete response|病理完全缓解|oncology
minimal residual disease|微小残留病|hematology
C-reactive protein|C反应蛋白|laboratory
procalcitonin|降钙素原|laboratory
erythrocyte sedimentation rate|红细胞沉降率|laboratory
white blood cell count|白细胞计数|laboratory
platelet count|血小板计数|laboratory
hemoglobin|血红蛋白|laboratory
hematocrit|血细胞比容|laboratory
creatinine|肌酐|laboratory
blood urea nitrogen|血尿素氮|laboratory
alanine aminotransferase|丙氨酸氨基转移酶|laboratory
aspartate aminotransferase|天冬氨酸氨基转移酶|laboratory
alkaline phosphatase|碱性磷酸酶|laboratory
gamma-glutamyl transferase|γ-谷氨酰转移酶|laboratory
total bilirubin|总胆红素|laboratory
direct bilirubin|直接胆红素|laboratory
D-dimer|D-二聚体|laboratory
fibrinogen|纤维蛋白原|laboratory
arterial blood gas|动脉血气分析|critical_care
oxygen saturation|血氧饱和度|critical_care
positive end-expiratory pressure|呼气末正压|critical_care
acute respiratory distress syndrome|急性呼吸窘迫综合征|critical_care
extracorporeal membrane oxygenation|体外膜肺氧合|critical_care
continuous renal replacement therapy|连续性肾脏替代治疗|critical_care
randomized controlled trial|随机对照试验|research
odds ratio|比值比|statistics
hazard ratio|风险比|statistics
relative risk|相对危险度|statistics
confidence interval|置信区间|statistics
standard deviation|标准差|statistics
interquartile range|四分位距|statistics
receiver operating characteristic|受试者工作特征曲线|statistics
area under the curve|曲线下面积|statistics
sensitivity|敏感度|statistics
specificity|特异度|statistics
positive predictive value|阳性预测值|statistics
negative predictive value|阴性预测值|statistics
""".strip()


@dataclass(frozen=True)
class TermSense:
    english: str
    chinese: str
    category: str


@dataclass(frozen=True)
class TermHit:
    display: str
    senses: tuple[TermSense, ...]
    kind: str
    position: int


def _parse_abbreviations() -> dict[str, tuple[TermSense, ...]]:
    buckets: dict[str, list[TermSense]] = {}
    for line in _ABBREVIATION_DATA.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, english, chinese, category = (part.strip() for part in line.split("|", 3))
        sense = TermSense(english, chinese, category)
        rows = buckets.setdefault(key, [])
        if sense not in rows:
            rows.append(sense)
    return {key: tuple(rows) for key, rows in buckets.items()}


def _parse_phrases() -> dict[str, TermSense]:
    result: dict[str, TermSense] = {}
    for line in _PHRASE_DATA.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        english, chinese, category = (part.strip() for part in line.split("|", 2))
        result.setdefault(english.casefold(), TermSense(english, chinese, category))
    return result


ABBREVIATION_SENSES = _parse_abbreviations()
PHRASE_ALIASES = _parse_phrases()

_SHORT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-zβ][A-Za-z0-9β+&./_-]{1,30})(?![A-Za-z0-9])"
)
_EXACT_SHORT = {
    key for key in ABBREVIATION_SENSES
    if len(key) <= 2 and key.replace("β", "B").isalpha()
}
_FOLDED_ABBREVIATIONS = {
    key.casefold(): key
    for key in ABBREVIATION_SENSES
    if key not in _EXACT_SHORT and " " not in key
}
_MULTIWORD_ABBREVIATIONS = tuple(
    sorted((key for key in ABBREVIATION_SENSES if " " in key), key=len, reverse=True)
)
_PHRASE_KEYS = tuple(sorted(PHRASE_ALIASES, key=len, reverse=True))


def _bounded_find(text: str, needle: str, *, case_sensitive: bool = False) -> int:
    if not needle:
        return -1
    flags = 0 if case_sensitive else re.IGNORECASE
    match = re.search(
        rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])",
        text,
        flags,
    )
    return -1 if match is None else int(match.start())


def _abbreviation_key(token: str) -> str:
    token = str(token or "").strip().rstrip(".")
    if token in _EXACT_SHORT:
        return token
    return _FOLDED_ABBREVIATIONS.get(token.casefold(), "")


def find_core_terms(text: str, limit: int = 32) -> tuple[TermHit, ...]:
    """Return terminology hits in source order, with ambiguity preserved."""

    source = str(text or "")
    if not source.strip():
        return ()

    hits: dict[tuple[str, str], TermHit] = {}

    for match in _SHORT_TOKEN_RE.finditer(source):
        raw = match.group(0)
        candidates = [raw.rstrip(".")]
        if "/" in raw:
            candidates.extend(part for part in raw.rstrip(".").split("/") if part)
        for token in candidates:
            key = _abbreviation_key(token)
            if not key:
                continue
            item = TermHit(key, ABBREVIATION_SENSES[key], "abbreviation", int(match.start()))
            hits.setdefault(("a", key.casefold()), item)

    for key in _MULTIWORD_ABBREVIATIONS:
        pos = _bounded_find(source, key)
        if pos >= 0:
            hits.setdefault(
                ("a", key.casefold()),
                TermHit(key, ABBREVIATION_SENSES[key], "abbreviation", pos),
            )

    lowered = source.casefold()
    for folded in _PHRASE_KEYS:
        if folded not in lowered:
            continue
        sense = PHRASE_ALIASES[folded]
        pos = _bounded_find(source, sense.english)
        if pos < 0:
            continue
        hits.setdefault(
            ("p", folded),
            TermHit(sense.english, (sense,), "phrase", pos),
        )

    ordered = sorted(
        hits.values(),
        key=lambda item: (item.position, 0 if item.kind == "abbreviation" else 1, -len(item.display)),
    )
    return tuple(ordered[: max(1, int(limit))])


def terms_for_text(text: str, limit: int = 48) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for hit in find_core_terms(text, limit=max(limit * 2, 48)):
        key = hit.display.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(hit.display)
        if len(result) >= limit:
            break
    return tuple(result)


def prompt_for_text(text: str, limit: int = 24) -> str:
    hits = find_core_terms(text, limit=max(1, int(limit)))
    if not hits:
        return ""

    lines = ["Phoenix核心医学术语（仅列当前文本命中项；歧义项必须结合上下文选择）："]
    for hit in hits:
        senses = hit.senses
        if len(senses) == 1:
            sense = senses[0]
            lines.append(f"- {hit.display}: {sense.english} = {sense.chinese}")
        else:
            joined = "；".join(
                f"{sense.english} = {sense.chinese}" for sense in senses[:6]
            )
            lines.append(f"- {hit.display} [歧义]: {joined}")
    return "\n".join(lines)


def acronym_seed() -> dict[str, tuple[tuple[str, str], ...]]:
    return {
        key: tuple((sense.english, sense.chinese) for sense in senses)
        for key, senses in ABBREVIATION_SENSES.items()
    }


def extend_acronym_seed(seed: dict) -> int:
    """Merge the core into the existing document acronym resolver in-place."""

    added = 0
    for key, senses in acronym_seed().items():
        rows = list(seed.get(key, ()))
        for pair in senses:
            if pair not in rows:
                rows.append(pair)
                added += 1
        seed[key] = tuple(rows)
    return added


def core_stats() -> dict[str, int]:
    return {
        "unique_abbreviations": len(ABBREVIATION_SENSES),
        "abbreviation_senses": sum(len(rows) for rows in ABBREVIATION_SENSES.values()),
        "phrase_aliases": len(PHRASE_ALIASES),
        "total_senses_and_aliases": sum(len(rows) for rows in ABBREVIATION_SENSES.values()) + len(PHRASE_ALIASES),
    }


def install() -> None:
    """Attach the core to the active contextual translation chain."""

    from . import medical_acronyms
    from . import translation_dual_route_release as contextual

    added = extend_acronym_seed(medical_acronyms.RADIOLOGY_SEED)

    old_terms = contextual._terms
    old_context = contextual._context

    if not bool(getattr(contextual, "_phoenix_terminology_core_installed", False)):
        def terms(text: str, limit: int = 48) -> tuple[str, ...]:
            merged: list[str] = []
            seen: set[str] = set()
            for item in terms_for_text(text, limit=limit):
                key = item.casefold()
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
            for item in old_terms(text, limit=limit):
                key = str(item).casefold()
                if key not in seen:
                    seen.add(key)
                    merged.append(str(item))
                if len(merged) >= limit:
                    break
            return tuple(merged[:limit])

        def context() -> str:
            base = old_context()
            try:
                ctx = contextual._CTX.get({}) or {}
                current = str(ctx.get("current_source", "") or "")
                core = prompt_for_text(current, limit=24)
            except Exception:
                core = ""
            if not core:
                return base
            return base + ("\n" if base else "") + core + "\n"

        contextual._terms = terms
        contextual._context = context
        contextual._phoenix_terminology_core_installed = True

    stats = core_stats()
    print(
        "[Phoenix][医学术语核心] 已启用："
        f"缩写/短词={stats['unique_abbreviations']}个、词义={stats['abbreviation_senses']}条、"
        f"英文别名/短语={stats['phrase_aliases']}条；"
        f"向原缩写解析器新增={added}条候选义。只注入当前文本命中项，不扩大常规prompt。",
        flush=True,
    )

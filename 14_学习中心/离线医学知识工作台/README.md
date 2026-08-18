# Phoenix 离线医学知识工作台

这是 Project Phoenix 的独立离线医学知识整理窗口。它不负责阅片推理，不接云端 API，不从互联网补充医学知识；知识来源限定为用户主动导入的 PDF 书籍、指南、规范和个人资料。

## 当前窗口

- **PDF资料库**：导入书籍、逐页索引、建立本地检索库。
- **PDF问答**：只依据已导入PDF回答，保留书名、页码和来源编号。
- **多书知识整理**：从全部已导入PDF跨书检索、去重、分批整理和分层合并。
- **整本书翻译**：默认从第1页翻到最后一页，也可指定起始页；逐页保存、断点续翻、多模型自动兜底。
- **TXT笔记整理**：读取TXT/MD或直接粘贴文字，一键整理成可保存、可复习、可打印的笔记。
- **打印**：PDF问答、多书整理、整本译本和TXT整理结果均提供打印/打印预览。

## 核心原则

1. **PDF 是事实来源，模型只负责整理。** 生成式知识回答必须引用当前检索到的 `[S编号]` 证据；没有有效引用的生成结果会被程序拦截并退回“原文证据模式”。
2. **运行阶段可完全断网。** PDF 解析、SQLite/FTS 检索、向量索引、问答、整理、翻译和笔记处理均读取本地文件与本地模型。
3. **长任务可恢复。** PDF 按页写入 SQLite；多书整理按批次保存 checkpoint；整本翻译每页单独落盘。
4. **来源可追溯。** 每个PDF知识块保存书名、文件路径、页码、块编号。
5. **不影响 Phoenix 阅片主链。** 本项目位于 `14_学习中心/离线医学知识工作台/`，与 CT/DR 推理代码解耦。

## SSD 目录

运行时自动从代码位置识别 Project Phoenix 根目录，不写死 D: 或 G:。

- PDF 原书：`14_学习中心/PDF资料/`
- SQLite/checkpoint：`14_学习中心/离线医学知识工作台_data/`
- 结构化 Docling 输出（可选）：`14_学习中心/离线医学知识工作台_data/docling_structure/`
- 本地模型：`04_AI模型/知识工作台/`
- 多书整理结果：`15_证据中心/PDF知识整理/`
- 整本译本：`15_证据中心/PDF知识整理/PDF整本翻译/`
- TXT整理笔记：`15_证据中心/PDF知识整理/TXT整理笔记/`

这些目录均属于本地资料、模型或运行数据，不提交到 GitHub。

## 最低可用版本

```bash
python -m pip install -r requirements-base.txt
python app.py
```

没有任何大模型时，程序仍可导入 PDF、按页建立可恢复索引、使用中文关键词回退检索并显示书名/页码原文证据。AI翻译和AI笔记整理需要至少一个对应本地模型。

## 本地模型

### 知识整理

- `Qwen/Qwen3.5-4B`：PDF问答、长期专题整理、TXT笔记整理、翻译最终兜底。
- `Qwen/Qwen3-Embedding-0.6B`：语义检索。
- `Qwen/Qwen3-Reranker-0.6B`：二阶段重排预留。

### 整本书翻译的多模型级联

整本英文医学书翻译默认按下面顺序工作：

1. `Helsinki-NLP/opus-mt-en-zh`：轻量英译中专用模型，第一模型。
2. `facebook/nllb-200-distilled-600M`：第二兜底；仅作为研究/非商业的故障恢复模型。
3. `Qwen/Qwen3.5-4B`：最后医学翻译兜底/复核模型。

不是每段都同时跑三个模型。第一模型输出后先做自动质量检查：

- 是否空译或明显漏译；
- 译文是否异常过短/过长；
- 数字、单位、CT/MRI参数是否丢失；
- 医学缩写是否大量丢失；
- 是否仍然大段保持英文；
- 是否出现重复输出。

检查失败才自动切到下一个模型。全部模型都失败时，程序不会让一整本书停止，而是在对应段落标记失败并保留原文，继续后面的页面，同时写入 audit JSON，之后可使用“重试警告页”。

## 模型下载

默认优先尝试 ModelScope：

```bash
python -m pip install -r requirements-ai.txt
python model_download.py translation_light --source modelscope
python model_download.py embedding --source modelscope
python model_download.py generator --source modelscope
```

翻译三模型全部准备：

```bash
python model_download.py translation --source modelscope
```

如果 ModelScope 某个模型线路不稳定，可只对该模型切换 Hugging Face：

```bash
python model_download.py translation_fast --source huggingface
python model_download.py translation_backup --source huggingface
```

下载结束后运行阶段可以完全断网。

## 整本书翻译

窗口中选择一本PDF后：

1. “从第几页开始”默认1；
2. 点击“开始/继续整本翻译”；
3. 程序逐页、逐段翻译；
4. 每完成一页立即保存；
5. 再次打开同一本书会自动跳过已完成页；
6. 有模型失败或质量异常的页面计入“警告页”；
7. 下载了更多模型后可点击“重试警告页”；
8. 全部完成后合成为一个完整TXT译本。

命令行：

```bash
python app.py --translate-book "D:/books/Core Radiology.pdf" --start-page 1 --no-gui
python app.py --translate-book "D:/books/Core Radiology.pdf" --start-page 1 --retry-warning-pages --no-gui
```

## 多书知识整理

多书整理不是一次把所有书塞进上下文。流程是：

1. 从全部PDF召回较大的相关证据池；
2. 做跨书籍覆盖，避免单本书垄断候选；
3. 分批生成带来源编号的高密度笔记；
4. 每批结束立即保存 checkpoint；
5. 使用分层合并避免超长上下文；
6. 最终文件保留来源索引。

电脑异常退出后，重新打开窗口点击“继续未完成任务”即可恢复。

## TXT笔记整理与打印

TXT/MD可以直接读取，也可以把零散笔记粘贴到窗口。长TXT会自动分段整理再分层合并，不要求一次塞进模型上下文。

结果默认保存为TXT，同时窗口提供：

- 保存TXT；
- 打印预览；
- 直接打印。

命令行：

```bash
python app.py --organize-txt "D:/notes/chest_ct.txt" --note-title "胸部CT复习笔记" --instruction "按征象、诊断、鉴别、陷阱和报告表达整理" --no-gui
```

## Embedding 索引

下载 `Qwen3-Embedding-0.6B` 并安装 `sentence-transformers` 后，可在窗口点击“生成向量索引”，或：

```bash
python app.py --build-embeddings --no-gui
```

没有 embedding 时系统自动使用 SQLite FTS + 中文关键词回退检索。

## 本地生成模型

默认寻找 `04_AI模型/知识工作台/Qwen3.5-4B/`。程序使用 `local_files_only=True` 加载，并设置 Transformers/Hugging Face 离线模式。新显卡可使用本地GPU；当前PyTorch不支持的老GPU自动保持CPU路径。

也可以连接本机启动的 OpenAI-compatible 服务：

```text
PHOENIX_KNOWLEDGE_LLM_URL=http://127.0.0.1:8080/v1/chat/completions
```

只允许 `localhost / 127.0.0.1 / ::1`，外部URL会被拒绝。

## Docling

基础索引优先使用 PyMuPDF。Docling 作为可选结构化增强，用于复杂版面、表格和OCR。只有明确设置 `PHOENIX_ENABLE_DOCLING=1` 时程序才尝试结构化导出，避免医院断网环境因为缺少额外资产而等待网络。

## 其他命令行示例

```bash
python app.py --ingest "D:/books/Core Radiology.pdf" --no-gui
python app.py --ask "整理肺磨玻璃结节的恶性CT征象和鉴别诊断" --no-gui
python app.py --organize "肺磨玻璃结节" --instruction "汇总全部PDF：征象、良恶性判断、鉴别、随访、漏诊点、报告模板；每条保留来源" --no-gui
python app.py --resume-task 3 --no-gui
python app.py --status --no-gui
```

## 当前边界

- 纯扫描PDF如果没有文本层，PyMuPDF可能提取不到文字；系统会标记OCR警告，不会把空白当作可靠医学证据或可靠译文。
- NLLB只作为故障恢复翻译模型，不作为医学专业翻译质量的最终裁决模型。
- Reranker已预留，但当前主检索采用FTS/中文关键词 + Qwen3 Embedding融合。
- 本工具只整理知识、翻译和笔记，不自动修改 Phoenix 主模型权重。

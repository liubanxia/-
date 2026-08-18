# Phoenix 离线医学知识工作台

这是 Project Phoenix 的独立离线医学知识整理窗口。它不负责阅片推理，不接云端 API，不从互联网补充医学知识；知识来源限定为用户主动导入的 PDF 书籍、指南、规范和个人资料。

## 核心原则

1. **PDF 是事实来源，模型只负责整理。** 生成式回答必须引用当前检索到的 `[S编号]` 证据；没有有效引用的生成结果会被程序拦截并退回“原文证据模式”。
2. **运行阶段可完全断网。** PDF 解析、SQLite/FTS 检索、向量索引、问答和整理都走本地文件；外部模型下载只在用户明确运行 `model_download.py` 时发生。
3. **长任务可恢复。** PDF 按页写入 SQLite，已经完成的页不会重复；深度整理按批次保存 checkpoint。窗口可直接“继续未完成任务”，命令行也可按 task ID 恢复。
4. **来源可追溯。** 每个知识块保存书名、文件路径、页码、块编号。最终专题输出写入 `15_证据中心/PDF知识整理/`。
5. **不影响 Phoenix 阅片主链。** 本项目位于 `14_学习中心/离线医学知识工作台/`，与 CT/DR 推理代码解耦。

## SSD 目录

运行时自动从代码位置识别 Project Phoenix 根目录，不写死 D: 或 G:。

- PDF 原书：`14_学习中心/PDF资料/`
- SQLite/checkpoint：`14_学习中心/离线医学知识工作台_data/`
- 结构化 Docling 输出（可选）：`14_学习中心/离线医学知识工作台_data/docling_structure/`
- 本地模型：`04_AI模型/知识工作台/`
- 深度整理结果：`15_证据中心/PDF知识整理/`

这些目录均是本地资料/模型/运行数据，不应提交到 GitHub。

## 最低可用版本

```bash
python -m pip install -r requirements-base.txt
python app.py
```

没有任何大模型时，程序仍可导入 PDF、按页建立可恢复索引、使用中文关键词回退检索、显示带书名和页码的原文证据，并让深度整理任务生成“证据包”。

## 推荐离线 AI 组件

模型不进入 Git 仓库，统一放在 `04_AI模型/知识工作台/`：

- `Qwen/Qwen3.5-4B`：离线问答和长期专题整理；
- `Qwen/Qwen3-Embedding-0.6B`：语义检索；
- `Qwen/Qwen3-Reranker-0.6B`：二阶段重排预留模型。

下载默认走 ModelScope：

```bash
python -m pip install modelscope huggingface_hub
python model_download.py embedding --source modelscope
python model_download.py generator --source modelscope
```

全部预下载：

```bash
python model_download.py all --source modelscope
```

ModelScope 不稳定时可切换 Hugging Face：

```bash
python model_download.py generator --source huggingface
```

下载器兼容 ModelScope 的直接目录和缓存快照两种落盘方式；下载结束后可以断网运行。

## Embedding 索引

下载 `Qwen3-Embedding-0.6B` 并安装 `sentence-transformers` 后，可在窗口点击“生成向量索引”，或命令行：

```bash
python app.py --build-embeddings --no-gui
```

没有 embedding 时系统自动使用 SQLite FTS + 中文关键词回退检索，不会把“缺模型”当成故障。

## 生成模型

默认寻找 `04_AI模型/知识工作台/Qwen3.5-4B/`。程序用 `local_files_only=True` 加载，并设置 Transformers/Hugging Face 离线模式。新显卡可自动使用本地 GPU；对当前 PyTorch CUDA 不适合的老显卡则保持 CPU 路径，不把模型错误送入不兼容 CUDA。

也可以连接本机启动的 OpenAI-compatible 服务：

```text
PHOENIX_KNOWLEDGE_LLM_URL=http://127.0.0.1:8080/v1/chat/completions
```

代码只允许 `localhost / 127.0.0.1 / ::1`，外部 URL 会被拒绝。

## 深度整理

深度整理不是一次把整本书塞给模型。流程是：

1. 从书库召回较大的相关证据池；
2. 做跨书籍覆盖，避免单本书垄断候选；
3. 每 8 个证据块为一批生成带 `[S编号]` 的笔记；
4. 每批结束立即写 checkpoint；
5. 批次数量很大时采用分层合并，避免一次超长上下文；
6. 最终 Markdown 保留来源索引。

电脑异常退出后，已完成批次不会丢失。重新打开窗口后点击“继续未完成任务”即可从最近 checkpoint 继续。

## Docling

基础索引优先使用 PyMuPDF，因为它完全本地、按页引用稳定。Docling 作为可选结构化增强，用于复杂版面/表格/OCR。只有明确设置 `PHOENIX_ENABLE_DOCLING=1` 时程序才尝试生成 Docling 结构化 Markdown，避免医院断网环境因为缺少模型资产而意外等待下载。

## 命令行

```bash
python app.py --ingest "D:/books/Core Radiology.pdf" --no-gui
python app.py --ask "整理肺磨玻璃结节的恶性CT征象和鉴别诊断" --no-gui
python app.py --organize "肺磨玻璃结节" --instruction "汇总全部PDF：征象、良恶性判断、鉴别、随访、漏诊点、报告模板；每条保留来源" --no-gui
python app.py --status --no-gui
python app.py --resume-task 3 --no-gui
```

## 当前版本边界

- 对纯扫描 PDF，基础 PyMuPDF 可能提取不到文字；程序会标记 OCR 警告，不会把空页当成医学证据。
- Reranker 已预留模型和目录，但第一版检索采用 FTS/中文关键词 + Qwen3 Embedding 融合；待真实书库建立后再根据召回质量启用 reranker。
- 本工具是知识整理与检索系统，不自动修改 Phoenix 主模型权重。

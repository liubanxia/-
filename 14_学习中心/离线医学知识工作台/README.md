# Phoenix 医学知识工作台

Phoenix 医学知识工作台是 Project Phoenix 中独立的离线医学资料学习、检索、问答、联合整理和整本翻译产品。正常使用不依赖 PACS 阅片链，也不会自动修改医学影像主模型权重。

## 正式发布基线

普通用户的默认入口是：

```text
启动Phoenix医学知识工作台.bat
```

启动器会自动：

1. 识别 SSD 上的 Project Phoenix 根目录，不写死 D: / G:；
2. 使用 SSD 自带 `02_开发环境/python.exe`；
3. 执行一次运行环境自检；
4. 缺少 Python 运行组件时一次性尝试修复，不再让用户逐个 `pip install`；
5. 算力默认 `auto`：可用 NVIDIA CUDA 时自动 GPU，否则自动 CPU；
6. 自检通过后直接打开工作台。

维护人员仍可使用：

```bash
python app.py
python app.py --status --no-gui
python real_acceptance.py
```

## 主功能

### 医学资料库

统一支持：

- PDF
- PPT / PPTX
- DOCX
- TXT / Markdown
- HTML
- XML / NXML / JATS
- NBIB
- RIS
- BibTeX
- CSL-JSON
- CAJ / NH / HN / KDH / TEB / C8

资料只在本机 SSD 解析。PDF 优先使用 PyMuPDF；扫描 PDF 在本地 OCR 条件满足时自动尝试 OCR。资料、SQLite、checkpoint、图片缓存和输出结果不提交到 GitHub。

导入完成后，如果 Qwen Embedding 模型和 `sentence-transformers` 运行组件均可用，Phoenix 会自动补齐缺失向量，不再要求普通用户手工点击“生成向量索引”。

### 资料问答

问答链：

```text
用户问题
→ SQLite FTS / 中文关键词召回
→ Qwen3 Embedding 语义召回
→ 融合证据
→ 可选 Qwen 智能归纳
→ 引用安全检查
→ 输出
```

所有生成式医学事实必须保留 `[S编号]`。如果模型生成内容没有合法来源编号，Phoenix 会阻止该答案并退回原始证据模式。

GUI 提供：

- 快速证据
- 智能1
- 智能2

智能1优先使用本地快速生成模型，智能2使用质量模型。

## 语义检索 READY 的定义

正式版不再以“Embedding 文件夹存在”作为 READY。

只有同时满足以下条件才显示真实 READY：

- Qwen3 Embedding 模型目录有效；
- `sentence-transformers` 可以实际加载；
- 已导入知识块全部拥有对应向量。

状态示例：

```text
语义模型未下载
语义组件缺失
语义索引 3200/5445
语义索引 5445/5445 READY
```

## 多资料 / 论文联合整理

联合整理不是把全部书籍一次塞进模型上下文。

流程：

```text
多个检索视角
→ 跨资料召回
→ 文档覆盖控制
→ 分批证据整理
→ 每批 checkpoint
→ 分层合并
→ 来源编号校验
→ 相关原图插入
→ 多格式输出
```

长任务持续显示当前阶段、已运行时间和最近一次模型响应时间。暂停后保留 checkpoint；恢复时从已完成批次继续。

最终自动生成：

- PDF
- DOCX
- Markdown
- TXT

任何一个多格式导出失败时，正文和 checkpoint 仍会保留，并明确提示失败原因。

## 整本医学 PDF 翻译

默认翻译策略：

1. Qwen 智能医学翻译；
2. Marian 英译中兜底；
3. NLLB 仅在 Development / Research 环境作为非商业故障恢复模型。

正式产品 / 商业模式会自动禁止 NLLB，因为其模型许可为 CC-BY-NC-4.0。

翻译安全门重点保护：

- 数字
- 单位
- 正负号
- 左右侧
- 阴性 / 阳性
- 否定
- 不确定性
- 良恶性
- 急慢性
- 增强 / 不增强
- 信号 / 密度方向变化
- 敏感度 / 特异度与数值绑定

每页翻译后立即写入 checkpoint。硬失败页不会再因为存在失败占位文本而被错误当成“已完成”；再次继续时会自动重新翻译硬失败页。

### PDF 成品

默认：

- 原 PDF 页面在上；
- 中文译文在下；
- 生成完整 PDF；
- 同时按指定页数生成 PDF 分册。

正式版 PDF 保存策略取消大文件上的 `garbage=3` 全文件重整理，改为：

```text
garbage=0
deflate=True
临时文件写入
原子替换
```

目标是避免数百页医学教材在 100% 翻译后长时间“假死”，同时避免关闭压缩造成 PDF 体积暴涨。

暂停发生在安全页面边界。用户在“暂停请求尚未完成”期间再次点击继续，Phoenix 会记录继续请求，当前页安全落盘后自动恢复。

## 算力设置

普通用户只显示两种模式：

- 本机自动（推荐）
- DeepSeek 云算力

“本机自动”会自动选择 CUDA / CPU，高级用户不需要手工选择 CUDA、DeepSpeed。

DeepSeek 需要 API Key，并且必须明确勾选本次运行允许发送学习资料文本。患者资料禁止使用云端算力。

高级设置默认隐藏，可配置服务地址和模型名称。当前 DeepSeek 默认：

- `deepseek-v4-flash`
- `deepseek-v4-pro`

API Key 只保存在当前运行进程，不写入 Phoenix 配置文件。

## SSD 目录

- 原始学习资料：`14_学习中心/PDF资料/`
- 工作台运行数据：`14_学习中心/离线医学知识工作台_data/`
- 本地模型：`04_AI模型/知识工作台/`
- 整理结果：`15_证据中心/PDF知识整理/`
- 整本译本：`15_证据中心/PDF知识整理/PDF整本翻译/`

路径均通过工程根目录动态解析，不依赖固定盘符。

## 运行依赖

基础依赖：

```bash
python -m pip install -r requirements-base.txt
```

完整日常运行依赖：

```bash
python -m pip install -r requirements-runtime.txt
```

模型下载和开发增强：

```bash
python -m pip install -r requirements-ai.txt
```

普通用户不需要手工执行这些命令，双击启动器会先做自检。

## 正式上线真实平台验收

在装有真实资料和真实本地模型的 SSD 上执行：

```bash
python real_acceptance.py
```

验收内容：

1. 真实资料库和完整语义向量覆盖；
2. 中文问题跨语言检索；
3. 真实资料问答及 `[S编号]`；
4. 本地医学翻译和安全校验；
5. 两页真实模型整本 PDF 翻译、压缩写入和分册；
6. 多资料联合整理；
7. PDF / DOCX / Markdown / TXT 输出。

只有最后出现：

```text
PHOENIX_RELEASE_ACCEPTANCE=PASS
```

才视为当前机器、当前 SSD、当前资料库和当前模型组合完成正式平台验收。

## 发布边界

- GitHub Actions 回归测试用于验证代码兼容性和回归，不等于真实 GPU / 本地大模型验收。
- 正式上线必须同时通过 Windows CI 和 `real_acceptance.py` 的真实机器验收。
- 本工具用于医学知识资料整理和辅助学习，不替代医生临床判断。

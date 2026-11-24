# Datasheet Copilot 部署与运维说明（MVP）

本文档描述第一版 MVP 的部署方式、运行方案、环境变量、安全检查与故障排查。
默认方案为 **D：Logical + P1 + recovered**，保守回退为 **C：Flat + P1 + legacy**。

## 1. 部署拓扑

最小部署单元：

```
客户端
  │ HTTP multipart
  ▼
FastAPI（app.main:app）
  │
  ├─ PyMuPDF：文本与坐标
  ├─ Docling：表格结构
  ├─ StructureRecovery：逻辑行恢复（D 方案）
  ├─ DeepSeek API：参数提取
  └─ Validator：legacy + recovered
  │
  ▼
JSON 响应 / 运行诊断
```

第一版是同步长请求。生产环境建议在前面加反向代理、超时与并发限制；
大流量场景应改为任务队列。

## 2. 系统要求

- Python >= 3.11
- 2 GB 以上内存（Docling CPU 推理建议 4 GB+）
- 可访问模型 API
- 有文本层的 PDF（当前不支持扫描件 OCR）
- Docling 表格模型文件

## 3. 安装

```bash
git clone <repo>
cd Datasheet-Copilot

uv sync --frozen
source .venv/bin/activate
```

或：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

`pip install -e .` / `uv sync` 会以 editable 方式安装 `app` 包；
仍可用 `PYTHONPATH=backend` 直接运行源码。构建分发包：

```bash
uv build
```

## 4. Docling 模型

方式一：让 Docling 自行下载。

方式二：把模型放到仓库内（模型权重不随公开仓库发布）：

```bash
export DOCLING_ARTIFACTS_PATH="$PWD/docling-models"
```

如果设置了该变量，请确保目录真实存在；否则 Docling 可能直接报路径
错误。`docling-models.sha256` 只提供文件校验清单，不包含模型权重。
准备模型目录后可校验：

```bash
sha256sum -c docling-models.sha256
```

CPU 部署：

```bash
export CUDA_VISIBLE_DEVICES=""
```

`TableParser` 会读取 `DOCLING_ARTIFACTS_PATH`，不会改变
`do_ocr / do_table_structure` 配置。

## 5. 环境变量

复制模板：

```bash
cp .env.example .env
```

| 变量 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- |
| `LLM_API_KEY` | 是 | - | 模型 API Key，只通过环境注入 |
| `LLM_BASE_URL` | 是 | - | OpenAI 兼容端点 |
| `LLM_MODEL` | 是 | - | 模型名 |
| `LLM_TEMPERATURE` | 否 | `0.0` | 固定采样温度 |
| `LLM_MAX_TOKENS` | 否 | `6000` | 单次输出上限 |
| `LLM_ROWS_PER_CHUNK` | 否 | `10` | 行级分块大小 |
| `LLM_MAX_SPLIT_DEPTH` | 否 | `6` | 截断二分最大深度 |
| `DATASHEET_RUN_PROFILE` | 否 | `d` | `d` 或 `c` |
| `DOCLING_ARTIFACTS_PATH` | 否 | - | 本地模型目录 |

受控实验覆盖变量（通常不需要）：

- `RECOVERED_EVIDENCE_LOGICAL_ROWS`
- `RECOVERED_EVIDENCE_RAW_STRUCTURE`
- `EXTRACTION_INPUT_MODE`
- `EVIDENCE_VALIDATION_MODE`
- `PROMPT_FAMILY_COVERAGE`

## 6. 启动服务

```bash
PYTHONPATH=backend uvicorn app.main:app \
  --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

## 7. 运行方案

### D（默认）

```bash
export DATASHEET_RUN_PROFILE=d
```

- 运行时真实执行 StructureRecovery；
- 使用 LogicalTableAdapter；
- recovered 校验；
- 如果显式指定的目标表恢复失败，会拒绝继续，而不是降级为 Flat 自动接受；
- 由选择器自动选表时，恢复失败的表不会以 Flat 形式进入 D 的可信输入。

### C（保守回退）

```bash
export DATASHEET_RUN_PROFILE=c
```

- 使用经过字形恢复的 Flat 表格；
- legacy 校验；
- 适合在 StructureRecovery 不可用时临时降级；
- 已知限制：复杂表可能触发输出截断，召回下降。

## 8. API 调用

只解析：

```bash
curl -F "file=@sample.pdf" \
  "http://localhost:8000/api/v1/documents/parse"
```

真实提取：

```bash
curl -F "file=@sample.pdf" \
  "http://localhost:8000/api/v1/documents/parse?extract=true&table_ids=4"
```

指定回退方案：

```bash
curl -F "file=@sample.pdf" \
  "http://localhost:8000/api/v1/documents/parse?extract=true&profile=c&table_ids=4"
```

响应中：

- `auto_accept` 仅包含 `validated`；
- `pending_review` 必须人工复核；
- `invalid` 阻止自动接受；
- `pipeline` 记录实际输入模式、校验模式与结构恢复结果。

## 9. 安全要求

1. **绝不提交 `.env` 或真实 Key**。`.gitignore` 已忽略 `.env`、`.env.*`，
   只保留 `.env.example`。
2. 不要把 Key 写进命令行参数、URL、日志或前端代码。
3. 容器部署使用 Secret Manager 或运行时环境注入，不要固化到镜像层。
4. 反向代理层限制上传大小（API 内部限制 20 MB）与并发数。
5. 日志只记录模型名、token、状态与耗时，不记录 Key。
6. `data/`、`docling-models/`、`*.extraction.json` 不进入版本库。
7. 对公网开放时增加鉴权、速率限制与 HTTPS。

## 10. 资源与超时

- Docling 首次加载模型较慢，建议进程常驻；
- 单个 PDF 的 E2E 时间通常为 1–5 分钟；
- API 客户端超时应大于 300 秒；
- 反向代理 `proxy_read_timeout` 建议 300–600 秒；
- 建议限制并发为 1–2，避免 Docling 内存峰值叠加；
- 模型输出截断时会自动二分，token 成本会上升。

## 11. 状态语义（验收必须遵守）

| 状态 | 自动接受 | 展示方式 |
| --- | --- | --- |
| `validated` | 是 | 正常结果 |
| `pending_review` | 否 | 单独“待复核”区域 |
| `invalid` | 否 | 拒绝并展示原因 |

任何把 `pending_review` 或 `invalid` 混入自动接受结果的行为都视为验收失败。

## 12. 验收检查

部署后执行：

```bash
# 1. 测试
#    公开环境（无 data/）：254 passed / 104 skipped（数据依赖自动跳过）。
#    内部环境（保留 fixture）：运行全部测试，内部基线 528 passed。
PYTHONPATH=backend python -m pytest -q

# 2. 内部验收演示（需自备 PDF、API Key 与本地 Gold/产物）
export PDF=/path/to/onsemi_ncp170.pdf
python backend/scripts/mvp_demo.py "${PDF}" \
  --profile d --table-ids 4 \
  --output-dir data/outputs/mvp_demo/onsemi_ncp170
```

内部验收样本为 `onsemi_ncp170.pdf` 与 `adi_lt3045.pdf`，PDF 本身不随
公开仓库发布；缺少 PDF 或 API Key 时脚本会明确报错。测试边界与
fixture 说明见 [`DATA_LAYOUT.md`](DATA_LAYOUT.md)。

检查：

- NCP170：`validated` 覆盖关键参数，`ground_current/shutdown_current`
  单位为 `µA`，`output_voltage_noise` 为 `µVrms`；
- ADI LT3045：`output_current_limit` 同名歧义保持 `invalid`；
- 两个演示的 `summary.md`、`parameters.csv`、`validation.json` 存在；
- `invalid` 没有出现在 `auto_accept` 中。

## 13. 冻结 Gold 与历史产物

以下内容只保留在内部环境，不随公开仓库发布：

- 冻结 Gold：
  `data/outputs/experiments/B3.5b/gold/gold.frozen.json`
  SHA256：
  `ed1088a165ef02afe6c84f76605e91cb3737e2feabf7e5c877c717b993015250`
- 不得覆盖 `data/outputs/experiments/B3.5c`、`B3.7`、`B3.8`、`B3.9`；
- 新实验只写新目录；
- 不重建 Gold，不修改历史评分口径。

公开版 B3.9 聚合指标见
[`evaluation/B3.9_REPORT.md`](evaluation/B3.9_REPORT.md)。

## 14. 故障排查

| 现象 | 排查 |
| --- | --- |
| `缺少 LLM_API_KEY` | 检查 `.env` 或环境注入 |
| Docling 下载失败 | 设置 `DOCLING_ARTIFACTS_PATH` |
| 显式 `table_ids` 恢复失败 | 检查该表是否有文本层与表格线段 |
| 全部参数 `pending_review` | 检查 recovered 证据是否与 PDF SHA 匹配 |
| `finish_reason=length` 频繁 | 降低 `LLM_ROWS_PER_CHUNK` 或提高 `LLM_MAX_TOKENS` |
| 大量 `output_current_limit` invalid | 预期行为：同名歧义不自动接受 |
| 服务超时 | 调整反向代理超时，降低并发 |

# Datasheet Copilot

Datasheet Copilot 是一个面向电子元器件 Datasheet 的参数提取与证据校验系统。
它把 PDF 中的表格恢复为可追溯的逻辑行，调用大模型提取 LDO 电气参数，
再用确定性 Validator 校验数值、单位、测试条件与来源，最终输出三种状态：

- `validated`：可以自动接受；
- `pending_review`：必须人工复核，不能当作成功；
- `invalid`：阻止自动接受。

第一版 MVP 的目标不是“所有 PDF 全自动解析”，而是给出一条
**可复现、可审计、不会把不可信结果伪装成成功** 的端到端链路。

> 当前在线验证范围仅两个型号：onsemi NCP170 与 ADI LT3045。
> 四厂商结果来自 B3.5c / B3.7 / B3.8 的历史与离线评测，
> 本版本不宣称新的四厂商在线泛化结果。

---

## 1. 项目定位与解决的问题

Datasheet 参数提取的难点不在“让模型读表格”，而在于：

1. PDF 表格经常有合并单元格、跨行单位、脚注、Unicode 字形错误；
2. 模型可能把 `µA` 读成 `mA`，或把最大值填进典型值列；
3. 同名参数（例如 Internal / Programmable Current Limit）语义不同，
   不能用一个名字自动接受；
4. 模型输出看起来合理，但没有可核对的来源证据。

本项目通过 **结构恢复 + 逻辑输入 + 证据校验 + 状态分级** 解决这些问题：

- 结构恢复保留 `table_id / row_index / page / cell_id / source_text`；
- Logical 输入让模型看到跨行合并后的完整逻辑行；
- Validator 不通过时不会自动接受；
- 所有真实模型输出、校验报告和运行日志可保存、可重放。

---

## 2. 核心架构与处理链路

```
PDF
 │
 ├─ PyMuPDF：逐页文本与坐标
 │
 ├─ Docling TableParser：表格结构（Flat）
 │    └─ Flat 字形恢复：仅修正已登记的 PDF/Docling 字形差异
 │
 ├─ StructureRecovery：PDF 几何证据 → logical_rows
 │    └─ 保留 source_row_indices / source_cell_ids / physical_row_band
 │
 ├─ LogicalTableAdapter：logical_rows → 模型可读输入
 │    └─ row_indices 映射回原始行号；needs_review 行默认不进入可信输入
 │
 ├─ DeepSeekExtractor：JSON 输出（P1 Prompt 族覆盖）
 │
 ├─ Evidence 校验：来源定位与原文引用
 │
 ├─ ParameterValidator（legacy 规则）
 │
 └─ RecoveredEvidenceValidator（逻辑行证据校验）
      ├─ validated（自动接受）
      ├─ pending_review（人工复核）
      └─ invalid（拒绝）
```

生产入口不会绕过上述链路：

- `backend/app/main.py` 的 `/api/v1/documents/parse?extract=true`；
- `backend/scripts/run_e2e.py` 的命令行入口；
- 两者都复用 `backend/app/pipeline.py` 的真实装配逻辑。

默认运行方案：

| 方案 | 输入 | Prompt | 校验 | 用途 |
| --- | --- | --- | --- | --- |
| **D** | Logical | P1 | recovered | MVP 默认 |
| C | Flat | P1 | legacy | 保守回退 |

通过 `DATASHEET_RUN_PROFILE=d`（默认）或 `c` 切换。
方案不是只改名字：D 会在运行时真实执行 StructureRecovery，
C 会使用经过字形恢复的 Flat 表格与 legacy 校验。

---

## 3. 各组件职责

### Docling / TableParser

- 把 PDF 表格导出为行列结构和原始单元格；
- 不负责判断参数语义；
- `backend/app/parsers/table_parser.py` 在解析后执行
  `_apply_flat_glyph_restoration`，只修正 Docling 文本与 PDF 文本
  仅相差已登记字形的单元格（例如 `mA` → `µA`），原始文本保留在
  `docling_cells` 供审计。

### StructureRecovery

- 用 PDF 文字坐标与表格线段，把 Docling 的物理单元格恢复为逻辑行；
- 处理跨行参数名、跨行单位、合并单元格中的多个数值；
- 证据不足时设置 `needs_review=true`，不自动猜测；
- 入口：`recover_from_pdf(raw_structure, table_id, pdf_path, page_no)`。

### LogicalTableAdapter

- 把 `logical_rows` 转换为模型输入列；
- `row_indices` 是最具体的原始行号，不是逻辑行号；
- 只包含显式请求的 `table_id`；
- `needs_review=true` 的逻辑行默认不进入可信输入。

### DeepSeekExtractor

- 只根据输入表格提取参数，要求 JSON 输出；
- 按行分块，遇到 `finish_reason=length` 自动二分重试；
- P1 开启低频参数族覆盖与 Logical 输入说明；
- 默认 `temperature=0.0`，实际模型名写入运行诊断；
- 不负责最终可信判定。

### Validator

- `app/validation/validator.py`：数值、Min/Typ/Max 关系等 legacy 规则；
- `app/validation/recovered_integration.py`：把 legacy 结果与
  RecoveredEvidenceValidator 合并；
- `app/validation/recovered_evidence.py`：身份、数值、单位、
  测试条件、来源映射校验；
- 只有 `validated` 可以自动接受；`pending_review` 单独展示；
  `invalid` 阻止自动接受。

---

## 4. 环境要求与依赖安装

- Python >= 3.11
- Linux / macOS（CPU 可运行；有 GPU 时 Docling 会更快）
- 可访问模型 API（OpenAI 兼容接口）
- 首次使用 Docling 需要表格模型文件

使用 `uv`（推荐，锁文件可复现）：

```bash
uv sync --frozen
source .venv/bin/activate
```

或使用 pip：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

安装后 `app` 包可用；也可以继续使用 `PYTHONPATH=backend` 运行源码。
构建分发包：

```bash
uv build
```

Docling 模型可选两种方式：

1. 让 Docling 自行解析/下载；
2. 把模型放到仓库内 `docling-models/`，并设置：

```bash
export DOCLING_ARTIFACTS_PATH="$PWD/docling-models"
```

CPU 环境建议：

```bash
export CUDA_VISIBLE_DEVICES=""
```

---

## 5. 环境变量与安全说明

复制示例文件：

```bash
cp .env.example .env
```

必填变量：

| 变量 | 说明 |
| --- | --- |
| `LLM_API_KEY` | 模型 API Key，禁止提交 |
| `LLM_BASE_URL` | OpenAI 兼容端点 |
| `LLM_MODEL` | 模型名 |
| `LLM_TEMPERATURE` | 默认 `0.0` |
| `DATASHEET_RUN_PROFILE` | `d`（默认）或 `c` |
| `DOCLING_ARTIFACTS_PATH` | 可选，本地 Docling 模型目录 |

安全约定：

- `.env`、`.env.*` 已被 `.gitignore` 忽略；只提交 `.env.example`；
- API Key 只从环境变量读取，不写入代码、日志或输出 JSON；
- `data/` 下的样本、Gold、运行产物与模型缓存不进入版本库；
- 生产部署应通过 Secret Manager / 环境注入，不要放在镜像层；
- 日志默认只记录模型名、token 数、状态，不记录 Key。

### 5.1 数据与公开发布边界

公开仓库只包含代码、测试、文档和 `docling-models.sha256`，**不包含**
PDF、Gold、原始模型响应、运行产物或模型权重。

- `data/` 下的 PDF、缓存、实验产物不进入版本库；
- 演示与在线指标复现需要用户自备 PDF、API Key 与可选模型目录；
- 数据依赖测试在缺少 `data/` 时会自动跳过，不会伪装成通过。

目录用途、fixture 恢复方式和测试边界见
[`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md)。

---

## 6. 本地运行、测试与演示命令

以下命令均在仓库根目录执行。

### 6.1 启动 API

```bash
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000
```

只解析、不调用模型（`PDF` 指向你自备的 Datasheet）：

```bash
export PDF=/path/to/your.pdf
curl -F "file=@${PDF}" \
  "http://localhost:8000/api/v1/documents/parse"
```

真实提取（默认 D）：

```bash
curl -F "file=@${PDF}" \
  "http://localhost:8000/api/v1/documents/parse?extract=true&table_ids=4"
```

切换到 C：

```bash
curl -F "file=@${PDF}" \
  "http://localhost:8000/api/v1/documents/parse?extract=true&profile=c&table_ids=4"
```

### 6.2 命令行 E2E

```bash
PYTHONPATH=backend python backend/scripts/run_e2e.py \
  "${PDF}" --profile d --table-ids 4
```

输出：

- `data/outputs/<stem>.extraction.json`：原始提取 + final extraction；
- `data/outputs/<stem>.pipeline.json`：运行方案、结构恢复、校验模式。

### 6.3 可复现演示

> 演示需要外部输入：Datasheet PDF、`LLM_API_KEY`/`LLM_BASE_URL`/
> `LLM_MODEL`，以及可选的本地 Docling 模型目录。公开仓库不提供
> `data/samples/b35b/*.pdf`，请替换为自备文件。
>
> 缺失 PDF 或 API Key 时脚本会以明确错误退出，不会静默成功。

```bash
python backend/scripts/mvp_demo.py "${PDF}" \
  --profile d --table-ids 4 \
  --output-dir data/outputs/mvp_demo/my_sample
```

不新增 API 调用、用最终 Validator 重放已保存输出：

```bash
python backend/scripts/mvp_demo.py "${PDF}" \
  --profile d --table-ids 4 \
  --output-dir data/outputs/mvp_demo/my_sample \
  --revalidate
```

内部验收使用的两个型号文件名是 `onsemi_ncp170.pdf` 与
`adi_lt3045.pdf`，对应命令与结果见
[`docs/evaluation/B3.9_REPORT.md`](docs/evaluation/B3.9_REPORT.md)。

### 6.4 测试

公开环境（无 `data/`）：自包含测试全部通过，数据依赖测试自动跳过。

```bash
# 默认命令：缺 data/ 时自动跳过数据依赖测试
PYTHONPATH=backend python -m pytest -q

# 显式只运行自包含测试
PYTHONPATH=backend python -m pytest -q -m "not requires_data"
```

当前公开基线：**254 passed / 104 skipped**（无 `data/` 的干净克隆）。
数据依赖测试只是被跳过，没有被删除；在保留 fixture 的环境中会实际运行。

内部完整环境（保留 `data/`、Gold 与实验产物）使用同一命令，运行全部
测试。内部基线为 **528 passed**；公开克隆不能直接复现该数字，也不会
把跳过的测试计入通过数。

只运行数据依赖测试：

```bash
PYTHONPATH=backend python -m pytest -q -m requires_data
```

测试边界与 fixture 恢复方式见
[`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md)。

---

## 7. JSON / CSV 输出字段

### 7.1 E2E artifact（`*.extraction.json`）

| 字段 | 说明 |
| --- | --- |
| `schema_version` | artifact 版本 |
| `source.pdf_path` / `pdf_sha256` | PDF 来源与指纹 |
| `source.model` | 实际使用的模型名 |
| `source.selected_table_ids` | 本次处理的表 |
| `extraction` | 模型原始结构化输出 |
| `final_extraction` | 重试纠正后的最终提取 |
| `retry_results` | 每次自动纠错结果 |

### 7.2 API 响应（`extract=true`）

| 字段 | 说明 |
| --- | --- |
| `page_count` / `pages` | 文本解析结果 |
| `table_count` / `tables` | Docling 表格（含 `glyph_evidence`） |
| `extraction` | 最终校验结果（`parameters` + `validation_summary`） |
| `auto_accept` | 仅 `validated` 参数 |
| `pending_review` | 待人工复核参数 |
| `invalid` | 被拒绝参数 |
| `pipeline` | 运行方案、表格选择、结构恢复、实际输入/校验模式 |

### 7.3 参数字段

| 字段 | 说明 |
| --- | --- |
| `name` | 参数名 |
| `value` / `unit` / `value_type` | 数值、单位、min/typ/max |
| `test_conditions` | 测试条件 |
| `evidence` | `table_id` / `row_index` / `page` / `source_text` |
| `validation_status` | validated / pending_review / invalid |
| `validation_source` | legacy / recovered |
| `detailed_recovery_status` | 恢复校验细分状态 |
| `recovered_logical_row_id` | 命中的逻辑行 ID |
| `identity_ambiguous` | 是否同名歧义 |
| `validation_errors` | 拒绝/待复核原因 |

### 7.4 演示 CSV（`parameters.csv`）

列：`name,value,unit,value_type,test_conditions,table_id,row_index,page,source_text,validation_status,validation_source,detailed_recovery_status,identity_ambiguous,recovered_logical_row_id,validation_errors`。

---

## 8. 真实评测结果与实验范围

### 8.1 B3.5c：Flat 基线

- 范围：四厂商 76 条 Gold 记录；
- 官方 micro：P 92.86% / R 34.21% / F1 50.00%；
- strict F1 48.07%；
- 主要问题：结构恢复行/单位错误 44 条、Validator 误拒 13 条。

### 8.2 B3.7：受控 2×2 A/B

- 范围：同一批四厂商 Gold，固定模型、温度、表格范围；
- 官方 F1：

| 方案 | 输入 + Prompt | 官方 P | 官方 R | 官方 F1 | strict F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Flat + P0 | 88.46% | 30.26% | 45.09% | 43.14% |
| B | Logical + P0 | 80.52% | 81.58% | 81.05% | 73.20% |
| C | Flat + P1 | 91.38% | 69.74% | 79.11% | 76.12% |
| D | Logical + P1 | 82.67% | 81.58% | **82.12%** | 74.17% |

### 8.3 B3.8：离线重评分（不新增模型调用）

复用 B3.7 已保存输出，修复结构恢复与 Validator 后重新评分：

- D 官方 F1 82.12%，validated/trusted F1 82.12%；
- C 官方 F1 79.11%，validated/trusted F1 79.11%；
- 说明 Validator 误拒被消除，但这是离线重放，不是新的在线实测。

### 8.4 B3.9：两型号真实在线验证

- 范围：仅 `onsemi_ncp170` 与 `adi_lt3045`，37 条 Gold 记录；
- 真实模型：`deepseek-v4-1-flash-261017`；
- 温度：0.0；API 调用 17 次；约 92k tokens。

| 口径 | D（Logical + P1） | C（Flat + P1） |
| --- | ---: | ---: |
| 官方 P / R / F1 | 63.89% / 62.16% / **63.01%** | 100% / 54.05% / **70.17%** |
| strict F1 | 52.06% | 66.67% |
| validated-only P / R / F1 | 100% / 62.16% / **76.67%** | 100% / 48.65% / **65.46%** |
| pending-inclusive F1（诊断） | 76.67% | 70.17% |

逐样本：

- NCP170：D 官方 19/19，validated-only 19/19；单位 `µA/µA/µVrms` 正确；
- NCP170：C 官方 19/19，validated-only 17/19（2 条 PSRR 为 pending）；
- ADI LT3045：D 提取 22 条，覆盖目标表 rows 1–6，4 条 shutdown_current
  自动接受；13 条 `output_current_limit` 同名歧义保持 invalid；
- ADI LT3045：C 仅提取 1 条（2 次 `finish_reason=length` 截断），
  17 条 Gold 未提取。

**不得把 B3.9 的两型号 micro 与 B3.7 的四厂商绝对值直接比较。**

---

## 9. 三种指标口径的区别

### 官方 F1

- 使用 B3.5c 冻结评分契约；
- in-scope 的所有预测都参与 P/R/F1；
- `invalid` 预测仍计为 FP；
- 适合看“模型内容是否找对”，不直接等于“生产可接受”。

### validated-only 自动接受口径

- 分母只包含 `validated` 预测；
- `pending_review` 与 `invalid` 不进入自动接受；
- 反映生产自动接受的真实精确率与召回；
- 因为排除了不确定结果，Precision 可能很高，Recall 可能下降。

### pending-inclusive 诊断口径

- 把内容匹配的 `pending_review` 也计入 TP；
- 只用于诊断“如果人工复核通过，上限在哪里”；
- **不代表自动接受，也不能作为验收通过率。**

### strict F1

- 在内容匹配之外，还要求测试条件与来源定位严格一致；
- 比官方 F1 更保守。

---

## 10. 已知限制与后续方向

1. **ADI 电流限制族同名歧义**：Internal / Programmable Current Limit
   语义不同，当前保持 `invalid`，不自动接受；这是有意的安全行为。
2. **合并单元格子行标签**：例如 ADI 的 Thermal Shutdown Hysteresis
   子行只继承父行符号，逻辑行缺少独立名称，可能被身份校验拒绝；
   后续需要在 StructureRecovery 中保留子行标签。
3. **C 方案输出截断**：ADI 复杂表会触发 `finish_reason=length`，
   自动二分后仍可能召回坍缩；C 只作为保守回退。
4. **在线范围有限**：只验证了两个型号，不能宣称新的四厂商泛化。
5. **扫描版 PDF**：当前 `do_ocr=false`，只支持有文本层的 PDF。
6. **API 为同步长请求**：Docling + LLM 耗时较长，生产部署建议加队列
   与超时控制。
7. **人工复核界面**：状态已区分，但第一版没有完整的前端复核工作流。

---

## 11. 相关文档

- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)：部署、环境变量、运维检查；
- [`docs/MVP_DELIVERY_REPORT.md`](docs/MVP_DELIVERY_REPORT.md)：
  第一版 MVP 验收报告与可复现命令；
- [`docs/evaluation/B3.9_REPORT.md`](docs/evaluation/B3.9_REPORT.md)：
  两型号在线验证公开摘要（聚合指标与已知限制）；
- [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md)：数据目录、测试边界与
  fixture 恢复说明。

内部运行产物（`data/outputs/mvp_demo/`、`data/outputs/experiments/`）
只保留在本地，不随公开仓库发布。

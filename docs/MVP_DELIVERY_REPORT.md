# Datasheet Copilot 第一版 MVP 最终交付报告

- 日期：2026-11-24
- 版本：`1.0.0`（tag `v1.0.0`）
- 发布分支：`main`（B3.4b～B3.5b 开发历史保留在 `experiment/*` 分支）
- 交付范围：LDO Datasheet 参数提取，两个型号真实在线演示
- 冻结 Gold SHA256：
  `ed1088a165ef02afe6c84f76605e91cb3737e2feabf7e5c877c717b993015250`

> 本报告不宣称新的四厂商在线泛化结果。B3.5c / B3.7 / B3.8 是四厂商
> 历史与离线评测，B3.9 与本次演示仅覆盖 onsemi NCP170 与 ADI LT3045。

> 公开仓库不包含 PDF、Gold、原始模型响应或运行产物。本文中的
> `528 passed` 是内部完整环境基线；公开克隆没有 `data/` fixture 时，
> 只运行自包含测试，数据依赖测试会自动跳过。边界说明见
> [`DATA_LAYOUT.md`](DATA_LAYOUT.md)。

## 1. 最终运行配置

| 项目 | 值 |
| --- | --- |
| 默认方案 | **D：Logical + P1 + recovered** |
| 回退方案 | C：Flat + P1 + legacy |
| 切换方式 | `DATASHEET_RUN_PROFILE=d` / `c` |
| 模型 | `deepseek-v4-1-flash-261017` |
| Base URL | `https://ark.cn-beijing.volces.com/api/v3` |
| Temperature | 0.0 |
| 输入模式 | `logical`（D）/ `flat`（C） |
| Prompt 族 | P1（`PROMPT_FAMILY_COVERAGE=1` 语义） |
| Validator | D 为 recovered，C 为 legacy |
| 自动接受 | 仅 `validated` |
| 待复核 | `pending_review` 单独展示，不算成功 |
| 拒绝 | `invalid` 阻止自动接受 |

### 生产链路是否真实接入

不是只改配置名。已新增 `backend/app/pipeline.py`，并在两个真实入口使用：

1. `backend/app/main.py`：
   `/api/v1/documents/parse?extract=true` 直接调用 `run_pipeline`；
2. `backend/scripts/run_e2e.py`：
   命令行入口调用同一 `run_pipeline`。

运行时行为：

- `TableParser` 真实解析 PDF，并执行 Flat 字形恢复；
- D 方案对目标表真实运行 `recover_from_pdf`（StructureRecovery）；
- 恢复结果通过 LogicalTableAdapter 装配为模型输入；
- `DeepSeekExtractor(family_coverage=True)` 使用 P1；
- `ParameterValidator` + `RecoveredEvidenceValidator` 依次执行；
- 显式指定的目标表恢复失败时拒绝继续，不会静默降级为 Flat 自动接受；
- 选择器模式下恢复失败的表不会以 Flat 形式进入 D 的可信输入。

新增测试覆盖：

- `backend/tests/test_mvp_pipeline.py`：方案解析、运行时恢复、
  显式 table_id 过滤、恢复失败拒绝、recovered 装配；
- `backend/tests/test_mvp_api.py`：API 真实调用生产流水线；
- `backend/tests/test_b39_identity_regressions.py`：
  hysteresis / EN 同义词身份回归。

## 2. E2E 演示

> 以下命令使用内部样本路径；`data/samples/b35b/*.pdf` 不随公开仓库
> 发布。公开环境请把路径替换为自备 Datasheet，并配置 API Key。

### 2.1 onsemi NCP170

命令：

```bash
python backend/scripts/mvp_demo.py \
  data/samples/b35b/onsemi_ncp170.pdf \
  --profile d --table-ids 4 \
  --output-dir data/outputs/mvp_demo/onsemi_ncp170
```

结果：

| 指标 | 值 |
| --- | ---: |
| StructureRecovery | 1 张表成功，19 条逻辑行 |
| 模型输出参数 | 28 |
| validated（自动接受） | 25 |
| pending_review | 3 |
| invalid | 0 |
| API 调用（含截断重试） | 4 |
| Token 总量 | 29,038 |
| 耗时 | 200.53 s |

关键验证：

- `ground_current`、`shutdown_current` 单位均为 `µA`；
- `output_voltage_noise` 单位为 `µVrms`；
- 合法 `mA` 参数（`output_current` 等）保留 `mA`；
- 字形恢复 4 个单元格（`mA→µA` ×2、`mVrms→µVrms`、
  条件文本 `mF→µF`）；
- 3 条 pending 为：2 条 `input_voltage` 缺少测试条件、
  1 条 `active_output_discharge_resistance` 无来源单位；
- 在线首次运行曾把 `enable_pin_current` 误判为 identity 冲突，
  已补同义词并用最终 Validator 重放：invalid 1 → 0，
  `validation_revalidated.json` 记录该变化，未新增 API 调用。

输出文件：

- `summary.md`：可读报告；
- `pipeline.json`：方案、选表、结构恢复、token；
- `validation.json`：在线原始校验；
- `validation_revalidated.json`：最终 Validator 重放结果；
- `revalidation_diff.json`：状态变化；
- `parameters.csv`：参数扁平表；
- `raw_responses.json`：真实模型输出；
- `run.log`：完整运行日志。

### 2.2 ADI LT3045

命令：

```bash
python backend/scripts/mvp_demo.py \
  data/samples/b35b/adi_lt3045.pdf \
  --profile d --table-ids 4 \
  --output-dir data/outputs/mvp_demo/adi_lt3045
```

结果：

| 指标 | 值 |
| --- | ---: |
| StructureRecovery | 1 张表成功，26 条逻辑行 |
| 模型输出参数 | 20 |
| validated（自动接受） | 6 |
| pending_review | 0 |
| invalid | 14 |
| API 调用（含截断重试） | 7 |
| Token 总量 | 41,165 |
| 耗时 | 172.16 s |

关键验证：

- 复杂表 rows 1–12 被触达，`shutdown_current` 4 条自动接受；
- `thermal_shutdown`、`thermal_regulation` 自动接受；
- 13 条 `output_current_limit` 因 **同名歧义** 保持 `invalid`，
  未被自动接受；
- 1 条 `thermal_shutdown_hysteresis` 因来源逻辑行只继承父行
  符号（“Thermal Shutdown”），身份不可区分而拒绝；这是已知
  结构恢复限制，不是模型数值错误。

输出文件与 NCP170 相同。

### 2.3 演示成本

两个演示合计：

- API 调用：11 次；
- Token：70,203；
- 总耗时：约 372.7 s（不含 Docling 重放）。

未使用写死参数，未绕过 Validator。

## 3. 指标口径

| 口径 | 含义 | 是否可用于验收 |
| --- | --- | --- |
| 官方 F1 | in-scope 预测全部计入，invalid 计 FP | 内容质量参考 |
| validated-only F1 | 仅 validated 预测进入分母 | 生产自动接受验收 |
| pending-inclusive F1 | 把 pending 中内容匹配项计入 TP | 仅诊断上限 |
| strict F1 | 额外要求条件与来源严格一致 | 更保守的质量参考 |

B3.9 两型号在线结果：

| 口径 | D | C |
| --- | ---: | ---: |
| 官方 F1 | 63.01% | 70.17% |
| strict F1 | 52.06% | 66.67% |
| validated-only F1 | **76.67%** | 65.46% |
| pending-inclusive F1 | 76.67% | 70.17% |

C 的官方 F1 更高是因为 ADI 只输出 1 条、几乎没有 FP，但 ADI
召回仅 1/18。生产自动接受口径下 D 更高，因此默认选 D。

## 4. 最终验收

| 检查项 | 结果 |
| --- | --- |
| 默认生产流程可运行 | PASS（真实 NCP170 / ADI 演示） |
| 内部全量 pytest | **528 passed** |
| 公开环境自包含测试 | 无 `data/` 时 **254 passed / 104 skipped** |
| 冻结 Gold SHA256 | 未变 |
| B3.5c / B3.7 / B3.8 / B3.9 历史产物 | 未覆盖 |
| `.gitignore` 防 Key / `.env` / 模型缓存 / 大文件 | PASS |
| 自动 Git Commit | 未执行 |

`git check-ignore` 验证：

- `.env` 被忽略；
- `data/outputs/**` 被忽略；
- `data/samples/**/*.pdf` 被忽略；
- `docling-models/` 被忽略。

## 5. 已知限制

1. **ADI 电流限制同名歧义**：
   `output_current_limit` 对应 Internal / Programmable 两种语义，
   当前一律 `invalid`，需要人工判断；这是有意的安全策略。
2. **合并单元格子行标签缺失**：
   ADI `thermal_shutdown_hysteresis` 子行只继承父行符号，
   身份校验无法区分；后续需在 StructureRecovery 保留子行标签。
3. **C 方案截断**：
   ADI 复杂表会触发 `finish_reason=length`，自动二分后召回仍可能下降。
4. **在线范围有限**：
   仅两个型号；四厂商结果来自历史与离线评测。
5. **扫描件不支持**：
   当前只支持有文本层的 PDF（`do_ocr=false`）。
6. **同步长请求**：
   API 未引入任务队列，生产高并发需要补队列、超时与限流。
7. **人工复核工作流**：
   状态已区分，但前端复核界面与操作审计尚未完成。

## 6. 第二版待办

1. 在 StructureRecovery 中保留合并单元格子行的独立参数名；
2. 为 current-limit 族提供人工选择 Internal / Programmable 的交互；
3. 为长表增加更强的分块与截断恢复策略；
4. 增加扫描件 OCR；
5. 引入异步任务队列与人工复核 UI；
6. 扩大在线验证到更多厂商与型号，再评估泛化结论。

## 7. 可复现命令

```bash
# 测试：内部完整环境为 528 passed；公开克隆会跳过数据依赖测试
PYTHONPATH=backend python -m pytest -q

# 真实演示：PDF 需自备，公开仓库不含样本
export PDF=/path/to/onsemi_ncp170.pdf
python backend/scripts/mvp_demo.py "${PDF}" \
  --profile d --table-ids 4 \
  --output-dir data/outputs/mvp_demo/onsemi_ncp170

# 不新增 API 调用，重放已保存输出的最终 Validator
python backend/scripts/mvp_demo.py "${PDF}" \
  --profile d --table-ids 4 \
  --output-dir data/outputs/mvp_demo/onsemi_ncp170 \
  --revalidate

# 检查冻结 Gold（仅内部环境存在）
sha256sum data/outputs/experiments/B3.5b/gold/gold.frozen.json
```

公开版 B3.9 聚合指标与限制见
[`evaluation/B3.9_REPORT.md`](evaluation/B3.9_REPORT.md)。

## 8. 交付结论

**符合第一版 MVP 交付条件。**

- D 作为默认方案，两型号核心 E2E 演示可运行；
- C 保留为可配置保守回退；
- 没有阻塞核心演示的硬缺陷；
- 已知限制已明确记录，不把 pending / invalid 伪装成成功；
- 可以转入 README、部署文档和最终交付。

本轮未重建 Gold、未启动 Prompt 调优、未扩大 PDF 样本范围、
未修改历史评分口径、未自动 Commit。

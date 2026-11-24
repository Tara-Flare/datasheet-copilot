# 脚本索引

本目录同时包含产品入口、可复现评测 harness 和历史实验脚本。生产代码
`backend/app/**` 不 import 本目录中的任何脚本。

## 产品入口

| 脚本 | 用途 |
| --- | --- |
| `run_e2e.py` | 真实端到端 CLI，复用 `app.pipeline.run_pipeline` |
| `mvp_demo.py` | MVP 演示，输出 pipeline / validation / CSV / summary |

## 可复现评测 harness

| 阶段 | 脚本 | 说明 |
| --- | --- | --- |
| B3.5c | `b35c_*` | Flat Baseline 预检、运行、评分与汇总 |
| B3.5d | `b35d_*` | 离线结构对照、Validator 复核、成本与报告 |
| B3.6 | `b36_*` | 修复后的结构恢复与 Validator 回归 |
| B3.7 | `b37_*` | 受控 2×2 A/B 运行与评分 |
| B3.8 | `b38_offline_replay.py` | 复用 B3.7 输出的离线重评分 |
| B3.9 | `b39_online_validation.py` | 两型号最终在线验证 |

这些脚本通常需要本地 `data/outputs/experiments/` 下的 Gold 或历史运行
产物；公开仓库不包含这些文件。真实运行 `b37_run_arm.py` /
`b39_online_validation.py` 会调用模型 API。

共享辅助：

- `b35b_docling_env.py`：离线 Docling 环境与 TableParser 构造；
- `b35b_review_common.py`：Gold 审核 XLSX/CSV 同步的共享逻辑。

## 历史实验与 Gold 工具

- `b35b_*`（除上述两个共享辅助）与 `test_b35b_*`：内部 Gold 构建、
  人工审核与冻结工具，不包含在公开发布范围；
- B3.2b～B3.4b 脚本（`build_*`、`run_*_experiment`、`re*_b34*`）：
  历史实验，保留用于追溯，不代表当前生产流程；
- `export_*`、`recover_logical_rows.py`、`run_*validation.py`、
  `visualize_table_evidence.py`：B0～B2 产物导出与离线校验工具。

## 运行约定

- 从仓库根目录执行，例如：

  ```bash
  PYTHONPATH=backend python backend/scripts/run_e2e.py --help
  ```

- 评测脚本默认只读冻结 Gold，不覆盖历史实验目录；
- 缺少 PDF、Gold 或 API Key 时应显式失败，不应静默降级。

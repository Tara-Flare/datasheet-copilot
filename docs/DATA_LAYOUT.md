# 数据目录、测试边界与公开发布策略

本文说明哪些数据随公开仓库发布，哪些只在内部环境保留，以及如何在
没有内部数据时运行测试和 E2E 演示。

## 1. 目录布局

```text
data/
├── samples/                     # 本地 PDF 样本（不发布）
│   └── b35b/
├── cache/                       # Docling/解析缓存（不发布）
├── outputs/                     # 运行产物（不发布）
│   ├── experiments/             # B3.x 实验、Gold、评分报告
│   └── mvp_demo/                # 两个真实 E2E 演示输出
└── outputs_gpu_test/            # 历史 GPU 测试产物（不发布）

docling-models/                  # 本地 Docling 模型权重（不发布）
docling-models.sha256            # 模型文件校验清单（发布）
```

`data/` 下的 PDF、Gold、原始模型响应和实验产物都**不进入版本库**。
公开仓库只提交代码、测试、文档和 `docling-models.sha256`。

原因：

- PDF 是第三方 Datasheet，公开再分发的版权状态未确认；
- Gold 与人工审核记录包含 Datasheet 原文与内部标注；
- 原始模型响应包含供应商返回内容与调用元数据；
- Docling 模型权重体积大，且许可证需单独确认。

## 2. 测试边界

测试分为两类：

| 类别 | 标记 | 依赖 | 公开克隆行为 |
| --- | --- | --- | --- |
| 自包含测试 | 无 | 仅代码与 `tmp_path` 合成数据 | 始终运行 |
| 数据依赖测试 | `pytest.mark.requires_data` | 本地 `data/` fixtures | 缺少 fixture sentinels 时自动跳过 |

`backend/tests/conftest.py` 负责注册标记，并用固定的 fixture 文件
（PDF、TPS7A02 产物、冻结 Gold）判断内部数据是否就绪；只有空
`data/` 目录不会误触发数据测试。少量模块在导入时读取 fixture，会
通过模块级 `pytest.skip` 明确跳过，不会产生 collection error。

### 公开环境

```bash
PYTHONPATH=backend python -m pytest -q
```

没有 `data/` 时，数据依赖测试显示为 skipped，自包含测试必须全部通过。
当前公开基线为 **254 passed / 104 skipped**。也可以显式只运行自包含测试：

```bash
PYTHONPATH=backend python -m pytest -q -m "not requires_data"
```

### 内部完整环境

服务器上保留 `data/`、Gold 与实验产物时，同一条命令会运行全部测试：

```bash
PYTHONPATH=backend python -m pytest -q
```

内部基线为 `528 passed`。公开克隆不会复现这个数字，README 不得把
528 描述为公开环境可直接复现的结果。

只运行数据依赖测试：

```bash
PYTHONPATH=backend python -m pytest -q -m requires_data
```

### 恢复本地 fixtures

从内部备份恢复以下目录后，数据依赖测试会自动启用：

```text
data/samples/                  # 测试 PDF
data/outputs/TPS7A02.*.json    # B1/B2 固定产物
data/cache/docling/            # 缓存表格
data/outputs/experiments/      # B3.x Gold 与实验产物
```

不要为了通过测试而修改 fixture 或删除断言；缺失 fixture 应保持跳过。

## 3. E2E 演示的外部输入

`run_e2e.py` 与 `mvp_demo.py` 需要用户自备：

1. 有文本层的 Datasheet PDF；
2. `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL`；
3. 可选：本地 Docling 模型目录（否则 Docling 自行下载）。

示例：

```bash
cp .env.example .env
# 编辑 .env，填入真实 Key 与端点

python backend/scripts/mvp_demo.py /path/to/your.pdf \
  --profile d \
  --output-dir data/outputs/mvp_demo/my_sample
```

脚本在 PDF 不存在或缺少 API Key 时会给出明确错误，不会静默失败。
公开仓库中的 `data/samples/b35b/*.pdf` 路径只是内部示例，文件本身
不随仓库发布。

## 4. 实验脚本的依赖层级

- `backend/app/**`：生产链路，不 import 任何 `backend/scripts` 脚本；
- `backend/scripts/run_e2e.py`、`mvp_demo.py`：产品入口；
- `backend/scripts/b35c_*` ～ `b39_*`：可复现评测 harness，需要本地
  Gold 或历史运行产物；
- `backend/scripts/b35b_*`、旧 B3.2b～B3.4b 脚本：内部 Gold 构建与
  历史实验，保留但不属于公开演示主路径。

公开仓库保留这些脚本用于审计与复现方法说明，但不会提交它们依赖的
PDF、Gold 或原始运行产物。

## 5. 发布检查

发布前确认：

```bash
git ls-files | git check-ignore --no-index --stdin   # 应为空
git status --short                                   # 不含 data/、PDF、模型、.env
```

`.gitignore` 已覆盖：

- `.env`、`.env.*`（保留 `.env.example`）；
- `data/**/*.pdf`；
- `data/cache/`；
- `data/outputs*/`（包括 `data/outputs_gpu_test/`）；
- `docling-models/`、`docling-models.tar.gz`；
- `*.extraction.json`、`*.log`、缓存目录。

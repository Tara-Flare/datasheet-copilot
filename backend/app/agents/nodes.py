import logging
from copy import deepcopy
from pathlib import Path

from langgraph.types import interrupt
from app.schemas.parameter import ElectricalParameter


from app.extraction.evidence import verify_evidence
from app.extraction.input_mode import (
    build_extraction_input,
    resolve_extraction_input_mode,
)
from app.extraction.logical_source import (
    canonicalize_parameters,
)
from app.validation.validator import ParameterValidator
from app.validation.recovered_integration import (
    build_recovered_validation_result,
    resolve_evidence_validation_mode,
)


logger = logging.getLogger(__name__)


LOGICAL_LEGACY_ERROR = (
    "Logical 输入参数必须使用 recovered 校验，"
    "legacy 模式不得自动通过"
)


def _guard_logical_with_legacy(validation_result):
    """Logical 输入 + legacy 校验时阻止错误放行。

    legacy 不校验测试条件，无法完整判断 Logical 恢复行。
    因此只要参数携带 Logical 来源映射，就必须由
    recovered 校验负责；本函数把它降级为 pending_review。
    """

    parameters = validation_result.get("parameters", [])
    blocked = 0

    for param in parameters:
        mapping_status = param.get(
            "source_mapping_status"
        )

        if mapping_status is None:
            continue

        param["logical_validation_required"] = True
        param["validation_source"] = "legacy_blocked"

        errors = list(
            param.get("validation_errors", [])
        )

        if LOGICAL_LEGACY_ERROR not in errors:
            errors.append(LOGICAL_LEGACY_ERROR)

        param["validation_errors"] = errors

        if param.get("validation_status") == "validated":
            param["validation_status"] = "pending_review"
            blocked += 1

    def count(status):
        return sum(
            1
            for param in parameters
            if param.get("validation_status") == status
        )

    summary = dict(
        validation_result.get("validation_summary", {})
    )
    summary["total"] = len(parameters)
    summary["validated"] = count("validated")
    summary["invalid"] = count("invalid")
    summary["pending_review"] = count("pending_review")
    summary["logical_legacy_blocked"] = blocked
    summary["logical_legacy_guard"] = True

    validation_result["validation_summary"] = summary
    validation_result["logical_legacy_guard"] = {
        "applied": True,
        "blocked_validated_count": blocked,
        "reason": LOGICAL_LEGACY_ERROR,
    }

    return validation_result


def _file_sha256(path):
    """计算 PDF SHA256，仅用于恢复证据来源一致性检查。"""

    import hashlib

    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()




def create_nodes(
    pdf_parser,
    table_parser,
    extractor,
    evidence_validation_mode=None,
    recovered_evidence=None,
    extraction_input_mode=None,
    logical_extraction=None,
):
    """
    创建工作流节点。

    支持传入真实工具或测试替身。
    evidence_validation_mode:
    - legacy（默认）：现有行为完全不变；
    - recovered：对逻辑行覆盖的参数使用严格恢复校验。
    extraction_input_mode:
    - flat（默认）：原始表格输入完全不变；
    - logical：已恢复表格使用 LogicalTableAdapter 输出，
      未恢复表格继续使用原始 Flat 数据。
    """

    validator = ParameterValidator()
    mode = resolve_evidence_validation_mode(
        evidence_validation_mode
    )
    input_mode = resolve_extraction_input_mode(
        extraction_input_mode
    )

    def _logical_config():
        """获取 Logical 输入配置；可复用 recovered 证据配置。"""

        config = dict(logical_extraction or {})

        if isinstance(recovered_evidence, dict):
            if config.get("logical_rows") is None:
                config["logical_rows"] = (
                    recovered_evidence.get("logical_rows")
                )

            if config.get("raw_structure") is None:
                config["raw_structure"] = (
                    recovered_evidence.get("raw_structure")
                )

        return config

    def parse_node(state):
        """解析 PDF 文本与表格。"""

        path = state["document_path"]

        document_info = pdf_parser.parse(path)
        table_result = table_parser.parse(path)

        return {
            "document_info": {
                "filename": document_info["filename"],
                "page_count": document_info["page_count"],
                "table_count": table_result["table_count"],
            },
            "tables": table_result["tables"],
            "status": "parsed",
        }

    def extract_node(state):
        """调用 LLM 提取参数。

        Flat 模式传入原始表格；
        Logical 模式只替换已恢复的 table_id，
        未恢复表格保持 Flat，且不修改 state["tables"]。
        """

        tables = state.get("tables", [])
        input_tables = tables
        effective_mode = input_mode
        logical_table_ids = []
        flat_table_ids = [
            table.get("table_id") for table in tables
        ]

        if input_mode == "logical":
            config = _logical_config()

            if isinstance(
                config.get("logical_rows"), dict
            ):
                prepared = build_extraction_input(
                    tables,
                    mode="logical",
                    logical_rows=config["logical_rows"],
                    raw_structure=config.get(
                        "raw_structure"
                    ),
                    table_ids=config.get("table_ids"),
                    include_needs_review=config.get(
                        "include_needs_review", False
                    ),
                )
                input_tables = prepared["tables"]
                effective_mode = prepared["effective_mode"]
                logical_table_ids = prepared[
                    "logical_table_ids"
                ]
                flat_table_ids = prepared[
                    "flat_table_ids"
                ]
            else:
                logger.warning(
                    "EXTRACTION_INPUT_MODE=logical，但未提供 "
                    "logical_rows；本次按 Flat 处理"
                )
                effective_mode = "flat_fallback"

        extraction = extractor.extract(input_tables)

        return {
            "extraction": extraction,
            "extraction_input_tables": input_tables,
            "extraction_input_mode": effective_mode,
            "extraction_input_summary": {
                "requested_mode": input_mode,
                "effective_mode": effective_mode,
                "logical_table_ids": logical_table_ids,
                "flat_table_ids": flat_table_ids,
            },
            "status": "extracted",
        }

    def validate_node(state):
        """执行来源校验与规则校验。"""

        extraction = state["extraction"]
        tables = state["tables"]

        # legacy 基线始终执行，保证默认行为与旧版本一致
        legacy_result = validator.validate(
            verify_evidence(extraction, tables)
        )

        if mode == "recovered":
            config = dict(recovered_evidence or {})

            if config.get("pdf_sha256") is None:
                document_path = state.get("document_path")

                if (
                    document_path
                    and Path(document_path).is_file()
                ):
                    config["pdf_sha256"] = _file_sha256(
                        document_path
                    )

            validation_result = (
                build_recovered_validation_result(
                    extraction,
                    legacy_result,
                    config,
                )
            )
        else:
            validation_result = legacy_result

            if input_mode == "logical":
                validation_result = (
                    _guard_logical_with_legacy(
                        validation_result
                    )
                )

        summary = validation_result[
            "validation_summary"
        ]

        all_valid = (
            summary["total"] > 0
            and summary["validated"] == summary["total"]
        )

        return {
            "validation_result": validation_result,
            "evidence_validation_mode": mode,
            "status": (
                "validated"
                if all_valid
                else "pending_review"
            ),
        }

    def finish_node(state):
        """标记任务完成。"""

        return {
            "status": "completed"
        }

    
    def review_node(state):
        """人工复核节点。"""

        # 兼容之前的自动化测试
        if not state.get("enable_human_review", False):
            return {
                "status": "pending_review"
            }

        validation_result = state["validation_result"]

        parameters = validation_result.get(
            "parameters", []
        )

        # 找出需要人工审核的参数
        review_items = []

        for index, param in enumerate(parameters):
            if param.get("validation_status") != "validated":
                review_items.append({
                    "index": index,
                    "parameter": param
                })

        # 暂停并等待人工输入
        human_input = interrupt({
            "message": "参数校验失败，需要人工复核",
            "review_items": review_items,
        })

        if not isinstance(human_input, dict):
            raise ValueError("人工复核输入格式错误")

        action = human_input.get("action")

        if action == "reject":
            return {
                "status": "rejected"
            }

        if action != "correct":
            raise ValueError("不支持的人工复核操作")

        corrections = human_input.get(
            "corrections", []
        )

        if not corrections:
            raise ValueError("没有提交任何参数修改")

        updated_parameters = deepcopy(parameters)

        # 只允许人工修改规定字段
        editable_fields = {
            "value",
            "unit",
            "value_type",
            "test_conditions",
            "evidence",
        }

        modified_indexes = set()

        for correction in corrections:
            index = correction.get("index")

            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(updated_parameters)
            ):
                raise ValueError("参数索引无效")

            if index in modified_indexes:
                raise ValueError("同一参数不能重复修改")

            modified_indexes.add(index)

            if (
                updated_parameters[index].get(
                    "validation_status"
                ) == "validated"
            ):
                raise ValueError(
                    "不能修改已通过校验的参数"
                )

            patch = {
                key: value
                for key, value in correction.items()
                if key in editable_fields
            }

            if not patch:
                raise ValueError("修改内容为空")

            candidate = {
                key: value
                for key, value
                in updated_parameters[index].items()
                if key not in {
                    "validation_status",
                    "validation_errors",
                }
            }

            candidate.update(patch)

            # 保留 schema 之外的 B3 元数据
            # （logical 来源、冲突、恢复状态等），
            # 避免 Review 重建参数时静默丢失。
            metadata = {
                key: value
                for key, value
                in updated_parameters[index].items()
                if key not in ElectricalParameter.model_fields
            }

            # 重新执行 Pydantic 结构验证
            checked = ElectricalParameter.model_validate(
                candidate
            )

            rebuilt = checked.model_dump(
                exclude_none=True
            )
            rebuilt.update(metadata)

            updated_parameters[index] = rebuilt

        # 人工修改后重新执行程序化 Logical 来源核对，
        # 避免 source_mapping_status 停留在修改前状态。
        updated_parameters = canonicalize_parameters(
            updated_parameters,
            state.get("extraction_input_tables") or [],
        )

        extraction = {
            **state["extraction"],
            "parameters": updated_parameters,
        }

        return {
            "extraction": extraction,
            "status": "reviewed",
        }



    def retry_node(state):
        """只重新提取校验失败的参数。"""

        validation_result = state["validation_result"]

        parameters = validation_result.get(
            "parameters", []
        )

        failed_parameters = [
            param
            for param in parameters
            if param.get("validation_status") == "invalid"
        ]

        retry_count = state.get("retry_count", 0) + 1
        max_retries = state.get("max_retries", 2)

        logger.info(
            "参数重试: retry_count=%s, max_retries=%s, "
            "失败参数=%s",
            retry_count,
            max_retries,
            len(failed_parameters),
        )

        # 向模型提供错误字段和来源表格。
        # Logical 模式使用与 extract 相同的输入表格，
        # 避免把逻辑行位置误当原始 row_index。
        retry_tables = state.get(
            "extraction_input_tables"
        ) or state["tables"]

        retry_result = extractor.retry_parameters(
            failed_parameters,
            retry_tables
        )

        corrected = retry_result.get(
            "parameters", []
        )

        # 用字段标识匹配，不根据模型返回顺序覆盖
        def field_key(param):
            return (
                param.get("name"),
                param.get("value_type"),
                param.get("test_conditions"),
            )

        # 只有唯一候选才接受纠错结果
        candidates = {}

        for param in corrected:
            key = field_key(param)
            candidates.setdefault(key, []).append(param)

        updated_parameters = []
        corrected_count = 0

        for original in parameters:
            if original.get(
                "validation_status"
            ) != "invalid":
                updated_parameters.append(original)
                continue

            matches = candidates.get(
                field_key(original), []
            )

            if len(matches) == 1:
                updated_parameters.append(matches[0])
                corrected_count += 1
            else:
                # 没有唯一纠错结果时保留旧值
                updated_parameters.append(original)

        logger.info(
            "参数重试完成: retry_count=%s, "
            "纠错成功=%s, 保留原值=%s",
            retry_count,
            corrected_count,
            len(failed_parameters) - corrected_count,
        )

        extraction = {
            **state["extraction"],
            "parameters": updated_parameters,
        }

        return {
            "extraction": extraction,
            "retry_count": retry_count,
            "status": "retrying",
        }


    return {
        "parse": parse_node,
        "extract": extract_node,
        "validate": validate_node,
        "finish": finish_node,
        "review": review_node,
        "retry": retry_node,
    }

    
    

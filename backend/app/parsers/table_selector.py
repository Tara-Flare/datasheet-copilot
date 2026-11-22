
import json


class TableSelector:
    """Datasheet 电气参数候选表格筛选器。"""

    KEYWORDS = {
        "electrical characteristics": 5,
        "dropout voltage": 5,
        "quiescent current": 5,
        "input voltage": 3,
        "output voltage": 3,
        "output current": 3,
        "power supply rejection": 4,
        "psrr": 4,
        "line regulation": 3,
        "load regulation": 3,
        "ground current": 3,
        "output voltage accuracy": 3,
    }

    def __init__(
        self,
        max_tables: int = 4,
        max_chars: int = 40000,
    ):
        self.max_tables = max_tables
        self.max_chars = max_chars

    def score_table(self, table: dict) -> int:
        """根据表格内容计算相关性分数。"""

        text = json.dumps(
            {
                "columns": table.get("columns", []),
                "rows": table.get("rows", []),
            },
            ensure_ascii=False,
        ).lower()

        score = 0

        for keyword, weight in self.KEYWORDS.items():
            if keyword in text:
                score += weight

        return score

    def select(self, tables: list[dict]) -> list[dict]:
        """选择相关表格，保留原始 ID 和行号。"""

        scored = []

        for table in tables:
            score = self.score_table(table)

            if score > 0:
                scored.append((score, table))

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        selected = []
        total_chars = 0

        for score, table in scored:
            if len(selected) >= self.max_tables:
                break

            size = len(
                json.dumps(table, ensure_ascii=False)
            )

            if total_chars + size > self.max_chars:
                continue

            selected.append(table)
            total_chars += size

        return selected


class SelectedTableParser:
    """
    为已有 TableParser 增加候选表格筛选。

    保持原来的 parse() 接口不变。
    可传入可选 cache 复用 Docling 解析结果。
    """

    def __init__(self, table_parser, selector, cache=None):
        self.table_parser = table_parser
        self.selector = selector
        self.cache = cache

    def _config_signature(self) -> dict:
        """返回影响 Docling 解析结果的配置签名。"""

        pipeline_options = getattr(
            self.table_parser,
            "pipeline_options",
            None,
        )

        return {
            "do_ocr": getattr(
                pipeline_options, "do_ocr", None
            ),
            "do_table_structure": getattr(
                pipeline_options,
                "do_table_structure",
                None,
            ),
        }

    def parse(self, pdf_path: str) -> dict:
        if self.cache is not None:
            key = self.cache.build_key(
                pdf_path,
                self._config_signature(),
            )

            result = self.cache.load(key)

            if result is None:
                result = self.table_parser.parse(pdf_path)
                self.cache.save(key, result)
        else:
            result = self.table_parser.parse(pdf_path)

        selected = self.selector.select(
            result["tables"]
        )

        return {
            **result,
            "original_table_count": result["table_count"],
            "table_count": len(selected),
            "tables": selected,
        }

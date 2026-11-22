
from app.parsers.table_selector import TableSelector


def test_table_selector():
    tables = [
        {
            "table_id": 1,
            "columns": ["Pin", "Description"],
            "rows": [
                ["IN", "Input pin"],
                ["GND", "Ground pin"],
            ],
        },
        {
            "table_id": 2,
            "columns": [
                "Parameter",
                "Typ",
                "Unit",
            ],
            "rows": [
                [
                    "Dropout Voltage",
                    "200",
                    "mV",
                ],
                [
                    "Quiescent Current",
                    "35",
                    "uA",
                ],
            ],
        },
    ]

    selector = TableSelector(max_tables=1)

    result = selector.select(tables)

    assert len(result) == 1
    assert result[0]["table_id"] == 2


def test_selector_keeps_table_id():
    tables = [
        {
            "table_id": 99,
            "page_numbers": [5],
            "columns": ["Parameter", "Typ"],
            "rows": [
                ["Dropout Voltage", "200"]
            ],
        }
    ]

    selector = TableSelector()

    result = selector.select(tables)

    assert result[0]["table_id"] == 99
    assert result[0]["page_numbers"] == [5]

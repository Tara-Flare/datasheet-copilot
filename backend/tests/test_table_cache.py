
from app.parsers.table_cache import TableParseCache


def test_cache_roundtrip(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 test content")

    cache = TableParseCache(str(tmp_path / "cache"))

    key = cache.build_key(
        str(pdf),
        {"do_ocr": False, "do_table_structure": True},
    )

    assert cache.load(key) is None

    data = {
        "table_count": 1,
        "tables": [
            {"table_id": 1, "rows": [["x"]]}
        ],
    }

    cache.save(key, data)

    assert cache.load(key) == data


def test_cache_key_varies_with_content_and_config(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4 content-a")

    cache = TableParseCache(str(tmp_path / "cache"))

    key1 = cache.build_key(
        str(pdf),
        {"do_ocr": False},
    )
    key2 = cache.build_key(
        str(pdf),
        {"do_ocr": True},
    )

    assert key1 != key2

    pdf.write_bytes(b"%PDF-1.4 content-b")

    key3 = cache.build_key(
        str(pdf),
        {"do_ocr": False},
    )

    assert key1 != key3


def test_from_env_disabled(monkeypatch):
    monkeypatch.delenv("TABLE_CACHE_DIR", raising=False)

    assert TableParseCache.from_env() is None


def test_from_env_enabled(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("TABLE_CACHE_DIR", str(cache_dir))

    cache = TableParseCache.from_env()

    assert cache is not None
    assert cache.cache_dir == cache_dir

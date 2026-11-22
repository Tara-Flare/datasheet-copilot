
import hashlib
import json
import os
from pathlib import Path


class TableParseCache:
    """
    Docling 表格解析结果本地缓存。

    缓存 key 由 PDF 文件内容哈希 + 解析配置签名共同决定：
    任一变化都会导致缓存失效，从而保证不会复用过期结果。

    仅缓存「未筛选」的原始表格解析结果；
    表格筛选始终在缓存之后实时执行，因此
    TableSelector 的配置变化不会命中旧缓存。
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)

    @classmethod
    def from_env(cls):
        """根据 TABLE_CACHE_DIR 环境变量创建缓存。

        未设置时返回 None（禁用缓存）。
        """

        cache_dir = os.getenv("TABLE_CACHE_DIR")

        if not cache_dir:
            return None

        return cls(cache_dir)

    def build_key(
        self,
        pdf_path,
        config_signature,
    ) -> str:
        """根据 PDF 内容与解析配置生成缓存 key。"""

        file_hash = self._file_hash(pdf_path)
        config_hash = hashlib.sha256(
            json.dumps(
                config_signature,
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

        return f"{file_hash}_{config_hash}.json"

    def load(self, key):
        """读取缓存，未命中或损坏时返回 None。"""

        path = self.cache_dir / key

        if not path.is_file():
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def save(self, key, data):
        """写入缓存；失败不阻断主流程。"""

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            tmp = self.cache_dir / (key + ".tmp")

            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

            tmp.replace(self.cache_dir / key)
        except OSError:
            pass

    @staticmethod
    def _file_hash(pdf_path) -> str:
        """计算 PDF 文件内容的 SHA-256。"""

        digest = hashlib.sha256()

        with open(pdf_path, "rb") as f:
            for chunk in iter(
                lambda: f.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest()

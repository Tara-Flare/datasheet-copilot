
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.parsers.pdf_parser import PDFParser
from app.parsers.table_parser import TableParser

from app.pipeline import (
    configure_docling_environment,
    parse_table_ids,
    run_pipeline,
)


configure_docling_environment()


app = FastAPI(
    title="Datasheet Copilot API",
    description="电子元器件 Datasheet 智能解析系统",
    version="1.0.0"
)

parser = PDFParser()
table_parser = TableParser()

MAX_FILE_SIZE = 20 * 1024 * 1024


@app.get("/")
def root():
    return {
        "message": "Datasheet Copilot API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/api/v1/documents/parse")
async def parse_document(
    file: UploadFile = File(...),
    extract: bool = False,
    profile: str = None,
    table_ids: str = None,
):
    """
    上传 PDF 并返回解析结果。

    extract=true 时使用真实生产链路：
    - 默认 D（Logical + P1 + recovered）
    - profile=c 可切换 C（Flat + P1 + legacy）
    """

    filename = Path(file.filename or "").name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="仅支持 PDF 文件"
        )

    try:
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "upload.pdf"

            total_size = 0

            with pdf_path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    total_size += len(chunk)

                    if total_size > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="PDF 文件不能超过 20MB"
                        )

                    output.write(chunk)

            # 检查 PDF 文件头
            with pdf_path.open("rb") as saved_file:
                if not saved_file.read(5) == b"%PDF-":
                    raise HTTPException(
                        status_code=400,
                        detail="文件不是有效的 PDF"
                    )

            if extract:
                # 真实生产链路：StructureRecovery +
                # LogicalTableAdapter + DeepSeek + Validator。
                # 只解析一次 PDF，避免 Docling 重复运行。
                run = await run_in_threadpool(
                    run_pipeline,
                    str(pdf_path),
                    profile=profile,
                    table_ids=parse_table_ids(table_ids),
                )

                result = {
                    "filename": filename,
                    "page_count": run["pdf_document"][
                        "page_count"
                    ],
                    "pages": run["pdf_document"]["pages"],
                    "table_count": len(
                        run["runtime"]["all_tables"]
                    ),
                    "tables": run["runtime"]["all_tables"],
                    "extraction": run["validation"],
                    "auto_accept": run["summary"][
                        "auto_accepted_parameters"
                    ],
                    "pending_review": run["summary"][
                        "pending_parameters"
                    ],
                    "invalid": run["summary"][
                        "invalid_parameters"
                    ],
                    "pipeline": run["diagnostics"],
                    "status": "extracted",
                }

                return result

            # 只解析、不提取时保持原有轻量返回结构。
            text_result = await run_in_threadpool(
                parser.parse,
                str(pdf_path)
            )

            table_result = await run_in_threadpool(
                table_parser.parse,
                str(pdf_path)
            )

            return {
                "filename": filename,
                "page_count": text_result["page_count"],
                "table_count": table_result["table_count"],
                "pages": text_result["pages"],
                "tables": table_result["tables"],
                "status": "parsed",
            }

    except HTTPException:
        raise

    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"PDF 解析失败: {str(exc)}"
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="服务器内部错误"
        )

    finally:
        await file.close()

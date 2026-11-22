
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.parsers.pdf_parser import PDFParser
from app.parsers.table_parser import TableParser

from app.extraction.extractor import ParameterExtractor
from app.extraction.evidence import verify_evidence


app = FastAPI(
    title="Datasheet Copilot API",
    description="电子元器件 Datasheet 智能解析系统",
    version="0.1.0"
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
    extract: bool = False
):
    """
    上传 PDF 并返回解析结果。
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

            # 1. PyMuPDF 提取文本与位置
            text_result = await run_in_threadpool(
                parser.parse,
                str(pdf_path)
            )

            # 2. Docling 识别表格结构
            table_result = await run_in_threadpool(
                table_parser.parse,
                str(pdf_path)
            )

            # 3. 合并解析结果
            result = {
                "filename": filename,
                "page_count": text_result["page_count"],
                "table_count": table_result["table_count"],
                "pages": text_result["pages"],
                "tables": table_result["tables"],
                "status": "parsed"
            }
            if extract:
                # 使用环境变量创建模型客户端
                extractor = ParameterExtractor()

                # 对表格进行参数提取
                extraction = await run_in_threadpool(
                    extractor.extract,
                    result["tables"]
                )

                # 检查参数来源
                verified = verify_evidence(
                    extraction,
                    result["tables"]
                )
                validator = ParameterValidator()

                validated_result = validator.validate(
                    verified
                )

                result["extraction"] = validated_result

                

                result["status"] = "extracted"

            result["filename"] = filename

            return result

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

"""FastAPI приложение для обработки документов."""

import uuid
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import sys
from pathlib import Path as PathLib

# Добавляем корневую директорию в путь для импортов
root_dir = PathLib(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from utils.logger import setup_logging
from utils.config import load_settings, Settings
from core.ai_client import AIClient
from storage.database import Database
from core.document_processor import DocumentProcessor

# Настройка логирования
setup_logging("INFO")
logger = logging.getLogger(__name__)

# Инициализация приложения
app = FastAPI(
    title="Document Processing API",
    description="API для обработки PDF счетов и других документов",
    version="1.0.0"
)

# Глобальные объекты (инициализируются при старте)
settings: Optional[Settings] = None
ai_client: Optional[AIClient] = None
db: Optional[Database] = None
processor: Optional[DocumentProcessor] = None


@app.on_event("startup")
async def startup_event():
    """Инициализация при старте приложения."""
    global settings, ai_client, db, processor
    
    logger.info("Инициализация приложения...")
    settings = load_settings()
    
    ai_client = AIClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        max_concurrency=settings.max_openai_concurrency
    )
    
    db = Database(settings.db_path)
    processor = DocumentProcessor(ai=ai_client, db=db)
    
    logger.info("Приложение готово к работе")


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении приложения."""
    logger.info("Завершение работы приложения")


# Модели ответов
class ProcessResponse(BaseModel):
    job_id: str
    status: str
    cached: bool
    result: Optional[dict] = None
    stats: dict
    billing: dict


class BatchProcessResponse(BaseModel):
    job_id: str
    status: str
    files_processed: int
    results: List[ProcessResponse]
    total_stats: dict
    total_billing: dict


class StatsResponse(BaseModel):
    total_runs: int
    cached_runs: int
    total_openai_requests: int
    total_tokens_in: int
    total_tokens_out: int
    total_cost_usd: float
    total_elapsed_ms: int
    avg_elapsed_ms: int
    savings_usd: float  # Экономия за счет кэша


def _calculate_cost(tokens_in: int, tokens_out: int, price_in: float, price_out: float) -> float:
    """Рассчитать стоимость в USD."""
    cost_in = (tokens_in / 1000.0) * price_in
    cost_out = (tokens_out / 1000.0) * price_out
    return cost_in + cost_out


def _process_file_internal(
    file: UploadFile,
    client_id: Optional[str] = None,
    job_id: Optional[str] = None
) -> ProcessResponse:
    """Внутренняя функция обработки файла."""
    if not processor or not ai_client or not db:
        raise HTTPException(status_code=500, detail="Приложение не инициализировано")
    
    if job_id is None:
        job_id = str(uuid.uuid4())
    
    start_time = time.time()
    
    # Сбрасываем счетчики перед обработкой
    ai_client.reset_counters()
    processor.reset_stats()
    
    # Читаем файл
    file_bytes = file.file.read()
    file_path = Path(file.filename)
    
    # Сохраняем во временный файл для обработки
    # Используем tempfile для кроссплатформенности
    temp_dir = Path("out/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"{job_id}_{file.filename}"
    try:
        temp_file.write_bytes(file_bytes)
        
        # Вычисляем хеш
        file_hash = processor.get_file_hash(temp_file)
        
        # Обрабатываем файл (process_file сам проверит кэш и увеличит счетчики)
        result = processor.process_file(temp_file)
        
        # Определяем, был ли использован кэш по статистике
        stats = processor.get_stats()
        cached = stats.get("cache_hit", 0) > 0
        
        if cached:
            logger.info(f"Cache hit for {file.filename} (job_id: {job_id})")
        else:
            logger.info(f"Cache miss for {file.filename} (job_id: {job_id})")
        
        # Получаем статистику
        elapsed_ms = int((time.time() - start_time) * 1000)
        processor_stats = processor.get_stats()
        openai_requests = ai_client.get_request_count()
        tokens_in, tokens_out = ai_client.get_tokens()
        
        # Рассчитываем стоимость
        cost_usd = _calculate_cost(
            tokens_in, tokens_out,
            settings.price_per_1k_in,
            settings.price_per_1k_out
        )
        
        # Сохраняем в базу
        db.save_run(
            job_id=job_id,
            client_id=client_id,
            filename=file.filename,
            file_hash=file_hash,
            cached=cached,
            openai_requests=openai_requests,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            elapsed_ms=elapsed_ms
        )
        
        return ProcessResponse(
            job_id=job_id,
            status="success",
            cached=cached,
            result=result,
            stats={
                "cache_hit": processor_stats.get("cache_hit", 0),
                "cache_miss": processor_stats.get("cache_miss", 0),
                "openai_requests": openai_requests,
                "elapsed_ms": elapsed_ms,
            },
            billing={
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": round(cost_usd, 6),
            }
        )
    except Exception as e:
        logger.error(f"Ошибка при обработке файла {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
    finally:
        # Удаляем временный файл
        if temp_file.exists():
            temp_file.unlink()


@app.get("/health")
async def health():
    """Проверка здоровья сервиса."""
    return {
        "status": "ok",
        "service": "document-processing-api"
    }


@app.post("/v1/process", response_model=ProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    x_client_id: Optional[str] = Header(None, alias="X-Client-ID")
):
    """
    Обработать один документ.
    
    - **file**: Файл для обработки (PDF, JPG, PNG и т.д.)
    - **X-Client-ID**: Опциональный идентификатор клиента
    """
    return _process_file_internal(file, client_id=x_client_id)


@app.post("/v1/process/batch", response_model=BatchProcessResponse)
async def process_batch(
    files: List[UploadFile] = File(...),
    x_client_id: Optional[str] = Header(None, alias="X-Client-ID")
):
    """
    Обработать несколько документов.
    
    - **files**: Список файлов для обработки
    - **X-Client-ID**: Опциональный идентификатор клиента
    """
    if not processor or not ai_client or not db:
        raise HTTPException(status_code=500, detail="Приложение не инициализировано")
    
    job_id = str(uuid.uuid4())
    results = []
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost_usd = 0.0
    total_elapsed_ms = 0
    
    for file in files:
        try:
            response = _process_file_internal(file, client_id=x_client_id, job_id=job_id)
            results.append(response)
            total_tokens_in += response.billing["tokens_in"]
            total_tokens_out += response.billing["tokens_out"]
            total_cost_usd += response.billing["cost_usd"]
            total_elapsed_ms += response.stats["elapsed_ms"]
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {file.filename}: {e}")
            results.append(ProcessResponse(
                job_id=job_id,
                status="error",
                cached=False,
                result=None,
                stats={"error": str(e)},
                billing={}
            ))
    
    return BatchProcessResponse(
        job_id=job_id,
        status="completed",
        files_processed=len(results),
        results=results,
        total_stats={
            "total_files": len(files),
            "successful": len([r for r in results if r.status == "success"]),
            "failed": len([r for r in results if r.status == "error"]),
            "total_elapsed_ms": total_elapsed_ms,
        },
        total_billing={
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_cost_usd": round(total_cost_usd, 6),
        }
    )


@app.get("/v1/stats", response_model=StatsResponse)
async def get_stats(
    x_client_id: Optional[str] = Header(None, alias="X-Client-ID")
):
    """
    Получить статистику обработки.
    
    - **X-Client-ID**: Опциональный идентификатор клиента для фильтрации
    """
    if not db:
        raise HTTPException(status_code=500, detail="Приложение не инициализировано")
    
    stats = db.get_runs_stats(client_id=x_client_id)
    
    # Рассчитываем экономию (если бы все запросы были без кэша)
    # Это упрощенная оценка - реальная экономия зависит от конкретных токенов
    cached_runs = stats["cached_runs"]
    avg_cost = stats["total_cost_usd"] / stats["total_runs"] if stats["total_runs"] > 0 else 0.0
    savings_usd = cached_runs * avg_cost  # Примерная экономия
    
    return StatsResponse(
        total_runs=stats["total_runs"],
        cached_runs=stats["cached_runs"],
        total_openai_requests=stats["total_openai_requests"],
        total_tokens_in=stats["total_tokens_in"],
        total_tokens_out=stats["total_tokens_out"],
        total_cost_usd=round(stats["total_cost_usd"], 6),
        total_elapsed_ms=stats["total_elapsed_ms"],
        avg_elapsed_ms=stats["avg_elapsed_ms"],
        savings_usd=round(savings_usd, 6)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


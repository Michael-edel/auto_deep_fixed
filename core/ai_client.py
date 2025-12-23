"""Клиент для работы с OpenAI API."""

import base64
import json
import logging
import threading
from typing import Optional

from openai import OpenAI, AsyncOpenAI

from models.document import DocumentData

logger = logging.getLogger(__name__)

class AIClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", max_concurrency: int = 2):
        if not api_key:
            raise ValueError("OPENAI_API_KEY не указан")

        api_key = self._clean_api_key(api_key)
        self._validate_api_key(api_key)

        self.api_key = api_key
        self.model = model
        self.sync_client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        # Semaphore для ограничения одновременных запросов к OpenAI
        self._semaphore = threading.Semaphore(max_concurrency)

        logger.info("AIClient инициализирован (model=%s, key=%s..., max_concurrency=%d)", 
                   model, api_key[:10], max_concurrency)

    @staticmethod
    def _clean_api_key(api_key: str) -> str:
        cleaned = (api_key or "").strip().replace("\n", "").replace("\r", "")
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]
        return cleaned

    @staticmethod
    def _validate_api_key(api_key: str) -> None:
        if not api_key.startswith(("sk-", "sk-proj-")):
            raise ValueError("Неверный формат API ключа (ожидается 'sk-' или 'sk-proj-').")
        try:
            api_key.encode("ascii")
        except UnicodeEncodeError as e:
            raise ValueError("API ключ содержит недопустимые символы (не ASCII).") from e

    @staticmethod
    def encode_image(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    def get_document_prompt(self) -> str:
        return """Ты — точный парсер бухгалтерских документов Казахстана. Твоя задача — извлечь данные БЕЗ выдумок и без изменения чисел.
Если чего-то нет в документе — ставь null или пустую строку.

Верни ОДИН JSON-объект со следующими ключами:

БАЗОВЫЕ ПОЛЯ:
1) document_type: строка (например: "счет на оплату", "накладная", "договор", "паспорт", "unknown")
2) iin_bin: строка из 12 цифр (ИИН/БИН компании-продавца/поставщика; если есть)
3) company_name: наименование ПРОДАВЦА/ПОСТАВЩИКА (кто выставил счет)
4) document_number: номер документа (например "ЦБ-11777")
5) document_date: дата документа (любой читаемый формат)
6) subtotal: сумма без НДС (если есть) иначе пусто
7) vat: сумма НДС (если есть) иначе пусто
8) total: итоговая сумма к оплате
9) items: массив позиций. Каждая позиция:
   - name
   - sku (если есть артикул/код)
   - quantity
   - unit
   - price
   - total

ДОПОЛНИТЕЛЬНО ДЛЯ СЧЕТА НА ОПЛАТУ (ОБЯЗАТЕЛЬНО ПЫТАЙСЯ НАЙТИ):
10) supplier: объект (поставщик) с полями: name, bin_iin, address, phone
11) buyer: объект (покупатель) с полями: name, bin_iin, address, phone
12) payment: объект реквизитов платежа:
    - beneficiary_bank_name (банк получателя)
    - beneficiary_bank_bik (БИК)
    - beneficiary_account_iban (счет/IBAN)
    - beneficiary_bin_iin (ИИН/БИН получателя)
    - kbe (КБе)
    - knp (КНП / код назначения платежа / "Код наз. пл." — если есть)
    - payment_code (код/уникальный код платежа, если указан)
    - payment_purpose (назначение платежа)
13) barcodes: массив строк (EAN-13/Code128/QR и т.п.), если присутствуют на документе или в строках товара.

КРИТИЧЕСКИЕ ПРАВИЛА:
- НИКОГДА не пересчитывай суммы и не "исправляй" totals/subtotal/vat. Верни как в документе.
- НИКОГДА не выдумывай реквизиты: если не уверен — null/"".
- company_name = поставщик (кто выставил документ), НЕ покупатель.
- Для document_number старайся вернуть именно номер счета из шапки (часто "Счет на оплату № ...").

Верни строго JSON, без пояснений и без текста вокруг.
"""

    async def analyze_document_async(self, image_bytes: bytes) -> DocumentData:
        try:
            base64_image = self.encode_image(image_bytes)
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.get_document_prompt() + ("\n\nТекст страницы (для надежности):\n" + extra_text if extra_text else "")},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                response_format={"type": "json_object"},
            )
            result_json = response.choices[0].message.content
            result_dict = json.loads(result_json)
            return DocumentData.from_dict(result_dict)
        except json.JSONDecodeError as e:
            logger.error("Ошибка декодирования JSON: %s", e)
            return DocumentData(document_type="unknown", error=f"Ошибка декодирования JSON: {e}")
        except Exception as e:
            logger.error("Ошибка при анализе документа: %s", e, exc_info=True)
            return DocumentData(document_type="unknown", error=self._format_error_message(e))

    def analyze_document_sync(self, image_bytes: bytes, extra_text: str | None = None) -> DocumentData:
        # Ограничение одновременных запросов к OpenAI
        with self._semaphore:
            try:
                base64_image = self.encode_image(image_bytes)
                response = self.sync_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.get_document_prompt() + ("\n\nТекст страницы (для надежности):\n" + extra_text if extra_text else "")},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            ],
                        }
                    ],
                    response_format={"type": "json_object"},
                )
                result_json = response.choices[0].message.content
                result_dict = json.loads(result_json)
                return DocumentData.from_dict(result_dict)
            except Exception as e:
                logger.error("Ошибка при синхронном анализе документа: %s", e, exc_info=True)
                return DocumentData(document_type="unknown", error=self._format_error_message(e))

    @staticmethod
    def _format_error_message(error: Exception) -> str:
        s = str(error)
        if "401" in s or "Unauthorized" in s or "invalid_api_key" in s:
            return "Ошибка авторизации OpenAI API (401). Проверьте ключ и баланс."
        if "codec" in s.lower() or "encode" in s.lower():
            return "Ошибка кодировки. Проверьте API ключ."
        return f"Ошибка при анализе документа: {s}"

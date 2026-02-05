from pathlib import Path
import base64
import re
from typing import Optional
import os
import pandas as pd

from openai import OpenAI

from model.llm_model_atod import (
    dataframe_to_flat_sequence,
    generate_plantuml_code,
)

from model.plantuml_runtime import render_plantuml


client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

MAX_FILE_MB = 5


# =========================================================
# Public API
# =========================================================

def analyze(
    *,
    file_path: Optional[str],
    input_type: str,
) -> dict:

    if not file_path:
        raise ValueError("file_path is required")

    _check_file_size(file_path)

    if input_type == "diagram":
        return _analyze_diagram(file_path)

    if input_type == "csv":
        return _analyze_csv(file_path)

    raise ValueError(f"Unsupported input_type: {input_type}")


# =========================================================
# CSV → PNG (production)
# =========================================================

def _analyze_csv(file_path: str) -> dict:

    df = pd.read_csv(file_path)

    flat_sequence = dataframe_to_flat_sequence(df)
    plantuml_code = generate_plantuml_code(flat_sequence)
    
    print("\n=== PLANTUML CODE ===\n")
    print(plantuml_code)
    print("\n=====================\n")

    
    if "@startuml" not in plantuml_code:
        plantuml_code = "@startuml\n" + plantuml_code
        
    if "@enduml" not in plantuml_code:
        plantuml_code = plantuml_code + "\n@enduml"

    png_path = render_plantuml(plantuml_code)
    
    png_base64 = base64.b64encode(png_path.read_bytes()).decode()

    return {
        "png_base64": png_base64
    }


# =========================================================
# Diagram → table (без изменений логики)
# =========================================================

def _analyze_diagram(file_path: str) -> dict:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        diagram_data = _image_to_base64(path)
        source_type = "image"
    else:
        diagram_data = path.read_text(encoding="utf-8", errors="ignore")
        source_type = _detect_text_diagram_type(diagram_data)

    filename = path.name

    if source_type == "image":
        system_prompt, user_prompt = build_table_prompt_from_img_atod()

        result = _call_llm_image(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_data=diagram_data,
        )
    else:
        prompt = build_table_prompt_from_txt_atod(
            source_type=source_type,
            filename=filename,
            diagram_data=diagram_data,
        )

        result = _call_llm_text(prompt)

    return {
        "filename": filename,
        "source_type": source_type,
        "raw": result["raw"],
    }


# =========================================================
# Helpers
# =========================================================

def _check_file_size(path: str):
    size_mb = Path(path).stat().st_size / 1024 / 1024
    if size_mb > MAX_FILE_MB:
        raise ValueError(f"File too large (> {MAX_FILE_MB} MB)")


def _image_to_base64(path: Path) -> str:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")

    ext = path.suffix.lower()
    mime = "image/png"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"

    return f"data:{mime};base64,{encoded}"


def _detect_text_diagram_type(text: str) -> str:
    t = text.lower()
    if "<bpmn" in t:
        return "bpmn_xml"
    if "<mxfile" in t:
        return "drawio_xml"
    if "@startuml" in t:
        return "plantuml"
    return "text_diagram"


def _safe_output(text: str) -> dict:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:csv|text|json)?", "", text)
        text = re.sub(r"```$", "", text).strip()

    return {"raw": text}


# =========================================================
# LLM
# =========================================================

def _call_llm_text(prompt: str) -> dict:
    MAX_CHARS = 2000   # ~1500 токенов

    if len(prompt) > MAX_CHARS:
        prompt = prompt[:MAX_CHARS] + "\n\n[TRUNCATED]"

    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-2B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return _safe_output(response.choices[0].message.content)


def _call_llm_image(*, system_prompt: str, user_prompt: str, image_data: str) -> dict:
    MAX_CHARS = 2000   # ~1500 токенов

    if len(system_prompt) > MAX_CHARS:
        system_prompt = system_prompt[:MAX_CHARS] + "\n\n[TRUNCATED]"
    if len(user_prompt) > MAX_CHARS:
        user_prompt = user_prompt[:MAX_CHARS] + "\n\n[TRUNCATED]"
    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-2B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        temperature=0.0,
    )
    return _safe_output(response.choices[0].message.content)


# =========================================================
# Prompts
# =========================================================

def build_table_prompt_from_txt_atod(source_type: str, filename: str, diagram_data: str) -> str:
    return f"""
Проанализируй данные диаграммы и представь результат в ВИДЕ ТЕКСТОВОЙ ТАБЛИЦЫ на русском языке.

ТИП ИСТОЧНИКА: {source_type}
ИМЯ ФАЙЛА: {filename}

ПРАВИЛА ФОРМАТИРОВАНИЯ ТАБЛИЦЫ:
1. Используй формат Markdown (символы | и - для разделителей)
2. Выравнивай колонки для читаемости
3. Не используй JSON, XML или код — только таблицу и краткое резюме сверху
4. Для длинных текстов в ячейках используй переносы через <br> или сокращения

ТИПЫ ТАБЛИЦ В ЗАВИСИМОСТИ ОТ ДИАГРАММЫ:

ДЛЯ ПРОЦЕССОВ (BPMN, Flowchart, UML Activity):
| № | Тип элемента       | Название / Действие      | Участник / Пул | Условие перехода | Предыдущий шаг | Следующий шаг |
|---|--------------------|--------------------------|----------------|------------------|----------------|---------------|
| 1 | Стартовое событие  | Открытие рабочего пространства | Пользователь   | —                | —              | Выбор шаблона |
| 2 | Шлюз (решение)     | Выбор шаблона            | Система        | «Выбран шаблон?» | Шаг 1          | Шаг 3 (Да)<br>Шаг 4 (Нет) |

ДЛЯ СТРУКТУРНЫХ ДИАГРАММ (C4 Model, UML Class):
| Уровень | Имя элемента    | Тип          | Описание                     | Связи / Зависимости       |
|---------|-----------------|--------------|------------------------------|---------------------------|
| Person  | Организатор     | Актор        | Создаёт страницы             | → Вики-платформа          |
| System  | Вики-платформа  | Система      | Хостинг совместных документов| ← Организатор<br>→ Участники |

ДЛЯ СЕТЕЙ / ГРАФОВ (Mermaid, Graphviz):
| Узел / Компонент | Тип    | Входящие связи | Исходящие связи | Описание          |
|------------------|--------|----------------|-----------------|-------------------|
| Сервер А         | Сервер | —              | → Сервер Б      | Обработка запросов|

ЭТАПЫ АНАЛИЗА:
1. Определи тип диаграммы по структуре данных (BPMN, C4, Flowchart и т.д.)
2. Извлеки все значимые элементы с их атрибутами (название, тип, участник, условия)
3. Определи связи между элементами (последовательность, зависимости)
4. Сгруппируй элементы по логическим категориям (пулы, уровни архитектуры)
5. Сформируй таблицу согласно шаблону выше

ВАЖНО:
- Анализируй ТОЛЬКО предоставленные данные — не выдумывай отсутствующие элементы
- Если данные неполные (обрезаны) — укажи это в резюме, но всё равно построй таблицу по доступной информации
- Для .vpd: ориентируйся на метаданные (тип диаграммы) и фрагменты данных
- Верни ТОЛЬКО таблицу в формате Markdown + краткое резюме сверху (2–3 предложения)
- Не добавляй пояснения под таблицей — только чистая таблица

РЕЗЮМЕ (2–3 предложения сверху таблицы):
[Кратко опиши тип диаграммы и её назначение]

ТАБЛИЦА ЭЛЕМЕНТОВ:
[Таблица в формате Markdown]

ДАННЫЕ ДЛЯ АНАЛИЗА:
{diagram_data}
"""


def build_table_prompt_from_img_atod():
    system_prompt = ("Ты — анализатор BPMN-диаграмм. ТВОЯ ЗАДАЧА: определить не только элементы, но и СТРУКТУРУ ПОТОКА — "
    "какие элементы соединены стрелками, в каком порядке выполняются шаги, и какие условия управляют переходами.\n\n"
    
    "СТРОГИЕ ПРАВИЛА:\n"
    "1. Анализируй ТОЛЬКО видимые стрелки/соединители на изображении.\n"
    "2. Если стрелка ведёт от элемента А к элементу Б — запиши это как переход.\n"
    "3. Если на стрелке есть текст условия («Да», «Нет», «Есть права») — извлеки его точно.\n"
    "4. НИКОГДА не выдумывай переходы между элементами, которые не соединены стрелкой на изображении.\n"
    "5. Для шлюзов (ромбы) укажи ВСЕ исходящие переходы с условиями.\n"
    "6. Верни ТОЛЬКО CSV без пояснений или примеров.")
    user_prompt = ("ЗАДАЧА: Проанализируй диаграмму и извлеки:\n"
    "а) Все элементы (блоки, события, шлюзы)\n"
    "б) Все переходы между элементами (стрелки) с условиями\n\n"
    
    "ФОРМАТ ВЫВОДА — ОДНА ТАБЛИЦА CSV:\n"
    "id,type,name,pool,prev_elements,next_elements,conditions\n\n"
    
    "ОПИСАНИЕ КОЛОНОК:\n"
    "• id: порядковый номер элемента (1, 2, 3...)\n"
    "• type: тип элемента (start_event, end_event, task, gateway_xor, gateway_and, event_message)\n"
    "• name: ТОЧНЫЙ текст из элемента\n"
    "• pool: название пула/лэйна (или '—' если нет)\n"
    "• prev_elements: ID предыдущих элементов (через запятую, например: '1,3')\n"
    "• next_elements: ID следующих элементов (через запятую, например: '4,5')\n"
    "• conditions: условия для переходов к следующим элементам (через запятую, например: 'Да,Нет')\n\n"
    
    "ПРАВИЛА ЗАПОЛНЕНИЯ ПЕРЕХОДОВ:\n"
    "• Для стартового события: prev_elements = '—', next_elements = ID следующего шага\n"
    "• Для конечного события: next_elements = '—', prev_elements = ID предыдущего шага\n"
    "• Для шлюза решения (gateway_xor):\n"
    "  - next_elements = '4,5' (два исходящих перехода)\n"
    "  - conditions = 'Да,Нет' (текст с каждой стрелки)\n"
    "• Для параллельного шлюза (gateway_and): conditions = '—'\n"
    "• Если переход без условия — укажи '—' в соответствующей позиции conditions\n"
    "• Если элемент не имеет предыдущих/следующих — укажи '—'\n\n"
    
    "КРИТИЧЕСКИ ВАЖНО:\n"
    "• Определяй переходы ТОЛЬКО по видимым стрелкам на изображении.\n"
    "• Не предполагай логику процесса — только то, что видно визуально.\n"
    "• МАКСИМУМ 25 элементов. После последней строки — немедленно остановись.\n\n"
    
    "ПРИМЕР ВАЛИДНОЙ СТРОКИ:\n"
    "2,gateway_xor,Выбран шаблон?,Организатор,1,3,4,Да,Нет")
    return system_prompt, user_prompt




# from .plantuml_runtime import render_puml_to_png
# from .debug_stubs import get_debug_table, get_example_puml


# class ModelService:
#     def csv_to_png(self, file_bytes: bytes) -> bytes:
#         puml = get_example_puml()
#         return render_puml_to_png(puml)

#     def diagram_to_table(self, file_bytes: bytes) -> str:
#         return get_debug_table()

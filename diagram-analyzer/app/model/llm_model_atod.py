import re
from pathlib import Path
from datetime import datetime
from typing import Tuple

import pandas as pd
from openai import OpenAI


client = OpenAI()


# =========================================================
# CSV → flat sequence
# =========================================================

def dataframe_to_flat_sequence(df: pd.DataFrame) -> str:
    """Сериализация таблицы процесса в линейное текстовое описание."""

    if df.empty:
        return "→ ШАГ: Пустой процесс"

    def safe_val(val) -> str:
        if pd.isna(val) or val in ["—", "-", ""]:
            return ""
        return str(val).strip()

    # колонка участника
    group_col = next(
        (
            col
            for col in df.columns
            if any(kw in col.lower() for kw in ["участник", "пул", "actor", "role", "lane"])
        ),
        None,
    )

    # колонка названия шага
    text_col = next(
        (
            col
            for col in df.columns
            if any(kw in col.lower() for kw in ["название", "действие", "имя", "name", "label", "activity"])
        ),
        df.columns[0],
    )

    steps: list[str] = []
    current_group = None

    for _, row in df.iterrows():
        group_val = safe_val(row[group_col]) if group_col else ""
        text_val = safe_val(row[text_col]) or "Без названия"

        attrs = []
        for col in df.columns:
            if col in (group_col, text_col):
                continue
            val = safe_val(row[col])
            if val:
                attrs.append(f"{col}={val}")

        if group_val and group_val != current_group:
            current_group = group_val
            steps.append(f"\n[УЧАСТНИК: {group_val}]")

        attr_str = " | " + " | ".join(attrs) if attrs else ""
        steps.append(f"→ ШАГ: {text_val}{attr_str}")

    return "\n".join(steps) if steps else "→ ШАГ: Процесс без шагов"


# =========================================================
# flat sequence → PlantUML (через LLM)
# =========================================================

def generate_plantuml_code(flat_sequence: str) -> str:
    prompt = f"""Преобразуй последовательность шагов в ВАЛИДНЫЙ код PlantUML.

КРИТИЧЕСКИ ВАЖНО:
1. НИКОГДА не используй квадратные скобки внутри двоеточия (:Текст; а НЕ :[Текст];)
2. Обязательно включи start в начале и stop в конце
3. Верни ТОЛЬКО код между @startuml и @enduml

ПРИМЕР ВАЛИДНОГО КОДА:
@startuml
|Пользователь|
start
:Открытие рабочего пространства;
if (Выбран шаблон?) then (Да)
  :Применить шаблон;
else (Нет)
  :Создать пустую страницу;
endif
:Добавить контент;
|Система|
:Получено приглашение;
:Открыть страницу;
if (Есть права?) then (Да)
  :Редактировать;
else (Нет)
  :Просмотреть;
endif
:Синхронизировать изменения;
stop
@enduml

ПОСЛЕДОВАТЕЛЬНОСТЬ:
{flat_sequence}

ВЕРНИ ТОЛЬКО ЧИСТЫЙ КОД PLANTUML.

"""

    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-2B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        # max_tokens=2048,
        top_p=1.0
    )

    code = response.choices[0].message.content.strip()

    # убрать markdown fences
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z]*", "", code)
        code = re.sub(r"```$", "", code).strip()

    # гарантируем start / stop
    if "start" not in code:
        code = code.replace("@startuml", "@startuml\nstart")

    if "stop" not in code:
        code = code.replace("@enduml", "stop\n@enduml")

    return code

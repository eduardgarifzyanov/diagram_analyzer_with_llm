import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

PLANTUML_JAR = Path("/runtime/plantuml.jar")
OUTPUT_DIR = Path("/runtime/output")


def ensure_runtime():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def render_plantuml(puml_text: str) -> Path:
    """
    Рендерит PlantUML → PNG.
    Гарантированно возвращает PNG даже при ошибке UML.
    """
    ensure_runtime()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        puml_path = tmp_dir / "diagram.puml"
        png_path = tmp_dir / "diagram.png"

        puml_path.write_text(puml_text, encoding="utf-8")

        result = subprocess.run(
            ["java", "-jar", str(PLANTUML_JAR), str(puml_path)],
            capture_output=True,
            text=True,
        )

        print("\n=== PLANTUML STDOUT ===\n", result.stdout)
        print("\n=== PLANTUML STDERR ===\n", result.stderr)

        # если PNG не создан — делаем fallback-диаграмму
        if not png_path.exists():
            fallback = """
@startuml
title Ошибка генерации диаграммы

:PlantUML не смог построить диаграмму;
:Проверь входные данные CSV;

@enduml
"""
            puml_path.write_text(fallback, encoding="utf-8")

            subprocess.run(
                ["java", "-jar", str(PLANTUML_JAR), str(puml_path)],
                check=True,
            )

            if not png_path.exists():
                raise RuntimeError("PlantUML failed even on fallback diagram")

        final_name = f"diagram_{datetime.now():%Y%m%d_%H%M%S}.png"
        final_path = OUTPUT_DIR / final_name
        final_path.write_bytes(png_path.read_bytes())

        return final_path


FROM python:3.11-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app


# system deps
RUN apt-get update && apt-get install -y \
default-jre \
graphviz \
curl \
&& rm -rf /var/lib/apt/lists/*


# install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# copy project
COPY . .


# runtime dir for plantuml
RUN mkdir -p /runtime


# download plantuml
RUN curl -L https://github.com/plantuml/plantuml/releases/latest/download/plantuml.jar \
-o /runtime/plantuml.jar


EXPOSE 8000


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
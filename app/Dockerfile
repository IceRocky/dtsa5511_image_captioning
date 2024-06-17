FROM tiangolo/uvicorn-gunicorn:python3.10

COPY requirements.txt /app/requirements.txt
COPY main.py /app/main.py
COPY model.py /app/model.py
COPY training_utils.py /app/training_utils.py
COPY data_processing.py /app/data_processing.py

COPY assets/XL_model_config.json /app/assets/XL_model_config.json
COPY assets/XL_model_weights.h5 /app/assets/XL_model_weights.h5
COPY assets/XL_tokenizer.keras /app/assets/XL_tokenizer.keras


WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8012

ENV MODULE_NAME="main"
ENV VARIABLE_NAME="app"
ENV PORT="8012"



CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8012"]
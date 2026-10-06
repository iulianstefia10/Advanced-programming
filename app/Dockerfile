FROM python:3.10-slim
WORKDIR /app
RUN pip install Flask
RUN pip install Flask-SQLAlchemy
RUN pip install numpy
RUN pip install keras
RUN pip install Pillow
RUN pip install matplotlib
RUN pip install influxdb-flask
RUN pip install tensorflow
RUN pip install Werkzeug
COPY . /app
EXPOSE 5004
CMD ["python", "app.py"]

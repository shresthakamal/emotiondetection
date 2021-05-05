FROM python:3.6

EXPOSE 5000

WORKDIR /app


COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

CMD ["python", "-m", "api.app"]
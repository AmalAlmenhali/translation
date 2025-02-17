docker build -t translation-api-server -f Dockerfile .

docker run -p 8000:8000 -it --rm translation-api-server

curl "http://localhost:8000/v1/translate?sentence=she%20is%20driving%20the%20truck"

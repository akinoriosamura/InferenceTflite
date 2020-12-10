# InferenceTflite
## prepare
```
docker-compose up -d
docker attach [contaqiner id]
pipenv install --system
```

## run
```
python inference.py -i img.png -m model.tflite
```
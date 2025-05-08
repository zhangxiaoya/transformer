# transformer

## Environment setting

1. for simple use
```
pip install -r requirements.txt
```

2. for GPU
```
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple torch==2.0.0
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple torchtext==0.15.1
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple torchdata
pip install 'portalocker>=2.0.0'
pip install spacy==3.2.6
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

3. For cpu
```
pip install torch==2.2.0
pip install torchtext==0.16.2
pip install numpy=1.26.4 thinc==8.3.3 spacy
pip install 'portalocker>=2.0.0'
pip install torchdata
```
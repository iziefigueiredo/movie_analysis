
## Getting Started  

### Create and activate a virtual environment  

```
python -m venv .venv

```

```
source .venv/bin/activate   # Linux/Mac

```

```
.venv\Scripts\activate      # Windows

```
### Install dependencies

```
pip install -r requirements.txt

```

### Run the pipeline  
```
python main.py
```



# Project structure
```
movie_analysis/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── imdb_predictor.pkl
│   └── model_features.pkl
├── reports/
│   ├── stats/
│   ├── viz/
├── src/
│   ├── __init__.py
│   ├── processing_imdb.py
│   ├── processing_oscar.py
│   ├── profile_imdb_oscar.py
│   ├── eda_imdb_oscar.py
│   ├── merge_data.py
│   ├── modeling.py
│   └── predict.py
├── notebooks/
│   ├── answers.ipynb
├── config.yaml
├── main.py
├── .gitignore
├── README.md
└── requirements.txt
```


# Movie Analysis Project

This repository implements a complete **data science pipeline** for movie analysis, combining data from **IMDB** and the **Oscars**. 
The primary goal is to **predict a movie's IMDB rating** based on features such as genre, box office revenue, directors, and actors. 
In addition, the project includes exploratory analyses to extract insights into what makes a movie successful.

## Visualizations




## Exploratory Data Analysis (EDA)

The exploratory analysis combines **descriptive statistics** and **visualizations**, 
generated with [`src/eda_imdb_oscar.py`](src/eda_imdb_oscar.py) and exported to 
[`reports/viz/index.html`](reports/viz/index.html).

Key analyses include:

- **Distributions** of IMDB ratings, number of votes, box office revenue, and metascore. 
- **Average IMDB rating by genre** → Dramas dominate in count, while *History* and *Biography* show stronger averages. 
- **Oscar win rate by genre** → *War* and *History* genres stand out with higher victory rates. 
- **Revenue comparison** between Oscar competitors and non-competitors. 
- **Boxplot of IMDB Ratings** contrasting winners vs non-winners. 
- **Correlation heatmap** showing a strong relationship between votes and revenue. 
- **Actors analysis**: 
  - Most frequent actors in the dataset. 
  - Actors most often linked with Oscar wins. 
  - Actors with the highest average IMDB ratings. 
  - Actor × Genre participation heatmap.


---

## Model Performance

I trained three regression models to predict the **IMDB Rating** based on genres, actors, directors, and additional features. 
The evaluation metrics used were **Mean Absolute Error (MAE)** and **R² (coefficient of determination)**. 

| Model              | MAE   | R²    |
|--------------------|-------|-------|
| Random Forest      | 0.164 | 0.493 |
| Gradient Boosting  | 0.157 | 0.531 |
| Linear Regression  | 0.170 | 0.447 |

### Key Takeaways
- **Gradient Boosting** achieved the best overall performance, with the **lowest MAE (0.157)** and the **highest R² (0.531)**. 
- **Random Forest** also performed reasonably well, with slightly higher error but more stability compared to Linear Regression. 
- **Linear Regression**, while interpretable, showed the weakest predictive power in this context. 

➡️ Errors (MAE ~0.16) are relatively small, meaning predictions are on average within **0.16 rating points** of the true IMDB rating. 

➡️ However, R² values below 0.6 suggest that while the models capture important patterns, there is still **significant unexplained variance** — likely due to subjective factors in movie ratings (e.g., cultural impact, critical reception, or time trends) that are not fully captured by the available features.

---



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
│   ├── raw/                  # Raw data files (imdb.csv, oscar.csv)
│   └── processed/            # Cleaned and unified data (imdb_oscar.csv)
├── models/                   # Trained models (.pkl)
│   ├── rf_model.pkl
│   ├── gb_model.pkl
│   ├── lr_model.pkl
│   └── features.pkl
├── reports/
│   ├── viz/                  # Generated visualizations (.png, .html)
│   └── stats/                # Summary tables and statistics (.csv)
├── src/
│   ├── __init__.py
│   ├── processing_imdb.py
│   ├── processing_oscar.py
│   ├── merge_data.py
│   ├── eda_imdb_oscar.py
│   ├── modeling_regression.py
│   ├── predict.py
│   └── text_classifier.py
├── main.py
├── .gitignore
└── requirements.txt

```

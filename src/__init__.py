# src/__init__.py

# --- Módulos de Processamento de Dados ---
from .processing_imdb import load_data as load_imdb, clean_data as clean_imdb
from .processing_oscar import load_data as load_oscar, clean_data as clean_oscar

# --- Módulo de União de Dados ---
from .merge_data import (
    transform_genres,
    merge_imdb_data,
    transform_oscar_data,
    merge_oscar_data,
)

# --- Módulos de EDA e Profiling ---
from .profile_imdb_oscar import run_profile as run_profile
from .eda_imdb_oscar import run as run_eda

# --- Módulo de Modelagem ---
from .modeling import main as run_modeling

# --- Módulo de Predição ---
from .predict import (
    load_models,
    load_features,
    prepare_data as prepare_data_for_prediction,
    make_prediction,
    main as run_prediction
)

# --- Módulo de Análise de Texto ---
from .text_classifier import (
    analyze_tokens as run_text_analysis
)

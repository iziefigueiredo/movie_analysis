# src/__init__.py

from .processing_imdb import load_data as load_imdb, clean_data as clean_imdb
from .processing_oscar import load_data as load_oscar, clean_data as clean_oscar

from .merge_data import (
    transform_genres,
    merge_imdb_data,
    transform_oscar_data,
    merge_oscar_data,
)

from .profile_imdb import run_profile as profile_imdb
from .profile_imdb_oscar import run_profile as profile_imdb_oscar  

from .eda_imdb import run_all as eda_imdb


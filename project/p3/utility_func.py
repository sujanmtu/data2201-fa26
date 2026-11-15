"""Provided utilities for P3. No assessed interval-function solutions are included.

Data preparation is deterministic. Learned medians, scaling and category levels
are fitted by the student's training operations or estimator.fit calls.
"""
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def remove_outliers(data, variable, lower=-np.inf, upper=np.inf):
    values = pd.to_numeric(data[variable], errors='coerce')
    return data.loc[values.gt(lower) & values.lt(upper)].copy()

def log_transform(data, col):
    result = data.copy()
    result['Log ' + col] = np.log(pd.to_numeric(result[col], errors='coerce'))
    return result

def add_total_bedrooms(data):
    result = data.copy()
    result['Bedrooms'] = pd.to_numeric(result['Description'].astype('string').str.extract(
        r'(\d+(?:\.\d+)?) of which are bedrooms', expand=False), errors='coerce')
    return result

def find_expensive_neighborhoods(data, n=3, metric=np.median):
    return data.groupby('Neighborhood Code')['Sale Price'].agg(metric).nlargest(n).index.tolist()

def add_in_expensive_neighborhood(data, neighborhoods):
    result = data.copy()
    result['in_expensive_neighborhood'] = result['Neighborhood Code'].isin(neighborhoods).astype(int)
    return result

def ohe_roof_material(data):
    result = data.copy()
    return pd.concat([result, pd.get_dummies(result['Roof Material'], prefix='Roof Material', dtype=float)], axis=1)

def load_course_csv(filename):
    # Provided loader: both loose CSVs and nested members in the original ZIP work.
    if Path(filename).exists():
        frame = pd.read_csv(filename)
    elif Path('cook_county_data.zip').exists():
        with zipfile.ZipFile('cook_county_data.zip') as archive:
            matches = [n for n in archive.namelist() if Path(n).name == filename]
            if len(matches) != 1:
                raise FileNotFoundError(f'Expected one {filename} inside cook_county_data.zip.')
            with archive.open(matches[0]) as stream:
                frame = pd.read_csv(stream)
    else:
        raise FileNotFoundError(f'Place {filename} or cook_county_data.zip beside this notebook.')
    index_columns = [c for c in frame if str(c).startswith('Unnamed:')]
    if index_columns:
        frame = frame.set_index(index_columns[0]).drop(columns=index_columns[1:], errors='ignore')
        frame.index.name = None
    return frame

def summarize_log_predictions(prediction_columns, actual_log):
    actual_dollars = np.exp(np.asarray(actual_log))
    rows = []
    for name, prediction in prediction_columns.items():
        log_error = np.asarray(actual_log) - np.asarray(prediction)
        dollar_error = actual_dollars - np.exp(np.asarray(prediction))
        rows.append([np.sqrt(np.mean(log_error**2)),np.sqrt(np.mean(dollar_error**2)),np.mean(np.abs(dollar_error))])
    return pd.DataFrame(rows,index=list(prediction_columns),columns=['valid_log_rmse','dollar_rmse','dollar_mae'])

def extract_base_features(data):
    area = pd.to_numeric(data['Building Square Feet'],errors='coerce').replace([np.inf,-np.inf],np.nan)
    lot = pd.to_numeric(data['Lot Size'],errors='coerce').replace([np.inf,-np.inf],np.nan)
    bedrooms = add_total_bedrooms(data)['Bedrooms'].replace([np.inf,-np.inf],np.nan)
    age = pd.to_numeric(data['Age'],errors='coerce').replace([np.inf,-np.inf],np.nan)
    return pd.DataFrame({'log_area':np.log(area.where(area>0)),
                         'log_lot':np.log(lot.where(lot>0)),
                         'Bedrooms':bedrooms.where(bedrooms>=0),
                         'Age':age.where(age>=0),
                         'Town Code':data['Town Code'].astype('string').fillna('Missing'),
                         'Garage 1 Size':data['Garage 1 Size'].astype('string').fillna('Missing')},index=data.index)

def rmse(predicted, actual):
    return float(np.sqrt(np.mean((np.asarray(actual)-np.asarray(predicted))**2)))


def prepare_labeled_partition(data):
    """Filter the stated labeled population; add log columns without fitting statistics."""
    numeric = data[['Sale Price','Building Square Feet']].apply(pd.to_numeric,errors='coerce').astype(float)
    keep = np.isfinite(numeric).all(axis=1) & numeric['Sale Price'].gt(499) & numeric['Building Square Feet'].gt(0)
    result = data.loc[keep].copy()
    result['Log Sale Price'] = np.log(numeric.loc[keep,'Sale Price'])
    result['Log Building Square Feet'] = np.log(numeric.loc[keep,'Building Square Feet'])
    return result

def prepare_project_views(data, base_train, base_valid):
    """Reuse the same split and retained population for the pipeline comparison."""
    train_b, valid_b = base_train.copy(), base_valid.copy()
    for frame in [train_b,valid_b]:
        frame['Sale Price'] = pd.to_numeric(frame['Sale Price'])
    retained = data.index.isin(base_train.index.union(base_valid.index))
    labeled_b = data.loc[retained].copy()
    labeled_b['Sale Price'] = pd.to_numeric(labeled_b['Sale Price'])
    return train_b, valid_b, labeled_b

def prepare_candidate_numeric(data):
    """Return raw/log area, age and bedroom candidates; no parameters are learned."""
    area = pd.to_numeric(data['Building Square Feet'],errors='coerce').replace([np.inf,-np.inf],np.nan)
    age = pd.to_numeric(data['Age'],errors='coerce').replace([np.inf,-np.inf],np.nan)
    bedrooms = add_total_bedrooms(data)['Bedrooms'].replace([np.inf,-np.inf],np.nan)
    return pd.DataFrame({'area':area.where(area>0),'log_area':np.log(area.where(area>0)),
                         'age':age.where(age>=0),'bedrooms':bedrooms.where(bedrooms>=0)},index=data.index)

def prepare_explicit_predictors(data):
    """Extract log area, Bedrooms and Age, keeping missing entries for train-only imputation."""
    result = add_total_bedrooms(data)[['Log Building Square Feet','Bedrooms','Age']].apply(pd.to_numeric,errors='coerce').astype(float)
    result = result.replace([np.inf,-np.inf],np.nan)
    result.loc[result['Bedrooms']<0,'Bedrooms'] = np.nan
    result.loc[result['Age']<0,'Age'] = np.nan
    result.columns = ['log_area','bedrooms','age']
    return result

def make_valuation_model(baseline=False, expanded=True):
    """Build OLS pipelines for log-area, structural, or engineered-feature comparisons.

    baseline=True: log_area only.
    baseline=False, expanded=False: structural numerics and categories.
    baseline=False, expanded=True: add area/lot density and nonlinear age.
    All learned preprocessing is fitted inside estimator.fit.
    """
    numeric_columns = ['log_area'] if baseline else ['log_area','log_lot','Bedrooms','Age']
    if not baseline and expanded:
        numeric_columns += ['extra_feature','age_squared']
    transformers = [('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),numeric_columns)]
    if not baseline:
        transformers.append(('categorical',OneHotEncoder(handle_unknown='ignore'),['Town Code','Garage 1 Size']))
    return make_pipeline(ColumnTransformer(transformers),lm.LinearRegression(fit_intercept=True))

def build_assessment_audit(actual_log, predicted_log):
    """Create the cached held-out frame and its two unrounded-median price groups."""
    actual_log = pd.Series(actual_log).copy()
    predicted_log = np.asarray(predicted_log)
    frame = pd.DataFrame({'True Log Sale Price':actual_log,
                          'Predicted Log Sale Price':predicted_log,
                          'True Sale Price':np.exp(actual_log),
                          'Predicted Sale Price':np.exp(predicted_log)},index=actual_log.index)
    cut = float(np.median(actual_log))
    return frame, frame.loc[frame['True Log Sale Price']<=cut].copy(), frame.loc[frame['True Log Sale Price']>cut].copy()

__all__ = [
    'remove_outliers','log_transform','add_total_bedrooms',
    'find_expensive_neighborhoods','add_in_expensive_neighborhood','ohe_roof_material',
    'load_course_csv','summarize_log_predictions','extract_base_features','rmse',
    'prepare_labeled_partition','prepare_project_views','prepare_candidate_numeric',
    'prepare_explicit_predictors','make_valuation_model','build_assessment_audit'
]

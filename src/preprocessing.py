# %%
import pandas as pd
import numpy as np
from pathlib import Path

# project root (parent of src/)
ROOT = Path(__file__).resolve().parents[1]

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)

# %%
df = pd.read_csv(ROOT / 'data/raw/Train_rev1.csv')

# %%
df.head()

# %%
df.info()

# %%
df.describe(include='all')

# %%
df.shape

# %%
"""
# 1. Drop columns
"""

# %%
df.drop(columns=['Id', 'SalaryRaw', 'LocationRaw'], inplace=True)

# %%
"""
# 2. Fill missing values
"""

# %%
print('Missing values:')
df.isna().sum()/df.shape[0]*100

# %%
def fill_missing(df):
    for col in df.columns:
        if df[col].dtype == 'O':  # object/string
            df = df.fillna(df[col].mode()[0])
        else:  # numbers
            df = df.fillna(df[col].mean())
    return df

# %%
df = fill_missing(df)

# %%
df.isna().sum()/df.shape[0]*100

# %%
"""
# 3. Duplicates
"""

# %%
df.duplicated().sum()

# %%
"""
# 4. Geostandarization
"""

# %%
"""
## 4.1. Get population data from geonames dataset
"""

# %%
# selecting data only for GB - turn on once (long)
# cols = [
#     'geonameid','name','asciiname','alternatenames','lat','lon',
#     'feature_class','feature_code','country_code','cc2','admin1',
#     'admin2','admin3','admin4','population','elevation','dem','tz','moddate'
# ]

# geonames = pd.read_csv(
#     "allCountries.txt",
#     sep="\t",
#     names=cols,
#     usecols=['asciiname', 'alternatenames', 'country_code', 'feature_code', 'feature_class', 'admin1', 'admin2', 'admin3', 'lon', 'lat', 'population'],
#     dtype=str,
#     header=None
# )

# geonames_gb = geonames[geonames['country_code'] == 'GB'].copy().reset_index(drop=True)
# geonames_gb = geonames_gb[geonames_gb['feature_class'].isin(['P', 'A'])].reset_index()
# geonames_gb.loc[geonames_gb['feature_code'] == 'PCLI', 'asciiname'] = 'UK'
# geonames_gb.to_csv(ROOT / 'data/external/geonames_gb.csv')

# %%
geonames_gb = pd.read_csv(ROOT / 'data/external/geonames_gb.csv')
geonames_gb.rename(columns={'asciiname': 'name'}, inplace=True)

# %%
"""
## 4.2. Get population for all locations where it is directly possible
"""

# %%
# get population for locations
pop_dict = geonames_gb['population'].copy()
pop_dict = geonames_gb.set_index(geonames_gb['name'].str.lower().str.strip())['population'].to_dict()

df['LocationPopulation'] = df['LocationNormalized'].str.lower().str.strip().map(lambda x: pop_dict.get(x))

# %%
def print_missing_info():
    print(f"Missing data in population of location: {round(df[df['LocationPopulation'].isna()]['LocationNormalized'].count() / len(df) * 100, 2)}%, {df[df['LocationPopulation'].isna()]['LocationNormalized'].count()} cases")
    print()
    print(df[df['LocationPopulation'].isna()]['LocationNormalized'].value_counts()[:5])

# %%
print_missing_info()

# %%
"""
## 4.3. Remove directions and assign population to other fitting names
"""

# %%
directions = ['North', 'South', 'East', 'West', 'Central']
df['LocationNormalized'] = df['LocationNormalized'].replace(
    directions, '', regex=True
).str.strip()

missing_mask = df['LocationPopulation'].isna()
missing_locations = df.loc[missing_mask, 'LocationNormalized'].str.lower().str.strip()

pop_dict_missing = {loc: pop_dict.get(loc, np.nan) for loc in missing_locations}

df.loc[missing_mask, 'LocationPopulation'] = missing_locations.map(pop_dict_missing)

# %%
print_missing_info()

# %%
"""
## 4.4. Find population for Midlands in NUT regions
"""

# %%
# remove locations out of GB
uk_lat_mask = (geonames_gb['lat'] >= 49) & (geonames_gb['lat'] <= 61)
uk_lon_mask = (geonames_gb['lon'] >= -10) & (geonames_gb['lon'] <= 2)
geonames_gb = geonames_gb[(geonames_gb['country_code'] == 'GB') & (uk_lat_mask) & (uk_lon_mask)]

# %%
nuts = pd.read_excel(ROOT / 'data/external/NUTS.xlsx')
nuts['NUTS118NM'] = nuts['NUTS118NM'].str.replace('(England)', '', regex=False).str.strip()
nuts = nuts.rename(columns={'NUTS118NM': 'name', 'LONG': 'lon', 'LAT': 'lat'})

# %%
# find the closest point in geonames in nuts
from scipy.spatial import cKDTree
tree = cKDTree(geonames_gb[['lat', 'lon']].values)
nuts_coords = nuts[['lat', 'lon']].values

distances, indices = tree.query(nuts_coords, k=1)  # k=1 -> 1 neighbour

nuts['population'] = geonames_gb.iloc[indices]['population'].values
nuts_population = dict(zip(nuts['name'], nuts['population']))

# %%
from typing import Counter

# combine West and East Midlands
nuts_population = {**{k: v for k, v in nuts_population.items() if 'Midlands' not in k},
                 **{'Midlands': sum(v for k, v in nuts_population.items() if 'Midlands' in k)}}

# %%
nuts_population

# %%
# impute nuts locations
population_from_dict = df['LocationNormalized'].map(nuts_population)

mask = ((df['LocationPopulation'].isnull()) | (df['LocationPopulation'] == 0)) & population_from_dict.notnull()

df.loc[mask, 'LocationPopulation'] = population_from_dict[mask]

# %%
print_missing_info()

# %%
"""
## 4.5. Cast rest of cases as 'UK'
"""

# %%
mask = df['LocationPopulation'].isna() | (df['LocationPopulation'] == 0)
uk_pop = df.loc[df['LocationNormalized'].str.lower().eq('uk'), 'LocationPopulation'].dropna().iloc[0] if any(df['LocationNormalized'].str.lower().eq('uk')) else np.nan
df.loc[mask, ['LocationNormalized', 'LocationPopulation']] = ['UK', uk_pop]

# %%
print_missing_info()

# %%
df['LocationPopulation'].value_counts().head()

# %%
df.drop(columns=['LocationNormalized'], inplace=True)

# %%
"""
# 5. Split data
"""

# %%
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3, random_state=42)

# %%
"""
# 6. One hot encoding
"""

# %%
# select most common source in category group
category_to_source = train.groupby('Category')['SourceName'].agg(lambda x: x.mode()[0]).to_dict()
train['SourceName'] = train['Category'].map(category_to_source)
test['SourceName'] = test['Category'].map(category_to_source)

# %%
train = pd.get_dummies(train, columns = ['ContractType', 'ContractTime', 'Category', 'SourceName'], drop_first=True, dtype=int)
test = pd.get_dummies(test, columns = ['ContractType', 'ContractTime', 'Category', 'SourceName'], drop_first=True, dtype=int)

# %%
"""
# 7. Target Encoding - mean salary of company instead of company name
"""

# %%
mean_company = train.groupby('Company')['SalaryNormalized'].mean()

train['CompanyEncoded'] = train['Company'].map(mean_company)
test['CompanyEncoded'] = test['Company'].map(mean_company)

train.drop(inplace=True, columns=['Company'])
test.drop(inplace=True, columns=['Company'])

# %%
"""
# 8. Standarization
"""

# %%
from sklearn.preprocessing import StandardScaler
numeric_cols = ['SalaryNormalized', 'CompanyEncoded', 'LocationPopulation']
scaler = StandardScaler()
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])

# %%
"""
# 9. Embeddings
"""

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

tfidf_desc = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_title = TfidfVectorizer(max_features=1000, stop_words='english')

X_train_desc = tfidf_desc.fit_transform(train["FullDescription"])
X_test_desc  = tfidf_desc.transform(test["FullDescription"])

X_train_title = tfidf_title.fit_transform(train["Title"])
X_test_title  = tfidf_title.transform(test["Title"])

# join
X_train_text = hstack([X_train_desc, X_train_title])
X_test_text  = hstack([X_test_desc, X_test_title])

# %%
from sklearn.decomposition import TruncatedSVD

# dimenshion reduction
svd = TruncatedSVD(n_components=50, random_state=42)

X_train_text = svd.fit_transform(X_train_text)
X_test_text  = svd.transform(X_test_text)

# %%
train.drop(columns=['Title', 'FullDescription'], inplace=True)
test.drop(columns=['Title', 'FullDescription'], inplace=True)

# %%
"""
# 10. Data saving
"""

# %%
train.to_csv(ROOT / 'data/processed/train_preprocessed.csv')
test.to_csv(ROOT / 'data/processed/test_preprocessed.csv')

# %%
train.head()

# %%
import joblib
np.save(ROOT / "data/processed/X_train_text.npy", X_train_text)
np.save(ROOT / "data/processed/X_test_text.npy", X_test_text)
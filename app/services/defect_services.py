import pandas as pd
import numpy as np
import string
import nltk
import spacy
import dask.dataframe as dd
import networkx as nx
import asyncio
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import inflect
from typing import List, Dict, Set
import json
from pymongo import MongoClient
from sqlalchemy import create_engine
import gc
import string
import datetime
from dask import delayed
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from typing import List, Tuple
print("data import  started...")
# Initialize necessary components
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")
le = LabelEncoder()
q = inflect.engine()
vectorizer = TfidfVectorizer()
import pandas as pd
from pymongo import MongoClient
from app.core.db_config import load_config
config= load_config()
engine=create_engine('postgresql+psycopg2://admin:admin123@35.154.117.132:5432/garud_ai_processed_data')
from app.services.estima_services import replace_nan_inf, float_round

def float_round(value):
    if pd.notna(value):  # Better check for non-null values
        return round(float(value), 2)
    return 0
def preprocess_text(text: str, preserve_symbols=[], words_to_remove=['DURING', 'INSPECTION', 'OBSERVED']) -> str:
    '''
    This function performs text preprocessing and returns processed text. 
    It will also accept a list of symbols to preserve.
    Input: text
    Output: text
    '''
    #Newly added code to address an NaN float error
    if isinstance(text, float) and np.isnan(text):  
        return ''  
    
    # Define symbols to preserve
    preserve_symbols = set(preserve_symbols)

    for word in words_to_remove:
        text = text.replace(word, ' ')
    
    # Remove punctuation, excluding specified symbols
    custom_translation = str.maketrans('', '', ''.join(set(string.punctuation) - preserve_symbols))
    text = text.translate(custom_translation)
    return text

def compute_tfidf(corpus: list, preserve_symbols=['-', '/']) -> list:
    # Ensure corpus is a list of strings
    corpus = [str(text) if isinstance(text, str) else "" for text in corpus]

    # Preprocess text
    preprocessed_corpus = [preprocess_text(text, preserve_symbols) for text in corpus]



    # Ensure corpus is not empty
    if all(text.strip() == "" for text in preprocessed_corpus):
        raise ValueError("All preprocessed texts are empty!")

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words=None)
    embeddings = vectorizer.fit_transform(preprocessed_corpus)

    return embeddings.toarray()


def threshold_transform(data: np.ndarray, threshold: float = 0.5, above_value: int = 1, below_value: int = 0) -> np.ndarray:
    """Apply a threshold transformation to data."""
    return np.where(np.array(data) > threshold, above_value, below_value)

def compute_cluster(mpd_defects_data,tasks):
    output_dataframe=pd.DataFrame()
    for task in tasks:
        exdata=mpd_defects_data[mpd_defects_data["source_cust_card"]==task]
    
        if len(exdata)>0:

            
            exdata['full_description']= exdata["description"]
            
            desc_correction_tf_idf_vec = compute_tfidf(exdata['full_description'].tolist(), preserve_symbols=['-', '/'])
            
            desc_correction_embeddings = pd.DataFrame(desc_correction_tf_idf_vec, index=exdata["work_order_and_item_number"].tolist())
            
            # Assuming desc_correction_embeddings is a pandas DataFrame
            desc_correction_embeddings = desc_correction_embeddings.T
            embeddings_array = desc_correction_embeddings.values
            
            # Calculate cosine similarity directly on the numpy array
            cos_sim_mat = cosine_similarity(embeddings_array.T)
            
            # Apply threshold directly to the numpy array
            cos_sim_mat = np.where(cos_sim_mat >= 0.5, cos_sim_mat, 0)
            
            # Create sparse representation
            rows, cols = np.where(cos_sim_mat > 0)
            values = cos_sim_mat[rows, cols]
            
            
            # Create the unpivoted dataframe directly from the sparse representation
            columns = desc_correction_embeddings.columns
            df_unpivoted = pd.DataFrame({
            'obsid_s': [columns[r] for r in rows],
            'obsid_d': [columns[c] for c in cols],
            'Value': values
            })
            
            # Set the index
            df_unpivoted.set_index('obsid_s', inplace=True)
            df_unpivot = df_unpivoted[df_unpivoted['obsid_d'] != 'level_0']
            
            df_unpivot.reset_index(inplace=True)
            
            combined_df = exdata.copy()
            
            # Merge df_unpivot with combined_df on 'Log Item #' to get 'sourcetask_s'
            df_unpivot = pd.merge(df_unpivot, combined_df[["work_order_and_item_number", 'source_cust_card']], left_on='obsid_s', right_on='work_order_and_item_number', how='left')
            df_unpivot.rename(columns={'source_cust_card': 'source_cust_card_s'}, inplace=True)
            df_unpivot.drop(columns="work_order_and_item_number", inplace=True)
            
            # Merge df_unpivot with combined_df again on 'Log Item #' to get 'sourcetask_d'
            df_unpivot = pd.merge(df_unpivot, combined_df[['work_order_and_item_number', 'source_cust_card']], left_on='obsid_d', right_on='work_order_and_item_number', how='left')
            df_unpivot.rename(columns={'source_cust_card': 'source_cust_card_d'}, inplace=True)
            df_unpivot.drop(columns="work_order_and_item_number", inplace=True)
            
            # Assuming df_unpivot is your original DataFrame
            # Splitting the DataFrame into three chunks
            chunk_size = max(len(df_unpivot) // 3,1)
            chunks = [df_unpivot[i:i+chunk_size] for i in range(0, len(df_unpivot), chunk_size)]
            
            # Define the custom function to update the 'Value' column based on conditions
            def update_value(row):
                
                if row['Value'] == 0:
                    return 0
                elif row['source_cust_card_s'] == row['source_cust_card_d']:
                    return 1
                else:
                    return 0
            
            # Process each chunk individually
            processed_chunks = []
            for chunk in chunks:
                # Create a copy of the chunk to work with
                chunk_copy = chunk.copy()
                # Apply the custom function row-wise to update the 'Value' column
                chunk_copy['Value'] = chunk_copy.apply(update_value, axis=1)
                # Append the processed chunk to the list
                processed_chunks.append(chunk_copy)
            
            # Concatenate the processed chunks into a single DataFrame
            df_unpivotchk = pd.concat(processed_chunks)
            df_sim = df_unpivotchk[df_unpivotchk['Value'] == 1].copy()
            df_sim1 = df_sim[df_sim['obsid_s'] != df_sim['obsid_d']].copy()
            # Assuming df_sim1 is your DataFrame
            
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add edges based on connections between obsid_s and obsid_d
            for index, row in df_sim1.iterrows():
                G.add_edge(row['obsid_s'], row['obsid_d'])
            
            # Assign groups using strongly connected components
            groups = {node: i for i, component in enumerate(nx.strongly_connected_components(G), start=1) for node in component}
            
            # Map the groups to the DataFrame
            df_sim1['Group'] = df_sim1['obsid_s'].map(groups)
            
            # Merge combined_df with df_group on 'Log Item #' and 'obsid_s' to get 'group' values
            group_df = pd.merge(combined_df, df_sim1[['obsid_s', 'Group']], left_on='work_order_and_item_number', right_on='obsid_s', how='left')
            
            # Rename the 'Group' column to 'group'
            group_df.rename(columns={'Group': 'group'}, inplace=True)
            
            group_df = group_df.drop_duplicates(subset=["work_order_and_item_number"]).copy()
            
            group_df = group_df.loc[:, ~group_df.columns.duplicated(keep='last')]
            
            exdata["group"] = float('nan')
            
            
            for i in group_df['work_order_and_item_number'].unique():  # Use unique values to avoid redundant operations
                group_values = group_df.loc[group_df['work_order_and_item_number'] == i, 'group'].values
            
                if len(group_values) > 0:  # Ensure at least one value exists
                    group_value = group_values[0]  # Take the first value
            
                # Assign scalar value correctly to all matching rows
                exdata.loc[exdata['work_order_and_item_number'] == i, "group"] = group_value
            
            
            def fillnull(row):
                if pd.isna(row["group"]):  # Correct way to check NaN
                    return row["work_order_and_item_number"]  # Return the new value
                return row["group"]  # Return original value if not NaN
            
            # Apply function to the 'group' column
            exdata["group"] = exdata.apply(fillnull, axis=1)
            exdata["group"] = exdata["group"].astype(str)  # Convert to string

            output_dataframe=pd.concat([output_dataframe,exdata])
            #print(output_dataframe.shape)
    print("clustering is computed")
    return output_dataframe
def prob(row):
    """
    if delta_tasks:
        if row["source_cust_card"] in not_available_tasks["task_number"].values:
            prob=len(row["packages_list"])/len(work_orders)*100
    else:"""
    prob = (len(row["packages_list"]) / len(row["work_orders"]))*100
    
    return prob
def manhours_prediction(mpd_defects_cluster_data):    
    group_level_mh = mpd_defects_cluster_data[[
    'source_cust_card',"full_description",
    'actual_man_hours',
    'skill_codes',
    "group",
    "work_order"
    ]]
    group_level_mh.drop_duplicates(inplace=True)
    # Get all unique package numbers once
    all_work_orders = group_level_mh["work_order"].unique()
    # First level aggregation
    group_level_mh = group_level_mh.groupby(
        ["source_cust_card", "group", "work_order"]
    ).agg(
        avg_actual_man_hours=("actual_man_hours", "sum"),
        max_actual_man_hours=("actual_man_hours", "sum"),
        min_actual_man_hours=("actual_man_hours", "sum"),
        description=("full_description", "first"),
        skill_number=('skill_codes', lambda x: list(set(x)))
    ).reset_index()
    
    group_level_mh["work_orders"]=group_level_mh["source_cust_card"].apply(
        lambda x: group_level_mh[group_level_mh["source_cust_card"] == x]["work_order"].unique().tolist()
    )
    
    # Second level aggregation
    aggregated = group_level_mh.groupby(
        ["source_cust_card", "group"]
    ).agg(
        avg_actual_man_hours=("avg_actual_man_hours", "mean"),
        max_actual_man_hours=("max_actual_man_hours", "max"),
        min_actual_man_hours=("min_actual_man_hours", "min"),
        description=("description", "first"),
        skill_number=("skill_number", lambda x: list(set(sum(x, []))))  # Flatten list of lists and remove duplicates
    ).reset_index()
            
    # Create a crosstab for package indicators (much faster than iterating)
    package_indicators = pd.crosstab(
        index=[group_level_mh["source_cust_card"], group_level_mh["group"]],
        columns=group_level_mh["work_order"]
    ).clip(upper=1)  # Convert counts to binary indicators
    
    # Merge the aggregated data with package indicators
    group_level_mh_result = pd.merge(
        aggregated,
        package_indicators,
        on=["source_cust_card", "group"]
    )
    # Step 1: Group and aggregate unique packages
    packages_by_group = (
        group_level_mh
        .groupby(["source_cust_card", "group"])["work_order"]
        .apply(lambda x: list(pd.unique(x)))
        .reset_index()
        .rename(columns={"work_order": "packages_list"})
    )
    
    
    # Step 2: Merge with the result DataFrame
    group_level_mh_result = group_level_mh_result.merge(
        packages_by_group,
        on=["source_cust_card", "group"],
        how="left"
    )
    
    print("work_orders  is computed")
    group_level_mh["work_orders"] = group_level_mh["work_orders"].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    
    group_level_mh_result = group_level_mh_result.merge(
    group_level_mh[["source_cust_card",  "work_orders"]].drop_duplicates(),
    on=["source_cust_card"],
    how="left"
    )
    print("prob being computed")
    
    # Apply the function row-wise using lambda
    group_level_mh_result["prob"] = group_level_mh_result.apply(
        lambda row: prob(row), axis=1
    )
    print("prob is being merged")
    group_level_mh = group_level_mh.merge(
    group_level_mh_result[["source_cust_card", "group","prob"]],
    on=["source_cust_card", "group"],
    how="left"
    )
    return group_level_mh
    
# Define parts price calculation
def prob(row):
    """
    if delta_tasks:
        if row["source_cust_card"] in not_available_tasks["task_number"].values:
            prob=len(row["packages_list"])/len(work_orders)*100
    else:"""
    prob = (len(row["packages_list"]) / len(row["work_orders"]))*100
    
    return prob
def parts_price(row):
    if row["unit_cost"] :
        return row["avg_used_qty"] * (row["unit_cost"])
    else:
        return 0
def parts_prediction(cluster_data,parts_data):

    cluster_data_updated = cluster_data.merge(
            parts_data[
                ['work_order_and_item_number', 'part_number',
                 'required_qty', 'group_code',
                 'unit_cost', 'currency','part_description']
            ],
            left_on='work_order_and_item_number',
            right_on='work_order_and_item_number',
            how="left"
        )
            # Group and aggregate
    if parts_data.empty or cluster_data.empty or cluster_data_updated.empty:
        return pd.DataFrame(columns=['source_cust_card', 'group', 'part_number', 'avg_used_qty',
       'max_used_qty', 'min_used_qty', 'unit_cost', 'total_used_qty',
       'part_description', 'group_code', 'work_orders', 'prob', 'total_cost','part_type'])
    group_level_parts = cluster_data_updated.groupby(
            ["source_cust_card", "group", "part_number", "work_order"]
        ).agg(
            group_code=("group_code", "first"),
            required_qty=("required_qty", "sum"),
            part_description=('part_description', "first"),
            unit_cost=('unit_cost', "first"),
            currency=('currency',"first")
        ).reset_index()
    group_level_parts["work_orders"] = group_level_parts.apply(
    lambda row: group_level_parts[
        (group_level_parts["source_cust_card"] == row["source_cust_card"]) &
        (group_level_parts["group"] == row["group"])
    ]["work_order"].unique().tolist(),
    axis=1
        )
    # Higher-level aggregation
    aggregated = group_level_parts.groupby(
        ["source_cust_card", "group", "part_number"]
    ).agg(
        avg_used_qty=("required_qty", 'mean'),
        max_used_qty=("required_qty", "max"),
        min_used_qty=("required_qty", "min"),
        unit_cost=("unit_cost", "first"),
        total_used_qty=("required_qty", "sum"),
        part_description=('part_description', "first"),
        group_code=("group_code", "first"),
    ).reset_index()

    # Ensure data types are strings for joining
    group_level_parts["source_cust_card"] = group_level_parts["source_cust_card"].astype(str)
    group_level_parts["part_number"] = group_level_parts["part_number"].astype(str)

    # Create binary package indicators (pivot-style)
    package_indicators = pd.crosstab(
        index=[
            group_level_parts["source_cust_card"],
            group_level_parts["group"],
            group_level_parts["part_number"]
        ],
        columns=group_level_parts["work_order"]
    ).clip(upper=1).reset_index()

    # Merge aggregated with package indicators
    group_level_parts_result = pd.merge(
        aggregated,
        package_indicators,
        on=["source_cust_card", "group", "part_number"]
    )

    # Create a packages_list mapping
    packages_by_group = group_level_parts.groupby(
        ["source_cust_card", "group", "part_number"]
    )["work_order"].apply(lambda x: list(pd.unique(x))).reset_index(name="packages_list")

    # Merge packages list to final result
    group_level_parts_result = pd.merge(
        group_level_parts_result,
        packages_by_group,
        on=["source_cust_card", "group", "part_number"],
        how="left"
    )
    group_level_parts["work_orders"] = group_level_parts["work_orders"].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    group_level_parts_result = group_level_parts_result.merge(
        group_level_parts[["source_cust_card", "group", "work_orders"]].drop_duplicates(),
        on=["source_cust_card", "group"],
        how="left"
    )

    # Apply probability
    group_level_parts_result["prob"] = group_level_parts_result.apply(
        lambda row: prob(row), axis=1
    )
    

    # Define parts price calculation
    def parts_price(row):
        if row["unit_cost"] :
            return row["avg_used_qty"] * (row["unit_cost"])
        else:
            return 0
    group_level_parts_result["total_cost"]=0
    # Apply parts price calculation
    group_level_parts_result["total_cost"] = group_level_parts_result.apply(parts_price, axis=1)
    group_level_parts_result["part_type"] = group_level_parts_result["group_code"].replace({
        "EXP": "expendables",
        "CONS": "consumables"
    })

    return group_level_parts_result[['source_cust_card', 'group', 'part_number', 'avg_used_qty',
       'max_used_qty', 'min_used_qty', 'unit_cost', 'total_used_qty',
       'part_description', 'group_code', 'work_orders', 'prob', 'total_cost','part_type']]
    
    

async def defects_prediction(tasks):
    
    defects_query = "SELECT * FROM mpd_defects_data WHERE source_cust_card IN %(tasks)s"
    mpd_defects_data = pd.read_sql(defects_query, engine, params={"tasks": tuple(tasks)})
    if mpd_defects_data.empty:
        return {
            "findings": [],
            "findings_manhours": 0.0,
            "findings_spare_parts_cost": 0.0
        }
    work_order_and_item_number_list=mpd_defects_data["work_order_and_item_number"].unique().tolist()
    parts_query="SELECT * FROM historical_parts_data WHERE work_order_and_item_number IN %(work_order_and_item_number_list)s"
    mpd_defects_parts_data = pd.read_sql( parts_query, engine, params={"work_order_and_item_number_list": tuple(work_order_and_item_number_list)})
    if mpd_defects_parts_data.empty:
        return {
            "findings": [],
            "findings_manhours": 0.0,
            "findings_spare_parts_cost": 0.0
        }
    cluster_data=compute_cluster(mpd_defects_data,tasks)
    if cluster_data.empty:
        return {
            "findings": [],
            "findings_manhours": 0.0,
            "findings_spare_parts_cost": 0.0
        }
    cluster_man_hours_data=manhours_prediction(cluster_data)
    if cluster_man_hours_data.empty:
         {
        "findings": findings,
        "findings_manhours": float_round(findings_manhours),
        "findings_spare_parts_cost": float_round(findings_spare_parts_cost)
        }
    cluster_parts_data=parts_prediction(cluster_data,mpd_defects_parts_data)
    findings=[]
    findings_manhours=0
    findings_spare_parts_cost=0
    for _, row in cluster_man_hours_data.iterrows():
        spare_parts = []
        # Calculate probability-adjusted values for disc_pred_list
        prob_factor = row["prob"] / 100
        spare_filtered = cluster_parts_data[cluster_parts_data["group"] == row["group"]] 
        for _, part in spare_filtered.iterrows():
            findings_spare_parts_cost += part["total_cost"] * prob_factor
            spare_parts.append({
                "partId": part["part_number"],
                "desc": part["part_description"],
                "qty": float_round(part["avg_used_qty"]),
                "price": float_round(part["total_cost"]),
                "part_type":part["part_type"],
                "prob": float_round(part["prob"])
            })

        finding = {
            
            "taskId": row["source_cust_card"],
            "chapterName": str(row["source_cust_card"])[:2],   
            "details": [{
                "cluster": f"{row['source_cust_card']}/{row['group']}",
                "description": row["description"],
                "skill": row["skill_number"],
                "mhs": {
                    "max": float_round(row["max_actual_man_hours"]),
                    "min": float_round(row["min_actual_man_hours"]),
                    "avg": float_round(row["avg_actual_man_hours"]),
                    "est": float_round(row["max_actual_man_hours"])
                },
                "prob": float_round(row["prob"]),
                "spare_parts": spare_parts
            }]
        }
        findings_manhours += row["avg_actual_man_hours"] * prob_factor
        
        findings.append(finding)
    result = {
        "findings": findings,
        "findings_manhours": float_round(findings_manhours),
        "findings_spare_parts_cost": float_round(findings_spare_parts_cost)
    }
    result = replace_nan_inf(result)
    return  result
    
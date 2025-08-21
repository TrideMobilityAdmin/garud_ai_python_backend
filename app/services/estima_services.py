import pandas as pd
import numpy as np
import string
import nltk
import spacy
import networkx as nx
import asyncio
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import inflect
from typing import List, Dict, Set
import json
from pymongo import MongoClient
import math
import re
import gc
import string
import datetime
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
database_details = config["database_details"]
client =MongoClient(database_details['connection_string'])
db = client[database_details['database_name']]
keywords_collection = db["keywords"]
keywords_df = pd.DataFrame(list(keywords_collection.find({})))
keywords_dict = keywords_df.set_index("Abbreviation")["Meaning"].to_dict()
keyword_set = set(keywords_df["Meaning"].str.lower().tolist())  # Normalize to lowercase
aircraft_collection = db["aircraft_details"]
aircraft_details = pd.DataFrame(list(aircraft_collection.find({})))
sub_task_description_max500mh_lhrh = db["sub_task_description_max500mh_lhrh"]
#sub_task_description_max500mh_lhrh = pd.DataFrame(list(db["sub_task_description_max500mh_lhrh"].find({})))
sub_task_parts_lhrh=db["sub_task_parts_lhrh"]
#sub_task_parts= pd.DataFrame(list(sub_task_parts_lhrh.find({})))
parts_master=db["parts_master"]
parts_master = list(parts_master.find({},))
parts_master= pd.DataFrame(parts_master)
parts_master = parts_master.drop(columns=["_id"], errors="ignore")
parts_master=parts_master.drop_duplicates()
aircraft_event_log= db["aircraft_event_log"]
aircraft_event_log = pd.DataFrame(list(aircraft_event_log.find({})))
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

def keyword_extraction(text: str,  n_keywords: int = 7) -> List[str]:
    """
    Extract top-n relevant keywords from text that exist in the MongoDB 'keywords' collection.

    Args:
        text (str): The input text from which to extract keywords.
        db: The MongoDB database object.
        n_keywords (int): The number of keywords to extract.

    Returns:
        List[str]: A list of filtered, high-scoring keywords.
    """

    # Preprocess the text
    preprocessed_text = preprocess_text(text,words_to_remove=[])
    for meaning, abbr in keywords_dict.items():
        # Use regex word boundary to avoid partial matches
        pattern = r'\b' + re.escape(meaning) + r'\b'
        preprocessed_text  = re.sub(pattern, abbr, preprocessed_text )
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the text
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])

    # Extract TF-IDF features and scores
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    keyword_scores = dict(zip(feature_names, scores))

    # Filter by keywords present in MongoDB
    filtered_keywords = {
        k: v for k, v in keyword_scores.items() if k.lower() in keyword_set
    }

    # Sort and return top n_keywords
    sorted_keywords = sorted(filtered_keywords.items(), key=lambda x: x[1], reverse=True)

    return [keyword for keyword, score in sorted_keywords[:n_keywords]]

def threshold_transform(data: np.ndarray, threshold: float = 0.5, above_value: int = 1, below_value: int = 0) -> np.ndarray:
    """Apply a threshold transformation to data."""
    return np.where(np.array(data) > threshold, above_value, below_value)

def packages_extract(aircraft_age,aircraft_model,check_category,customer_name,customer_name_consideration):
    age_cap=3
        # Convert aircraft_age to float
    aircraft_age = float(aircraft_age)
    aircraft_details['aircraft_age'] = aircraft_details['aircraft_age'].astype(float)
    aircraft_model_family = []
    
    A320_family = ["A319", "A320", "A321"]
    Boeing_NG = ["B737 NG", "B737-800(BCF)"]
    others = ["ATR42", "ATR72", "Q400", "B737 MAX"]
    
    if aircraft_model in A320_family:
        aircraft_model_family = A320_family
    elif aircraft_model in Boeing_NG:
        aircraft_model_family = Boeing_NG
    elif aircraft_model in others:
        aircraft_model_family = [aircraft_model]  # Corrected this line
    else:
        aircraft_model_family = aircraft_details['aircraft_model'].unique().tolist()


    normalized_customer_name = customer_name.upper()

    # Filter based on customer name consideration
    if normalized_customer_name in ["AIR INDIA", "AIR ASIA", "AIR INDIA EXPRESS", "VISTARA"]:
        customer_name_list = ["AIR INDIA", "AIR ASIA", "AIR INDIA EXPRESS", "VISTARA"]
    else:
        customer_name_list = [customer_name]

    # Normalize 'customer_name' column in dataframe for case-insensitive matching
    aircraft_details["customer_name_upper"] = aircraft_details["customer_name"].str.upper()
    train_packages=[]
    if aircraft_age> 0.0:
        # Continue increasing age_cap until we get at least 5 packages or reach the maximum age limit
        while len(train_packages) < 5:  # Changed from >4 to <5 to keep looping until we have at least 5 packages
            if customer_name_consideration:
                # Filter based on customer name consideration
                # Normalize customer_name to uppercase for comparison
                train_packages = aircraft_details[
                    (aircraft_details["aircraft_model"].isin(aircraft_model_family)) & 
                    (aircraft_details["check_category"].isin(check_category)) & 
                    (aircraft_details["customer_name_upper"].isin([name.upper() for name in customer_name_list])) &
                    (aircraft_details["aircraft_age"].between(max(aircraft_age - age_cap, 0), min(aircraft_age + age_cap, 31)))    
                ]["package_number"].unique().tolist()


            else:
            
                train_packages = aircraft_details[
                    (aircraft_details["aircraft_model"].isin(aircraft_model_family)) & 
                    (aircraft_details["check_category"].isin(check_category)) & 
                    (aircraft_details["aircraft_age"].between(max(aircraft_age - age_cap,0), min(aircraft_age + age_cap,31)))
                ]["package_number"].unique().tolist()
                
            # If we found at least 5 packages, we can exit the loop
            if len(train_packages) >= 5:
                break
            
            # Increase age_cap by 1
            age_cap += 1
            
            # Check if we've reached the maximum age limit
            if aircraft_age + age_cap > 30:
                break 
    else:
        if customer_name_consideration:
            train_packages = aircraft_details[
                (aircraft_details["aircraft_model"].isin(aircraft_model_family)) & 
                (aircraft_details["check_category"].isin(check_category)) & 
                (aircraft_details["customer_name_upper"].isin([name.upper() for name in customer_name_list]))     
            ]["package_number"].unique().tolist()
    
        else:
            train_packages = aircraft_details[
                    (aircraft_details["aircraft_model"] .isin(aircraft_model_family)) & 
                    (aircraft_details["check_category"].isin(check_category))
                ]["package_number"].unique().tolist()
    return train_packages
def compute_cluster(sub_task_description_max500mh_lhrh,tasks):
    output_dataframe=pd.DataFrame()
    for task in tasks:
        exdata=sub_task_description_max500mh_lhrh[sub_task_description_max500mh_lhrh["source_task_discrepancy_number_updated"]==task]
    
        if len(exdata)>0:
            if "source_task_discrepancy_number_updated" in exdata.columns:
                exdata["source_task_discrepancy_number_updated"] = exdata["source_task_discrepancy_number_updated"].astype(str)
            
            if "source_task_discrepancy_number" in exdata.columns:
                exdata["source_task_discrepancy_number"] = exdata["source_task_discrepancy_number"].astype(str)
            
            
            def update(row):
                if row["source_task_discrepancy_number"] != row["source_task_discrepancy_number_updated"]:
                    return row["source_task_discrepancy_number_updated"]
                return row["source_task_discrepancy_number"]  # Return the original value if no change
            
            # Apply function correctly
            exdata["source_task_discrepancy_number"] = exdata.apply(update, axis=1)
            
            exdata['full_description']= exdata["task_description"]+" "+exdata["corrective_action"]
            
            desc_correction_tf_idf_vec = compute_tfidf(exdata['full_description'].tolist(), preserve_symbols=['-', '/'])
            
            desc_correction_embeddings = pd.DataFrame(desc_correction_tf_idf_vec, index=exdata['log_item_number'].tolist())
            
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
            df_unpivot = pd.merge(df_unpivot, combined_df[['log_item_number', 'source_task_discrepancy_number']], left_on='obsid_s', right_on='log_item_number', how='left')
            df_unpivot.rename(columns={'source_task_discrepancy_number': 'source_task_discrepancy_number_s'}, inplace=True)
            df_unpivot.drop(columns='log_item_number', inplace=True)
            
            # Merge df_unpivot with combined_df again on 'Log Item #' to get 'sourcetask_d'
            df_unpivot = pd.merge(df_unpivot, combined_df[['log_item_number', 'source_task_discrepancy_number']], left_on='obsid_d', right_on='log_item_number', how='left')
            df_unpivot.rename(columns={'source_task_discrepancy_number': 'source_task_discrepancy_number_d'}, inplace=True)
            df_unpivot.drop(columns='log_item_number', inplace=True)
            
            # Assuming df_unpivot is your original DataFrame
            # Splitting the DataFrame into three chunks
            chunk_size = max(len(df_unpivot) // 3,1)
            chunks = [df_unpivot[i:i+chunk_size] for i in range(0, len(df_unpivot), chunk_size)]
            
            # Define the custom function to update the 'Value' column based on conditions
            def update_value(row):
                
                if row['Value'] == 0:
                    return 0
                elif row['source_task_discrepancy_number_s'] == row['source_task_discrepancy_number_d']:
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
            group_df = pd.merge(combined_df, df_sim1[['obsid_s', 'Group']], left_on='log_item_number', right_on='obsid_s', how='left')
            
            # Rename the 'Group' column to 'group'
            group_df.rename(columns={'Group': 'group'}, inplace=True)
            
            group_df = group_df.drop_duplicates(subset=['log_item_number']).copy()
            
            group_df = group_df.loc[:, ~group_df.columns.duplicated(keep='last')]
            
            exdata["group"] = float('nan')
            
            
            for i in group_df['log_item_number'].unique():  # Use unique values to avoid redundant operations
                group_values = group_df.loc[group_df['log_item_number'] == i, 'group'].values
            
                if len(group_values) > 0:  # Ensure at least one value exists
                    group_value = group_values[0]  # Take the first value
            
                # Assign scalar value correctly to all matching rows
                exdata.loc[exdata['log_item_number'] == i, "group"] = group_value
            
            
            def fillnull(row):
                if pd.isna(row["group"]):  # Correct way to check NaN
                    return row["log_item_number"].replace("HMV","0") # Return the new value
                return row["group"]  # Return original value if not NaN
            
            # Apply function to the 'group' column
            exdata["group"] = exdata.apply(fillnull, axis=1)
            exdata["group"] = exdata["group"].astype(str)  # Convert to string

            output_dataframe=pd.concat([output_dataframe,exdata])
            print(output_dataframe.shape)
            print("clustering is computed")
    return output_dataframe

def prob(row):
    """
    if delta_tasks:
        if row["source_task_discrepancy_number"] in not_available_tasks["task_number"].values:
            prob=len(row["packages_list"])/len(package_numbers)*100
    else:"""
    prob = (len(row["packages_list"]) / len(row["package_numbers"]))*100
    
    return prob
def manhours_prediction(cluster_data):
    group_level_mh = cluster_data[[
    "source_task_discrepancy_number","full_description",
    "actual_man_hours",
    "skill_number",
    "group",
    "package_number"
    ]]
    group_level_mh.drop_duplicates(inplace=True)
    # Get all unique package numbers once
    all_package_numbers = group_level_mh["package_number"].unique()
    # First level aggregation
    group_level_mh = group_level_mh.groupby(
        ["source_task_discrepancy_number", "group", "package_number"]
    ).agg(
        avg_actual_man_hours=("actual_man_hours", "sum"),
        max_actual_man_hours=("actual_man_hours", "sum"),
        min_actual_man_hours=("actual_man_hours", "sum"),
        description=("full_description", "first"),
        skill_number=("skill_number", lambda x: list(set(x)))
    ).reset_index()
    
    group_level_mh["package_numbers"]=group_level_mh["source_task_discrepancy_number"].apply(
        lambda x: group_level_mh[group_level_mh["source_task_discrepancy_number"] == x]["package_number"].unique().tolist()
    )
    
    # Second level aggregation
    aggregated = group_level_mh.groupby(
        ["source_task_discrepancy_number", "group"]
    ).agg(
        avg_actual_man_hours=("avg_actual_man_hours", "mean"),
        max_actual_man_hours=("max_actual_man_hours", "max"),
        min_actual_man_hours=("min_actual_man_hours", "min"),
        description=("description", "first"),
        skill_number=("skill_number", lambda x: list(set(sum(x, []))))  # Flatten list of lists and remove duplicates
    ).reset_index()
            
    # Create a crosstab for package indicators (much faster than iterating)
    package_indicators = pd.crosstab(
        index=[group_level_mh["source_task_discrepancy_number"], group_level_mh["group"]],
        columns=group_level_mh["package_number"]
    ).clip(upper=1)  # Convert counts to binary indicators
    
    # Merge the aggregated data with package indicators
    group_level_mh_result = pd.merge(
        aggregated,
        package_indicators,
        on=["source_task_discrepancy_number", "group"]
    )
    # Step 1: Group and aggregate unique packages
    packages_by_group = (
        group_level_mh
        .groupby(["source_task_discrepancy_number", "group"])["package_number"]
        .apply(lambda x: list(pd.unique(x)))
        .reset_index()
        .rename(columns={"package_number": "packages_list"})
    )
    
    
    # Step 2: Merge with the result DataFrame
    group_level_mh_result = group_level_mh_result.merge(
        packages_by_group,
        on=["source_task_discrepancy_number", "group"],
        how="left"
    )
    
    print("package_numbers  is computed")
    group_level_mh["package_numbers"] = group_level_mh["package_numbers"].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    
    group_level_mh_result = group_level_mh_result.merge(
    group_level_mh[["source_task_discrepancy_number",  "package_numbers"]].drop_duplicates(),
    on=["source_task_discrepancy_number"],
    how="left"
    )
    print("prob being computed")
    
    # Apply the function row-wise using lambda
    group_level_mh_result["prob"] = group_level_mh_result.apply(
        lambda row: prob(row), axis=1
    )
    print("prob is being merged")
    group_level_mh = group_level_mh.merge(
    group_level_mh_result[["source_task_discrepancy_number", "group","prob"]],
    on=["source_task_discrepancy_number", "group"],
    how="left"
    )
    group_level_mh.sort_values(by="prob", ascending=False)
    return group_level_mh
# Define parts price calculation

def parts_price(row):
    if row["latest_price"] :
        return row["avg_used_qty"] * (row["latest_price"])
    else:
        return 0
def parts_prediction(cluster_data,sub_task_parts_lhrh):
    
    if cluster_data.empty or sub_task_parts_lhrh.empty :
        return pd.DataFrame(columns=['source_task_discrepancy_number', 'group', 'issued_part_number',
       'avg_used_qty', 'max_used_qty', 'min_used_qty', 'latest_price',
       'total_billable_value_usd', 'total_used_qty', 'part_description','prob', 'billable_value_usd'])    
    
    group_level_parts = cluster_data.merge(
            sub_task_parts_lhrh[
                ['task_number', 'issued_part_number','part_description',
                 'issued_unit_of_measurement', 'used_quantity', 'base_price_usd',
                 'billable_value_usd','latest_price']
            ],
            left_on="log_item_number",
            right_on="task_number",
            how="left"
        ).drop(columns=["task_number"])  # Drop duplicate column
            # Group and aggregate
        # Get all unique package numbers

    if group_level_parts.empty :
        return pd.DataFrame(columns=['source_task_discrepancy_number', 'group', 'issued_part_number',
       'avg_used_qty', 'max_used_qty', 'min_used_qty', 'latest_price',
       'total_billable_value_usd', 'total_used_qty', 'part_description','prob', 'billable_value_usd'])    

    # Group and aggregate
    group_level_parts = group_level_parts.groupby(
        ["source_task_discrepancy_number", "group", "issued_part_number", "package_number"]
    ).agg(
        billable_value_usd=("billable_value_usd", "sum"),
        used_quantity=("used_quantity", "sum"),
        part_description=('part_description', "first"),
        latest_price=('latest_price', "first"),
        issued_unit_of_measurement=('issued_unit_of_measurement', "first")
    ).reset_index()
    
    group_level_parts["package_numbers"] = group_level_parts.apply(
        lambda row: group_level_parts[
            (group_level_parts["source_task_discrepancy_number"] == row["source_task_discrepancy_number"]) &
            (group_level_parts["group"] == row["group"])
        ]["package_number"].unique().tolist(),
        axis=1
    )
    # Higher-level aggregation
    aggregated = group_level_parts.groupby(
        ["source_task_discrepancy_number", "group", "issued_part_number"]
    ).agg(
        avg_used_qty=("used_quantity", 'mean'),
        max_used_qty=("used_quantity", "max"),
        min_used_qty=("used_quantity", "min"),
        latest_price=("latest_price", "first"),
        total_billable_value_usd=("billable_value_usd", "sum"),
        total_used_qty=("used_quantity", "sum"),
        part_description=('part_description', "first"),
        issued_unit_of_measurement=('issued_unit_of_measurement', "first")
    ).reset_index()

    # Ensure data types are strings for joining
    group_level_parts["source_task_discrepancy_number"] = group_level_parts["source_task_discrepancy_number"].astype(str)
    group_level_parts["issued_part_number"] = group_level_parts["issued_part_number"].astype(str)

    # Create binary package indicators (pivot-style)
    package_indicators = pd.crosstab(
        index=[
            group_level_parts["source_task_discrepancy_number"],
            group_level_parts["group"],
            group_level_parts["issued_part_number"]
        ],
        columns=group_level_parts["package_number"]
    ).clip(upper=1).reset_index()

    # Merge aggregated with package indicators
    group_level_parts_result = pd.merge(
        aggregated,
        package_indicators,
        on=["source_task_discrepancy_number", "group", "issued_part_number"]
    )

    # Create a packages_list mapping
    packages_by_group = group_level_parts.groupby(
        ["source_task_discrepancy_number", "group", "issued_part_number"]
    )["package_number"].apply(lambda x: list(pd.unique(x))).reset_index(name="packages_list")

    # Merge packages list to final result
    group_level_parts_result = pd.merge(
        group_level_parts_result,
        packages_by_group,
        on=["source_task_discrepancy_number", "group", "issued_part_number"],
        how="left"
    )
    group_level_parts["package_numbers"] = group_level_parts["package_numbers"].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    group_level_parts_result = group_level_parts_result.merge(
        group_level_parts[["source_task_discrepancy_number", "group", "package_numbers"]].drop_duplicates(),
        on=["source_task_discrepancy_number", "group"],
        how="left"
    )

    # Apply probability
    print("the shape of the parts data is", group_level_parts_result.shape)
    if group_level_parts_result.empty:
        return pd.DataFrame(columns=['source_task_discrepancy_number', 'group', 'issued_part_number',
       'avg_used_qty', 'max_used_qty', 'min_used_qty', 'latest_price',
       'total_billable_value_usd', 'total_used_qty', 'part_description','prob', 'billable_value_usd'])
    group_level_parts_result["prob"] = group_level_parts_result.apply(
        lambda row: prob(row), axis=1
    )
    
    discrete_units = ["EA", "JR", "BOTTLE", "TU", "PAC", "BOX", "GR", "PC", "NO", "PINT", "PAIR", "GAL"]
    
    for idx, part_row in group_level_parts_result.iterrows():
        unit_of_measurement = str(part_row["issued_unit_of_measurement"]).strip().upper()
        if unit_of_measurement in discrete_units:
            group_level_parts_result.at[idx, 'avg_used_qty'] = round(float(part_row['avg_used_qty']), 0)
        else:
            group_level_parts_result.at[idx, 'avg_used_qty'] = round(float(part_row['avg_used_qty']), 3)

    # Define parts price calculation
    def parts_price(row):
        if row["latest_price"] :
            return row["avg_used_qty"] * (row["latest_price"])
        else:
            return 0

    # Apply parts price calculation
    group_level_parts_result["billable_value_usd"] = group_level_parts_result.apply(parts_price, axis=1)
    
    return group_level_parts_result[['source_task_discrepancy_number', 'group', 'issued_part_number',
       'avg_used_qty', 'max_used_qty', 'min_used_qty', 'latest_price',
       'total_billable_value_usd', 'total_used_qty', 'part_description','prob', 'billable_value_usd']]


async def most_probable_defects(aircraft_age: float, aircraft_model_family: List[str], check_category: List[str], customer_name_list: List[str], customer_name_consideration: bool):
    train_packages=packages_extract(aircraft_age,aircraft_model_family,check_category,customer_name_list,customer_name_consideration)
    sub_task_description_max500mh_lhrh=pd.DataFrame(list(db["sub_task_description_max500mh_lhrh"].find({"package_number": {"$in": train_packages}})))
    #sub_task_description_max500mh_lhrh=sub_task_description_max500mh_lhrh[sub_task_description_max500mh_lhrh['package_number'].isin(train_packages)]
    tasks=sub_task_description_max500mh_lhrh["source_task_discrepancy_number_updated"].unique().tolist()
    defects_list = sub_task_description_max500mh_lhrh["log_item_number"].unique().tolist()
    sub_task_description_max500mh_lhrh=sub_task_description_max500mh_lhrh[['log_item_number',
                'task_description', 'corrective_action',
            'source_task_discrepancy_number','source_task_discrepancy_number_updated', 'estimated_man_hours', 
            'actual_man_hours', 'skill_number',"package_number"]]
    cluster_data=compute_cluster(sub_task_description_max500mh_lhrh,tasks)
    sub_task_parts_lhrh = pd.DataFrame(
    list(db["sub_task_parts_lhrh"].find({
        "package_number": {"$in": train_packages},
        "task_number": {"$in": defects_list}
    }))
    )
    sub_task_parts_lhrh["latest_price"] = sub_task_parts_lhrh["issued_part_number"].apply(
    lambda x: parts_master.loc[parts_master["issued_part_number"] == x, "latest_total_billable_price"].values[0] if x in parts_master["issued_part_number"].values else 0
    )
    print("the shape of the parts data is",sub_task_parts_lhrh.shape)
    cluster_parts_data=parts_prediction(cluster_data[["log_item_number","source_task_discrepancy_number","group", "package_number"]],sub_task_parts_lhrh)
    cluster_parts_data["parts_cost"] = cluster_parts_data["billable_value_usd"] * (cluster_parts_data["prob"] / 100)

    # Group by 'issued_part_number' and 'part_description', then sum the 'parts_cost'
    top_failing_parts = cluster_parts_data.groupby(
        ["issued_part_number", "part_description"], as_index=False
    ).agg(
        total_cost=("parts_cost", "sum")
    )


    top_failing_parts = top_failing_parts.sort_values(by="total_cost", ascending=False).head(10)
        
    cluster_manhours_data=manhours_prediction(cluster_data[[
    "source_task_discrepancy_number","full_description",
    "actual_man_hours",
    "skill_number",
    "group",
    "package_number"
    ]])
    findings=[]
    findings_manhours=0
    findings_spare_parts_cost=0
    cluster_manhours_data = cluster_manhours_data.sort_values(by="prob", ascending=False)
    for index, row in cluster_manhours_data.iterrows():
        spare_parts = []
        spare_filtered = cluster_parts_data[cluster_parts_data["group"] == row["group"]] 
        prob_factor = row["prob"] / 100
        
        for _, part in spare_filtered.iterrows():
            part_cost = part["billable_value_usd"] * part["prob"] / 100* prob_factor
            findings_spare_parts_cost += part_cost
            
            spare_parts.append({
                "partId": part["issued_part_number"],
                "desc": part["part_description"],
                "qty": float_round(part["avg_used_qty"]),
                "price": float_round(part["billable_value_usd"]),
                "prob": float_round(part["prob"])
            })

        finding = {
            "taskId": row["source_task_discrepancy_number"],
            "details": [{
                "cluster": f"{row['source_task_discrepancy_number']}/{row['group']}",
                "description": row["description"],
                "skill": row["skill_number"],
                "mhs": {
                    "max": float_round(row["avg_actual_man_hours"])*np.random.uniform(1.1, 1.3),  # Randomly increase max by 10-30%
                    "min": float_round(row["avg_actual_man_hours"])*np.random.uniform(0.7, 0.9),  # Randomly decrease min by 10-30%
                    "avg": float_round(row["avg_actual_man_hours"]),
                    "est": float_round(row["max_actual_man_hours"])  # Consider using a different field for estimate
                },
                "prob": float_round(row["prob"]),
                "spare_parts": spare_parts
            }]
        }
        
        findings_manhours += row["avg_actual_man_hours"] * prob_factor
        
        # Fixed condition: should be <= 11 to get first 12 items, or remove condition entirely
        if index <= 10:  # or remove this condition if you want all findings
            findings.append(finding)

    # Fixed typo: "reliability" not "reliablity"
    reliability_score = (
        85 if findings_manhours < 1000
        else 75 if findings_manhours < 3000
        else 50 if findings_manhours < 5000
        else 25
    )


    result = {
        "findings": findings,
        "findings_summary": {
            "reliability_score": reliability_score,  # Fixed typo
            "total_manhours": float_round(findings_manhours),
            "total_spare_parts_cost": float_round(findings_spare_parts_cost),
            "total_clusters": len(cluster_data["group"].unique())
        },
        "top_failing_parts": top_failing_parts.to_dict(orient='records')
    }

    result = replace_nan_inf(result)
    print(result)
    return result

async def estima_defects_prediction(tasks):
    """
    Predicts defects for a list of tasks.

    Args:
        tasks (List[str]): List of task numbers to predict defects for.

    Returns:
        Dict: A dictionary containing defect predictions.
    """
    if not tasks:
        return {
            "findings": [],
            "findings_manhours": 0.0,
            "findings_spare_parts_cost": 0.0
        }

    # Convert tasks to a DataFrame
    sub_task_description_max500mh_lhrh = pd.DataFrame(
        list(db["sub_task_description_max500mh_lhrh"].find(
            {"source_task_discrepancy_number_updated": {"$in": tasks}},
            {"_id": 0}
        ))
    )

    if sub_task_description_max500mh_lhrh.empty:
        return {
            "findings": [],
            "findings_manhours": 0.0,
            "findings_spare_parts_cost": 0.0
        }
    
    # Compute clusters
    sub_task_description_max500mh_lhrh = sub_task_description_max500mh_lhrh[['log_item_number',
                'task_description', 'corrective_action',
            'source_task_discrepancy_number','source_task_discrepancy_number_updated', 'estimated_man_hours', 
            'actual_man_hours', 'skill_number',"package_number"]]
    
    cluster_data = compute_cluster(sub_task_description_max500mh_lhrh, tasks)
    
    if cluster_data.empty:
        return {
            "findings": [],
            "findings_manhours": 0.0,
            "findings_spare_parts_cost": 0.0
        }
    
    defects_list = sub_task_description_max500mh_lhrh["log_item_number"].unique().tolist()
    
    sub_task_parts_lhrh = pd.DataFrame(
        list(db["sub_task_parts_lhrh"].find({
            "task_number": {"$in": defects_list}
        }))
    )
    
    # Add latest_price column
    sub_task_parts_lhrh["latest_price"] = sub_task_parts_lhrh["issued_part_number"].apply(
        lambda x: parts_master.loc[parts_master["issued_part_number"] == x, "latest_total_billable_price"].values[0] 
        if x in parts_master["issued_part_number"].values else 0
    )
    
    print("the shape of the parts data is", sub_task_parts_lhrh.shape)
    
    cluster_parts_data = parts_prediction(
        cluster_data[["log_item_number","source_task_discrepancy_number","group", "package_number"]], 
        sub_task_parts_lhrh
    )
    
    cluster_manhours_data = manhours_prediction(cluster_data[[
        "source_task_discrepancy_number","full_description",
        "actual_man_hours",
        "skill_number",
        "group",
        "package_number"
    ]])
    
    # Prepare the result
    findings = []
    findings_manhours = 0
    findings_spare_parts_cost = 0
    
    for _, row in cluster_manhours_data.iterrows():
        spare_parts = []
        # Calculate probability-adjusted values
        prob_factor = row["prob"] / 100
        spare_filtered = cluster_parts_data[cluster_parts_data["group"] == row["group"]] 
        
        for _, part in spare_filtered.iterrows():
            # Use billable_value_usd instead of total_cost
            part_cost = part.get("billable_value_usd", 0)
            findings_spare_parts_cost += part_cost * prob_factor
            
            spare_parts.append({
                "partId": part["issued_part_number"],  # Use issued_part_number instead of part_number
                "desc": part["part_description"],
                "qty": float_round(part["avg_used_qty"]),
                "price": float_round(part_cost),  # Use billable_value_usd
                "part_type": part.get("issued_unit_of_measurement", "Unknown"),  # Use available column or default
                "prob": float_round(part["prob"])
            })

        finding = {
            "taskId": row["source_task_discrepancy_number"],  # Use available column
            "chapterName": str(row["source_task_discrepancy_number"])[:2],  # Use available column
            "details": [{
                "cluster": f"{row['source_task_discrepancy_number']}/{row['group']}",
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
    return result

async def defect_investigator(task_number: List[str], log_item_number: str, defect_desc: str, corrective_action: str):
    # Input variables


    # If you're matching a single task number, use a string instead of a list
    task_number_str = task_number[0]
    print("data is being processed for task number:", task_number_str)

    # Filter the dataframe correctly
    exdata = pd.DataFrame(
        list(sub_task_description_max500mh_lhrh.find(
            {"source_task_discrepancy_number_updated": task_number_str},
            {"_id": 0}
        ))
    )


    # Prepare input data as a DataFrame row
    input_data = {
        'log_item_number': log_item_number,
        'task_description': defect_desc,
        'corrective_action': corrective_action,
        'source_task_discrepancy_number': task_number_str,
        'source_task_discrepancy_number_updated': task_number_str,
        'estimated_man_hours': 0,
        'actual_man_hours': 0,
        'skill_number': "UNKNOWN",
        'package_number': "TEST"
    }

    # Append the input_data as a new row to exdata (returns a new DataFrame)
    exdata = pd.concat([exdata, pd.DataFrame([input_data])], ignore_index=True)
    print("cluster data is being computed for task number:", task_number_str)
    # Assuming `tasks` is defined somewhere above or is a column name in the DataFrame
    cluster_data = compute_cluster(exdata , task_number)
    print("cluster data is computed for task number:", task_number_str)
    # Ensure log_item_number is a string if comparing with string column
    log_item_number = str(log_item_number)
    # Filter the cluster data where 'log_item_number' matches
    group_number=cluster_data[cluster_data["log_item_number"] == log_item_number]["group"].iloc[0]

    similar_defect_data=cluster_data[cluster_data["group"] == group_number]
    similar_defect_data=similar_defect_data[similar_defect_data["log_item_number"] != log_item_number]
    defects_list = similar_defect_data["log_item_number"].unique().tolist()
    sub_task_parts_lhrh = pd.DataFrame(
    list(db["sub_task_parts_lhrh"].find({
        "task_number": {"$in": defects_list}
    }))
    )
    if not sub_task_parts_lhrh.empty:
        sub_task_parts_lhrh["latest_price"] = sub_task_parts_lhrh["issued_part_number"].apply(
        lambda x: parts_master.loc[parts_master["issued_part_number"] == x, "latest_total_billable_price"].values[0] if x in parts_master["issued_part_number"].values else 0
        )
    print("the shape of the parts data is",sub_task_parts_lhrh.shape)
    cluster_parts_data=parts_prediction(similar_defect_data[["log_item_number","source_task_discrepancy_number","group", "package_number"]],sub_task_parts_lhrh)
    cluster_manhours_data=manhours_prediction(similar_defect_data[[
    "source_task_discrepancy_number","full_description",
    "actual_man_hours",
    "skill_number",
    "group",
    "package_number"
    ]])
    findings=[]
    findings_manhours=0
    findings_spare_parts_cost=0
    # Join all full_description entries into one text string
    findings_text = similar_defect_data["full_description"].str.cat(sep=" ")

    # Extract keywords (assuming db is available in context)
    keywords_list = keyword_extraction(findings_text, n_keywords=10)
    cluster_manhours_data = cluster_manhours_data.sort_values(by="prob", ascending=False)
    for _, row in cluster_manhours_data.iterrows():
        spare_parts = []
        spare_filtered = cluster_parts_data[cluster_parts_data["group"] == row["group"]]
        prob_factor = row["prob"] / 100 
        for _, part in spare_filtered.iterrows():
            findings_spare_parts_cost += (part["billable_value_usd"]  * (part["prob"] / 100) * prob_factor)
            spare_parts.append({
                "partId": part["issued_part_number"],
                "desc": part["part_description"],
                "qty": float_round(part["avg_used_qty"]),
                "price": float_round(part["billable_value_usd"]),
                "prob": float_round(part["prob"])
            })

        # Calculate probability-adjusted values for disc_pred_list

        finding = {
            "taskId": row["source_task_discrepancy_number"],
            "details": [{
                "cluster": f"{row['source_task_discrepancy_number']}/{row['group']}",
                "description": row["description"],
                "skill": row["skill_number"],
                "mhs": {
                    "max": float_round(row["avg_actual_man_hours"])*np.random.uniform(1.1, 1.3),  # Randomly increase max by 10-30%
                    "min": float_round(row["avg_actual_man_hours"])*np.random.uniform(0.7, 0.9),  # Randomly decrease min by 10-30%
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
        "findings_summary": {
            "total_findings": len(findings),
            "total_manhours": float_round(findings_manhours),
            "total_spare_parts_cost": float_round(findings_spare_parts_cost),
            "total_clusters": len(cluster_data["group"].unique())
        },
        "keywords": keywords_list,
        "findings_raw_data":similar_defect_data.to_dict(orient='records')
        
   
        
    }
    # Convert the result to a dictionary
    print("preprocessing the result for task number:", task_number_str)
    result = replace_nan_inf(result)
    print(result)
    return result
    
async def event_log_management():
    response = {
        "planned": [],
        "closed": [],
        "prepared": [],
        "in_progress": []
    }

    for index, row in aircraft_event_log.iterrows():
        event_details = {
            'system_component': row['system_component_csdd'],
            'aircraft_reg': row['aircraft_reg'],
            'description': row['description'],
            'source_system': row['source_system'],
            'reference': row['reference'],
            'category': row['category'],
            'status': row['status'],
            'severity': row['severity'],
            'model': row['model'],
            'age': row['age'],
            'mel_cdl_applicable': row['mel_cdl_applicable'],
            'dispatch_allowed': row['dispatch_allowed'],
            'estimated_man_hours': row['approx_resolution_time'],
            'limitation_action': row['limitation_action'],
            'diagnosis_1': row['diagnosis_1'],
            'diagnosis_1_probability': row['diagnosis_1_probability'],
            'diagnosis_2': row['diagnosis_2'],
            'diagnosis_2_probability': row['diagnosis_2_probability']
        }

        status = row['status'].lower()

        if status == "open":
            response["planned"].append(event_details)
        elif status == "closed":
            event_details["actual_man_hours"] = row['approx_resolution_time']*1.5
            response["closed"].append(event_details)
        elif status == "prepared":
            response["prepared"].append(event_details)
        elif status in ["in_progress", "in progress"]:  # handle both forms
            response["in_progress"].append(event_details)

    return response



def replace_nan_inf(obj):
    """
    Recursively replace numpy data types, NaN, and Inf values with Python native types
    to ensure JSON serialization works properly.
    Compatible with NumPy 2.0+
    """
    
    if isinstance(obj, dict):
        return {k: replace_nan_inf(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_inf(v) for v in obj]
    # Updated NumPy integer types
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    # Updated NumPy float types
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return float('inf') if obj > 0 else float('-inf')
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return replace_nan_inf(obj.tolist())
    elif isinstance(obj, pd.DataFrame):
        return replace_nan_inf(obj.to_dict('records'))
    elif isinstance(obj, pd.Series):
        return replace_nan_inf(obj.to_dict())
    elif obj is pd.NA or pd.isna(obj):
        return None
    return obj

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import requests
import pandas as pd
import numpy as np
import tarfile
import io
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from tqdm import tqdm
import shutil

Base = declarative_base()

class DatabaseManager:
    def __init__(self, db_type, dbapi, host, user, password, database, port):
        self.db_type = db_type
        self.dbapi = dbapi
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port
        self.engine = None
        self.session = None

    def create_engine(self):
        database_url = f"{self.db_type}+{self.dbapi}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        try:
            self.engine = create_engine(database_url, echo=False)
            with self.engine.connect() as connection:
                print("Database connection established.")
        except Exception as e:
            print(f"Failed to connect to the database. Error: {e}")

    def create_session(self):
        if self.engine:
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            print("Database session created.")
        else:
            print("Database engine not found. Please create the engine first.")

    def create_tables(self):
        Base.metadata.create_all(self.engine)
        print("Database tables created.")

    def upload_dataframes_to_sql(self, data_dict, chunksize=500):
        for table_name, dataframe in data_dict.items():
            table_name_lower = table_name.lower()
            if self.engine.dialect.has_table(self.engine, table_name_lower):
                with self.engine.connect() as conn:
                    conn.execute(f"DROP TABLE IF EXISTS {table_name_lower} CASCADE")
                    print(f"Table {table_name_lower} dropped.")

            total_rows = len(dataframe)
            chunks = range(0, total_rows, chunksize)

            for start in tqdm(chunks, desc=f"Uploading {table_name_lower}", total=len(chunks)):
                end = start + chunksize
                chunk = dataframe.iloc[start:end]
                chunk.to_sql(
                    table_name_lower,
                    con=self.engine,
                    if_exists='append' if start > 0 else 'replace',
                    index=False,
                    method='multi'
                )
            #print(f"Table {table_name_lower} uploaded to the database.")

class DatasetDownloader:
    def __init__(self, file_urls, dest_folder):
        self.file_urls = file_urls
        self.dest_folder = dest_folder

    def download_file(self, url):
        if not os.path.exists(self.dest_folder):
            os.makedirs(self.dest_folder)
            print(f"Destination folder {self.dest_folder} created.")

        local_filename = url.split('/')[-1]
        path = os.path.join(self.dest_folder, local_filename)
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"File {local_filename} downloaded successfully.")
        else:
            print(f"Failed to download file {local_filename}.")
        return path

    def download_all_files(self):
        for url in self.file_urls:
            self.download_file(url)
        print("All files downloaded.")

class NPASSDataset:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.data_frames = {}
        self.NPASS = {}

    def clean_filename(self, filename):
        name = os.path.splitext(filename)[0]
        name = (name.replace('NPASSv2.0_download_', '')
                    .replace('_', ' '))
        return name

    def load_data(self, file_path):
        try:
            df = pd.read_csv(file_path, sep='\t', engine='python')
            print(f"Data loaded from {file_path}.")
            return df
        except Exception as e:
            print(f"Failed to load data from {file_path}. Error: {e}")

    def load_all_data(self):
        for filename in os.listdir(self.dataset_folder):
            file_path = os.path.join(self.dataset_folder, filename)
            if os.path.isfile(file_path):
                df_name = self.clean_filename(filename)
                self.data_frames[df_name] = self.load_data(file_path)
        print("All data loaded.")

    def rename_and_store_dataframes(self, rename_mapping):
        for old_name, new_name in rename_mapping.items():
            if old_name in self.data_frames:
                self.NPASS[new_name] = self.data_frames[old_name]
                print(f"DataFrame {old_name} renamed to {new_name}.")

    def process_dataframes(self):
        for df_name, df in self.NPASS.items():
            initial_row_count = df.shape[0]
            df.drop_duplicates(inplace=True)
            final_row_count = df.shape[0]
            duplicates_dropped = initial_row_count - final_row_count
            df.replace('N.A.', np.nan, inplace=True)
            self.NPASS[df_name] = df
            print(f"DataFrame {df_name} processed. Duplicates dropped: {duplicates_dropped}.")

class NPASSProcessor:
    def __init__(self, n_pass):
        self.n_pass = n_pass

    def find_primary_key_candidates(self):
        for df_name, df in self.n_pass.items():
            primary_key_candidates = [col for col in df.columns if df[col].is_unique]
            print(f"Primary key candidates for {df_name}: {primary_key_candidates}")

    def combine_gi_and_structure(self):
        if 'GI_NP' in self.n_pass and 'Structure_NP' in self.n_pass:
            compound_df = self.n_pass['GI_NP']
            structure_df = self.n_pass['Structure_NP']
            merged_df = pd.merge(compound_df, structure_df, on='np_id', how='inner')
            self.n_pass['GI_NP'] = merged_df
            del self.n_pass['Structure_NP']
            print("GI_NP and Structure_NP DataFrames merged.")
        else:
            print("GI_NP or Structure_NP DataFrame not found. Skipping merge.")



class NAEBDataLoader:
    def __init__(self, tar_url, extract_folder):
        self.tar_url = tar_url
        self.extract_folder = os.path.abspath(extract_folder)  # Convert to absolute path
        self.NAEB = {}

    def download_and_extract_tar(self):
        response = requests.get(self.tar_url)
        response.raise_for_status()

        # Create a temporary directory for extraction
        temp_extract_folder = os.path.join(self.extract_folder, 'temp')
        if not os.path.exists(temp_extract_folder):
            os.makedirs(temp_extract_folder)
        
        with tarfile.open(fileobj=io.BytesIO(response.content), mode='r:gz') as thetar:
            thetar.extractall(temp_extract_folder)
        print(f"NAEB tar file downloaded and extracted to temporary folder {temp_extract_folder}.")

        # Move files to the correct directory
        for root, dirs, files in os.walk(temp_extract_folder):
            for file in files:
                shutil.move(os.path.join(root, file), self.extract_folder)
                print(f"Moved {file} to {self.extract_folder}")

        # Clean up temporary directory
        shutil.rmtree(temp_extract_folder)
        print(f"Temporary folder {temp_extract_folder} deleted.")

    def rename_and_store_dataframes(self, rename_mapping):
        for file_name, key in rename_mapping:
            file_path = os.path.join(self.extract_folder, file_name)
            if os.path.exists(file_path):
                self.NAEB[key] = pd.read_csv(file_path)
                print(f"DataFrame {file_name} stored with key {key}.")
            else:
                print(f"File {file_path} not found.")

class NAEBProcessor:
    def __init__(self, naeb):
        self.naeb = naeb

    def process_use_subcategory(self):
        if 'Uses_NB' in self.naeb:
            df = self.naeb['Uses_NB']

            if 'use_subcategory' in df.columns:
                df['use_subcategory'].fillna(0, inplace=True)
                df['use_subcategory'] = df['use_subcategory'].astype(int)
                self.naeb['Uses_NB'] = df
                print("Use subcategory column processed.")
            else:
                print("Use subcategory column not found in Uses_NB DataFrame.")
        else:
            print("Uses_NB DataFrame not found in NAEB dictionary.")

class USDAProcessor:
    def __init__(self, usda_path, naeb, npass):
        self.usda_df = pd.read_csv(usda_path)
        self.naeb = naeb
        self.npass = npass
        self.common_usda_codes = set()
        self.common_usda_codes_list = []

    def find_common_usda_codes(self):
        if 'Species_NB' in self.naeb:
            species_nb_df = self.naeb['Species_NB']
            non_null_usda_codes = species_nb_df['usda_code'].dropna()
            self.common_usda_codes = set(self.usda_df['Symbol']).intersection(set(non_null_usda_codes))
            self.common_usda_codes_list = list(self.common_usda_codes)
            print(f"Found {len(self.common_usda_codes)} common USDA codes.")
        else:
            print("Species_NB DataFrame not found in NAEB dictionary.")

    def count_unique_common_usda_codes(self):
        unique_count = len(self.common_usda_codes)
        print(f"Number of unique common USDA codes: {unique_count}")
        return unique_count

    def add_usda_codes_to_taxonomic_info(self):
        if 'TaxonomicInfo_NP' in self.npass:
            taxonomicinfo_np_df = self.npass['TaxonomicInfo_NP']
            filtered_usda_df = self.usda_df[self.usda_df['Symbol'].isin(self.common_usda_codes_list)]
            taxonomicinfo_np_df['USDA_code'] = None

            for index, npass_row in tqdm(taxonomicinfo_np_df.iterrows(), total=taxonomicinfo_np_df.shape[0], desc="Checking matches"):
                org_name = npass_row['org_name'].lower().strip()
                matches = filtered_usda_df[filtered_usda_df['Scientific Name with Author'].str.contains(org_name, case=False, regex=False, na=False)]
                if not matches.empty:
                    taxonomicinfo_np_df.at[index, 'USDA_code'] = matches.iloc[0]['Symbol']
            print("USDA codes added to TaxonomicInfo_NP DataFrame.")
        else:
            print("TaxonomicInfo_NP DataFrame not found in NPASS dictionary.")

# ORM classes for NAEB
class Species_NB(Base):
    __tablename__ = 'species_nb'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    common_names = Column(String)
    usda_code = Column(String)
    family = Column(String)
    family_apg = Column(String)

class Tribes_NB(Base):
    __tablename__ = 'tribes_nb'
    id = Column(Integer, primary_key=True)
    name = Column(String)

class Use_Categories_NB(Base):
    __tablename__ = 'use_categories_nb'
    id = Column(Integer, primary_key=True)
    name = Column(String)

class Use_Subcategories_NB(Base):
    __tablename__ = 'use_subcategories_nb'
    id = Column(Integer, primary_key=True)
    parent = Column(Integer, ForeignKey('use_categories_nb.id'))
    name = Column(String)

class Sources_NB(Base):
    __tablename__ = 'sources_nb'
    id = Column(Integer, primary_key=True)
    refcode = Column(String)
    type = Column(String)
    fulltext = Column(String)
    address = Column(String)
    author = Column(String)
    booktitle = Column(String)
    comment = Column(String)
    edition = Column(String)
    editor = Column(String)
    journal = Column(String)
    month = Column(String)
    note = Column(String)
    number = Column(Float)
    pages = Column(String)
    publisher = Column(String)
    school = Column(String)
    title = Column(String)
    url = Column(String)
    volume = Column(String)
    year = Column(Integer)

class Uses_NB(Base):
    __tablename__ = 'uses_nb'
    id = Column(Integer, primary_key=True)
    species = Column(Integer, ForeignKey('species_nb.id'))
    tribe = Column(Integer, ForeignKey('tribes_nb.id'))
    source = Column(Integer, ForeignKey('sources_nb.id'))
    pageno = Column(String)
    use_category = Column(Integer, ForeignKey('use_categories_nb.id'))
    use_subcategory = Column(Integer, ForeignKey('use_subcategories_nb.id'))
    notes = Column(Text)
    rawsource = Column(Text)

# ORM classes for NPASS
class GI_NP(Base):
    __tablename__ = 'gi_np'
    np_id = Column(String, primary_key=True)
    pref_name = Column(String)
    iupac_name = Column(String)
    chembl_id = Column(String)
    pubchem_cid = Column(String)
    num_of_organism = Column(Float)
    num_of_target = Column(Float)
    num_of_activity = Column(Float)
    if_has_Quantity = Column(String)
    InChI = Column(String)
    InChIKey = Column(String)
    SMILES = Column(String)
    activities = relationship("Activity_NP", back_populates="gi_np")
    species = relationship("Species_NP", back_populates="gi_np")

class Activity_NP(Base):
    __tablename__ = 'activity_np'
    artificial_id = Column(Integer, primary_key=True)
    np_id = Column(String, ForeignKey('gi_np.np_id'))
    target_id = Column(String)
    activity_type_grouped = Column(String)
    activity_relation = Column(String)
    activity_type = Column(String)
    activity_value = Column(Float)
    activity_units = Column(String)
    assay_organism = Column(String)
    assay_tax_id = Column(String)
    assay_strain = Column(String)
    assay_tissue = Column(String)
    assay_cell_type = Column(String)
    ref_id = Column(String)
    ref_id_type = Column(String)
    gi_np = relationship("GI_NP", back_populates="activities")

class Species_NP(Base):
    __tablename__ = 'species_np'
    artificial_id = Column(Integer, primary_key=True)
    src_org_pair = Column(String)
    org_id = Column(String, ForeignKey('taxonomicinfo_np.org_id'))
    np_id = Column(String, ForeignKey('gi_np.np_id'))
    new_cp_found = Column(String)  
    org_isolation_part = Column(String)
    org_collect_location = Column(String)
    org_collect_time = Column(String)
    ref_type = Column(String)
    ref_id = Column(String)
    ref_id_type = Column(String)
    ref_url = Column(String)  
    gi_np = relationship("GI_NP", back_populates="species")
    taxonomicinfo = relationship("TaxonomicInfo_NP", back_populates="species")

class TaxonomicInfo_NP(Base):
    __tablename__ = 'taxonomicinfo_np'
    org_id = Column(String, primary_key=True)
    org_name = Column(String)  
    org_tax_level = Column(String)
    org_tax_id = Column(String)
    subspecies_tax_id = Column(String)
    subspecies_name = Column(String)
    species_tax_id = Column(String)
    species_name = Column(String)
    genus_tax_id = Column(String)
    genus_name = Column(String)
    family_tax_id = Column(String)
    family_name = Column(String)
    kingdom_tax_id = Column(String)
    kingdom_name = Column(String)
    superkingdom_tax_id = Column(String)
    superkingdom_name = Column(String) 
    USDA_code = Column(String)
    species = relationship("Species_NP", back_populates="taxonomicinfo")

class Targets_NP(Base):
    __tablename__ = 'targets_np'
    artificial_id = Column(Integer, primary_key=True)
    target_id = Column(String)
    target_type = Column(String)
    target_name = Column(String)
    target_organism_tax_id = Column(String)
    target_organism = Column(String)
    uniprot_id = Column(String)

def main():
      
    db_manager = DatabaseManager(
        db_type='postgresql',
        dbapi='psycopg2',
        host='localhost',
        user='postgres',  
        password='123',
        database='dbtest',
        port=5432
    )

    # Create engine and session
    db_manager.create_engine() 
    db_manager.create_session()

    # Create the tables in the database
    db_manager.create_tables()

    # NPASS dataset URLs and renaming map
    npass_file_urls = [
        "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_generalInfo.txt",
        "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_structureInfo.txt",
        "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_activities.txt", 
        "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_species_pair.txt",
        "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_targetInfo.txt",
        "https://bidd.group/NPASS/downloadFiles/NPASSv2.0_download_naturalProducts_speciesInfo.txt"
    ]

    npass_rename_mapping = {
        'naturalProducts generalInfo': 'GI_NP',  
        'naturalProducts structureInfo': 'Structure_NP',
        'naturalProducts activities': 'Activity_NP',
        'naturalProducts species pair': 'Species_NP',
        'naturalProducts targetInfo': 'Targets_NP',  
        'naturalProducts speciesInfo': 'TaxonomicInfo_NP'
    }

    # NAEB dataset URL and renaming map
    naeb_tar_url = 'https://naeb.louispotok.com/static/naeb.tar.gz'
    naeb_rename_mapping = [
        ('uses.csv', 'Uses_NB'), 
        ('species.csv', 'Species_NB'),
        ('tribes.csv', 'Tribes_NB'),
        ('use_categories.csv', 'Use_Categories_NB'), 
        ('use_subcategories.csv', 'Use_Subcategories_NB'),
        ('sources.csv', 'Sources_NB')  
    ]

    # Initialize and process NPASS dataset
    npass_downloader = DatasetDownloader(npass_file_urls, './Datasets/NPASS Dataset')
    npass_downloader.download_all_files() 

    npass_dataset = NPASSDataset('./Datasets/NPASS Dataset')
    npass_dataset.load_all_data()
    npass_dataset.rename_and_store_dataframes(npass_rename_mapping) 
    npass_dataset.process_dataframes()

    npass_processor = NPASSProcessor(npass_dataset.NPASS)
    npass_processor.find_primary_key_candidates()
    npass_processor.combine_gi_and_structure()

    # Initialize and process NAEB dataset 
    naeb_loader = NAEBDataLoader(naeb_tar_url, './naeb_data/data/naeb_dump')
    naeb_loader.download_and_extract_tar()
    naeb_loader.rename_and_store_dataframes(naeb_rename_mapping)

    naeb_processor = NAEBProcessor(naeb_loader.NAEB)  
    naeb_processor.process_use_subcategory()

    # Initialize and process USDA data
    usda_processor = USDAProcessor('D:/Download/usda.csv', naeb_loader.NAEB, npass_dataset.NPASS)
    usda_processor.find_common_usda_codes()
    usda_processor.count_unique_common_usda_codes() 
    usda_processor.add_usda_codes_to_taxonomic_info()

    # Upload processed data to database
    db_manager.upload_dataframes_to_sql(npass_dataset.NPASS)
    db_manager.upload_dataframes_to_sql(naeb_loader.NAEB)

    print("Data processing and upload completed successfully.")

if __name__ == "__main__":
    main()


# In[ ]:





## pydqc functions and parameters

#### pydqc.infer_schema.infer_schema(_data, fname, output_root='', sample_size=1.0, type_threshold=0.5, n_jobs=1, base_schema=None, base_schema_feature_colname='column', base_schema_dtype_colname='type')  

**function:** infer data types for all columns for the input table  
  
**Parameters:** 

* **_data**: pandas DataFrame  
	- data table to infer  
* **fname**: string  
	- the output file name  
* **output_root**: string, default=''  
	- the root directory for the output file  
* **sample_size**: int or float(<= 1.0), default=1.0  
	- int: number of sample rows to infer the data type (useful for large tables)  
	- float: sample size in percentage  
* **type_threshold**: float(<= 1.0), default=0.5  
	- threshold for inferring data type  
* **n_jobs**: int, default=1  
	- the number of jobs to run in parallel  
* **base_schema**: pandas DataFrame, default=None  
	- data schema to base on  
* **base_schema_feature_colname**: string  
	- feature_colname in base schema  
* **base_schema_dtype_colname**: string  
	- dtype_colname in base schema  

**Example:**  
```python
import pandas as pd
from pydqc import infer_schema, data_summary, data_compare

data_2016 = pd.read_csv('data/properties_2016.csv')
infer_schema.infer_schema(_data=data_2016, fname='properties_2016', output_root='output/', 
                          sample_size=1.0, type_threshold=0.5, n_jobs=1, 
                          base_schema=None, base_schema_feature_colname='column', base_schema_dtype_colname='type')
			  
# with base schema
data_2016_schema = pd.read_excel('output/data_schema_properties_2016_mdf.xlsx')
data_2017 = pd.read_csv('data/properties_2017.csv')
infer_schema.infer_schema(_data=data_2017, fname='properties_2017_sample', output_root='output/', 
                          sample_size=0.1, type_threshold=0.5, n_jobs=1, 
                          base_schema=data_2016_schema, base_schema_feature_colname='column', 
			  base_schema_dtype_colname='type')
```
-------------------------------------------------------------------------------------------------------
  
#### pydqc.data_summary.data_summary(table_schema, _table, fname, sample_size=1.0, feature_colname='column', dtype_colname='type', output_root='', n_jobs=1)

**function:** summary basic information of all columns in a data table based on the provided data schema  

**Parameters:** 
	
* **table_schema**: pandas DataFrame  
	- schema of the table, should contain data types of each column  
* **_table**: pandas DataFrame  
	- the data table  
* **fname**: string  
	- the output file name  
* **sample_size**: integer or float(<=1.0), default=1.0  
	- int: number of sample rows to do the summary (useful for large tables)  
	- float: sample size in percentage  
* **feature_colname**: string  
	- name of the column for feature  
* **dtype_colname**: string  
	- name of the column for data type  
* **output_root**: string, default=''  
	- the root directory for the output file  
* **n_jobs**: int, default=1  
	- the number of jobs to run in parallel  

**Example:**  
```python
import pandas as pd
from pydqc import infer_schema, data_summary, data_compare

data_2016 = pd.read_csv('data/properties_2016.csv')
# we should use the modified data schema
data_2016_schema = pd.read_excel('output/data_schema_properties_2016_mdf.xlsx')

data_summary.data_summary(table_schema=data_2016_schema, _table=data_2016, fname='properties_2016', 
                          sample_size=1.0, feature_colname='column', dtype_colname='type', 
			  output_root='output/', n_jobs=1)
```
-------------------------------------------------------------------------------------------------------
  
#### pydqc.data_summary.data_summary_notebook(table_schema, _table, fname, sample=False, feature_colname='column', dtype_colname='type', output_root='')

**function:** automatically generate ipynb for data summary 

**Parameters:** 
	
* **table_schema**: pandas DataFrame  
	- schema of the table, should contain data types of each column  
* **_table**: pandas DataFrame  
	- the data table  
* **fname**: string  
	- the output file name  
* **sample**: boolean, default=False  
	- whether to do sampling on the original data  
* **feature_colname**: string  
	- name of the column for feature  
* **dtype_colname**: string  
	- name of the column for data type  
* **output_root**: string, default=''  
	- the root directory for the output file  

**Example:**  
```python
import pandas as pd
from pydqc import infer_schema, data_summary, data_compare

data_2016 = pd.read_csv('data/properties_2016.csv')
# we should use the modified data schema
data_2016_schema = pd.read_excel('output/data_schema_properties_2016_mdf.xlsx')

data_summary.data_summary_notebook(table_schema=data_2016_schema, _table=data_2016, fname='properties_2016',
                                   sample=False, feature_colname='column', dtype_colname='type', output_root='output/')
```
-------------------------------------------------------------------------------------------------------
  
#### pydqc.data_summary.distribution_summary_pretty(_value_df, col, figsize=None, date_flag=False)

**function:** draw pretty distribution graph for a column

**Parameters:** 
	
* **_value_df**: pandas DataFrame  
	- slice of dataframe containing enough information to check  
* **col**: string  
	- name of column to check 
* **figsize**: tuple, default=None 
	- figure size 
* **date_flag**: bool, default=False 
	- whether it is checking date features  

**Example:**  
```python
import pandas as pd
from pydqc import infer_schema, data_summary, data_compare

table = pd.read_csv('../data/properties_2016.csv')
col="basementsqft"
value_df = table[[col]].copy()
distribution_summary_pretty(value_df, col, figsize=None, date_flag=False)
```
-------------------------------------------------------------------------------------------------------
  
#### pydqc.data_compare.data_compare(_table1, _table2, _schema1, _schema2, fname, sample_size=1.0, feature_colname1='column', feature_colname2='column', dtype_colname1='type', dtype_colname2='type', output_root='', n_jobs=1)

**function:** compare values of same columns between two tables

**Parameters:** 
	
* **_table1**: pandas DataFrame  
	- one of the two tables to compare  
* **_table2**: pandas DataFrame  
	- one of the two tables to compare  
* **_schema1**: pandas DataFrame  
	- data schema (contains column names and corresponding data types) for _table1  
* **_schema2**: pandas DataFrame  
	- data schema (contains column names and corresponding data types) for _table2  
* **fname**: string  
	- the output file name  
* **sample_size**: integer or float(<=1.0), default=1.0  
	- int: number of sample rows to do the comparison (useful for large tables)  
	- float: sample size in percentage  
* **feature_colname1**: string, default='column'  
	- name of the column for feature of _table1  
* **feature_colname2**: string, default='column'  
	- name of the column for feature of _table2  
* **dtype_colname1**: string, default='type'  
	- name of the column for data type of _table1  
* **dtype_colname2**: string, default='type'  
	- name of the column for data type of _table2  
* **output_root**: string, default=''  
	- the root directory for the output file  
* **n_jobs**: int, default=1  
	- the number of jobs to run in parallel  

**Example:**  
```python
import pandas as pd
from pydqc import infer_schema, data_summary, data_compare

data_2016 = pd.read_csv('data/properties_2016.csv')
data_2017 = pd.read_csv('data/properties_2017.csv')

# we should use the modified data schema
data_2016_schema = pd.read_excel('output/data_schema_properties_2016_mdf.xlsx')
data_2017_schema = pd.read_excel('output/data_schema_properties_2017_mdf.xlsx')

data_compare.data_compare(_table1=data_2016, _table2=data_2017, _schema1=data_2016_schema, _schema2=data_2017_schema,
                          fname='properties_2016', sample_size=1.0, feature_colname1='column', feature_colname2='column',
                          dtype_colname1='type', dtype_colname2='type', output_root='output/', n_jobs=1)
```
-------------------------------------------------------------------------------------------------------
  
#### pydqc.data_compare.data_compare_notebook(_table1, _table2, _schema1, _schema2, fname, sample=False, feature_colname1='column', feature_colname2='column', dtype_colname1='type', dtype_colname2='type', output_root='')

**function:** automatically generate ipynb for data comparison

**Parameters:** 
	
* **_table1**: pandas DataFrame  
	- one of the two tables to compare  
* **_table2**: pandas DataFrame  
	- one of the two tables to compare  
* **_schema1**: pandas DataFrame  
	- data schema (contains column names and corresponding data types) for _table1  
* **_schema2**: pandas DataFrame  
	- data schema (contains column names and corresponding data types) for _table2  
* **fname**: string  
	- the output file name  
* **sample**: boolean, default=False
	- whether to do sampling on the original data  
* **feature_colname1**: string, default='column'  
	- name of the column for feature of _table1  
* **feature_colname2**: string, default='column'  
	- name of the column for feature of _table2  
* **dtype_colname1**: string, default='type'  
	- name of the column for data type of _table1  
* **dtype_colname2**: string, default='type'  
	- name of the column for data type of _table2  
* **output_root**: string, default=''  
	- the root directory for the output file  

**Example:**  
```python
import pandas as pd
from pydqc import infer_schema, data_summary, data_compare

data_2016 = pd.read_csv('data/properties_2016.csv')
data_2017 = pd.read_csv('data/properties_2017.csv')

# we should use the modified data schema
data_2016_schema = pd.read_excel('output/data_schema_properties_2016_mdf.xlsx')
data_2017_schema = pd.read_excel('output/data_schema_properties_2017_mdf.xlsx')

data_compare.data_compare_notebook(_table1=data_2016, _table2=data_2017, _schema1=data_2016_schema, _schema2=data_2017_schema,
                                   fname='properties_2016', sample=False, feature_colname1='column', feature_colname2='column', 
                                   dtype_colname1='type', dtype_colname2='type', output_root='output/')
```
-------------------------------------------------------------------------------------------------------
  
#### pydqc.data_compare.distribution_compare_pretty(_df1, _df2, col, figsize=None, date_flag=False)

**function:** draw pretty distribution graph for comparing a column between two tables

**Parameters:** 
	
* **_df1**: pandas DataFrame  
	- slice of table1 containing enough information to check  
* **_df2**: pandas DataFrame  
	- slice of table2 containing enough information to check  
* **col**: string  
	- name of column to check  
* **figsize**: tuple, default=None  
	- figure size  
* **date_flag**: bool, default=False  
	- whether it is checking date features  

**Example:**  
```python
import pandas as pd
from pydqc import infer_schema, data_summary, data_compare

table1 = pd.read_csv('data/properties_2016.csv')
table2 = pd.read_csv('data/properties_2017.csv')

# we should use the modified data schema
col="bathroomcnt"
df1 = table1[[col]].copy()
df2 = table2[[col]].copy()

distribution_compare_pretty(df1, df2, col, figsize=None, date_flag=False)
```
-------------------------------------------------------------------------------------------------------

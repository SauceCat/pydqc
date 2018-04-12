# pydqc
[![PyPI version](https://badge.fury.io/py/pydqc.svg)](https://badge.fury.io/py/pydqc)

Python automatic data quality check toolkit. Aims to relieve the pain of writing tedious codes for general data understanding by:    
- Automatically generate data summary report, which contains useful statistical information for each column in a data table. (useful for general data understanding)
- Automatically summarize the statistical difference between two data tables. (useful for comparing training set with test set, comparing the same data table from two different snapshot dates, etc.)  
- But still need some help from human for data types inferring. :see_no_evil:

## Motivation
"Today I don't feel like doing anything about data quality check, I just wanna lay in my bed. Don't feel like writing any tedious codes. So build a tool runs on its own." :microphone: :musical_note: :notes:       
-modified **The Lazy Song**   
  
<img src="https://github.com/SauceCat/pydqc/blob/master/images/rescue-cat-sleeps-doll-bed-sophie-4.jpg" width="15%">  


## Install pydqc
  - install [py2nb](https://github.com/sklam/py2nb)
  - install dependents `pip install -r requirements.txt`
  - install pydqc
  
```bash
git clone https://github.com/SauceCat/pydqc.git
cd pydqc
python setup.py install
```

## How does it work?
<img src="https://github.com/SauceCat/pydqc/blob/master/images/pydqc_process.jpg" width="50%">   

For an input data table (pandas dataframe): 
### Step 1: data schema
- **function: pydqc.infer_schema.infer_schema(data, fname, output_root='', sample_size=1.0, type_threshold=0.5, n_jobs=1, base_schema=None)**  
  Infer data types for each column. pydqc recognizes four basic data types, including 'key', 'str', 'date', 'numeric'.  
    - **'key'**: column that doesn't have concrete meaning itself, but acts as 'key' to link with other tables.  
    - **'str'**: categorical column
    - **'date'**: datetime column
    - **'numeric'**: numeric column  
    
  After inferring, an excel file named 'data_schema_XXX.xlsx' (XXX here represents the 'fname' parameter) is generated. We should check the generated file and modify the 'type' column when it is necessary (when the infer_schema function makes some mistakes). But it is easy because we can do the modification by selecting from a drop down list.  
  <br><img src="https://github.com/SauceCat/pydqc/blob/master/images/infer_schema_drop_down_list.png" width="60%">  

  You can also modify the 'include' column to exclude some features for further checking.  
  <br><img src="https://github.com/SauceCat/pydqc/blob/master/images/infer_schema_drop_down_list_include.png" width="60%">  
 
  ***For this version, pydqc is not able to infer the 'key' type, so it always needs human modification.***  
  After necessary modification, it is better to save the schema as 'data_schema_XXX_mdf.xlsx' or with some other names different from the original one.  
  - example output for infer_schema: [raw schema](https://github.com/SauceCat/pydqc/blob/master/test/output/data_schema_properties_2016.xlsx), [modified schema](https://github.com/SauceCat/pydqc/blob/master/test/output/data_schema_properties_2016_mdf.xlsx)
  
### Step 2 (option 1): data summary  
- **function: pydqc.data_summary.data_summary(table_schema, table, fname, sample_size=1.0, sample_rows=100, output_root='', n_jobs=1)**   
  Summary basic statistical information for each column based on the provided data type.  
    - **'key' and 'str'**: sample value, rate of nan values, number of unique values, top 10 value count.  
    **example output:**  
      <img src="https://github.com/SauceCat/pydqc/blob/master/images/data_summary_key.png" width="40%">   
    
    *the only different between summary for 'key' and 'str' is pydqc doesn't do sampling for 'key' columns. (check 'sample_size' parameter)*
    - **'date'**: sample value, rate of nan values, number of unique values, minimum numeric value, mean numeric value, median numeric value, maximum numeric value, maximum date value, minimum date value, distribution graph for numeric values.  
    **example output:**  
      <img src="https://github.com/SauceCat/pydqc/blob/master/images/data_summary_date.png" width="80%">  
      
    *numeric value for 'date' column is calculated as the time difference between the date value and today in months.*  
    - **'numeric'**: sample value, rate of nan values, number of unique values, minimum value, mean value, median value, maximum value, distribution graph (log10 scale automatically when absolute maximum value is larger than 1M).  
    **example output:**   
      <img src="https://github.com/SauceCat/pydqc/blob/master/images/data_summary_numeric.png" width="80%">   
      <img src="https://github.com/SauceCat/pydqc/blob/master/images/data_summary_numeric_log10.png" width="80%">  
 
    ***You can also turn the whole data summmary process into a jupyter notebook by function data_summary_notebook()***  
  - example output for data summary: [data summary report](https://github.com/SauceCat/pydqc/blob/master/test/output/data_summary_properties_2016.xlsx)  
  - example output for data summary notebook: [data summary notebook](https://github.com/SauceCat/pydqc/blob/master/test/output/data_summary_notebook_properties_2016.ipynb)  
      
### Step 2 (option 2): data compare 
- **function: data_compare(table1, table2, schema1, schema2, fname, sample_size=1.0, output_root='', n_jobs=1)**   
  Compare statistical characteristics of the same columns between two different tables based on the provided data type. (It might be useful when we want to compare training set with test set, or sample table from two different snapshot dates)  
    - **'key'**: compare sample value, rate of nan values, number of unique values, size of intersection set, size of set only in table1, size of set only in table2, venn graph.  
    **example output:**   
      <img src="https://github.com/SauceCat/pydqc/blob/master/images/data_compare_key.png" width="80%">   
      
    - **'str'**: compare sample value, rate of nan values, number of unique values, size of intersection set, size of set only in table1, size of set only in table2, top 10 value counts.  
    **example output:**  
      <img src="https://github.com/SauceCat/pydqc/blob/master/images/data_compare_str.png" width="60%"> 
      
    - **'date'**: compare sample value, rate of nan values, number of unique values, minimum numeric value, mean numeric value, median numeric value, maximum numeric value, maximum date value, minimum date value, distribution graph for numeric values.  
    **example output:**   
      <img src="https://github.com/SauceCat/pydqc/blob/master/images/data_compare_date.png" width="80%">
      
    *numeric value for 'date' column is calculated as the time difference between the date value and today in months.*  
    - **'numeric'**: compare sample value, rate of nan values, number of unique values, minimum value, mean value, median value, maximum value,distribution graph.  
    **example output:**   
      <img src="https://github.com/SauceCat/pydqc/blob/master/images/data_compare_numeric.png" width="80%"> 

   ***You can also turn the whole data compare process into a jupyter notebook by function data_compare_notebook()***  
  - example output for data compare: [data compare report](https://github.com/SauceCat/pydqc/blob/master/test/output/data_compare_properties_2016.xlsx)  
  Inside the excel report, there is a worksheet called 'summary'. This worksheet summarizes the basic information regarding the comparing result, including a 'corr' field that indicates correlation of the same column between different tables.  
      <img src="https://github.com/SauceCat/pydqc/blob/master/images/data_compare_summary.PNG" width="45%">  
      - **key**: 'corr' = rate of overlap
      - **str**: 'corr' = Spearman rank-order correlation coefficient between not-nan value counts
      - **numeric** and **date**: 'corr' = Spearman rank-order correlation coefficient between not-nan value counts (when the number of unique values is small) or between not-nan value distribution (use 100-bin histogram)  
  
  - example output for data compare notebook: [data compare notebook](https://github.com/SauceCat/pydqc/blob/master/test/output/data_compare_notebook_properties_2016.ipynb)  
  
### Step 2 (option 3): data consist 
- **function: data_consist(table1, table2, key1, key2, schema1, schema2, fname, sample_size=1.0, output_root='', keep_images=False, n_jobs=1)**   
  Check consistency of the same columns between two different tables by merging tables on the provided keys. (It might be useful when we want to compare training set with test set, or sample table from two different snapshot dates)  
    - **'key'**: same as data_compare for key type  
      
    - **'str'**: check whether two values of the same key is the same between two tables.  
    **example output:**  
      <img src="https://github.com/SauceCat/pydqc/blob/master/images/data_consist_str.PNG" width="45%"> 
      
    - **'numeric'**: calculate a Spearman rank-order correlation coefficient between values of the same key between two tables, calculate the minimum, mean, median, maximum difference rate between two values.  
    **example output:**   
      <img src="https://github.com/SauceCat/pydqc/blob/master/images/data_consist_numeric.PNG" width="80%"> 

   ***You can also turn the whole data compare process into a jupyter notebook by function data_consist_notebook()***  
  - example output for data consist: [data consist report](https://github.com/SauceCat/pydqc/blob/master/test/output/data_consist_properties_2016.xlsx)  
  Inside the excel report, there is a worksheet called 'summary'. This worksheet summarizes the basic information regarding the consistency checking result, including a 'corr' field that indicates correlation of the same column between different tables.  
      <img src="https://github.com/SauceCat/pydqc/blob/master/images/data_consist_summary.PNG" width="45%">  
      - **key**: 'corr' = rate of overlap
      - **str**: 'corr' = rate of consistency
      - **numeric** and **date**: 'corr' = Spearman rank-order correlation coefficient between not-nan value pairs
  - example output for data consist notebook: [data consist notebook](https://github.com/SauceCat/pydqc/blob/master/test/output/data_consist_notebook_properties_2016.ipynb)  


## Documentation
For details about the ideas, please refer to [Introducing pydqc](https://medium.com/@SauceCat/introducing-pydqc-7f23d04076b3).  
For description about the functions and parameters, please refer to [pydqc functions and parameters](https://github.com/SauceCat/pydqc/blob/master/parameters.md).  
For test and demo, please refer to https://github.com/SauceCat/pydqc/tree/master/test.  
For example outputs, please refer to https://github.com/SauceCat/pydqc/tree/master/test/output.  


## Contribution
If you have other ideas for general automatic data quality check, push requests are always welcome! :raising_hand:


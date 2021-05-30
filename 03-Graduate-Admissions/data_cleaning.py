"""
This python file is a collection for data cleaning"""

def drop_feature(data, features):
    return data.drop(columns= features)

def column_log(data, features):
    data_log = data.loc[features].applymap(np.log).add_suffix('_log') 
    return data_log
    
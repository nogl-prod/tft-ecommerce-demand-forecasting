#!/usr/bin/env python
# coding: utf-8

# In[1]:


from configparser import ConfigParser


# In[2]:


def config(section, filename='/opt/ml/processing/input/databaseAWSRDS.ini'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db
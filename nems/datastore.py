"""
Store model parameters and function objectives during optimization

This module provides a tools for saving model fitting results (parameters and function evaluations) to a Mongo database
during optimization. It is useful for visualizing learned model parameters in real time (using, for example, a separate
web interface). It is also useful for storing results of a large model fitting procedure such that you have the results
regardless of if python or the machine happens to crash.

.. warning:: Work in progress!

Classes
-------
- `Datastore` -- A class for managing a connection to a Mongo database that lets you store model parameters and function
    objectives.

"""

# imports
from pymongo import MongoClient
from datetime import datetime
import numpy as np


class Datastore(object):
    """
    Store parameters and objective values in a mongodb during optimization
    """

    def __init__(self, name, desc, headers, host='localhost', port=27017, dbname='expts'):
        """
        initialize modelfitting datastore object

        Arguments
        ---------
        name        -- name of the simulation
        desc        -- text describing the simulation
        headers     -- tuple containing the names of the columns of data

        """

        # initialize Mongo client
        self.host = host
        self.port = port
        self.client = MongoClient(self.host, self.port)

        # connect to database
        self.db = self.client[dbname]

        # collection for storing experiment metadata
        self.metadata = self.db['metadata']

        # collection for storing experiment results
        self.results = self.db['results']

        # experiment initialization
        self.name = name.replace(" ", "")
        self.desc = desc
        self.headers = headers

        # insert metadata object into database
        self.metadata.insert({
            'name': self.name,
            'desc': self.desc,
            'headers': self.headers,
            'createdOn': datetime.now().isoformat()
        })

    def insert(self, idx, data, spread=None):
        """
        Insert results for this experiment

        Arguments
        ---------
        idx         -- a number (index) corresponding to this data point
        data        -- a tuple consisting of float values to store

        """
        assert len(data) == len(self.headers), 'Data is not the same length as headers! These need to match up.'

        if spread is None:
            spread = tuple(-1*np.ones(len(data)))

        self.results.insert({
            'index': idx,
            'exptname': self.name,
            'date': datetime.now().isoformat(),
            'data': data,
            'spread': spread
        })

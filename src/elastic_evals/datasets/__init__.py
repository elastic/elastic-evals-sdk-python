# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

"""Dataset stores: where the runner gets its examples and their ids from."""

from .kibana import KibanaDatasetStore
from .memory import InMemoryDatasetStore

__all__ = ["InMemoryDatasetStore", "KibanaDatasetStore"]

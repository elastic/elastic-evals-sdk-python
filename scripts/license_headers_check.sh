#!/usr/bin/env sh
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License 2.0;
# you may not use this file except in compliance with the Elastic License 2.0.

if [ $# -eq 0 ]
then
   FILES=$(find . \( -name "*.py" -o -name "*.c" -o -name "*.sh" \) -size +1c -not -path "./dist/*" -not -path "./build/*" -not -path "./*venv*/*")
else
    FILES=$@
fi

LICENSE_HEADER="Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one"

MISSING=$(grep --files-without-match "$LICENSE_HEADER" ${FILES})

if [ -z "$MISSING" ]
then
    exit 0
else
    echo "Files with missing copyright header:"
    echo $MISSING
    exit 1
fi
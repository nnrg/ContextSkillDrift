#!/usr/bin/env bash

if [ -d "/lusr/opt/python-3.6.4/" ]; then
    export PATH=/lusr/opt/python-3.6.4/bin:$PATH
fi

python3 "$@"
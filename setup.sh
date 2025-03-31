# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 22:48:58 2025

@author: Suprava Modak
"""
mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml

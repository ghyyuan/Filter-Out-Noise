#!/bin/bash
export PATH="/home/ubuntu/.local/bin:$PATH"
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
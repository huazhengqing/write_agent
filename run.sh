#!/bin/bash


docker-compose up -d

source venv/bin/activate


streamlit run ui/home.py



#
# http://127.0.0.1:8000
# http://127.0.0.1:8000/docs
#
uvicorn api.main:app --reload




npm install


PORT=3000 npm start


#
# http://localhost:5173
#
npm run dev

npm run build



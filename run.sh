#!/bin/bash


docker-compose up -d

source venv/bin/activate


#
# http://127.0.0.1:8000
# http://127.0.0.1:8000/docs
#
uvicorn api.main:app --reload


#
# http://localhost:5173
#
cd ui
npm run dev

npm run build



PORT=3000 npm start

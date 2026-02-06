## Run Locally

1. Install dependencies
pip install -r requirements.txt

2. Create vector database
python run_pipeline.py

3. Start API
uvicorn app.main:app --reload

4. Open browser
http://127.0.0.1:8000/docs

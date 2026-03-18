# 1. base image 
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy only requirements first (This makes future builds 10x faster)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Now copy the rest of your project files
COPY . .

# 6. Streamlit's actual default port
EXPOSE 8501

# 7. Run Streamlit and bind it to all network interfaces
CMD ["streamlit", "run", "stream.py", "--server.port=8501", "--server.address=0.0.0.0"] 
# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Set a default command that keeps the container alive if needed, 
# though we will override this via docker-compose run
CMD ["echo", "Container ready. Use docker-compose run --rm app python src/train.py"]
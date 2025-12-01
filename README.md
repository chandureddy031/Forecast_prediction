# Build image
docker build -t ml-pipeline-app .

# Run container
docker run -p 8000:8000 ml-pipeline-app

# Or use docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
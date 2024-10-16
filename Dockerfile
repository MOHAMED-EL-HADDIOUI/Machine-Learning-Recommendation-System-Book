# Use the official Python image as the base image for the application
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt ./

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Stage 2: Serve the application using nginx
FROM nginx:alpine

# Copy the configuration file for Nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built application from the previous stage
COPY --from=base /app /usr/share/nginx/html

# Expose port 80 (the default port for nginx)
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]

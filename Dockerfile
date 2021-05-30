FROM python:3.8-buster

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN sed -i 's/cu101/cpu/' requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy only the relevant directories
COPY . /repo/

# Run the web server
EXPOSE 5000
ENV PYTHONPATH /repo
CMD python /repo/app1.py
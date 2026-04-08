# CSC 575 - Final Project - Recommender System

## Install Dependencies

1. Install [Docker Desktop](https://www.docker.com/)
2. Install [Git](https://git-scm.com/downloads)
3. (Optional) Verify Installs in Terminal
    * Run `docker --version` and you should see something like `Docker version 28.0.4, build b8034c0`
    * Run `docker compose version` and you should see something like `Docker Compose version v2.34.0-desktop.1`
    * Run `git --version` and you should see something like `git version 2.49.0`

## Clone the Repository
NOTE: You will need access to the repo
1. Open a terminal
2. Navigate to a directory that you wish to keep the project files in
3. Clone the project repository `git clone https://github.com/ablaser5/csc575-final-project.git`

For Professor/Grader: Skip as you should already have the code

## Running the Application

1. Start the docker desktop application
2. Run `cd csc575-final-project` to change into the project directory
3. Run `docker compose build` to build the container
4. Run `docker compose up` to run the container
5. (Optional) You can build and run the container in one command with: `docker compose up --build`

## Setting up the Application

1. Keep your container running and open a new terminal window
2. Run `docker compose exec web bash` to run a shell inside the container
3. Run `python manage.py migrate` to initialize the database
4. Run `python manage.py createsuperuser` to create the first super user in the app

## Importing the data

1. Create a virtual environment using the  `requirements.txt` file and activate it
2. Run `pip install -r requirements.txt` to install the requirements
3. In order to run the import script, create a `.env` file in the root directory for this project. Add the following:
    ```
    DB_NAME=mydb
    DB_USER=myuser
    DB_PASSWORD=mypassword
    DB_HOST=localhost
    DB_PORT=5432
    ```

    This will allow connecting to the container's database without running the script in the container
4. Make sure the container is running
5. Outside of the container, in a separate terminal, navigate to the root of the project
6. Run `python scripts/import_movie_data.py`

## Running the experiments

1. Repeat steps 1-5 from the importing data section if you haven't already
2. Run `python experiments/[name_of_experiment_file]`

The files for the experiments are:
- `experiments/experiment_clustering.py`
- `experiments/experiment_collaborative_based.py`
- `experiments/experiment_content_based.py`
- `experiments/experiment_sentiment_analysis.py`


## Using the Web App
1. Make sure the container is running
2. Visit `http://0.0.0.0:8000/` in your browser
3. Login to the super user account you created earlier
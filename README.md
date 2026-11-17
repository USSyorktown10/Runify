# Runify

Important note: Due to the fact that the product name "Runify" is already in use, Runify will be renamed again. When this will happen is unknown, as well as if this will happen, as Runify is built more to be a open-source project instead of a product (for now).

## What's this?

Runify is a Strava-esque platform with a goal of making many advanced metrics paywalled by large companies more widely available, and to focus the social aspect onto advanced stats rather than basic activity tracking.
It also fixes many of the nice-to-have features that Strava does not (better workout visibility). We also plan to provide tools for coaches to assign workouts and
for runners to mark their workouts as completing coach-defined workouts, as well as 3d visualizations of runs, and comparisons of metrics between athletes and activities.

## Current Status
This project is currently in very early development. Algorithms, API spec, schema, and endpoints and finished or nearly finished, with algs needing more attention. 
Now that the listed objectives have been completed, we are now building the frontend and backend for Runify.

To see our progress and plans, check out our [Google Doc](https://docs.google.com/document/d/1otQ1p44epb4aDC_gV2EMzUMwYKy_1-d5wA-t6tC3zEU/view)

## Stack

DB: Postgres

API Server: FastAPI/uvicorn

Web Server: Uvicorn 

Note that to ease development (*cough* have a language that everyone knows), we will be writing the backend in Python. This is (obviously) not ideal performance-wise, but should be "good enough" for the scale of this project, and noting that the needs of the project are mostly calculation and data-based, and the speed of FastAPI (good name right?) should match the level needed.

### Contributors

**[USSyorktown10](https://github.com/USSyorktown10)** (Creator, Developer) <-- Doing most of the work on this branch

**[quantumbagel](https://github.com/quantumbagel)** (Developer)

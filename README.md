# forecasting_recommendation
This is a forecasting system used to propose future ADB and CC AR metrics. The system is composed of four classes: Backend, Frontend, Functional Forms, and Recommender.

## Backend 
The backend is responsible for extracting and extrapolating data actuals, tracking the miss/beats between the actuals and forecasted data of metrics for the current month.

## Frontend
The bread and butter of the entire forecasting system. The frontend is responsible for processing the base dataset to be used in functional forms and the recommender.

## Functional Forms
Builds the processed data from the frontend to be analyzed and plotted throughout an obsrvation period. Includes a global plotting function as well.

## Recommender
Applies user input forecasting methodology to change the projected metrics forecast for the remainder of the year. Includes a global summary to show the present vs projected change

## User Interface
A .ipynb that allows users to interact with the forecasted system. 

# State of Health (SoH) Estimation for Battery Cells
Estimating the State of Health (SoH) of batteries is crucial for Battery Management Systems (BMS) as it provides vital information about the battery's condition and guides its usage. This repository offers a data-driven method for SoH estimation of NMC cells using two different types of voltage responses as input:

-  Constant Current (CCCV) Voltage Response
-  Pulse Current Voltage Response
  
The code calculates the Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE) for both input types, with observations indicating that the second input type consistently delivers superior performance.

# Repository Structure
Rawdata: Contains voltage profiles and state values of 48 NMC cells.
Code: Contains Python code for processing the data and estimating SoH. Further information and detailed instructions are available within the folder.

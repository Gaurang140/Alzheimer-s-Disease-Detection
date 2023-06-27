# Alzheimer’s Disease Detection : 
* Alzheimer's disease is a progressive and irreversible neurodegenerative disorder that causes memory loss, cognitive impairment, and a decline in various brain functions, leading to a loss of independence in daily life. It is estimated that the number of people affected by Alzheimer's disease will increase from 47 million to 152 million by 2050, resulting in significant economic, medical, and societal consequences. There is currently no cure or treatment available that can halt the disease progression, and its pathophysiology remains unknown. Patients with amnestic moderate cognitive impairment (MCI) are at a higher risk of developing Alzheimer's disease, emphasizing the importance of early detection through MCI screening to facilitate better care and the development of new treatments.

# Objective
* The main objective is to detect Alzheimer's disease and mild cognitive impairment early through the proposed CNN network approach, which retrieves features from brain MRI data to differentiate between healthy cognition, clinical diagnosed AD, and MCI, and identifying the patterns of MRI brain changes that characterize AD and MCI.



#workflow
fatch data from mongodb 
create split train and test
validation of the data
model training
model evaluation
model pusher in production


# ML flow 
```bash 

set MLFLOW_TRACKING_URI=https://dagshub.com/Gaurang140/Alzheimer-s-Disease-Detection.mlflow 
set MLFLOW_TRACKING_USERNAME= username 
set MLFLOW_TRACKING_PASSWORD= password 

````
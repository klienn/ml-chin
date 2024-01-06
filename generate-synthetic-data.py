import numpy as np
import pandas as pd

np.random.seed(0)
n_samples = 1000

age = np.random.randint(1, 20, n_samples)  
oil_quality = np.random.uniform(70, 95, n_samples)  
temperature = np.random.uniform(25, 40, n_samples) 
load = np.random.uniform(50, 100, n_samples)  
humidity = np.random.uniform(40, 80, n_samples) 
voltage_fluctuations = np.random.uniform(1, 10, n_samples) 
frequency_fluctuations = np.random.uniform(0.01, 0.2, n_samples)

maintenance_need = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

data = pd.DataFrame({
    'Age': age,
    'Oil_Quality': oil_quality,
    'Temperature': temperature,
    'Load': load,
    'Humidity': humidity,
    'Voltage_Fluctuations': voltage_fluctuations,
    'Frequency_Fluctuations': frequency_fluctuations,
    'Maintenance_Need': maintenance_need
})

data.to_csv('power-transformer-maintenance/power_transformer_maintenance_dataset.csv', index=False)

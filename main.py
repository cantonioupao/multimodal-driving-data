from load_data import load_data
import os

print("So let's start the task")
data_path = os.path.join("./data", "data.p")
data = load_data(data_path)
velodyne =  data['velodyne']
print(velodyne.shape)
import os
import pandas as pd
from current_voltage import Fitting
downloads_path = os.path.expanduser("C:/Users/amira/Downloads/master teza/iv mjerenja")
file_name1 = "IV_187.txt"
file_name2 = "IV_200.txt"
file_name3 = "IV_250.txt"
file_name4 = "IV_288.txt"
file_name5 = "IV_300.txt"
file_name6 = "IV_321.txt"
file_name7 = "IV_350.txt"
file_names = [file_name1, file_name2, file_name3, file_name4, file_name5, file_name6, file_name7]
dataframes = []
for name in file_names:
    file_path = os.path.join(downloads_path,name)
    with open(file_path,'r') as file:
        content = file.readlines()
        data = [line.strip().split('\t') for line in content]
        columns = ['Anode I', 'Anode V']
        dtype = {'Anode I': float, 'Anode V': float}
        df = pd.DataFrame(data[1:], columns=columns).astype(dtype)
        dataframes.append(df)


linear_fit = Fitting(dataframes)
#linear_fit.lin_fit()
#linear_fit.plot_ln()
#linear_fit.plot_ln()
linear_fit.rich_graph()
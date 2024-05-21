import csv

data = [
    
['Country',	'Age',	'Salary',	'Purchased'],
['France',	'44',	'72000',	'No'],
['Spain',	'27',	'48000',	'Yes'],
['Germany',	'30',	'54000',	'No'],
['Spain',   '38',	'61000',	'No'],
['Germany',	'40',	  '',      	'Yes'],
['France',	'35',	'58000',	'Yes'],
['Spain',	'',    '52000',	    'No'],
['France',	'48',   '79000',	'Yes'],
['Germany',	'50',   '83000',	'No'],
['France',	'37',	'67000',	'Yes'],
    
]

filename = '/Users/jogbaner/Machine_Learning_Demo/demo.csv'

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Dummy data has been successfully written to {filename}.")
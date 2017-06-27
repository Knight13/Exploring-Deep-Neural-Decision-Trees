import numpy as np

data = []
with open('covtype.data', 'r+') as f:
    for line in f:
        line = line.rstrip('\n')
        i = line.split(',')
        data.append(i)

    f.closed

for entry in data:
    for (index, attribute) in enumerate(entry):
        try:
            entry[index] = float(attribute)
        except ValueError:
            data.remove(entry)
           
        
for (n,i) in enumerate(data):
    if i[-1:][0] == 1:
        data[n][len(i)-1] = [1,0,0,0,0,0,0]

    elif i[-1:][0] == 2:
        data[n][len(i)-1] = [0,1,0,0,0,0,0]
    
    elif i[-1:][0] == 3:
        data[n][len(i)-1] = [0,0,1,0,0,0,0]
    
    elif i[-1:][0] == 4:
        data[n][len(i)-1] = [0,0,0,1,0,0,0]
        
    elif i[-1:][0] == 5:
        data[n][len(i)-1] = [0,0,0,0,1,0,0]
        
    elif i[-1:][0] == 6:
        data[n][len(i)-1] = [0,0,0,0,0,1,0]
        
    else:
        data[n][len(i)-1] = [0,0,0,0,0,0,1]        

feature = np.vstack(np.array(i[:-1], dtype = np.float32) for i in data)
label = np.vstack(np.array(i[-1:], dtype = np.float32) for i in data)


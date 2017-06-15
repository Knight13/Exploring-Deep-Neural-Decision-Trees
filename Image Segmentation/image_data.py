import numpy as np

data = []
with open('data.txt', 'r+') as f:
    for line in f:
        line = line.rstrip('\n')
        i = line.split(',')
        data.append(i)

    f.closed

for entry in data:
    for (index, attribute) in enumerate(entry):
        if index > 0:
            try:
                entry[index] = float(attribute)
            except ValueError:
                data.remove(entry)
            
for (n,i) in enumerate(data):
    if i[0] == 'BRICKFACE':
        data[n][0] = [1,0,0,0,0,0,0]

    elif i[0] == 'SKY':
        data[n][0] = [0,1,0,0,0,0,0]
    
    elif i[0] == 'FOLIAGE':
        data[n][0] = [0,0,1,0,0,0,0]
    
    elif i[0] == 'CEMENT':
        data[n][0] = [0,0,0,1,0,0,0]
        
    elif i[0] == 'WINDOW':
        data[n][0] = [0,0,0,0,1,0,0]
        
    elif i[0] == 'PATH':
        data[n][0] = [0,0,0,0,0,1,0]
        
    else:
        data[n][0] = [0,0,0,0,0,0,1]


feature = np.vstack(np.array(i[1:], dtype = np.float32) for i in data)
label = np.vstack(np.array(i[0], dtype = np.float32) for i in data)


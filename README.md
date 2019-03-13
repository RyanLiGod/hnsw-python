# hnsw-python

HNSW implemented by python. 

#### Supported distances:

| Distance          | parameter | Equation                                                |
| ----------------- | --------- | ------------------------------------------------------- |
| Squared L2        | 'l2'      | d = sum((Ai-Bi)^2)                                      |
| Cosine similarity | 'cosine'  | d = 1.0 - sum(Ai\*Bi) / sqrt(sum(Ai\*Ai) \* sum(Bi*Bi)) |

#### examples

```python
import time
from progressbar import *
import pickle
from hnsw import HNSW

dim = 200
num_elements = 10000

data = np.array(np.float32(np.random.random((num_elements, dim))))
hnsw = HNSW('cosine', m0=16, ef=128)
widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(), ' ', ETA()]

# show progressbar
pbar = ProgressBar(widgets=widgets, maxval=train_len).start()
for i in range(len(data)):
    hnsw.add(data[i])
    pbar.update(i + 1)
pbar.finish()

# save index
with open('glove.ind', 'wb') as f:
    picklestring = pickle.dump(hnsw, f, pickle.HIGHEST_PROTOCOL)

# load index
fr = open('glove.ind','rb')
hnsw_n = pickle.load(fr)

add_point_time = time.time()
idx = hnsw_n.search(np.float32(np.random.random((1, 200))), 10)
search_time = time.time()
print("Searchtime: %f" % (search_time - add_point_time))
```


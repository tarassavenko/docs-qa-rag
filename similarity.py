import numpy as np 
def cosine_similarity(a,b):
    a=np.array(a)
    b=np.array(b)

    product_dot=np.dot(a,b) #To eswteriko ginomeno twn 2 arrays

    norm_a=np.linalg.norm(a) #To mhkos tou dianusmatos a
    norm_b=np.linalg.norm(b) #To mhkos tou dianusmatos b

    return product_dot/(norm_a * norm_b)
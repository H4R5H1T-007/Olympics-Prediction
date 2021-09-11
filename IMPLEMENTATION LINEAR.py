#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

def test(x, y):
    n = np.size(x)

    mx = np.mean(x)
    my = np.mean(y)
    
    p = np.sum(y*x) - n*my*mx
    q = np.sum(x*x) - n*mx*mx

    b1 = p / q
    b0 = my - b1*mx

    return (b0, b1)

def plot_regression_line(x, y, b):
    
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
  
    
    y_pred = b[0] + b[1]*x
  
   
    plt.plot(x, y_pred, color = "g")
  
    
    plt.xlabel('x')
    plt.ylabel('y')
  
    
    plt.show()
def main():
    
    x = np.array([1,3,5,6,8,12,14])
    y = np.array([3,7,11,12,13,14,16])

    
    b = test(x, y)
    print("Estimated coefficients:\nb_0 = {}     \nb_1 = {}".format(b[0], b[1]))

    
    plot_regression_line(x, y, b)

if __name__ == "__main__":
    main()


# In[ ]:





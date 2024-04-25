#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install tensorflow')


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# In[4]:


(x_train, y_train) , (x_test, y_test) = mnist.load_data()


# In[5]:


x_train[0]


# In[6]:


plt.imshow(x_train[7])


# In[7]:


y_train[7]


# In[8]:


#1 can be removed in 3channels
print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape) 
print("Test data shape:", x_train.shape) 
print("Test labels shape:", y_test.shape) 


# In[9]:


data_type = x_train.dtype
print(data_type)


# In[10]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# In[11]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# In[12]:


x_train 
x_test 


# In[13]:


y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# In[14]:


data_type = x_train.dtype
print(data_type)


# In[15]:


y_train[7]


# In[16]:


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()


# In[17]:


model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[18]:


model.fit(x_train, y_train, epochs=10, batch_size=200, verbose=1)


# In[19]:


loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# In[20]:


y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)


# In[21]:


y_pred[:3]


# In[22]:


y_pred_classes[:3]


# In[23]:


y_true = np.argmax(y_test, axis=1)


# In[24]:


y_test[:3]


# In[25]:


y_true[:3]


# In[26]:


report = classification_report(y_true, y_pred_classes, target_names=[str(i) for i in range(10)])
print("Classification Report:\n", report)


# In[27]:


confusion = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





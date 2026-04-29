## Logistic Regression from Scratch (Iris Dataset)

This project implements **logistic regression from scratch** and visualizes the model using:

- Decision boundary  
- Probability heatmap  

---

### Goal
Build a model that correctly distinguishes between:

- **Iris-setosa**
- **Iris-versicolor**

---

### Features

The model uses all 4 features:

- Sepal length  
- Sepal width  
- Petal length  
- Petal width  

---

### Decision Boundary & Heatmap

![Decision Boundary](https://github.com/user-attachments/assets/e0250349-e3ef-4de7-9a0a-83cd8d5eb88a)

We can see that although some outliers exist, they don’t affect the model too much. Predictions remain correct.

Interestingly the slope of the decision boundarydoes not align with the gradient seen in the probability heatmap.

---

### Feature Correlation

![Feature Correlation](https://github.com/user-attachments/assets/18ba3859-22be-415f-b71c-c2a48267de53)

All features show strong correlation with each other.

This likely affects how the model distributes weights.



# Task 10: Table Analysis Using PaddleOCR

## Introduction

The objective of this task is to employ PaddleOCR, an Optical Character Recognition (OCR) toolkit, to extract and analyze tabular data from images or scanned documents. The process includes detecting tables, extracting data into a structured format, performing statistical analyses, and visualizing the results.


## Implementation

### Step 1: Initialize PaddleOCR and Load Image

We will begin by initializing PaddleOCR and loading the image containing the tabular data.

### Step 2: Perform OCR and Extract Data

We will utilize PaddleOCR to perform OCR on the image and extract text data. 

### Step 3: Create a DataFrame for Analysis

Extracted data will be structured into a Pandas DataFrame for further analysis.

### Step 4: Statistical Computations

We will compute summary statistics and value counts of the extracted data.

### Step 5: Data Visualization

A pie chart will be created to visualize the value counts of the extracted data.

### Example Code

Here’s the complete code implementing the pipeline:

```python
from paddleocr import PaddleOCR 
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')
image_path = '/path/to/your/image.jpg'  # Input image path here
image = cv2.imread(image_path)

# Perform OCR
result = ocr.ocr(image_path, cls=True)
extracted_data = []

# Extract text from OCR result
for line in result:
    for word_info in line:
        extracted_data.append(word_info[1][0])  # Extracting detected text

# Create a DataFrame from extracted data
df = pd.DataFrame(extracted_data, columns=['Extracted Text'])

# Perform statistical computations
summary_statistics = df.describe()
value_counts = df['Extracted Text'].value_counts()

print("Summary Statistics:")
print(summary_statistics)
print("\nValue Counts:")
print(value_counts)

# Create a pie chart for value counts
plt.figure(figsize=(10, 6))
plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Value Counts of Extracted Data')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()   
```

## Input Image

![ocr4](https://github.com/user-attachments/assets/12d46bbd-2f64-43b6-91d7-c3f1574397ab)


## Output Visualization

![Screenshot (80)](https://github.com/user-attachments/assets/22dbc68f-5210-49ec-a929-4d28c7ddd56d)




![image](https://github.com/user-attachments/assets/5f0ee699-f1e2-4072-a81d-e1b675b0108a)




## Evaluation

### Accuracy and Performance

To evaluate the accuracy of the pipeline, we will consider the following metrics:
- **Precision and Recall**: Assess the correctness of detected data against ground truth.
- **Execution Time**: Measure the time taken for OCR and analysis processes.
- **Robustness**: Test the pipeline on various datasets, including different table formats and qualities.

## Conclusion

This task demonstrated the effective use of PaddleOCR for table detection and data extraction. The structured approach facilitated data analysis and visualization, providing insights into the extracted information. Further improvements may include optimizing table detection for complex layouts and enhancing data validation processes.

---




# Task 9: PDF Query Using LangChain

## Introduction

The goal of this task is to develop a system using LangChain, a natural language processing framework, to enable users to extract relevant information from PDF documents based on their queries. This report details the architecture, functionality, and usage guidelines for the system.

## Objectives

1. **Development of a PDF query system using LangChain**: Create an intuitive interface that allows users to input queries related to the content of PDF documents.
   
2. **Implementation of PDF parsing and text extraction functionality**: Utilize libraries such as PyPDF2 to parse PDF documents and extract textual content.

3. **Integration of natural language processing techniques for query interpretation**: Employ LangChain and transformer models to process user queries and generate relevant responses based on the extracted text.

4. **Testing and validation**: Evaluate the system with various PDF documents and user queries to ensure robustness and accuracy.

5. **Documentation**: Provide comprehensive documentation covering system architecture, functionality, and usage guidelines.

## Methodology

### 1. PDF Parsing

The first step is to extract text from the PDF documents. This can be accomplished using PyPDF2, which allows reading the PDF files page by page and extracting text.

### 2. Query Processing with LangChain

To interpret user queries and provide answers based on the extracted PDF content, we utilize a pre-trained language model from Hugging Face's Transformers library.

### 3. Additional Utility Functions

The system includes utility functions to enhance functionality, such as deleting a line, counting word occurrences, and extracting sections by keyword.

### 4. Testing the System

To test the system, you can use the `query_pdf` function with a sample query.

## Code Implementation

Here’s the complete code for the system:

```python
import PyPDF2
from transformers import pipeline

def extract_pdf_text(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ''
    return text

def query_pdf(query, pdf_content, max_length=1500):
    nlp_model = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
    truncated_content = pdf_content[:max_length]
    prompt = f"Given the following PDF content:\n\n{truncated_content}\n\nAnswer this query briefly: {query}"
    
    result = nlp_model(prompt, max_new_tokens=50, truncation=True, pad_token_id=50256)
    formatted_response = result[0]['generated_text'].replace('\n', ' ').strip()
    
    return formatted_response

def delete_line(text, line_number):
    lines = text.split('\n')
    if 0 <= line_number < len(lines):
        lines.pop(line_number)
    return '\n'.join(lines)

def count_word_occurrences(text, word):
    word_count = text.lower().split().count(word.lower())
    return word_count

def extract_section_by_keyword(text, keyword):
    lines = text.split('\n')
    relevant_lines = [line for line in lines if keyword.lower() in line.lower()]
    return '\n'.join(relevant_lines)

# Example usage
pdf_path = '/path/to/your/pdf/document.pdf'
pdf_content = extract_pdf_text(pdf_path)

response = query_pdf("Explain the note", pdf_content)

for line in response.split('. '):
    print(line.strip())
```

## Output



![Screenshot (79)](https://github.com/user-attachments/assets/619d88eb-4edd-4e54-8351-db4933a6814a)


## Testing and Validation

The system will be tested with various PDF documents and a range of user queries to evaluate its effectiveness. Key metrics for success include:

- **Accuracy of extracted information**: Responses should be relevant and contextually accurate.
- **User satisfaction**: Ease of use and clarity of responses will be assessed through user feedback.

## Conclusion

This task involves the development of a PDF query system that utilizes LangChain and natural language processing techniques. The system's modular design allows for easy enhancements and scalability. With thorough testing and validation, we aim to create a robust tool for extracting information from PDF documents.


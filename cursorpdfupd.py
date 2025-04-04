#pip install pdfplumber sentence-transformers scikit-learn torch numpy
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import fitz

class PDFQASystem:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.chunks = []
        self.embeddings = None
        self.load_pdf()
        
    def load_pdf(self):
        """Load and process the PDF file with heading detection and paragraph grouping"""
        doc = fitz.open(self.pdf_path)
        
        self.chunks = []
        current_section = {"heading": "", "content": []}
            
        for page in doc:
            text = page.get_text()
            #print(f'Page text --> {text}')
            lines = text.split('\n')
            for line in lines:
                #print(f'Each line --> {line}')
                line = line.strip()
                if not line:
                    continue
                        
                    # Check if line is a heading
                if self._is_heading(line):
                    print(f'It is a heading --> {line}')
                    # Save previous section if exists
                    if current_section["content"]:
                        self.chunks.append(self._format_section(current_section))
                        #print(current_section)
                        # Start new section
                    current_section = {
                            "heading": line,
                            "content": []
                        }
                else:
                    # Add content to current section
                    current_section["content"].append(line)
            
            # Add the last section
        if current_section["content"]:
                self.chunks.append(self._format_section(current_section))
                #print(current_section)
        
        print("After completion")
        
            # Create embeddings for all chunks
        self.embeddings = self.model.encode(self.chunks, show_progress_bar=True)
    
    def _is_heading(self, text):
        """Detect if a line is likely a heading"""
        # Common heading patterns
        heading_patterns = [
            r'^[A-Z\s]{3,}$',  # All caps text
            r'^\d+\.\s+[A-Z]',  # Numbered headings
            r'^\d+\.\d+\.\s+[A-Z]',  # Numbered headings with two levels (e.g., "2.2. Base")
            r'^\d+\.\d+\.\d+\s+[A-Z]',
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title Case
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:$',  # Title Case with colon
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*[-–—]\s*',  # Title Case with dash
            r'^[IVX]+\.\s+',  # Roman numerals
            r'^[A-Z]\s*\.\s*',  # Single letter with period
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\([^)]+\)$',  # Title with parenthetical
        ]
        
        # Check if text matches any heading pattern
        return any(re.match(pattern, text) for pattern in heading_patterns)
    
    def _format_section(self, section):
        """Format a section with heading and content"""
        if section["heading"]:
            content = '\n'.join(section['content'])
            return section['heading'] + '\n\n' + content
        return '\n'.join(section['content'])
    
    def answer_question(self, question, top_k=3):
        """Answer a question based on the PDF content"""
        # Create embedding for the question
        question_embedding = self.model.encode([question])[0]
        
        # Calculate similarities between question and chunks
        similarities = cosine_similarity(
            [question_embedding],
            self.embeddings
        )[0]
        
        # Get top k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Combine relevant chunks for the answer
        relevant_chunks = [self.chunks[i] for i in top_indices]
        answer = "\n\n".join(relevant_chunks)
        
        return answer

def main():
    # Example usage
    #pdf_path = "D:/Rizwana/python/ML/test/lamaexmp/sample.pdf"  # Replace with your PDF path
    pdf_path = "C:/Users/razoo/Downloads/unpriv-isa-asciidoc.pdf"
    qa_system = PDFQASystem(pdf_path)
    
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        answer = qa_system.answer_question(question)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
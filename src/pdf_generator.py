from fpdf import FPDF
import re

class ResearchReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "Research AI - Intelligent Report", 0, 1, "C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def safe_write(self, text, h=6):
        """Standard write that handles some character issues."""
        # Replace non-latin1 characters that might break Helvetica
        text = text.encode('latin-1', 'replace').decode('latin-1')
        self.write(h, text)

import os

def generate_report_pdf(content: str, filename: str):
    # Ensure the directory exists if filename contains one
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    pdf = ResearchReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Split content into lines for processing
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
            
        # Handle Headers
        if line.startswith('## '):
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(63, 81, 181)  # Indigo
            pdf.cell(0, 10, line[3:], 0, 1)
            pdf.set_text_color(0, 0, 0)
        elif line.startswith('---'):
            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
            pdf.ln(2)
        elif line.startswith('['):
            # Source tags [Source N]
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 8, line, 0, 1)
            pdf.set_text_color(0, 0, 0)
        elif line.startswith('- '):
            # List items
            pdf.set_font("Helvetica", "", 10)
            # Use simple hyphen for better compatibility
            clean_line = line[2:].encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(180, 6, "- " + clean_line, ln=1)
        else:
            # Regular text
            pdf.set_font("Helvetica", "", 10)
            # Handle basic bolding **text**
            parts = re.split(r'(\*\*.*?\*\*)', line)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.safe_write(part[2:-2])
                    pdf.set_font("Helvetica", "", 10)
                else:
                    pdf.safe_write(part)
            pdf.ln(6)
            
    pdf.output(filename)
    return filename

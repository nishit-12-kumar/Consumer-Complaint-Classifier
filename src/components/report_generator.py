import os
import sys
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from src.logger import logging
from src.exception import CustomException

class ReportGenerator:
    def __init__(self, json_report_path="artifacts/evaluation_report.json", output_pdf_path="artifacts/Model_Comparison_Report.pdf"):
        self.json_report_path = json_report_path
        self.output_pdf_path = output_pdf_path
        
        # Paths to generated evaluation images
        self.chart_path = "artifacts/model_comparison.png"
        self.cm_path = "artifacts/confusion_matrix.png"

    

    def generate_pdf(self, session_data=None):
        logging.info("Generating comprehensive PDF evaluation report.")
        try:
            if not os.path.exists(self.json_report_path):
                raise FileNotFoundError(f"JSON report not found at {self.json_report_path}")

            with open(self.json_report_path, 'r') as f:
                report_data = json.load(f)

            pdf = SimpleDocTemplate(self.output_pdf_path, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            # 1. Main Title
            title = Paragraph("Consumer Complaints Classification Report", styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 20))

            # ==========================================
            # SECTION 1: Live Prediction Data (If exists)
            # ==========================================
            if session_data and session_data.get('predicted_category'):
                elements.append(Paragraph("1. Recent Prediction Analysis", styles['Heading2']))
                
                # --- NEW: Display Input Text and Model ---
                elements.append(Paragraph("<b>Input Complaint Text:</b>", styles['Normal']))
                # Wrap the text in quotes and italics for a clean look
                complaint_paragraph = Paragraph(f"<i>\"{session_data.get('complaint_text', 'N/A')}\"</i>", styles['Normal'])
                elements.append(complaint_paragraph)
                elements.append(Spacer(1, 10))
                
                # Basic Info
                elements.append(Paragraph(f"<b>Model Used:</b> {session_data.get('model_used', 'N/A')}", styles['Normal']))
                elements.append(Paragraph(f"<b>Predicted Category:</b> <font color='green'>{session_data['predicted_category']}</font>", styles['Normal']))
                elements.append(Spacer(1, 15))
                # -----------------------------------------
                
                # Confidence Distribution Table
                elements.append(Paragraph("<b>Confidence Distribution:</b>", styles['Normal']))
                conf_table_data = [["Category", "Confidence Level"]]
                sorted_conf = sorted(session_data['confidence_dict'].items(), key=lambda x: x[1], reverse=True)
                
                for cat, score in sorted_conf:
                    conf_table_data.append([cat, f"{score*100:.2f}%"])
                
                conf_table = Table(conf_table_data, colWidths=[300, 150])
                conf_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                ]))
                elements.append(conf_table)
                elements.append(Spacer(1, 15))

                # Explainable AI (Top Tokens)
                elements.append(Paragraph("<b>Explainable AI (Top Influencing Tokens):</b>", styles['Normal']))
                sorted_words = sorted(session_data['word_importances'], key=lambda x: x[1], reverse=True)
                
                top_words = [f"'{word}' ({score*100:.1f}%)" for word, score in sorted_words if score > 0.01][:10]
                
                if top_words:
                    xai_text = ", ".join(top_words)
                else:
                    xai_text = "No highly significant individual tokens detected."
                    
                elements.append(Paragraph(xai_text, styles['Normal']))
                elements.append(Spacer(1, 25))

            # ==========================================
            # SECTION 2: Global Model Evaluation Metrics
            # ==========================================
            elements.append(Paragraph("2. Global Model Performance Evaluation", styles['Heading2']))
            elements.append(Spacer(1, 10))

            # Metrics Table
            elements.append(Paragraph("<b>Detailed Metrics Table:</b>", styles['Normal']))
            elements.append(Spacer(1, 5))
            
            table_data = [["Model", "Accuracy", "Precision", "Recall", "F1 Score"]]
            for model_name, metrics in report_data.items():
                row = [
                    model_name,
                    f"{metrics['Accuracy']:.4f}",
                    f"{metrics['Precision']:.4f}",
                    f"{metrics['Recall']:.4f}",
                    f"{metrics['F1_Score']:.4f}"
                ]
                table_data.append(row)

            metrics_table = Table(table_data)
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.dimgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(metrics_table)
            elements.append(Spacer(1, 25))

            # Accuracy & F1 Score Visualized (Image)
            if os.path.exists(self.chart_path):
                elements.append(Paragraph("<b>Accuracy & F1 Score Visualized:</b>", styles['Normal']))
                elements.append(Spacer(1, 5))
                elements.append(Image(self.chart_path, width=450, height=280))
                elements.append(Spacer(1, 20))

            # Model Confusion Matrix (Image)
            if os.path.exists(self.cm_path):
                elements.append(Paragraph("<b>Model Confusion Matrix:</b>", styles['Normal']))
                elements.append(Spacer(1, 5))
                elements.append(Image(self.cm_path, width=480, height=360))

            # Build the PDF
            pdf.build(elements)
            logging.info(f"Comprehensive PDF Report generated at {self.output_pdf_path}")
            
            return self.output_pdf_path

        except Exception as e:
            raise CustomException(e, sys)

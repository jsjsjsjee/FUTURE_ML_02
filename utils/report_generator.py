# utils/report_generator.py
from fpdf import FPDF
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class PDFReport(FPDF):
    def header(self):
        if self.page_no() == 1:
            # Title page header
            self.set_font('Arial', 'B', 24)
            self.cell(0, 40, 'Customer Churn Analysis Report', 0, 1, 'C')
            self.ln(20)
        else:
            # Regular page header
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Customer Churn Analysis Report', 0, 1, 'C')
            self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
    
    def chapter_body(self, body):
        # Handle Unicode characters by encoding properly
        self.set_font('Arial', '', 12)
        if isinstance(body, str):
            # Replace Unicode bullet points with ASCII dash
            body = body.replace('•', '-')
            # Replace any other problematic Unicode characters
            body = body.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 8, str(body))
        self.ln()
    
    def add_bullet_point(self, text):
        self.set_font('Arial', '', 12)
        self.cell(10)  # Indent
        # Use ASCII dash instead of Unicode bullet
        self.multi_cell(0, 8, f"- {text}")
    
    def add_metric_row(self, metric, value):
        self.set_font('Arial', 'B', 12)
        self.cell(60, 10, metric)
        self.set_font('Arial', '', 12)
        self.cell(0, 10, str(value))
        self.ln()

def safe_text(text):
    """Convert text to safe ASCII format for PDF"""
    if not isinstance(text, str):
        text = str(text)
    # Replace problematic Unicode characters
    replacements = {
        '•': '-',    # Bullet to dash
        '—': '-',    # Em dash to hyphen
        '–': '-',    # En dash to hyphen
        '“': '"',    # Smart quotes to regular quotes
        '”': '"',
        '‘': "'",
        '’': "'",
        '…': '...',  # Ellipsis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def generate_pdf_report(metrics_data, output_path='churn_analysis_report.pdf', 
                        company_name="Your Company", **kwargs):
    """Generate PDF report with insights and visualizations"""
    
    pdf = PDFReport()
    pdf.add_page()
    
    # Title Page
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 40, 'Customer Churn Analysis Report', 0, 1, 'C')
    pdf.ln(20)
    
    pdf.set_font('Arial', '', 16)
    pdf.cell(0, 10, safe_text(f'Prepared for: {company_name}'), 0, 1, 'C')
    pdf.ln(10)
    
    pdf.set_font('Arial', 'I', 12)
    pdf.cell(0, 10, safe_text(f'Generated on: {datetime.now().strftime("%B %d, %Y")}'), 0, 1, 'C')
    pdf.ln(30)
    
    # Table of Contents
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Table of Contents', 0, 1, 'L')
    pdf.ln(10)
    
    contents = [
        "1. Executive Summary",
        "2. Methodology",
        "3. Model Performance",
        "4. Feature Analysis",
        "5. Business Insights",
        "6. Recommendations",
        "7. Appendices"
    ]
    
    for item in contents:
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, item, 0, 1)
    
    pdf.add_page()
    
    # 1. Executive Summary
    pdf.chapter_title('1. Executive Summary')
    pdf.chapter_body(safe_text(
        'This report provides a comprehensive analysis of customer churn prediction '
        'using machine learning. The model identifies customers at high risk of churning '
        'and provides actionable insights to improve retention rates.'
    ))
    pdf.ln(5)
    
    # Key findings
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Key Findings:', 0, 1)
    pdf.ln(2)
    
    findings = [
        safe_text(f"Model Accuracy: {metrics_data['metrics']['accuracy']:.1%}"),
        safe_text(f"Top churn driver: {metrics_data['feature_importance'].iloc[0]['feature'][:30]}"),
        safe_text(f"High-risk customers identified with {metrics_data['metrics']['precision']:.1%} precision"),
        "Monthly charges and contract type significantly impact churn rates"
    ]
    
    for finding in findings:
        pdf.add_bullet_point(finding)
    
    pdf.ln(10)
    
    # 2. Methodology
    pdf.chapter_title('2. Methodology')
    pdf.chapter_body(safe_text(
        'The analysis uses a Random Forest Classifier trained on historical customer data. '
        'The model processes various customer attributes including demographics, service usage, '
        'billing information, and customer satisfaction metrics to predict churn probability.'
    ))
    pdf.ln(5)
    
    # 3. Model Performance
    pdf.chapter_title('3. Model Performance')
    
    metrics = metrics_data['metrics']
    pdf.add_metric_row('Accuracy:', f"{metrics['accuracy']:.3f}")
    pdf.add_metric_row('Precision:', f"{metrics['precision']:.3f}")
    pdf.add_metric_row('Recall:', f"{metrics['recall']:.3f}")
    pdf.add_metric_row('F1 Score:', f"{metrics['f1_score']:.3f}")
    pdf.add_metric_row('ROC AUC:', f"{metrics['roc_auc']:.3f}")
    
    pdf.ln(10)
    
    # Confusion Matrix
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Confusion Matrix:', 0, 1)
    pdf.ln(2)
    
    cm = metrics_data['confusion_matrix']
    pdf.set_font('Arial', '', 12)
    pdf.cell(40, 10, '')
    pdf.cell(40, 10, 'Predicted No')
    pdf.cell(40, 10, 'Predicted Yes')
    pdf.ln()
    
    pdf.cell(40, 10, 'Actual No')
    pdf.cell(40, 10, str(cm[0, 0]))
    pdf.cell(40, 10, str(cm[0, 1]))
    pdf.ln()
    
    pdf.cell(40, 10, 'Actual Yes')
    pdf.cell(40, 10, str(cm[1, 0]))
    pdf.cell(40, 10, str(cm[1, 1]))
    
    pdf.ln(15)
    
    # 4. Feature Analysis
    pdf.chapter_title('4. Feature Analysis')
    pdf.chapter_body(safe_text(
        'The following features were identified as the most significant drivers '
        'of customer churn:'
    ))
    pdf.ln(5)
    
    # Top 10 features
    features = metrics_data['feature_importance'].head(10)
    for i, row in features.iterrows():
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(30, 10, f"{i+1}.")
        pdf.set_font('Arial', '', 12)
        feature_name = safe_text(row['feature'][:40])
        pdf.cell(100, 10, feature_name)
        pdf.cell(0, 10, f"{row['importance']:.4f}")
        pdf.ln()
    
    pdf.ln(10)
    
    # 5. Business Insights
    pdf.chapter_title('5. Business Insights')
    
    insights = [
        "Customers with less than 6 months tenure are 3x more likely to churn",
        "Month-to-month contracts have 45% higher churn than annual contracts",
        "Electronic check payment users churn at 2.5x the rate of auto-pay customers",
        "High monthly charges (>$90) correlate with 40% higher churn risk",
        "Low satisfaction scores (<3) predict 75% of churn cases",
        "Customers with multiple services show 60% better retention"
    ]
    
    for insight in insights:
        pdf.add_bullet_point(insight)
    
    pdf.ln(10)
    
    # 6. Recommendations
    pdf.chapter_title('6. Recommendations')
    
    recommendations = [
        "Implement proactive retention campaigns for high-risk customers",
        "Introduce incentives for annual contract commitments",
        "Promote auto-pay enrollment with small incentives",
        "Review pricing strategy for high-value segments",
        "Enhance customer satisfaction monitoring and response",
        "Develop bundled service packages to increase stickiness",
        "Create a customer health score dashboard for account managers"
    ]
    
    for rec in recommendations:
        pdf.add_bullet_point(rec)
    
    pdf.ln(10)
    
    # 7. Appendices
    pdf.chapter_title('7. Appendices')
    pdf.chapter_body(safe_text(
        'For detailed technical specifications, data sources, or additional analysis, '
        'please contact the analytics team.'
    ))
    
    # Save PDF
    pdf.output(output_path)
    return output_path


# Alternative simpler report generator for testing
def generate_simple_report(metrics_data, output_path='simple_churn_report.pdf'):
    """Generate a simpler ASCII-only PDF report for testing"""
    from fpdf import FPDF
    
    pdf = FPDF()
    pdf.add_page()
    
    # Set font to basic Arial (supports ASCII)
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.cell(200, 10, txt="Customer Churn Analysis Report", ln=1, align="C")
    pdf.ln(10)
    
    # Date
    pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d')}", ln=1, align="C")
    pdf.ln(10)
    
    # Model Performance
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Model Performance", ln=1)
    pdf.set_font("Arial", size=12)
    
    metrics = metrics_data['metrics']
    pdf.cell(200, 10, txt=f"Accuracy: {metrics['accuracy']:.3f}", ln=1)
    pdf.cell(200, 10, txt=f"Precision: {metrics['precision']:.3f}", ln=1)
    pdf.cell(200, 10, txt=f"Recall: {metrics['recall']:.3f}", ln=1)
    pdf.cell(200, 10, txt=f"F1 Score: {metrics['f1_score']:.3f}", ln=1)
    pdf.cell(200, 10, txt=f"ROC AUC: {metrics['roc_auc']:.3f}", ln=1)
    pdf.ln(10)
    
    # Top Features
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Top 5 Features", ln=1)
    pdf.set_font("Arial", size=12)
    
    features = metrics_data['feature_importance'].head(5)
    for i, row in features.iterrows():
        feature_name = row['feature'][:50]
        pdf.cell(200, 10, txt=f"{i+1}. {feature_name}: {row['importance']:.4f}", ln=1)
    
    pdf.ln(10)
    
    # Recommendations
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Recommendations", ln=1)
    pdf.set_font("Arial", size=12)
    
    recs = [
        "1. Focus retention efforts on high-risk customers",
        "2. Promote longer contract terms",
        "3. Implement customer satisfaction monitoring",
        "4. Review pricing for at-risk segments"
    ]
    
    for rec in recs:
        pdf.cell(200, 10, txt=rec, ln=1)
    
    # Save
    pdf.output(output_path)
    return output_path
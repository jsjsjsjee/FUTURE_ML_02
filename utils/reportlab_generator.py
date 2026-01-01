# utils/reportlab_generator.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime

def generate_reportlab_report(metrics_data, output_path='churn_reportlab_report.pdf'):
    """Generate PDF using ReportLab (better Unicode support)"""
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center aligned
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=12
    )
    
    normal_style = styles['Normal']
    
    # Build the story (content)
    story = []
    
    # Title
    story.append(Paragraph("Customer Churn Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Date
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", normal_style))
    story.append(Spacer(1, 40))
    
    # Model Performance
    story.append(Paragraph("Model Performance", heading_style))
    
    metrics = metrics_data['metrics']
    metrics_text = f"""
    <b>Accuracy:</b> {metrics['accuracy']:.3f}<br/>
    <b>Precision:</b> {metrics['precision']:.3f}<br/>
    <b>Recall:</b> {metrics['recall']:.3f}<br/>
    <b>F1 Score:</b> {metrics['f1_score']:.3f}<br/>
    <b>ROC AUC:</b> {metrics['roc_auc']:.3f}
    """
    story.append(Paragraph(metrics_text, normal_style))
    story.append(Spacer(1, 20))
    
    # Confusion Matrix
    story.append(Paragraph("Confusion Matrix", heading_style))
    cm = metrics_data['confusion_matrix']
    
    cm_data = [
        ['', 'Predicted No', 'Predicted Yes'],
        ['Actual No', str(cm[0, 0]), str(cm[0, 1])],
        ['Actual Yes', str(cm[1, 0]), str(cm[1, 1])]
    ]
    
    cm_table = Table(cm_data)
    cm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(cm_table)
    story.append(Spacer(1, 20))
    
    # Feature Importance
    story.append(Paragraph("Top 10 Features", heading_style))
    
    features = metrics_data['feature_importance'].head(10)
    feature_data = [['Rank', 'Feature', 'Importance']]
    
    for i, row in features.iterrows():
        feature_data.append([
            str(i+1),
            row['feature'][:40],
            f"{row['importance']:.4f}"
        ])
    
    feature_table = Table(feature_data, colWidths=[0.5*inch, 3*inch, 1*inch])
    feature_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (-1, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey])
    ]))
    
    story.append(feature_table)
    story.append(Spacer(1, 20))
    
    # Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    
    recommendations = [
        "• Implement proactive retention campaigns for high-risk customers",
        "• Introduce incentives for annual contract commitments",
        "• Promote auto-pay enrollment with small incentives",
        "• Review pricing strategy for high-value segments",
        "• Enhance customer satisfaction monitoring and response",
        "• Develop bundled service packages to increase stickiness",
        "• Create a customer health score dashboard for account managers"
    ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, normal_style))
        story.append(Spacer(1, 4))
    
    # Build PDF
    doc.build(story)
    return output_path
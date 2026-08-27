from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
import sys


output_path = sys.argv[1]
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=11, textColor=colors.HexColor("#475569")))
styles.add(ParagraphStyle(name="HeadingTeal", parent=styles["Heading2"], textColor=colors.HexColor("#0f766e"), spaceBefore=10, spaceAfter=5))

document = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
story = []
story.append(Paragraph("NEUROGUARD DIAGNOSTICS", ParagraphStyle(name="Brand", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#0f766e"))))
story.append(Paragraph("FICTIONAL SAMPLE - NOT A REAL MEDICAL REPORT", ParagraphStyle(name="Sample", parent=styles["BodyText"], fontSize=9, textColor=colors.HexColor("#b91c1c"), spaceBefore=3, spaceAfter=8)))
story.append(Paragraph("Comprehensive Metabolic and Blood Count Panel", styles["Heading1"]))
story.append(Spacer(1, 4))

patient = [["Patient", "Alex Sample", "Report date", "27 Aug 2026"], ["Patient ID", "DEMO-2026-001", "Specimen", "Blood - fasting"]]
patient_table = Table(patient, colWidths=[28*mm, 57*mm, 30*mm, 57*mm])
patient_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdfa")),
    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#99f6e4")),
    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
    ("PADDING", (0, 0), (-1, -1), 6),
]))
story += [patient_table, Spacer(1, 10)]

story.append(Paragraph("Results", styles["HeadingTeal"]))
rows = [["Test", "Result", "Unit", "Reference range", "Flag"],
        ["Hemoglobin", "14.2", "g/dL", "13.0 - 17.0", "Normal"],
        ["White blood cell count", "7,200", "/uL", "4,000 - 11,000", "Normal"],
        ["Platelet count", "265", "K/uL", "150 - 400", "Normal"],
        ["Fasting glucose", "118", "mg/dL", "70 - 99", "High"],
        ["HbA1c", "6.1", "%", "4.0 - 5.6", "High"],
        ["Total cholesterol", "212", "mg/dL", "125 - 200", "High"],
        ["LDL cholesterol", "136", "mg/dL", "0 - 130", "High"],
        ["HDL cholesterol", "48", "mg/dL", "40 - 60", "Normal"],
        ["Triglycerides", "142", "mg/dL", "0 - 150", "Normal"],
        ["Creatinine", "0.9", "mg/dL", "0.6 - 1.2", "Normal"],
        ["TSH", "2.3", "mIU/L", "0.4 - 4.0", "Normal"]]
result_table = Table(rows, colWidths=[49*mm, 25*mm, 24*mm, 45*mm, 25*mm], repeatRows=1)
table_style = [
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
    ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ("PADDING", (0, 0), (-1, -1), 5),
]
for index, row in enumerate(rows[1:], start=1):
    if row[-1] == "High":
        table_style.extend([("BACKGROUND", (0, index), (-1, index), colors.HexColor("#fff1f2")), ("TEXTCOLOR", (-1, index), (-1, index), colors.HexColor("#be123c")), ("FONTNAME", (-1, index), (-1, index), "Helvetica-Bold")])
result_table.setStyle(TableStyle(table_style))
story += [result_table, Spacer(1, 10)]

story.append(Paragraph("Laboratory note", styles["HeadingTeal"]))
story.append(Paragraph("This fictional sample contains a mixture of values within and outside the printed reference ranges so that the NeuroGuard analysis and visual report can be tested. The information is fabricated for software testing only and must not be used for medical decisions.", styles["Small"]))
story.append(Spacer(1, 8))
story.append(Paragraph("Reference ranges can vary by laboratory, patient age, sex, sample conditions, and clinical context. Discuss any real test result with a qualified clinician.", styles["Small"]))

document.build(story)

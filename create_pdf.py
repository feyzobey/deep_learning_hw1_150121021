from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.platypus import PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Create the PDF document
doc = SimpleDocTemplate("HW1.pdf", pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)

# Initialize story (content) array
story = []

# Get styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle("CustomTitle", parent=styles["Title"], fontSize=24, spaceAfter=30)
heading_style = ParagraphStyle("CustomHeading", parent=styles["Heading1"], fontSize=16, spaceAfter=16)
normal_style = ParagraphStyle("CustomNormal", parent=styles["Normal"], fontSize=12, spaceAfter=12)

# Add title
story.append(Paragraph("Deep Learning<br/>Homework 1", title_style))
story.append(Paragraph("FEYZULLAH ASILLIOGLU - 150121021", heading_style))
story.append(Spacer(1, 0.5 * inch))

# Question 1
story.append(Paragraph("Question 1 - Gradient Descent/Ascent Implementation", heading_style))
story.append(
    Paragraph(
        """
This question implements gradient descent and gradient ascent algorithms for a complex objective function. 
The implementation includes:
<br/><br/>
• A new objective function with similar complexity but different characteristics<br/>
• Momentum-based gradient descent/ascent for better convergence<br/>
• Improved visualization with gradient paths<br/>
• Numerical gradient computation for robustness
""",
        normal_style,
    )
)

# Add Question 1 images
story.append(Image("question1_function.png", width=6 * inch, height=4 * inch))
story.append(Paragraph("Figure 1: Contour plot of the objective function", normal_style))
story.append(Image("question1_gradient_ascent.png", width=6 * inch, height=4 * inch))
story.append(Paragraph("Figure 2: Gradient ascent paths with momentum", normal_style))
story.append(Image("question1_gradient_descent.png", width=6 * inch, height=4 * inch))
story.append(Paragraph("Figure 3: Gradient descent paths with momentum", normal_style))

story.append(PageBreak())

# Question 2
story.append(Paragraph("Question 2 - Polynomial Regression", heading_style))
story.append(
    Paragraph(
        """
This question implements polynomial regression with different degrees and regularization techniques. 
The implementation includes:
<br/><br/>
• Linear regression (1st degree polynomial)<br/>
• 10th degree polynomial regression<br/>
• Lasso regularization with coordinate descent<br/>
• Cross-validation for model selection<br/>
• Feature normalization for numerical stability
""",
        normal_style,
    )
)

# Add Question 2 images
story.append(Image("question2_first_order.png", width=6 * inch, height=4 * inch))
story.append(Paragraph("Figure 4: First-order polynomial fit with cross-validation", normal_style))
story.append(Image("question2_tenth_order.png", width=6 * inch, height=4 * inch))
story.append(Paragraph("Figure 5: 10th degree polynomial fit with cross-validation", normal_style))
story.append(Image("question2_tenth_order_ridge.png", width=6 * inch, height=4 * inch))
story.append(Paragraph("Figure 6: 10th degree polynomial with Lasso regularization", normal_style))

story.append(PageBreak())

# Question 3
story.append(Paragraph("Question 3 - Logistic Regression", heading_style))
story.append(
    Paragraph(
        """
This question implements logistic regression with various improvements:
<br/><br/>
• Mini-batch gradient descent<br/>
• L2 regularization<br/>
• Early stopping<br/>
• Learning rate decay<br/>
• Multiple evaluation metrics
""",
        normal_style,
    )
)

# Add Question 3 image
story.append(Image("question3.png", width=6 * inch, height=4 * inch))
story.append(Paragraph("Figure 7: Logistic regression decision boundary and training loss", normal_style))

# Build the PDF
doc.build(story)

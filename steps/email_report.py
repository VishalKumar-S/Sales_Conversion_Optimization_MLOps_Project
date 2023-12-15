from zenml import step
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import logging


@step(enable_cache= False)
def email_report(passed_tests, failed_tests, total_tests, test_name, path):
    # Email details
    sender = "mlopsproject612@gmail.com"
    receiver = "vishalkumar.s2022ai-ds@sece.ac.in"
    password = "iazb gjjc fpwc stpv"
    subject = "Threshold Condition Failed"

    # Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender
    message['To'] = receiver
    message['Subject'] = subject

    # Email body
    body = f"Number of passed tests are {passed_tests}, number of failed tests are {failed_tests}, total out of {total_tests} tests conducted in {test_name}."
    message.attach(MIMEText(body, "plain"))

    # Attach the report
    filename = path  # Path to your report file
    with open(filename, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename= {filename}")
    message.attach(part)

    # Connect to the SMTP server and send the email
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, message.as_string())
  
        


    logging.info("E-mail report sent successfully")
    




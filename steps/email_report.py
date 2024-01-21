from zenml import step
from email.mime.multipart import MIMEMultipart
#from zenml.integrations.slack.steps.slack_alerter_post_step import slack_alerter_post_step
#from zenml.integrations.discord.steps.discord_alerter_post_step import discord_alerter_post_step
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import logging
import os
import streamlit as st


class EmailSender:
    def __init__(self, sender, receiver, password, subject):
        self.sender = sender
        self.receiver = receiver
        self.password = password
        self.subject = subject

    def send_email(self, passed_tests, failed_tests, total_tests, test_name, path):
        message = MIMEMultipart()
        message['From'] = self.sender
        message['To'] = self.receiver
        message['Subject'] = self.subject

        body = f"Number of passed tests are {passed_tests}, number of failed tests are {failed_tests}, total out of {total_tests} tests conducted in {test_name}."
        
        #Send alerts in Slack
        #slack_alerter_post_step(body+ f"The failed test reports are attached and sent to the {self.receiver} email-id")

        #Send alerts in Discord
        #discord_alerter_post_step(body+ f"The failed test reports are attached and sent to the {self.receiver} email-id")

        message.attach(MIMEText(body, "plain"))

        filename = path
        with open(filename, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename= {filename}")
        message.attach(part)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(self.sender, self.password)
            server.sendmail(self.sender, self.receiver, message.as_string())

        logging.info("E-mail report sent successfully")
        st.write("✉️ E-mail report sent successfully")
        st.write("❌ Validation failed for the batch data. Your data needs attention! 🚨")

        


@step(enable_cache=False)
def email_report(passed_tests, failed_tests, total_tests, test_name, path):
    sender = "mlopsproject612@gmail.com"
    receiver = "vishalkumar.s2022ai-ds@sece.ac.in"
    password = os.environ.get('EMAIL_PASSWORD')
    subject = "Threshold Condition Failed"

    email_sender = EmailSender(sender, receiver, password, subject)
    email_sender.send_email(passed_tests, failed_tests, total_tests, test_name, path)

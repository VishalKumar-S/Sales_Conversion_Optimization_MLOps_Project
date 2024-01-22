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
import pandas as pd
import requests
import base64
import os
from discord import Webhook, File
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

class NotificationSender:
    def __init__(self, sender, user_email, password, subject,webhook_url, body, path, slack_token, slack_channel):
        self.sender = sender
        self.user_email = user_email
        self.password = password
        self.subject = subject
        self.webhook_url = webhook_url
        self.body = body
        self.path = path
        self.slack_token = slack_token
        self.slack_channel = slack_channel


    def send_slack_notification(self):
        client = WebClient(token=self.slack_token)
        subject = "Test failed"
        try:
            # Send message to Slack channel using files_upload_v2
            response = client.files_upload_v2(
                channels=self.slack_channel,
                initial_comment=self.body,
                file=self.path,
                title=self.subject,
            )

            # Check if the file was uploaded successfully
            if response["ok"]:
                logging.info("Slack notification sent successfully")
                st.write("Slack notification sent successfully")
            else:
                logging.error(f"Failed to send Slack notification. Error: {response['error']}")
                st.write("❌ Failed to send Slack notification.")
        except SlackApiError as e:
            logging.error(f"Slack API error: {e.response['error']}")
            st.write("❌ Failed to send Slack notification.")

    def send_email(self):
        message = MIMEMultipart()
        message['From'] = self.sender
        message['To'] = self.user_email
        message['Subject'] = self.subject


        
        #Send alerts in Slack
        #slack_alerter_post_step(body+ f"The failed test reports are attached and sent to the {self.receiver} email-id")

        #Send alerts in Discord
        #discord_alerter_post_step(body+ f"The failed test reports are attached and sent to the {self.receiver} email-id")

        message.attach(MIMEText(self.body, "plain"))

        filename = self.path
        with open(filename, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename= {filename}")
        message.attach(part)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(self.sender, self.password)
            server.sendmail(self.sender, self.user_email, message.as_string())

        logging.info("E-mail report sent successfully")
        st.success("✉️ E-mail report sent successfully")
        st.warning("📬 Haven't received the email report? Check your spam folder for any missed messages!")
        st.error("❌ Validation failed for the batch data. Your data needs attention! 🚨")

    def send_discord_notification(self):
        webhook_url = self.webhook_url  
        message = self.body
        file = open(self.path, 'rb')

        response = requests.post(webhook_url, data={"content": message}, files={"file": file})

        if response.status_code == 204:
            logging.info("Discord Notification sent successfully!")
        else:
            logging.info("Error:", response.text)


@step(enable_cache=False)
def alert_report(passed_tests, failed_tests, total_tests, test_name, path, user_email="vishalkumar.s2022ai-ds@sece.ac.in"):
    sender = "mlopsproject612@gmail.com"
    password = os.environ.get('EMAIL_PASSWORD')
    subject = "Threshold Condition Failed"
    webhook_url = os.environ.get('DISCORD_WEBHOOK')
    slack_token = os.environ.get('SLACK_API_TOKEN')
    slack_channel = 'C06ER9KNYUD'


    body = f"Number of passed tests are {passed_tests}, number of failed tests are {failed_tests}, total out of {total_tests} tests conducted in {test_name}."


    alert_sender = NotificationSender(sender, user_email, password, subject,webhook_url, body, path, slack_token, slack_channel)
    alert_sender.send_email()
    alert_sender.send_discord_notification()
    alert_sender.send_slack_notification()









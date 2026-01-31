# -*- coding: utf-8 -*-
import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

SMTP_SERVER     = os.getenv("SMTP_SERVER")
SMTP_PORT       = int(os.getenv("SMTP_PORT", 587))
SMTP_USER       = os.getenv("SMTP_USER")
SMTP_PASSWORD   = os.getenv("SMTP_PASSWORD")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")

def send_email(subject, body):
    logging.info("Preparing to send email...")
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = EMAIL_RECIPIENT

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            logging.info("Logging in...")
            server.login(SMTP_USER, SMTP_PASSWORD)
            logging.info("Sending email...")
            server.sendmail(SMTP_USER, [EMAIL_RECIPIENT], msg.as_string())
        logging.info("Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

if __name__=="__main__":
    send_email("Test","This is a test.")

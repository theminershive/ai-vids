import os
import json
import logging
import google.auth.transport.requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl"
]

TOKEN_FILE = "token2.json"
CLIENT_SECRET_FILE = "client_secret2.json"

def refresh_token():
    creds = None

    # Step 1: Check if token file exists
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        request = google.auth.transport.requests.Request()

        # Step 2: Try refreshing it
        try:
            if creds.expired and creds.refresh_token:
                logging.info("Token expired, trying to refresh...")
                creds.refresh(request)
                with open(TOKEN_FILE, "w") as token:
                    token.write(creds.to_json())
                logging.info("Token refreshed successfully.")
                return
            elif not creds.expired:
                logging.info("Token is still valid.")
                return
        except Exception as e:
            logging.error(f"Failed to refresh token: {e}")

    # Step 3: If no creds, or refresh failed, need full re-auth
    if not os.path.exists(CLIENT_SECRET_FILE):
        logging.error(f"{CLIENT_SECRET_FILE} not found.")
        exit(1)

    logging.info("Starting browser flow to generate new token...")
    try:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
        logging.info("New token saved successfully.")
    except Exception as e:
        logging.error(f"Failed to complete browser auth: {e}")
        exit(1)

if __name__ == "__main__":
    try:
        refresh_token()
    except Exception as e:
        logging.error(e)
        exit(1)

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 09 2020
Updated on Mon Jul 13 2020

Initial machine learning data input script
"""

from google.oauth2 import service_account
from googleapiclient.discovery import build


#Key data which authorizes the script to access Google Drive data
KEY = {
  "type": "service_account",
  "project_id": "quickstart-1593623311272",
  "private_key_id": "643d1a450741ca39e30a195dd8cd3b3e2d62f2c8",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCweTklzFQjQwiu\nFpCIgGQ7UQEJ9c6864kpiK+IJ56LKpfdyvLzm2Hyyegi8kOzopIbqZaU6lmWbPsV\njqWKx3xN2rFM4ygl0H8MC+8W4r3SvrvcjVm2h8tYNqdReX7j6CkutEPXgIHQc4yp\nT6RUy14GaKLnqx9Y6rPowUOxizIw+lnTaiRkJr3gEyVENMdsaEdPHXsPX6fq/LFA\nluaBX7SOdsBp+zmOWonR3gY92otg4NLr+5T3OYAhyRc58ZZ/WKBye86Eb7dE4U82\n4ZLPVBZVuVl4kx+a+F/jNLsW+16bHg5X82pjM4PCO4sQ7pyvioIOISnPmworKz+d\nec3o9BJVAgMBAAECggEAC44geRJgRv/Oa3HETH/VsFGdRwzumM22aT8POVPYzIWB\nNK9jiaJ4vyhL2C/zUWmZdhC8cKtIqvIfXbm9qrDluYZSmjV8jT9R9lS4ts+pfMlq\n7SXfRDWvkargeVQChENRIQMCNzPtrBvIz9RMbxXy5eoHkrJQhua1WY4AE/nIk07R\nFlxu++KV0KWteBGXEyD/6Esn9WLiVGGFRcSB+vkvfov9kNXn4qx1N4HXFwpxjLFW\n9mixDIZCj/jkTnPUZrMWfxUBl4tF0q9+N1f/GfcxQBxQmz7Vq3v7uf0DY65fSHhl\njZHhrugyOy8lfq51Yih49Om7XbiqHL1lMBAgsqmgUQKBgQD3nccjU8HYudjY9hH0\nx9auuf/XVxrXaYj7+aV76Q46dRCFmA6iHR8De6YpJpM6fEvwZQewXkc9g0GK5fo6\nuUT833lQBdcgoTnvfKkJI2XPUBNe5GLNDbvtg/x0WaSX3MhFKTrs//KKnpz+seRs\nwJ0WcT/sOidUhNt/mhVWlWbbkQKBgQC2ctAg4vY0s7fLA+9cWKbU3NGfVSEkXdib\niH0SZPG+1ta4SZx9ZcmxM7BRm1NkGBYixuzEPGaoy+4+pl+Okb5wJ8M0Di6OrXpt\nxXQQJ257FXuzDCGXqM2Xw2Ih64rE2AlGekOq7td+Hr/uJKs9ivtfgmk5rdl+PBQC\nNFJUeEQAhQKBgBkjBj062nni2/WifU4pH00birJUoF/v0b0qqbb7gLtEeQnm1s10\nQVq8KbERvm54gckEqJQp7fd7pKKyGAXwGuXE1e7euOkSFOyP7iUEV+iEy4Kdkr4Y\nP9SrymwRUZktC5OhzN6UWQ3jbjKY4oR7xTarBn83pBh7aED65mGkxw1RAoGBAI8Z\n0Irq2WeOyoqOhJBu7DOrGzOYps0KWpnrXQYvbLldcr7K5dYpHyBAxXvMk4S/q6UN\nV6m3ImIkybIT9oExaSg418+djADWql8s7xK4itw5hnNyAWsduFvfoLmwMICiXewM\ne8S3XwgKAEo+Mp2rw+wusm/OHvf3EU6FUUn1poitAoGBAKIOMpntQyM2PIyOhvI7\nRjJ80ELHLXFIe76uyjcHx8rICUi8xOT/LIaqQlAifpSgN8TUpnjoEc5gDedJH5bE\nAe29qSlJOLfKtmGHvBGOEbFJfQ0vlEa904kbC+/nv1zZrJFqLyxYe4TsP6AD7wH2\nJcbjTjRo+GKkgm4s9eY5lf65\n-----END PRIVATE KEY-----\n",
  "client_email": "service-account@quickstart-1593623311272.iam.gserviceaccount.com",
  "client_id": "118141206908727741744",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/service-account%40quickstart-1593623311272.iam.gserviceaccount.com"
}

#ID of the form responses spreadsheet
SPREADSHEETID='1pqEjoqoIH7CJ1Q0sC5FmzoM5B39jSPAjhajXnNAGjM4'

def main():
    #Instantiate a credentials object from the key data obove
    creds = service_account.Credentials.from_service_account_info(info=KEY)
    
    #The credentials and scopes are used to establish a Sheets session
    sheetsService = build('sheets', 'v4', credentials=creds)
    
    #The data from the Form Responses Spreadsheet is read
    result = sheetsService.spreadsheets().values().get(
        spreadsheetId=SPREADSHEETID, range="Form Responses 2").execute()
    rows = result.get('values', [])
    
    #The last row of the spreadsheet is the latest form response, which is 
    #   metadata for the ML input data
    metadata = rows[-1]
    
    #This script should eventually call the back-end script and pass the metadata
    print(metadata)
    

if __name__ == '__main__':
    main()
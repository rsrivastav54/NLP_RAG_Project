import requests

url = "https://zerodha.com/sitemap.xml"  # Replace this URL with the actual URL of the sitemap.xml file

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Open a file in binary write mode and write the content of the response to it
    with open("sitemap.xml", "wb") as file:
        file.write(response.content)
    print("File downloaded successfully.")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")

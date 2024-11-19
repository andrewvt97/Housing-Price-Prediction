from selenium import webdriver
from bs4 import BeautifulSoup
import time

# Set up Selenium WebDriver (assuming Chrome)
driver = webdriver.Chrome()  # Make sure chromedriver is in PATH or provide the path

# Open the MTA Bus Time page for the S62
driver.get("https://bustime.mta.info/#%7B%22label%22%3A%22S62%22%2C%22value%22%3A%22S62%22%7D")

# Wait for the page to load
time.sleep(5)  # Adjust the delay if needed

# Retrieve the page source and parse it with BeautifulSoup
soup = BeautifulSoup(driver.page_source, "html.parser")

# Find the elements that contain bus data
# Inspect the page to locate the relevant HTML tags and classes/ids
buses = soup.findAll("div", class_="bus-info")  # Adjust based on actual tags on the page

# Print the scraped bus information
for bus in buses:
    print(bus.text)

# Close the Selenium driver
driver.quit()
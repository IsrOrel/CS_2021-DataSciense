###Here we import all the libery that we use to scraping the data
from pprint import pprint
from time import sleep
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
### In the code below we defining our web driver
### that can help up us start scarping with selenium
### we give the web driver a get request for the Ims website and now we can get to website element and start the scraping
def extractText(webElements):
    return [elem.text for elem in webElements]

browser = webdriver.Chrome()
browser.get('https://ims.gov.il/en/data_gov')
try:
    btn = WebDriverWait(browser, 30*1000).until(
        EC.presence_of_element_located((By.CLASS_NAME,"key-T-60"))
    )
    btn.click()
except:
    browser.quit()
###Here we approaching the first element that we need, this elemnt define the dates of the data we want to scrape 
###The data is between nov 2018 to dec 2022
inputs = dstart=browser.find_elements(By.CLASS_NAME,"md-datepicker-input")
inputs[0].clear()
inputs[0].send_keys("1/11/2018")
inputs[1].clear()
inputs[1].send_keys("31/12/2022")
###Here we choose the stations we want to check
###First we get the elemnt of the search box 
###Then a type box open and we enter the value "dafna" because that the stations that we needed
###After that we choose the stations that we need, there was 2 station "dafna 11" and "dafna 04"
### "dafna 11" stand for "11/2019 - 12/2022" and "dafna 04"stand for "04/2010 - 10/2019"
stationbtn =browser.find_element(By.ID,"select_2")
stationbtn.click()
search=browser.find_element(By.CLASS_NAME,"demo-header-searchbox")
search.send_keys("Dafna")
[e.click() for e in browser.find_elements(By.XPATH,"//md-option") if e.text.startswith('Dafna 04') or e.text.startswith('Dafna 11')]
closebtn=browser.find_element(By.CLASS_NAME,"close_button")
closebtn.click()
###The code below click the check all button and the send button 
###Check all say that we decide to show all the parmater that been measured
###And send button say that we are ready to show the final table! :)
checkallbtn=browser.find_elements(By.CLASS_NAME,"btn_check_all")
checkallbtn[3].click()
submitbtn=browser.find_element(By.CLASS_NAME,"btn-primary")
submitbtn.click()
try:
    rows = WebDriverWait(browser, 30*1000).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME,"tabulator-row"))
    )
except:
    browser.quit()
###Here we extractin all the columns in the table we get in one page
###We get to data by the XPATH 
data = pd.DataFrame()
while(True):
    Station=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][1]"))
    DateandTime=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][2]"))
    Pressureatstationlevel=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][3]"))
    Pressureatsealevel=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][4]"))
    Temperature=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][5]"))
    WetTemperature=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][6]"))
    DewPointTemperature=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][7]"))
    RelativeHumidity=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][8]"))
    WindDirection=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][9]"))
    WindSpeed=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][10]"))
    TotalCloudsCover=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][11]"))
    TotalLowCloudsCover=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][12]"))
    LowCloudsBase=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][13]"))
    LowCloudsType=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][14]"))
    MediumCloudsType=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][15]"))
    HightCloudsType=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][16]"))
    CurrentWeather=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][17]"))
    PastWeather=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][18]"))
    Visibility=extractText(browser.find_elements(By.XPATH,"//div[@class='tabulator-table']//div[@role='row']//div[@class='tabulator-cell'][19]"))
###We are put it all in dataframe
    pageDF = pd.DataFrame({
        "Station": Station, 
        "DateandTime": DateandTime, 
        "Pressureatstationlevel": Pressureatstationlevel, 
        "Pressureatsealevel": Pressureatsealevel, 
        "Temperature": Temperature, 
        "WetTemperature": WetTemperature,
        "DewPointTemperature": DewPointTemperature,
        "RelativeHumidity": RelativeHumidity,
        "WindDirection": WindDirection,
        "WindSpeed": WindSpeed,
        "TotalCloudsCover": TotalCloudsCover,
        "TotalLowCloudsCover": TotalLowCloudsCover,
        "LowCloudsBase": LowCloudsBase,
        "LowCloudsType": LowCloudsType,
        "MediumCloudsType": MediumCloudsType,
        "HightCloudsType": HightCloudsType,
        "CurrentWeather": CurrentWeather,
        "PastWeather": PastWeather,
        "Visibility": Visibility})
###And keep doing it for all the pages
    data = pd.concat([data, pageDF])
    print(f"Number of rows: {len(data. index)}")
    nextbtn=browser.find_element(By.XPATH,"//button[@data-page='next']")
    if not nextbtn.is_enabled():
        break

    nextbtn.click()
###Now we got our data ready and we can start exploring it:)
data.to_csv('project1', index=False)   

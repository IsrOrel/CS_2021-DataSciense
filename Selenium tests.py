from pprint import pprint
from time import sleep
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

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
inputs = dstart=browser.find_elements(By.CLASS_NAME,"md-datepicker-input")
inputs[0].clear()
inputs[0].send_keys("1/11/2019")

inputs[1].clear()
inputs[1].send_keys("31/12/2022")
stationbtn =browser.find_element(By.ID,"select_2")
stationbtn.click()
search=browser.find_element(By.CLASS_NAME,"demo-header-searchbox")
search.send_keys("Dafna 11")
# try:
#    search1 = WebDriverWait(browser, 1*1000).until(
#        EC.presence_of_element_located((By.CLASS_NAME,"md-checkbox-enable"))
#     )
#    search1.click()
# except:
#    browser.quit()
serch1=browser.find_element(By.XPATH,"//md-option")
# pprint([s.text for s in serch1])
serch1.click()
closebtn=browser.find_element(By.CLASS_NAME,"close_button")
closebtn.click()
checkallbtn=browser.find_elements(By.CLASS_NAME,"btn_check_all")
checkallbtn[3].click()
submitbtn=browser.find_element(By.CLASS_NAME,"btn-primary")
submitbtn.click()
# WebDriverWait(browser, 70000*1000)
# rows=browser.find_element(By.CLASS_NAME,"tabulator-table")
try:
    rows = WebDriverWait(browser, 30*1000).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME,"tabulator-row"))
    )
except:
    browser.quit()
# for row in rows:
#     cells=browser.find_elements(By.CLASS_NAME,"tabulator-cell")
#     for cel in cells:
#         print(cells[0])

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

    data = pd.concat([data, pageDF])
    print(f"@@@@@@@@@@@@ Number of rows: {len(data. index)}")
    nextbtn=browser.find_element(By.XPATH,"//button[@data-page='next']")
    if not nextbtn.is_enabled():
        break

    nextbtn.click()

data.to_csv('project', index=False)   

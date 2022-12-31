from pprint import pprint
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

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
inputs[0].send_keys("1/11/2016")

inputs[1].clear()
inputs[1].send_keys("31/10/2019")
stationbtn =browser.find_element(By.ID,"select_2")
stationbtn.click()
search=browser.find_element(By.CLASS_NAME,"demo-header-searchbox")
search.send_keys("Dafna 04")
# try:
#    search1 = WebDriverWait(browser, 1*1000).until(
#        EC.presence_of_element_located((By.CLASS_NAME,"md-checkbox-enable"))
#     )
#    search1.click()
# except:
#    browser.quit()
serch1=browser.find_elements(By.ID,"select_listbox_4")
pprint([s.text for s in serch1])
serch1[0].click()
closebtn=browser.find_element(By.CLASS_NAME,"close_button")
closebtn.click()
checkallbtn=browser.find_elements(By.CLASS_NAME,"btn_check_all")
checkallbtn[3].click()
submitbtn=browser.find_element(By.CLASS_NAME,"btn-primary")
submitbtn.click()
x = 5
#serch1=browser.find_element(By.XPATH,"//option[div[contains(text(), 'Dafna 04/2010-10/2019')]]")
#serch1=browser.find_element(By.CLASS_NAME,"md-ripple-container")

# browser.find_element(By.XPATH("//html")).click()
# actions = ActionChains(browser)
# actions.move_to_element_with_offset(browser.find_element_by_tag_name('body'),0,0).click().perform()
# checkallbtn.click()

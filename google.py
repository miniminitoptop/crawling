from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

chrome_options=Options()
Service = Service(executable_path=ChromeDriverManager().install())

# 브라우저 꺼짐 방지
chrome_options.add_experimental_option("detach",True)

# 불필요한 에러 메세지 없애기
chrome_options.add_experimental_option("excludeSwitches",["enable-logging"])

#크롬드라이버 설정
driver = webdriver.Chrome(service=Service, options=chrome_options)


mainkeyword = ["NottinghamForestFC","NewcastleUnitedFC","LiverpoolFC","ManchesterCityFC","ManchesterUnitedFC","BurnleyFC","AFCBournemouth","Brighton&HoveAlbionFC","BrentfordFC","SheffieldUnitedFC","ArsenalFC","AstonVillaFC","EvertonFC","WolverhamptonWanderersFC","WestHamUnitedFC","ChelseaFC","TottenhamHotspurFC","CrystalPalaceFC","FulhamFC"]
subkeyword = "2022-2023Homekit"

for i in mainkeyword:

    keyword = i + subkeyword
    url = 'https://www.google.com/imghp?hl=ko&tab=ri&authuser=0&ogbl'
    driver.get(url)

    elem = driver.find_element(By.CLASS_NAME, "gLFyf")
    elem.send_keys(keyword+Keys.ENTER)

    SCROLL_PAUSE_TIME = 1

    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        #스크롤 아래로 내리기
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
        #새로운 페이지가 load되기를 기다리기
        time.sleep(SCROLL_PAUSE_TIME)
        #새로운 스크롤 높이 구하여 이전 스크롤 높이와 비교하기
        new_heigth = driver.execute_script("return document.body.scrollHeight")
        if new_heigth == last_height:
            try:
                driver.find_element(By.CSS_SELECTOR,".mye4qd").click()
            except:
                break
        last_height = new_heigth

    import os 

    folder = './'
    img_folder = folder+keyword

    print(img_folder)

    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    images = driver.find_elements(By.CLASS_NAME,"rg_i.Q4LuWd")
    count = 1
    for image in images[1:100]:
        try:
            image.click()
            time.sleep(2)
            imgUrl = driver.find_element(By.CLASS_NAME,"r48jcc.pT0Scc.iPVvYb").get_attribute("src")
            urllib.request.urlretrieve(imgUrl, f'{img_folder}/{str(count)}.jpg')
            count = count +1
        except:
            pass

    
driver.close()


# class="rg_i Q4LuWd"
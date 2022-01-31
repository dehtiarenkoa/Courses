from selenium import webdriver
import time
import unittest





    
class TestAbs(unittest.TestCase):
    def temp(self, link):
        browser = webdriver.Firefox()
        browser.get(link)
        input1 = browser.find_element_by_css_selector("form div.first_block div.first_class input")
        input1.send_keys("Ivan")
        input2 = browser.find_element_by_css_selector("form div.first_block div.second_class input")
        input2.send_keys("Petrov")
        input3 = browser.find_element_by_css_selector("form div.first_block div.third_class input")
        input3.send_keys("email")
        button = browser.find_element_by_css_selector("button.btn")
        button.click()
        # Проверяем, что смогли зарегистрироваться
        # ждем загрузки страницы
        time.sleep(1)        # находим элемент, содержащий текст
        welcome_text_elt = browser.find_element_by_tag_name("h1")
        # записываем в переменную welcome_text текст из элемента welcome_text_elt
        time.sleep(5)
        browser.quit()
        return welcome_text_elt.text
    def test_t1(self):
        link = "http://suninjuly.github.io/registration1.html"        
        self.assertEqual(self.temp(link),"Congratulations! You have successfully registered!",'NOTOK1')
    def test_t3(self):
        link = "http://suninjuly.github.io/registration2.html"        
        self.assertEqual(self.temp(link),"Congratulations! You have successfully registered!",'NOTOK2')
    # ожидание чтобы визуально оценить результаты прохождения скрипта


if __name__ == '__main__':
    unittest.main()

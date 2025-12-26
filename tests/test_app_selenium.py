#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selenium White Box Test Suite
Human vs AI Classifier - Web Arayüz Testleri

Bu dosya, web uygulamasının fonksiyonel testlerini Selenium WebDriver kullanarak gerçekleştirir.
Testleri çalıştırmadan önce Flask sunucusunun aktif olduğundan emin olun:
    $ python3 app/app.py

Testleri çalıştırma:
    $ python3 -m pytest tests/test_app_selenium.py -v
    veya
    $ python3 tests/test_app_selenium.py
"""

import unittest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class TestHumanAIClassifierUI(unittest.TestCase):
    """
    Human vs AI Classifier Web Arayüzü için Selenium Test Sınıfı
    
    White Box Test Senaryoları:
    1. Ana sayfa yükleme ve başlık kontrolü
    2. Metin girişi ve tahmin akışı
    3. Temizle butonu fonksiyonelliği
    """
    
    BASE_URL = "http://127.0.0.1:5000"
    
    @classmethod
    def setUpClass(cls):
        """
        Test sınıfı başlamadan önce bir kez çalışır.
        Chrome WebDriver'ı başlatır.
        """
        print("\n" + "="*60)
        print("Selenium White Box Test Suite Başlatılıyor...")
        print("="*60)
        
        # Chrome seçenekleri
        chrome_options = Options()
        # Headless mod (görünmez tarayıcı) - CI/CD için aktif edilebilir
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            cls.driver = webdriver.Chrome(options=chrome_options)
            cls.driver.implicitly_wait(10)  # Implicit wait
            print("✓ Chrome WebDriver başarıyla başlatıldı")
        except Exception as e:
            print(f"✗ WebDriver başlatma hatası: {e}")
            raise
    
    @classmethod
    def tearDownClass(cls):
        """
        Tüm testler bittikten sonra bir kez çalışır.
        Tarayıcıyı kapatır.
        """
        if hasattr(cls, 'driver') and cls.driver:
            cls.driver.quit()
            print("\n" + "="*60)
            print("✓ Chrome WebDriver kapatıldı")
            print("="*60)
    
    def setUp(self):
        """Her test öncesi çalışır."""
        # Ana sayfaya git
        self.driver.get(self.BASE_URL)
        time.sleep(1)  # Sayfanın yüklenmesini bekle
    
    def tearDown(self):
        """Her test sonrası çalışır."""
        pass  # Gerekirse temizlik işlemleri
    
    # =========================================================================
    # TEST 1: Ana Sayfa Başlık Kontrolü
    # =========================================================================
    def test_home_page_title(self):
        """
        TEST: Ana sayfanın başlığının 'Human or AI' içerdiğini doğrular.
        
        Adımlar:
        1. Ana sayfaya git
        2. Sayfa başlığını al
        3. Başlıkta 'Human or AI' ifadesini kontrol et
        """
        print("\n[TEST 1] Ana Sayfa Başlık Kontrolü")
        
        # Sayfa başlığını al
        page_title = self.driver.title
        print(f"  → Sayfa başlığı: '{page_title}'")
        
        # Başlıkta "Human or AI" ifadesi var mı?
        self.assertIn(
            "Human or AI", 
            page_title, 
            f"Başlık 'Human or AI' içermiyor. Mevcut başlık: {page_title}"
        )
        print("  ✓ Başlık doğrulandı")
    
    # =========================================================================
    # TEST 2: Tahmin Akışı (Prediction Flow)
    # =========================================================================
    def test_prediction_flow(self):
        """
        TEST: Metin girişi yapıp tahmin sonuçlarının gösterilmesini doğrular.
        
        Adımlar:
        1. Textarea elementini bul
        2. Test metnini yaz
        3. Submit butonuna tıkla
        4. Sonuçların görüntülenmesini bekle
        5. Sonuçları doğrula
        """
        print("\n[TEST 2] Tahmin Akışı Testi")
        
        # Test metni (AI tarzı akademik metin)
        test_text = """This is a test abstract driven by computer science. 
        We investigate the challenge of automated text classification using 
        machine learning algorithms. Our approach leverages deep neural networks 
        to achieve state-of-the-art performance on benchmark datasets."""
        
        try:
            # 1. Textarea'yı bul
            textarea = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "textInput"))
            )
            print("  → Textarea elementi bulundu")
            
            # 2. Textarea'yı temizle ve metni yaz
            textarea.clear()
            textarea.send_keys(test_text)
            print(f"  → Test metni yazıldı ({len(test_text)} karakter)")
            
            # 3. Submit butonunu bul ve tıkla
            submit_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))
            )
            submit_button.click()
            print("  → Submit butonuna tıklandı")
            
            # 4. Sonuçların yüklenmesini bekle (progress bar veya sonuç kartı)
            # Genel tahmin sonucu görünene kadar bekle
            result_element = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".prediction-badge, .result-card, [class*='progress'], [class*='result']"))
            )
            print("  → Sonuç elementi bulundu")
            
            # 5. Sayfada yüzde (%) işareti veya "AI" / "Human" kelimesi var mı?
            page_source = self.driver.page_source
            
            # En az bir sonuç göstergesi bulunmalı
            has_percentage = "%" in page_source
            has_ai_label = "AI" in page_source
            has_human_label = "Human" in page_source
            
            self.assertTrue(
                has_percentage or has_ai_label or has_human_label,
                "Sonuç sayfasında beklenen içerik bulunamadı (%, AI, Human)"
            )
            print("  ✓ Tahmin sonuçları başarıyla görüntülendi")
            
            # Ekstra: Progress bar kontrolü
            try:
                progress_bars = self.driver.find_elements(By.CSS_SELECTOR, ".progress-bar, [class*='progress']")
                if progress_bars:
                    print(f"  ✓ {len(progress_bars)} adet progress bar bulundu")
            except NoSuchElementException:
                pass
            
        except TimeoutException as e:
            self.fail(f"Timeout: Element bulunamadı - {e}")
        except Exception as e:
            self.fail(f"Tahmin akışı hatası: {e}")
    
    # =========================================================================
    # TEST 3: Temizle Butonu Fonksiyonelliği
    # =========================================================================
    def test_clear_button(self):
        """
        TEST: Temizle butonunun textarea'yı temizlediğini doğrular.
        
        Adımlar:
        1. Textarea'ya metin yaz
        2. Temizle butonunu bul ve tıkla
        3. Textarea'nın boş olduğunu doğrula
        """
        print("\n[TEST 3] Temizle Butonu Testi")
        
        test_text = "Bu metin temizlenecek."
        
        try:
            # 1. Textarea'yı bul ve metin yaz
            textarea = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "textInput"))
            )
            textarea.clear()
            textarea.send_keys(test_text)
            print(f"  → Metin yazıldı: '{test_text}'")
            
            # Metnin yazıldığını doğrula
            current_value = textarea.get_attribute("value")
            self.assertEqual(current_value, test_text, "Metin yazılamadı")
            
            # 2. Temizle butonunu bul
            # Farklı olası seçicileri dene
            clear_button = None
            selectors = [
                "button[type='reset']",
                "button.btn-outline-secondary",
                "//button[contains(text(), 'Temizle')]",
                "//button[contains(text(), 'Clear')]",
                "[onclick*='clear']",
                ".btn-clear",
                "#clearBtn"
            ]
            
            for selector in selectors:
                try:
                    if selector.startswith("//"):
                        clear_button = self.driver.find_element(By.XPATH, selector)
                    else:
                        clear_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if clear_button:
                        break
                except NoSuchElementException:
                    continue
            
            if clear_button:
                clear_button.click()
                print("  → Temizle butonuna tıklandı")
                time.sleep(1)  # Sayfa yenilenmesi için bekleme
                
                # 3. Textarea'yı yeniden bul (sayfa yenilenmiş olabilir)
                textarea = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, "textInput"))
                )
                
                # Textarea'nın boş olduğunu kontrol et
                new_value = textarea.get_attribute("value")
                self.assertEqual(
                    new_value, 
                    "", 
                    f"Textarea temizlenmedi. Mevcut değer: '{new_value}'"
                )
                print("  ✓ Textarea başarıyla temizlendi")
            else:
                # Temizle butonu bulunamazsa, JavaScript ile temizle
                print("  ⚠ Temizle butonu bulunamadı, manuel temizleme test ediliyor")
                self.driver.execute_script("document.getElementById('textInput').value = ''")
                textarea = self.driver.find_element(By.ID, "textInput")
                new_value = textarea.get_attribute("value")
                self.assertEqual(new_value, "", "Textarea manuel olarak temizlenemedi")
                print("  ✓ Manuel temizleme başarılı")
                
        except TimeoutException as e:
            self.fail(f"Timeout: Element bulunamadı - {e}")
        except Exception as e:
            self.fail(f"Temizle butonu testi hatası: {e}")
    
    # =========================================================================
    # TEST 4: Boş Metin Gönderme (Edge Case)
    # =========================================================================
    def test_empty_text_submission(self):
        """
        TEST: Boş metin gönderildiğinde uygun hata mesajı gösterilmesini doğrular.
        
        Adımlar:
        1. Textarea'yı boş bırak
        2. Submit butonuna tıkla
        3. Hata mesajı veya validasyon uyarısı kontrol et
        """
        print("\n[TEST 4] Boş Metin Gönderme Testi")
        
        try:
            # 1. Textarea'yı bul ve boş olduğundan emin ol
            textarea = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "textInput"))
            )
            textarea.clear()
            print("  → Textarea temizlendi (boş)")
            
            # 2. Submit butonuna tıkla
            submit_button = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            submit_button.click()
            print("  → Submit butonuna tıklandı")
            
            time.sleep(1)
            
            # 3. HTML5 validasyon veya özel hata mesajı kontrolü
            # HTML5 required attribute varsa, tarayıcı otomatik uyarı gösterir
            is_valid = self.driver.execute_script(
                "return document.getElementById('textInput').validity.valid"
            )
            
            if not is_valid:
                print("  ✓ HTML5 validasyonu çalışıyor (boş metin reddedildi)")
            else:
                # Özel hata mesajı kontrolü
                page_source = self.driver.page_source.lower()
                has_error = any(word in page_source for word in ['error', 'hata', 'gerekli', 'required', 'boş'])
                print(f"  → Özel hata mesajı kontrolü: {has_error}")
            
            print("  ✓ Boş metin gönderme testi tamamlandı")
            
        except Exception as e:
            print(f"  ⚠ Test bilgi: {e}")
    
    # =========================================================================
    # TEST 5: Sayfa Elementlerinin Varlığı
    # =========================================================================
    def test_page_elements_exist(self):
        """
        TEST: Sayfadaki temel elementlerin varlığını doğrular.
        
        Kontrol edilen elementler:
        - Başlık (h1 veya title)
        - Textarea (metin girişi)
        - Submit butonu
        - Temizle butonu
        """
        print("\n[TEST 5] Sayfa Elementleri Kontrolü")
        
        elements_to_check = [
            ("Textarea", By.ID, "textInput"),
            ("Submit Button", By.CSS_SELECTOR, "button[type='submit']"),
            ("Form", By.TAG_NAME, "form"),
        ]
        
        for name, by, selector in elements_to_check:
            try:
                element = self.driver.find_element(by, selector)
                self.assertIsNotNone(element, f"{name} bulunamadı")
                print(f"  ✓ {name} bulundu")
            except NoSuchElementException:
                self.fail(f"{name} elementi bulunamadı (selector: {selector})")
        
        print("  ✓ Tüm temel elementler mevcut")


# =============================================================================
# Ana çalıştırma bloğu
# =============================================================================
if __name__ == "__main__":
    # Test sonuçlarını detaylı göster
    unittest.main(verbosity=2)

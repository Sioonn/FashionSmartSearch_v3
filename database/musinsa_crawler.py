import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# ==================== 설정 ====================
MAX_ITEMS = 1000  # 카테고리당 수집할 상품 개수
# =============================================

def setup_driver():
    """Chrome WebDriver 설정"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # 브라우저 창 없이 실행
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = None
    
    # webdriver_manager 사용하여 ChromeDriver 자동 다운로드
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        import warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        # ChromeDriver 다운로드 및 경로 가져오기
        driver_path = ChromeDriverManager().install()
        print(f"ChromeDriver 다운로드 완료: {driver_path}")
        
        # options 파라미터 사용 (DeprecationWarning 방지)
        driver = webdriver.Chrome(executable_path=driver_path, options=chrome_options)
        print("ChromeDriver 초기화 성공!")
        return driver
        
    except ValueError as e:
        # urllib3 호환성 문제 - 해결 방법 안내
        if "Timeout value" in str(e):
            print(f"\nurllib3 버전 호환성 문제 발견: {e}")
            print("\n해결 방법:")
            print("pip install urllib3==1.26.15")
            print("또는")
            print("pip install --upgrade selenium")
            raise Exception(
                "urllib3 버전 문제입니다.\n"
                "다음 명령어를 실행하세요:\n"
                "pip install urllib3==1.26.15"
            )
        raise
        
    except Exception as e:
        print(f"ChromeDriver 자동 설치 실패: {e}")
        print("\n대안: 수동으로 ChromeDriver를 다운로드해주세요.")
        print("1. https://chromedriver.chromium.org/downloads 방문")
        print("2. Chrome 브라우저 버전에 맞는 ChromeDriver 다운로드")
        print("3. 다운로드한 chromedriver.exe를 이 스크립트와 같은 폴더에 위치")
        print("4. 또는 PATH 환경변수에 chromedriver.exe 경로 추가")
        raise Exception("ChromeDriver를 찾을 수 없습니다. 위 안내를 따라주세요.")

def extract_product_info(item):
    """각 상품 아이템에서 정보 추출"""
    product_data = {}
    
    try:
        # 1. 상품 구매 링크
        link_element = item.find_element(By.CSS_SELECTOR, 'a.sc-cOpnSz.keqfmf')
        product_data['product_url'] = link_element.get_attribute('href')
    except NoSuchElementException:
        product_data['product_url'] = None
    
    try:
        # 2. 브랜드 이름
        brand_element = item.find_element(By.CSS_SELECTOR, 'span.text-etc_11px_semibold.sc-hwkwBN.sc-kNOymR')
        product_data['brand_name'] = brand_element.text
    except NoSuchElementException:
        product_data['brand_name'] = None
    
    try:
        # 3. 상품 이름
        product_name_element = item.find_element(By.CSS_SELECTOR, 'span.text-body_13px_reg.sc-hwkwBN.sc-dYwGCk')
        product_data['product_name'] = product_name_element.text
    except NoSuchElementException:
        product_data['product_name'] = None
    
    try:
        # 4. 가격 (할인된 가격)
        price_elements = item.find_elements(By.CSS_SELECTOR, 'span.text-body_13px_semi.sc-jJLAfE.gsJKfg.font-pretendard')
        # 두 번째 요소가 실제 가격 (첫 번째는 할인율)
        if len(price_elements) >= 2:
            product_data['price'] = price_elements[1].text
        elif len(price_elements) == 1:
            product_data['price'] = price_elements[0].text
        else:
            product_data['price'] = None
    except NoSuchElementException:
        product_data['price'] = None
    
    try:
        # 5. 상품 이미지 URL
        img_element = item.find_element(By.CSS_SELECTOR, 'img.max-w-full.w-full.absolute')
        image_url = img_element.get_attribute('src')
        # 쿼리 파라미터 제거 (?w=260 등)
        if image_url and '?' in image_url:
            image_url = image_url.split('?')[0]
        product_data['image_url'] = image_url
    except NoSuchElementException:
        product_data['image_url'] = None
    
    return product_data

def crawl_musinsa(url, output_file='musinsa_products.json', max_items=50):
    """무신사 페이지 크롤링 메인 함수
    
    Args:
        url: 크롤링할 무신사 페이지 URL
        output_file: 결과를 저장할 JSON 파일명
        max_items: 수집할 최대 상품 개수 (기본값: 50)
    """
    driver = setup_driver()
    products = []
    collected_urls = set()  # 중복 방지를 위한 URL 세트
    collected_count = 0
    scroll_count = 0
    
    try:
        print(f"페이지 로딩 중: {url}")
        driver.get(url)
        
        # 페이지 로드 대기
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "sc-hdBJTi")))
        time.sleep(2)  # 초기 로딩 대기 시간 증가
        
        print(f"\n=== 크롤링 시작 (목표: {max_items}개) ===\n")
        
        no_new_items_count = 0
        
        while collected_count < max_items:
            # 스크롤 전 약간의 대기 (DOM 업데이트 시간 확보)
            time.sleep(1)
            
            # 현재 화면에 로드된 모든 상품 찾기
            items = driver.find_elements(By.CSS_SELECTOR, 'div.sc-hdBJTi.gAWtWT')
            current_item_count = len(items)
            
            print(f"[디버그] 현재 DOM에서 찾은 총 요소 수: {current_item_count}개")
            
            # 새로운 상품만 추출 (URL 기반 중복 제거)
            new_items_found = 0
            
            for item in items:
                if collected_count >= max_items:
                    break
                
                try:
                    # URL로 중복 체크
                    link_element = item.find_element(By.CSS_SELECTOR, 'a.sc-cOpnSz.keqfmf')
                    product_url = link_element.get_attribute('href')
                    
                    # 이미 수집한 상품이면 스킵
                    if product_url in collected_urls:
                        continue
                    
                    # 새 상품 정보 추출
                    product_data = extract_product_info(item)
                    
                    # 유효한 데이터인 경우만 추가
                    if product_data.get('product_url'):
                        products.append(product_data)
                        collected_urls.add(product_url)
                        collected_count += 1
                        new_items_found += 1
                        
                        # 10개마다 진행 상황 출력
                        if collected_count % 10 == 0:
                            print(f"  [{collected_count}/{max_items}] {product_data.get('brand_name', 'N/A')} - {product_data.get('product_name', 'N/A')[:40]}...")
                            
                except NoSuchElementException:
                    continue
                except Exception as e:
                    print(f"  [오류] 상품 추출 실패: {str(e)}")
                    continue
            
            if new_items_found > 0:
                print(f"[배치 {scroll_count + 1}] 전체 요소: {current_item_count}개, 새로 수집: {new_items_found}개, 누적: {collected_count}개\n")
                no_new_items_count = 0
                
                if collected_count >= max_items:
                    print(f"\n✓ 목표 개수 {max_items}개 도달!")
                    break
            else:
                no_new_items_count += 1
                print(f"[수집 완료: {collected_count}/{max_items}개] 새 상품 없음 (연속 {no_new_items_count}회)")
                
                # 10회 연속 새 상품이 없으면 종료
                if no_new_items_count >= 10:
                    print(f"\n⚠ 10회 연속 새 상품이 발견되지 않아 종료합니다.")
                    print(f"최종 수집 개수: {collected_count}개")
                    break
                
                # 스크롤
                print(f"  → 스크롤하여 추가 상품 로딩 시도 중...")
                
                current_scroll = driver.execute_script("return window.pageYOffset;")
                scroll_height = driver.execute_script("return document.body.scrollHeight")
                
                # 한 번에 1000px씩 스크롤 (더 작은 단위로 변경)
                target_position = current_scroll + 2000
                
                print(f"  → 현재 위치: {current_scroll}px → {target_position}px (전체: {scroll_height}px)")
                
                driver.execute_script(f"window.scrollTo(0, {target_position});")
                time.sleep(2.5)  # 로딩 대기 시간 증가
                
                scroll_count += 1
                print()
        
        # JSON 파일로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*50}")
        print(f"✓ 크롤링 완료!")
        print(f"✓ 수집된 상품: {len(products)}개")
        print(f"✓ 저장 위치: {output_file}")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"\n✗ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        driver.quit()
    
    return products

if __name__ == "__main__":
    # database 폴더 생성 (없으면)
    import os
    os.makedirs('database', exist_ok=True)
    
    # 크롤링할 카테고리 목록 (영어 key)
    categories = {
        # 상의
        "knit_sweater": "https://www.musinsa.com/category/001006?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "sweatshirt": "https://www.musinsa.com/category/001005?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "long_sleeve_tshirt": "https://www.musinsa.com/category/001010?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "hoodie": "https://www.musinsa.com/category/001004?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "shirt": "https://www.musinsa.com/category/001002?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "polo_tshirt": "https://www.musinsa.com/category/001003?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "short_sleeve_tshirt": "https://www.musinsa.com/category/001001?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "other_tops": "https://www.musinsa.com/category/001008?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "sleeveless_tshirt": "https://www.musinsa.com/category/001011?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        # 아우터
        "short_padding": "https://www.musinsa.com/category/002012?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "blouson": "https://www.musinsa.com/category/002001?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "safari_jacket": "https://www.musinsa.com/category/002014?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "fleece": "https://www.musinsa.com/category/002023?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "trucker_jacket": "https://www.musinsa.com/category/002017?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "coach_jacket": "https://www.musinsa.com/category/002006?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "winter_other_coat": "https://www.musinsa.com/category/002009?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "training_jacket": "https://www.musinsa.com/category/002018?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "blazer": "https://www.musinsa.com/category/002003?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "other_outer": "https://www.musinsa.com/category/002015?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "winter_double_coat": "https://www.musinsa.com/category/002024?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "winter_single_coat": "https://www.musinsa.com/category/002007?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "leather_jacket": "https://www.musinsa.com/category/002002?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "shearling": "https://www.musinsa.com/category/002025?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "spring_coat": "https://www.musinsa.com/category/002008?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "anorak": "https://www.musinsa.com/category/002019?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "padding_vest": "https://www.musinsa.com/category/002016?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "zip_up_hoodie": "https://www.musinsa.com/category/002022?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "vest": "https://www.musinsa.com/category/002021?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "long_padding": "https://www.musinsa.com/category/002013?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "stadium_jacket": "https://www.musinsa.com/category/002004?gf=M&sortCode=SALE_ONE_YEAR_COUNT",
        "cardigan": "https://www.musinsa.com/category/002020?gf=M&sortCode=SALE_ONE_YEAR_COUNT"
    }
    
    # 카테고리 한글 이름 (출력용)
    category_names_kr = {
        # 상의
        "knit_sweater": "니트/스웨터",
        "sweatshirt": "맨투맨",
        "long_sleeve_tshirt": "긴소매 티셔츠",
        "hoodie": "후드",
        "shirt": "셔츠",
        "polo_tshirt": "피케/카라 티셔츠",
        "short_sleeve_tshirt": "반소매 티셔츠",
        "other_tops": "기타 상의",
        "sleeveless_tshirt": "민소매 티셔츠",
        # 아우터
        "short_padding": "숏패딩/헤비 아우터",
        "blouson": "블루종",
        "safari_jacket": "사파리/헌팅 재킷",
        "fleece": "플리스/뽀글이",
        "trucker_jacket": "트러커 재킷",
        "coach_jacket": "나일론/코치 재킷",
        "winter_other_coat": "겨울 기타 코트",
        "training_jacket": "트레이닝 재킷",
        "blazer": "수트/블레이저 재킷",
        "other_outer": "기타 아우터",
        "winter_double_coat": "겨울 더블 코트",
        "winter_single_coat": "겨울 싱글 코트",
        "leather_jacket": "레더/라이더스 자켓",
        "shearling": "무스탕/퍼",
        "spring_coat": "환절기 코트",
        "anorak": "아노락 재킷",
        "padding_vest": "패딩 베스트",
        "zip_up_hoodie": "후드 집업",
        "vest": "베스트",
        "long_padding": "롱패딩/헤비 아우터",
        "stadium_jacket": "스타디움 재킷",
        "cardigan": "카디건"
    }
    
    # 카테고리별로 상품을 저장할 딕셔너리
    products_by_category = {}
    items_per_category = MAX_ITEMS  # 상단의 MAX_ITEMS 상수 사용
    
    print("="*60)
    print(f"무신사 크롤러 시작")
    print(f"카테고리 수: {len(categories)}개")
    print(f"카테고리당 수집 개수: {items_per_category}개")
    print(f"예상 총 수집 개수: {len(categories) * items_per_category}개")
    print("="*60)
    print()
    
    # 각 카테고리별로 크롤링
    for idx, (category_key, url) in enumerate(categories.items(), 1):
        category_name_kr = category_names_kr[category_key]
        
        print(f"\n{'#'*60}")
        print(f"[{idx}/{len(categories)}] {category_name_kr} ({category_key}) 카테고리 크롤링 시작")
        print(f"{'#'*60}\n")
        
        # temp 파일명 (database 폴더에 저장)
        temp_output_file = f'database/temp_{category_key}.json'
        
        try:
            # temp 파일에 저장 (메모리 절약)
            products = crawl_musinsa(url, output_file=temp_output_file, max_items=items_per_category)
            
            print(f"\n✓ {category_name_kr} ({category_key}): {len(products)}개 수집 완료\n")
            
        except Exception as e:
            print(f"\n✗ {category_name_kr} ({category_key}) 크롤링 실패: {str(e)}\n")
            continue
    
    # temp 파일들을 읽어서 통합 JSON 생성
    print("\n" + "="*60)
    print("temp 파일들을 통합하는 중...")
    print("="*60)
    
    for category_key in categories.keys():
        category_name_kr = category_names_kr[category_key]
        temp_file = f'database/temp_{category_key}.json'
        
        if os.path.exists(temp_file):
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    products = json.load(f)
                    products_by_category[category_key] = products
                    print(f"  ✓ {category_key} ({category_name_kr}): {len(products)}개 로드")
            except Exception as e:
                print(f"  ✗ {category_key} 로드 실패: {str(e)}")
                products_by_category[category_key] = []
        else:
            print(f"  ✗ {category_key}: temp 파일 없음")
            products_by_category[category_key] = []
    
    # 통합 JSON 파일로 저장 (database 폴더에)
    output_file = 'database/musinsa_products_db.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(products_by_category, f, ensure_ascii=False, indent=2)
    
    # 최종 결과 출력
    print("\n" + "="*60)
    print("크롤링 완료!")
    print("="*60)
    
    total_products = sum(len(products) for products in products_by_category.values())
    print(f"총 수집된 상품 수: {total_products}개")
    print(f"저장 위치: {output_file}")
    print("="*60)
    
    # 카테고리별 수집 통계
    print("\n[카테고리별 수집 통계]")
    for category_key, products in products_by_category.items():
        category_name_kr = category_names_kr[category_key]
        print(f"  {category_key} ({category_name_kr}): {len(products)}개")
    
    print("\n[데이터 구조]")
    print(f"{{")
    for category_key in list(products_by_category.keys())[:3]:
        print(f"  \"{category_key}\": [")
        print(f"    {{ 상품1 정보 }},")
        print(f"    {{ 상품2 정보 }},")
        print(f"    ...")
        print(f"  ],")
    print(f"  ...")
    print(f"}}")
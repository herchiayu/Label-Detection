"""
OCR 文字萃取模組
提供 Tesseract OCR 功能：多語言文字識別、結果輸出
"""

import os
import json
import time
import pytesseract
from PIL import Image

# Tesseract 配置
TESSERACT_LOCAL = os.path.join(os.path.dirname(__file__), 'OCR model', 'tesseract', 'tesseract.exe')
TESSERACT_SYSTEM = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if os.path.exists(TESSERACT_LOCAL):
    TESSERACT_CMD = TESSERACT_LOCAL
    TESSDATA_DIR = os.path.join(os.path.dirname(__file__), 'OCR model', 'tesseract', 'tessdata')
else:
    TESSERACT_CMD = TESSERACT_SYSTEM
    TESSDATA_DIR = None

# 語言設定
SUPPORTED_LANGUAGES = {'eng', 'fra', 'spa'}
DEFAULT_LANGUAGE = 'eng'

def initialize_tesseract():
    """初始化 Tesseract OCR"""
    try:
        # 設定 Tesseract 執行檔路徑
        if os.path.exists(TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
            
            # 如果使用本地 Tesseract，設定 tessdata 路徑
            if TESSDATA_DIR and os.path.exists(TESSDATA_DIR):
                os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR
            
            return True
        else:
            print(f"[ERROR] 找不到 Tesseract 執行檔: {TESSERACT_CMD}")
            return False
    except Exception as e:
        print(f"[ERROR] Tesseract OCR 初始化失敗: {e}")
        return False

def extract_text_from_image(image_path, language='eng', confidence_threshold=0.3, verbose=True):
    """從圖片萃取文字
    Returns: list of {'text', 'confidence', 'bbox', 'polygon'}
    """
    # 初始化檢查
    if not hasattr(pytesseract.pytesseract, 'tesseract_cmd') or not pytesseract.pytesseract.tesseract_cmd:
        if not initialize_tesseract():
            return []
    
    try:
        # 語言驗證
        tesseract_lang = language if language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE
        
        # 讀取圖像
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
        
        # OCR 處理
        ocr_data = pytesseract.image_to_data(
            image, 
            lang=tesseract_lang,
            output_type=pytesseract.Output.DICT,
            config='--psm 6'
        )
        
        # 組織文字結果
        lines = {}
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            confidence = float(ocr_data['conf'][i])
            text = ocr_data['text'][i].strip()
            line_num = ocr_data['line_num'][i]
            word_num = ocr_data['word_num'][i]
            
            # 字符安全處理
            if text:
                try:
                    text.encode('utf-8')
                except UnicodeEncodeError:
                    text = text.encode('utf-8', errors='ignore').decode('utf-8')
                
            if text and confidence > 0:
                confidence_normalized = confidence / 100.0
                
                if confidence_normalized >= confidence_threshold:
                    x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                    
                    if line_num not in lines:
                        lines[line_num] = {'texts': [], 'boxes': [], 'confidences': [], 'words': []}
                    
                    lines[line_num]['texts'].append(text)
                    lines[line_num]['boxes'].append([x, y, x+w, y+h])
                    lines[line_num]['confidences'].append(confidence_normalized)
                    lines[line_num]['words'].append({
                        'text': text,
                        'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                        'confidence': confidence_normalized
                    })
                    
                    if verbose:
                        try:
                            print(f"    [OK] '{text}' (信心度: {confidence_normalized:.3f})")
                        except UnicodeEncodeError:
                            print(f"    [OK] '<特殊字符>' (信心度: {confidence_normalized:.3f})")
        
        # 組合文字行
        ocr_texts = []
        for line_num in sorted(lines.keys()):
            line_data = lines[line_num]
            combined_text = ' '.join(line_data['texts'])
            
            all_boxes = line_data['boxes']
            if all_boxes:
                min_x = min(box[0] for box in all_boxes)
                min_y = min(box[1] for box in all_boxes)
                max_x = max(box[2] for box in all_boxes)
                max_y = max(box[3] for box in all_boxes)
                
                avg_confidence = sum(line_data['confidences']) / len(line_data['confidences'])
                
                bbox_points = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
                
                ocr_texts.append({
                    'text': combined_text,
                    'confidence': avg_confidence,
                    'bbox': {'x': min_x, 'y': min_y, 'width': max_x-min_x, 'height': max_y-min_y},
                    'polygon': bbox_points,
                    'words': line_data['words']  # 個別詞的詳細資訊
                })
        
        if verbose:
            print(f"  識別結果: 共 {len(ocr_texts)} 行文字（信心度 >= {confidence_threshold}）")
        return ocr_texts
        
    except Exception as e:
        if verbose:
            print(f"  [ERROR] Tesseract OCR 處理失敗: {e}")
        return []

def save_extracted_texts_txt(tesseract_results, result_path, confidence_threshold, language='eng'):
    """保存 OCR 結果為 TXT 文件"""
    # 創建結果目錄
    os.makedirs(result_path, exist_ok=True)
    
    # 文件名處理
    if tesseract_results:
        image_name = tesseract_results[0]['image']
        image_basename = os.path.splitext(image_name)[0]
        output_file = os.path.join(result_path, f"{image_basename}_ocr_text_{language}.txt")
    else:
        output_file = os.path.join(result_path, f"ocr_text_{language}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if tesseract_results:
            result = tesseract_results[0]  # 處理第一個結果
            image_name = result['image']
            texts = result['texts']
            processing_time = result['processing_time']
            
            f.write("=== TESSERACT OCR 文字萃取結果 ===\n\n")
            f.write(f"目標圖片: {image_name}\n")
            f.write(f"語言模型: {language}\n")
            f.write(f"信心度閾值: {confidence_threshold}\n")
            f.write(f"處理時間: {processing_time:.2f} 秒\n")
            f.write(f"識別文字數量: {len(texts)} 行\n")
            f.write(f"處理時間: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 34 + "\n\n")
            
            if texts:
                # 排序文字
                sorted_texts = sorted(texts, key=lambda x: (x['bbox']['y'], x['bbox']['x']))
                
                f.write("📝 識別的文字 (逐行顯示):\n\n")
                
                for i, text_info in enumerate(sorted_texts, 1):
                    f.write(f"{i:2d}. {text_info['text']}\n")
                
                f.write("-" * 34 + "\n\n")
                
                f.write("📊 詳細資訊:\n\n")
                
                for i, text_info in enumerate(sorted_texts, 1):
                    f.write(f"{i:2d}. '{text_info['text']}'\n")
                    f.write(f"    信心度: {text_info['confidence']:.3f}\n")
                    f.write(f"    位置: ({text_info['bbox']['x']}, {text_info['bbox']['y']})\n")
                    f.write(f"    尺寸: {text_info['bbox']['width']}×{text_info['bbox']['height']}\n\n")
            else:
                f.write("未識別到任何文字\n")
    
    return output_file

def save_extracted_texts_json(tesseract_results, result_path, confidence_threshold, language='eng'):
    """保存 OCR 結果為 JSON 文件"""
    # 創建結果目錄
    os.makedirs(result_path, exist_ok=True)
    
    # 獲取圖片名稱（不含副檔名）
    if tesseract_results:
        image_name = tesseract_results[0]['image']  # 獲取第一張圖片的檔名
        image_basename = os.path.splitext(image_name)[0]  # 移除副檔名
        output_file = os.path.join(result_path, f"{image_basename}_ocr_text_{language}.json")
    else:
        output_file = os.path.join(result_path, f"ocr_text_{language}.json")
    
    # JSON 數據結構
    json_data = {
        "metadata": {
            "processing_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "confidence_threshold": confidence_threshold,
            "total_images": len(tesseract_results),
            "ocr_engine": "Tesseract"
        },
        "images": []
    }
    
    total_texts = 0
    total_processing_time = 0
    
    for result in tesseract_results:
        image_name = result['image']
        texts = result['texts']
        processing_time = result['processing_time']
        total_texts += len(texts)
        total_processing_time += processing_time
        
        # 按Y座標排序 (從上到下)
        sorted_texts = sorted(texts, key=lambda x: (x['bbox']['y'], x['bbox']['x']))
        
        # 圖片結果
        image_result = {
            "filename": image_name,
            "texts": []
        }
        
        for text_info in sorted_texts:
            text_entry = {
                "text": text_info['text'],
                "confidence": round(text_info['confidence'], 3),
                "bbox": {
                    "x": text_info['bbox']['x'],
                    "y": text_info['bbox']['y'],
                    "width": text_info['bbox']['width'],
                    "height": text_info['bbox']['height']
                },
                "words": text_info.get('words', [])
            }
            image_result["texts"].append(text_entry)
        
        json_data["images"].append(image_result)
    
    # 寫入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    return output_file

def initialize_ocr_module():
    """OCR 模組初始化"""
    return initialize_tesseract()

def main(test_image, test_language='eng', confidence_threshold=0.3, result_path="result"):
    """執行 OCR 測試流程"""
    # 初始化 OCR
    if not initialize_ocr_module():
        return False
    
    # 檢查圖像
    if not os.path.exists(test_image):
        return False
    
    # 執行 OCR
    start_time = time.time()
    ocr_results = extract_text_from_image(
        image_path=test_image,
        language=test_language,
        confidence_threshold=confidence_threshold,
        verbose=False
    )
    processing_time = time.time() - start_time
    
    # 保存結果
    try:
        # 準備數據
        results_data = [{
            'image': os.path.basename(test_image),
            'texts': ocr_results,
            'processing_time': processing_time
        }]
        
        # 保存 TXT 和 JSON
        save_extracted_texts_txt(results_data, result_path, confidence_threshold, test_language)
        save_extracted_texts_json(results_data, result_path, confidence_threshold, test_language)
        return True
        
    except Exception as e:
        return False

# 測試區域

if __name__ == "__main__":
    print("=== text_extraction OCR 模組測試 ===")
    
    # 測試參數
    TEST_IMAGE = r"input data\target\Label_clean.png"
    TEST_LANGUAGES = ['eng', 'fra', 'spa']
    CONFIDENCE_THRESHOLD = 0.3
    RESULT_PATH = r"result"
    
    # 依序測試各語言
    all_success = True
    for language in TEST_LANGUAGES:
        print(f"\n🔍 測試語言: {language.upper()}")
        
        success = main(
            test_image=TEST_IMAGE,
            test_language=language,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            result_path=RESULT_PATH
        )
        
        if success:
            print(f"✅ {language.upper()} 語言辨識成功")
        else:
            print(f"❌ {language.upper()} 語言辨識失敗")
            all_success = False
    
    print("\n" + "="*50)
    if all_success:
        print("🎉 所有語言測試完成，全部成功！")
    else:
        print("⚠️ 部分語言測試失敗，請檢查錯誤訊息")
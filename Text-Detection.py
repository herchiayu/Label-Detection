import cv2
import numpy as np
import os
import json
from pathlib import Path
import glob
import pandas as pd
import pytesseract
from PIL import Image

# ============================
# GLOBAL PARAMETERS
# ============================
# OCR 信心度閾值 (0.0-1.0)
OCR_CONFIDENCE_THRESHOLD = 0.3

# 文字匹配設定
IGNORE_CASE = True                # 忽略大小寫
TEXT_MATCH_MODE = "exact"         # 完全匹配模式

# 標記樣式設定 (BGR格式)
BBOX_COLOR = (0, 0, 255)         # 紅色
BBOX_THICKNESS = 2               # 底線粗細
TEXT_COLOR = (255, 255, 255)     # 白色文字
TEXT_BACKGROUND_COLOR = (0, 0, 255)  # 紅色背景

# 輸入輸出路徑
INPUT_TARGET_PATH = "input data/target"
INPUT_TEXT_PATH = "input data/text/text.txt"
RESULT_PATH = "result"

# Tesseract OCR 設定
# 優先使用項目目錄中的 Tesseract（用於打包部署）
# 如果不存在，則使用系統安裝的 Tesseract
TESSERACT_LOCAL = os.path.join(os.path.dirname(__file__), 'tesseract', 'tesseract.exe')
TESSERACT_SYSTEM = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if os.path.exists(TESSERACT_LOCAL):
    TESSERACT_CMD = TESSERACT_LOCAL
    TESSDATA_DIR = os.path.join(os.path.dirname(__file__), 'tesseract', 'tessdata')
else:
    TESSERACT_CMD = TESSERACT_SYSTEM
    TESSDATA_DIR = None

OCR_LANGUAGE = 'eng'  # 英文

def create_result_directory():
    """創建結果目錄"""
    os.makedirs(RESULT_PATH, exist_ok=True)

def initialize_ocr():
    """
    初始化 Tesseract OCR
    """
    print("正在初始化 Tesseract OCR...")
    try:
        # 設定 Tesseract 執行檔路徑
        if os.path.exists(TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
            
            # 如果使用本地 Tesseract，設定 tessdata 路徑
            if TESSDATA_DIR and os.path.exists(TESSDATA_DIR):
                os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR
                print(f"[OK] Tesseract OCR 初始化成功（使用項目目錄）")
                print(f"[OK] Tesseract 路徑: {TESSERACT_CMD}")
                print(f"[OK] Tessdata 路徑: {TESSDATA_DIR}")
            else:
                print(f"[OK] Tesseract OCR 初始化成功（使用系統安裝）")
                print(f"[OK] Tesseract 路徑: {TESSERACT_CMD}")
            
            print(f"[OK] 語言: {OCR_LANGUAGE}")
            print(f"[OK] 信心度閾值設為: {OCR_CONFIDENCE_THRESHOLD}")
            
            # 測試 Tesseract 是否可用
            version = pytesseract.get_tesseract_version()
            print(f"[OK] Tesseract 版本: {version}")
            return True
        else:
            print(f"[ERROR] 找不到 Tesseract 執行檔")
            print(f"已嘗試: {TESSERACT_CMD}")
            if TESSERACT_CMD == TESSERACT_LOCAL:
                print(f"備用路徑: {TESSERACT_SYSTEM}")
            print("請確認 Tesseract 已正確安裝或複製到項目目錄")
            return False
    except Exception as e:
        print(f"[ERROR] Tesseract OCR 初始化失敗: {e}")
        print("請確認已正確安裝 Tesseract OCR")
        import traceback
        traceback.print_exc()
        return False

def load_target_texts():
    """
    載入要檢測的目標文字
    """
    target_texts = []
    
    try:
        with open(INPUT_TEXT_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            text = line.strip()
            if text:  # 忽略空行
                target_texts.append(text)
                
        print(f"已載入 {len(target_texts)} 個目標文字")
        for i, text in enumerate(target_texts, 1):
            print(f"  {i}. {text}")
            
        return target_texts
        
    except FileNotFoundError:
        print(f"錯誤: 找不到文字檔案 {INPUT_TEXT_PATH}")
        return []
    except Exception as e:
        print(f"錯誤: 載入文字檔案失敗 - {e}")
        return []

def perform_ocr_on_image(ocr_initialized, image_path):
    """
    使用 Tesseract OCR 對圖像執行 OCR 識別
    """
    try:
        print(f"  使用 Tesseract OCR 識別圖像: {os.path.basename(image_path)}")
        
        # 讀取圖像
        image = Image.open(image_path)
        
        # 使用 Tesseract 進行 OCR，獲取詳細數據
        # 使用 image_to_data 獲取邊界框和信心度
        ocr_data = pytesseract.image_to_data(
            image, 
            lang=OCR_LANGUAGE,
            output_type=pytesseract.Output.DICT,
            config='--psm 6'  # PSM 6: 假設單一均勻文本塊
        )
        
        ocr_texts = []
        n_boxes = len(ocr_data['text'])
        
        # 遍歷所有識別結果
        for i in range(n_boxes):
            # 獲取信心度（Tesseract 返回 -1 表示未識別）
            confidence = float(ocr_data['conf'][i])
            text = ocr_data['text'][i].strip()
            
            # 過濾空白和低信心度結果
            if text and confidence > 0:
                # 轉換信心度為 0-1 範圍
                confidence_normalized = confidence / 100.0
                
                if confidence_normalized >= OCR_CONFIDENCE_THRESHOLD:
                    # 獲取邊界框
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    # 計算四角座標
                    bbox_points = [
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ]
                    
                    ocr_texts.append({
                        'text': text,
                        'confidence': confidence_normalized,
                        'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                        'polygon': bbox_points
                    })
                    
                    print(f"    [OK] 識別到: '{text}' (信心度: {confidence_normalized:.3f})")
                else:
                    print(f"    [!] 低信心度: '{text}' (信心度: {confidence_normalized:.3f}) - 已過濾")
        
        print(f"  識別結果: 共 {len(ocr_texts)} 個文字（信心度 >= {OCR_CONFIDENCE_THRESHOLD}）")
        return ocr_texts
        
    except Exception as e:
        print(f"  [ERROR] Tesseract OCR 處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return []

def match_texts(ocr_results, target_texts):
    """
    匹配 OCR 結果與目標文字（改進版 - 支援精確的多詞組合匹配）
    """
    import re
    matches = []
    
    print(f"\n  開始文字匹配 (共 {len(target_texts)} 個目標文字)")
    
    for target_text in target_texts:
        found = False
        best_match = None
        best_confidence = 0
        best_word_count = 999  # 優先選擇詞數最少的匹配
        
        target_clean = target_text.strip().lower()
        target_alphanum = re.sub(r'[^a-z0-9]', '', target_clean)
        target_words = target_clean.split()
        
        # 方法 1: 單個 OCR 結果匹配
        for ocr_result in ocr_results:
            detected_text = ocr_result['text']
            confidence = ocr_result['confidence']
            detected_clean = detected_text.strip().lower()
            detected_alphanum = re.sub(r'[^a-z0-9]', '', detected_clean)
            
            match = False
            match_type = ""
            
            # 1. 完全匹配
            if target_clean == detected_clean or target_alphanum == detected_alphanum:
                match = True
                match_type = "完全匹配"
            # 2. 包含匹配（單個OCR結果包含完整目標文字）
            elif target_clean in detected_clean or target_alphanum in detected_alphanum:
                match = True
                match_type = "包含匹配"
            
            if match:
                if confidence > best_confidence or (confidence == best_confidence and 1 < best_word_count):
                    best_match = {
                        'target_text': target_text,
                        'detected_text': detected_text,
                        'confidence': confidence,
                        'bbox': ocr_result['bbox'],
                        'polygon': ocr_result['polygon'],
                        'match_type': match_type
                    }
                    best_confidence = confidence
                    best_word_count = 1
                    found = True
        
        # 方法 2: 組合多個 OCR 結果（如果單個匹配失敗且目標有多個單詞）
        if not found and len(target_words) > 1:
            # 按 Y 座標排序（同一行的文字）
            sorted_results = sorted(ocr_results, key=lambda x: (x['bbox']['y'], x['bbox']['x']))
            
            # 嘗試組合連續的 OCR 結果
            for i in range(len(sorted_results)):
                combined_text = []
                combined_boxes = []
                
                # 嘗試不同長度的組合（從目標詞數到目標詞數+2）
                for j in range(i, min(i + len(target_words) + 3, len(sorted_results))):
                    ocr_text = sorted_results[j]['text'].strip()
                    if not ocr_text:
                        continue
                    
                    combined_text.append(ocr_text)
                    combined_boxes.append(sorted_results[j])
                    
                    # 組合文字
                    combined_str = ' '.join(combined_text).lower()
                    combined_alphanum = re.sub(r'[^a-z0-9]', '', combined_str)
                    
                    # 計算匹配程度
                    exact_match = (target_clean == combined_str) or (target_alphanum == combined_alphanum)
                    contains_match = (target_clean in combined_str) or (target_alphanum in combined_alphanum)
                    
                    # 檢查是否為精確匹配（包含目標但不包含太多額外內容）
                    if exact_match or contains_match:
                        # 計算多餘詞數
                        extra_words = len(combined_text) - len(target_words)
                        
                        # 只接受精確匹配或稍微多一點的匹配
                        if exact_match or (contains_match and extra_words <= 1):
                            # 計算平均信心度
                            avg_confidence = sum(box['confidence'] for box in combined_boxes) / len(combined_boxes)
                            
                            # 優先選擇信心度高且詞數少的匹配
                            is_better = False
                            if not found:
                                is_better = True
                            elif exact_match and best_match['match_type'].startswith('組合'):
                                is_better = True
                            elif len(combined_text) < best_word_count:
                                is_better = True
                            elif len(combined_text) == best_word_count and avg_confidence > best_confidence:
                                is_better = True
                            
                            if is_better:
                                # 計算組合邊界框
                                min_x = min(box['bbox']['x'] for box in combined_boxes)
                                min_y = min(box['bbox']['y'] for box in combined_boxes)
                                max_x = max(box['bbox']['x'] + box['bbox']['width'] for box in combined_boxes)
                                max_y = max(box['bbox']['y'] + box['bbox']['height'] for box in combined_boxes)
                                
                                match_type = "組合匹配" if exact_match else "包含組合匹配"
                                
                                best_match = {
                                    'target_text': target_text,
                                    'detected_text': ' '.join(combined_text),
                                    'confidence': avg_confidence,
                                    'bbox': {'x': min_x, 'y': min_y, 'width': max_x - min_x, 'height': max_y - min_y},
                                    'polygon': [
                                        [min_x, min_y],
                                        [max_x, min_y],
                                        [max_x, max_y],
                                        [min_x, max_y]
                                    ],
                                    'match_type': f"{match_type}({len(combined_boxes)}個詞)"
                                }
                                best_confidence = avg_confidence
                                best_word_count = len(combined_text)
                                found = True
        
        if found:
            matches.append(best_match)
            print(f"  ✓ 找到匹配: '{target_text}' -> '{best_match['detected_text']}' ({best_match['match_type']}, 信心度: {best_confidence:.3f})")
        else:
            print(f"  ✗ 未找到: '{target_text}'")
    
    return matches

def draw_text_detections(image, matches):
    """
    在圖像上繪製文字檢測結果（僅紅色底線）
    """
    result_image = image.copy()
    
    for match in matches:
        bbox = match['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        # 繪製紅色底線（在文字底部）
        line_y = y + h - 2
        cv2.line(result_image, (x, line_y), (x + w, line_y), BBOX_COLOR, BBOX_THICKNESS)
    
    return result_image

def save_text_detection_results(image, matches, target_filename):
    """
    保存文字檢測結果
    """
    # 保存結果圖像
    result_image_path = os.path.join(RESULT_PATH, f"text_result_{target_filename}")
    cv2.imwrite(result_image_path, image)
    print(f"結果圖像已保存: {result_image_path}")
    
    # 準備JSON結果
    json_result = {
        'target_image': target_filename,
        'total_matches': len(matches),
        'matches': []
    }
    
    for match in matches:
        match_info = {
            'target_text': match['target_text'],
            'detected_text': match['detected_text'],
            'confidence': float(match['confidence']),
            'position': {
                'x': int(match['bbox']['x']),
                'y': int(match['bbox']['y']),
                'width': int(match['bbox']['width']),
                'height': int(match['bbox']['height'])
            }
        }
        json_result['matches'].append(match_info)
    
    # 保存JSON結果
    json_result_path = os.path.join(RESULT_PATH, f"text_result_{Path(target_filename).stem}.json")
    with open(json_result_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)
    
    print(f"JSON結果已保存: {json_result_path}")
    return json_result

def generate_text_detection_report(all_results, target_texts):
    """
    生成文字檢測報告
    """
    report_data = []
    
    # 遍歷每個目標圖像的結果
    for result in all_results:
        target_image = result['target_image']
        matches = result['matches']
        
        # 為每個目標文字創建一行記錄
        for target_text in target_texts:
            # 尋找該文字的檢測結果
            found_matches = [m for m in matches if m['target_text'] == target_text]
            
            if found_matches:
                # 如果找到多個，取信心度最高的
                best_match = max(found_matches, key=lambda x: x['confidence'])
                report_data.append({
                    '目標圖像': target_image,
                    '目標文字': target_text,
                    '檢測狀態': '成功',
                    '識別文字': best_match['detected_text'],
                    '信心度': f"{best_match['confidence']:.3f}",
                    '位置(x,y)': f"({best_match['position']['x']}, {best_match['position']['y']})",
                    '尺寸(w×h)': f"{best_match['position']['width']}×{best_match['position']['height']}"
                })
            else:
                # 沒有檢測到
                report_data.append({
                    '目標圖像': target_image,
                    '目標文字': target_text,
                    '檢測狀態': '失敗',
                    '識別文字': 'N/A',
                    '信心度': 'N/A',
                    '位置(x,y)': 'N/A',
                    '尺寸(w×h)': 'N/A'
                })
    
    # 創建DataFrame並保存
    df = pd.DataFrame(report_data)
    report_path = os.path.join(RESULT_PATH, "text_detection_report.csv")
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"文字檢測報告已保存: {report_path}")
    
    return df

def process_all_targets():
    """
    處理所有目標圖像
    """
    print("=== 開始文字辨識程式 ===")
    print(f"參數設定:")
    print(f"  OCR信心度閾值: {OCR_CONFIDENCE_THRESHOLD}")
    print(f"  忽略大小寫: {IGNORE_CASE}")
    print(f"  匹配模式: {TEXT_MATCH_MODE}")
    
    # 創建結果目錄
    create_result_directory()
    
    # 初始化 OCR
    ocr = initialize_ocr()
    if not ocr:
        print("錯誤: 無法初始化 Tesseract OCR！")
        print("建議解決方案:")
        print("1. 下載安裝 Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. 確保安裝路徑為: C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
        return None
    
    # 載入目標文字
    target_texts = load_target_texts()
    if not target_texts:
        print("錯誤: 沒有找到任何目標文字！")
        return None
    
    # 獲取所有目標圖像
    target_files = glob.glob(os.path.join(INPUT_TARGET_PATH, "*"))
    target_files = [f for f in target_files if os.path.isfile(f)]
    
    if not target_files:
        print("錯誤: 沒有找到任何目標圖像！")
        return None
    
    print(f"\n找到 {len(target_files)} 個目標圖像")
    
    # 處理每個目標圖像
    all_results = []
    
    for target_file in target_files:
        print(f"\n正在處理: {os.path.basename(target_file)}")
        
        # 讀取目標圖像
        target_image = cv2.imread(target_file)
        if target_image is None:
            print(f"警告: 無法載入圖像 {target_file}")
            continue
        
        # 執行OCR識別
        print("  正在執行OCR識別...")
        ocr_results = perform_ocr_on_image(ocr, target_file)
        print(f"  OCR識別到 {len(ocr_results)} 個文字區域")
        
        # 匹配目標文字
        print("  正在匹配目標文字...")
        matches = match_texts(ocr_results, target_texts)
        
        print(f"  匹配結果: 找到 {len(matches)} 個符合的文字")
        
        # 繪製檢測結果
        result_image = draw_text_detections(target_image, matches)
        
        # 保存結果
        target_filename = os.path.basename(target_file)
        json_result = save_text_detection_results(result_image, matches, target_filename)
        all_results.append(json_result)
    
    # 生成檢測報告
    if all_results:
        detection_df = generate_text_detection_report(all_results, target_texts)
    
    # 保存總結報告
    summary_report = {
        'processing_summary': {
            'total_targets_processed': len(all_results),
            'total_target_texts': len(target_texts),
            'parameters': {
                'ocr_confidence_threshold': OCR_CONFIDENCE_THRESHOLD,
                'ignore_case': IGNORE_CASE,
                'text_match_mode': TEXT_MATCH_MODE
            }
        },
        'results': all_results
    }
    
    summary_path = os.path.join(RESULT_PATH, "text_summary_report.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 處理完成 ===")
    print(f"總結報告已保存: {summary_path}")
    print(f"共處理 {len(all_results)} 個目標圖像")
    print(f"檢測 {len(target_texts)} 個目標文字")
    
    return all_results

def main():
    """主函數"""
    try:
        results = process_all_targets()
        print("\n文字辨識程式執行完成！")
        return results
    except Exception as e:
        print(f"程式執行時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
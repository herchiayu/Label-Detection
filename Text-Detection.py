import cv2
import numpy as np
import os
import json
from pathlib import Path
import glob
import pandas as pd
import easyocr

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

# EasyOCR 語言設定
OCR_LANGUAGES = ['en', 'es', 'fr', 'la']  # 英文、西班牙文、法文、拉丁文

def create_result_directory():
    """創建結果目錄"""
    os.makedirs(RESULT_PATH, exist_ok=True)

def initialize_ocr():
    """
    初始化 EasyOCR
    """
    print("正在初始化 EasyOCR...")
    try:
        # 初始化 EasyOCR，支援多語言，優化參數
        reader = easyocr.Reader(
            OCR_LANGUAGES, 
            gpu=False,
            model_storage_directory='easyocr_models',
            download_enabled=True
        )
        print(f"✓ EasyOCR 初始化成功 (語言: {', '.join(OCR_LANGUAGES)})")
        print(f"✓ 信心度閾值設為: {OCR_CONFIDENCE_THRESHOLD}")
        return reader
    except Exception as e:
        print(f"✗ EasyOCR 初始化失敗: {e}")
        print("請安裝 EasyOCR: pip install easyocr")
        return None

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

def perform_ocr_on_image(reader, image_path):
    """
    使用 EasyOCR 對圖像執行 OCR 識別
    """
    try:
        print(f"  使用 EasyOCR 識別圖像: {os.path.basename(image_path)}")
        
        # 執行 OCR，使用優化參數
        results = reader.readtext(
            image_path,
            detail=1,
            paragraph=False,
            width_ths=0.7,
            height_ths=0.7,
            decoder='greedy',
            beamWidth=5,
            batch_size=1
        )
        
        ocr_texts = []
        for result in results:
            # EasyOCR 格式: (bbox, text, confidence)
            bbox_points = result[0]  # 四個角點座標
            text = result[1]
            confidence = result[2]
            
            # 只保留信心度達標的結果
            if confidence >= OCR_CONFIDENCE_THRESHOLD:
                # 計算邊界框
                x_coords = [point[0] for point in bbox_points]
                y_coords = [point[1] for point in bbox_points]
                
                x = int(min(x_coords))
                y = int(min(y_coords))
                w = int(max(x_coords) - min(x_coords))
                h = int(max(y_coords) - min(y_coords))
                
                ocr_texts.append({
                    'text': text.strip(),
                    'confidence': confidence,
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'polygon': bbox_points
                })
                
                print(f"    ✓ 識別到: '{text}' (信心度: {confidence:.3f})")
        
        # 顯示所有低信心度的結果
        for result in results:
            text = result[1].strip()
            confidence = result[2]
            if confidence < OCR_CONFIDENCE_THRESHOLD and text:
                print(f"    ⚠ 低信心度: '{text}' (信心度: {confidence:.3f}) - 已過濾")
        
        return ocr_texts
        
    except Exception as e:
        print(f"  EasyOCR 處理失敗: {e}")
        return []

def match_texts(ocr_results, target_texts):
    """
    匹配 OCR 結果與目標文字（改進版）
    """
    matches = []
    
    print(f"\n  開始文字匹配 (共 {len(target_texts)} 個目標文字)")
    
    for target_text in target_texts:
        found = False
        best_match = None
        best_confidence = 0
        
        for ocr_result in ocr_results:
            detected_text = ocr_result['text']
            confidence = ocr_result['confidence']
            
            # 文字匹配（忽略大小寫和前後空白）
            target_clean = target_text.strip().lower()
            detected_clean = detected_text.strip().lower()
            
            match = False
            match_type = ""
            
            # 1. 完全匹配
            if target_clean == detected_clean:
                match = True
                match_type = "完全匹配"
            # 2. 包含匹配（目標文字包含在識別文字中）
            elif target_clean in detected_clean:
                match = True
                match_type = "包含匹配"
            # 3. 被包含匹配（識別文字包含在目標文字中）
            elif detected_clean in target_clean:
                match = True
                match_type = "部分匹配"
            # 4. 模糊匹配（去除特殊字符後比較）
            else:
                import re
                target_alphanum = re.sub(r'[^a-z0-9]', '', target_clean)
                detected_alphanum = re.sub(r'[^a-z0-9]', '', detected_clean)
                if target_alphanum and detected_alphanum:
                    if target_alphanum == detected_alphanum:
                        match = True
                        match_type = "模糊匹配"
                    elif len(target_alphanum) > 5 and len(detected_alphanum) > 5:
                        # 長文字的相似度匹配
                        common_chars = len(set(target_alphanum) & set(detected_alphanum))
                        similarity = common_chars / max(len(target_alphanum), len(detected_alphanum))
                        if similarity > 0.7:
                            match = True
                            match_type = f"相似匹配({similarity:.2f})"
            
            if match and confidence > best_confidence:
                best_match = {
                    'target_text': target_text,
                    'detected_text': detected_text,
                    'confidence': confidence,
                    'bbox': ocr_result['bbox'],
                    'polygon': ocr_result['polygon'],
                    'match_type': match_type
                }
                best_confidence = confidence
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
        print("錯誤: 無法初始化 EasyOCR！")
        print("建議解決方案:")
        print("1. 執行: pip install easyocr")
        print("2. 確保有網路連線（首次使用需下載模型）")
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
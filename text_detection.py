"""
OCR 文字比對程式 - 多語言支援版本
功能: 圖片逐一處理 -> 多語言OCR -> 語言匹配 -> 視覺標記 -> 結果報告
"""

import os, csv, time, json, sys
from PIL import Image, ImageDraw
from text_extraction import main as text_extraction_main

# 全域參數
INPUT_TARGET_DIR = r"input data\target"
INPUT_TEXT_FILE = r"input data\text\Checklist_listed_text.txt"
RESULT_DIR = r"result"
OCR_CONFIDENCE_THRESHOLD = 0.3
IGNORED_CHARS = ['=']

def load_images():
    """載入圖片檔案"""
    print(f"載入圖片: {INPUT_TARGET_DIR}")
    if not os.path.exists(INPUT_TARGET_DIR):
        print(f"  [ERROR] 目錄不存在: {INPUT_TARGET_DIR}")
        return []
    
    formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    images = [os.path.join(INPUT_TARGET_DIR, f) for f in os.listdir(INPUT_TARGET_DIR) 
             if os.path.splitext(f.lower())[1] in formats]
    print(f"  [OK] 找到 {len(images)} 張圖片")
    return images

def parse_target_texts():
    """解析目標文字，回傳目標列表和語言集合"""
    print(f"解析目標文字: {INPUT_TEXT_FILE}")
    if not os.path.exists(INPUT_TEXT_FILE):
        print(f"  [ERROR] 檔案不存在: {INPUT_TEXT_FILE}")
        return [], set()
    
    targets = []
    languages = set()
    
    with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            if ',' in line:
                text, language = line.rsplit(',', 1)
                text, language = text.strip(), language.strip()
            else:
                text, language = line, 'eng'
                print(f"  [WARNING] 第{line_num}行缺少語言標記，預設為eng")
            
            targets.append({'text': text, 'language': language, 'line_number': line_num})
            languages.add(language)
    
    print(f"  [OK] 目標: {len(targets)}個, 語言: {sorted(languages)}")
    return targets, languages

def run_multi_language_ocr(image_path, languages):
    """對單張圖片執行多語言OCR"""
    image_name = os.path.basename(image_path)
    print(f"  多語言OCR處理: {image_name}")
    
    # 對單張圖片執行OCR處理
    for language in languages:
        print(f"    {language.upper()}...", end=" ")
        try:
            success = text_extraction_main(
                test_image=image_path,
                test_language=language,
                confidence_threshold=OCR_CONFIDENCE_THRESHOLD,
                result_path=RESULT_DIR
            )
            
            if success:
                print("✓")
            else:
                print("✗")
        except Exception as e:
            print(f"✗ ({str(e)[:30]})")

def match_texts_comprehensive(image_name, targets):
    """綜合匹配分析，生成詳細JSON資料"""
    print(f"  文字匹配分析...", end=" ")
    image_basename = os.path.splitext(image_name)[0]
    
    # 內部輔助函數
    def calculate_bbox_from_words(words):
        """從單詞列表計算統一邊界框"""
        if not words:
            return None
        
        min_x = min(w['bbox']['x'] for w in words)
        max_x = max(w['bbox']['x'] + w['bbox']['width'] for w in words)
        min_y = min(w['bbox']['y'] for w in words)
        max_y = max(w['bbox']['y'] + w['bbox']['height'] for w in words)
        
        return {
            'x': min_x,
            'y': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }
    
    def normalize_text(text):
        """文字正規化：移除特殊字符並標準化空格"""
        normalized = text
        for char in IGNORED_CHARS:
            normalized = normalized.replace(char, ' ')
        return ' '.join(normalized.split())
    
    def find_target_in_words(target_text, words):
        """在單詞列表中尋找目標文字的精確邊界框"""
        target_normalized = normalize_text(target_text)
        target_words = target_normalized.split()
        
        if len(target_words) == 0:
            return None
        
        # 建立完整的單詞文字串列
        word_texts = [normalize_text(w.get('text', '')) for w in words]
        
        # 尋找目標文字的起始位置
        for i in range(len(word_texts)):
            # 檢查從第i個單詞開始是否匹配目標文字
            if i + len(target_words) <= len(word_texts):
                match_words = word_texts[i:i+len(target_words)]
                if match_words == target_words:
                    # 找到匹配，使用統一函數計算邊界框
                    matched_words = words[i:i+len(target_words)]
                    return calculate_bbox_from_words(matched_words)
        
        return None
    
    def calculate_target_bbox(target_text, full_text, full_bbox, words=None):
        """計算目標文字在完整文字中的邊界框位置"""
        # 如果有單詞級邊界框，優先使用精確匹配
        if words:
            target_bbox = find_target_in_words(target_text, words)
            if target_bbox:
                return target_bbox
        
        # 找到目標文字在完整文字中的起始位置
        start_index = full_text.find(target_text)
        if start_index == -1:
            return full_bbox  # 如果找不到，返回完整邊界框
        
        # 計算目標文字的長度比例
        target_length = len(target_text)
        full_length = len(full_text)
        
        if full_length == 0:
            return full_bbox
        
        # 計算相對位置和寬度
        start_ratio = start_index / full_length
        length_ratio = target_length / full_length
        
        # 計算新的邊界框
        new_x = full_bbox['x'] + int(full_bbox['width'] * start_ratio)
        new_width = max(int(full_bbox['width'] * length_ratio), 10)  # 最小寬度10像素
        
        return {
            'x': new_x,
            'y': full_bbox['y'],
            'width': new_width,
            'height': full_bbox['height']
        }
    
    def calculate_combo_word_positions(target_normalized, combo_texts):
        """計算組合匹配中目標文字在各行的位置"""
        word_positions = []
        target_words = target_normalized.split()
        
        # 建立所有行的詞語列表
        all_words = []
        
        for line_idx, line_data in enumerate(combo_texts):
            line_words = line_data.get('words', [])
            for word in line_words:
                all_words.append({
                    'text': normalize_text(word.get('text', '')),
                    'bbox': word.get('bbox', {}),
                    'line_idx': line_idx
                })
        
        # 在所有詞語中尋找目標文字序列
        for i in range(len(all_words) - len(target_words) + 1):
            match_words = [w['text'] for w in all_words[i:i+len(target_words)]]
            if match_words == target_words:
                # 找到匹配，按行組織位置
                matched_words = all_words[i:i+len(target_words)]
                
                # 按行分組
                line_groups = {}
                for word in matched_words:
                    line_idx = word['line_idx']
                    if line_idx not in line_groups:
                        line_groups[line_idx] = []
                    line_groups[line_idx].append(word)
                
                # 為每行計算邊界框
                for line_idx, words_in_line in line_groups.items():
                    if words_in_line:
                        line_text = ' '.join([w['text'] for w in words_in_line])
                        line_bbox = calculate_bbox_from_words(words_in_line)
                        
                        word_positions.append({
                            'text': line_text,
                            'bbox': line_bbox
                        })
                break
        
        return word_positions
    
    def add_visual_markup(target_text, bbox, match_type, segment_text=None):
        """統一的視覺標記添加函數"""
        markup_data = {
            'target_text': target_text,
            'bbox': bbox,
            'match_type': match_type
        }
        if segment_text:
            markup_data['segment_text'] = segment_text
        comprehensive_data['visual_markup_data']['markup_positions'].append(markup_data)
    
    def process_match(target_text, extracted_text, confidence, match_type, processing_note, target_bbox, word_positions=None):
        """統一處理匹配結果"""
        match_result.update({
            'status': '成功',
            'extracted_text': extracted_text,
            'confidence': confidence,
            'match_type': match_type,
            'processing_note': processing_note
        })
        
        # 添加視覺標記
        if match_type == '組合匹配' and word_positions:
            for word_pos in word_positions:
                add_visual_markup(target_text, word_pos['bbox'], match_type, word_pos['text'])
        else:
            add_visual_markup(target_text, target_bbox, match_type)
        
        return True
    
    # 詳細匹配資料結構
    comprehensive_data = {
        'image_info': {
            'filename': image_name,
            'basename': image_basename,
            'processing_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_targets': len(targets)
        },
        'matching_results': {
            'matches': [],
            'summary_stats': {'total_targets': len(targets), 'matched_count': 0, 'failed_count': 0}
        },
        'visual_markup_data': {
            'markup_positions': []
        }
    }
    
    for target in targets:
        target_text, target_language = target['text'], target['language']
        json_file = os.path.join(RESULT_DIR, f"{image_basename}_ocr_text_{target_language}.json")
        
        match_result = {
            'target_text': target_text,
            'language': target_language,
            'ocr_source_file': os.path.basename(json_file) if os.path.exists(json_file) else 'N/A',
            'status': '失敗',
            'extracted_text': 'N/A',
            'confidence': 'N/A',
            'match_type': 'N/A',
            'processing_note': '無處理'
        }
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)
                
                target_normalized = normalize_text(target_text)
                
                for img_info in ocr_data['images']:
                    matched = False
                    
                    # 單行匹配
                    for extracted in img_info['texts']:
                        text, confidence, bbox = extracted['text'], extracted['confidence'], extracted['bbox']
                        words = extracted.get('words', [])
                        text_normalized = normalize_text(text)
                        
                        processing_note = '忽略特殊字符/空格標準化' if (target_text != target_normalized or text != text_normalized) else '無處理'
                        
                        if target_normalized == text_normalized:
                            match_type = '完全匹配'
                            target_bbox = bbox
                        elif target_normalized in text_normalized:
                            match_type = '包含匹配'
                            target_bbox = calculate_target_bbox(target_normalized, text_normalized, bbox, words)
                        else:
                            continue
                        
                        matched = process_match(target_text, text, confidence, match_type, processing_note, target_bbox)
                        break
                    
                    # 組合匹配 (如果單行匹配失敗)
                    if not matched:
                        for combo_size in [2, 3]:  # 2行組合, 3行組合
                            if matched:  # 如果已經匹配，跳出外層循環
                                break
                            for i in range(len(img_info['texts']) - combo_size + 1):
                                combo_texts = img_info['texts'][i:i+combo_size]
                                combined_text = ' '.join([t['text'] for t in combo_texts])
                                combined_normalized = normalize_text(combined_text)
                                
                                if target_normalized in combined_normalized:
                                    word_positions = calculate_combo_word_positions(target_normalized, combo_texts)
                                    if word_positions:
                                        first_bbox = combo_texts[0]['bbox']
                                        avg_confidence = sum(t['confidence'] for t in combo_texts) / len(combo_texts)
                                        matched = process_match(target_text, combined_text, avg_confidence, '組合匹配', '無處理', first_bbox, word_positions)
                                        break
                    
                    if matched:
                        break
                        
            except Exception as e:
                print(f"      [ERROR] 讀取{os.path.basename(json_file)}失敗: {e}")
        
        comprehensive_data['matching_results']['matches'].append(match_result)
        
        # 更新統計
        stats = comprehensive_data['matching_results']['summary_stats']
        if match_result['status'] == '成功':
            stats['matched_count'] += 1
        else:
            stats['failed_count'] += 1
    
    # 保存詳細匹配資料為JSON
    comprehensive_json_file = os.path.join(RESULT_DIR, f"{image_basename}_comprehensive_match_data.json")
    with open(comprehensive_json_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_data, f, ensure_ascii=False, indent=2)
    
    success_count = comprehensive_data['matching_results']['summary_stats']['matched_count']
    print(f"{success_count}/{len(targets)}")
    return comprehensive_data, comprehensive_json_file

def generate_csv_from_comprehensive_data(comprehensive_json_file):
    """從詳細匹配JSON生成CSV報告"""
    print(f"  生成CSV報告...", end=" ")
    
    try:
        with open(comprehensive_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_basename = data['image_info']['basename']
        csv_file = os.path.join(RESULT_DIR, f"{image_basename}_text_detection_report.csv")
        
        fieldnames = ['目標文字', '語言', '匹配JSON檔', '檢測狀態', 
                      '識別文字', '信心度', '匹配類型', '處理標記']
        
        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for match in data['matching_results']['matches']:
                writer.writerow({
                    '目標文字': match['target_text'],
                    '語言': match['language'],
                    '匹配JSON檔': match['ocr_source_file'],
                    '檢測狀態': match['status'],
                    '識別文字': match['extracted_text'],
                    '信心度': match['confidence'],
                    '匹配類型': match['match_type'],
                    '處理標記': match['processing_note']
                })
        
        match_count = len(data['matching_results']['matches'])
        print(f"✓ ({match_count}筆)")
        return csv_file
        
    except Exception as e:
        print(f"✗ 錯誤: {e}")
        return None

def create_visual_markup_from_comprehensive_data(image_path, comprehensive_json_file):
    """從詳細匹配JSON在圖片上標記匹配結果"""
    image_name = os.path.basename(image_path)
    print(f"  視覺標記...", end=" ")
    
    try:
        # 讀取詳細匹配資料
        with open(comprehensive_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # 使用視覺標記資料進行繪圖
        markup_positions = data['visual_markup_data']['markup_positions']
        marked_targets = set()
        
        for markup in markup_positions:
            bbox = markup['bbox']
            x1, x2 = bbox['x'], bbox['x'] + bbox['width']
            y = bbox['y'] + bbox['height'] + 2
            draw.line([x1, y, x2, y], fill=(255, 0, 0), width=2)
            marked_targets.add(markup['target_text'])
        
        image_basename = os.path.splitext(image_name)[0]
        marked_path = os.path.join(RESULT_DIR, f"{image_basename}_marked.png")
        img.save(marked_path)
        print(f"✓ ({len(marked_targets)}個目標)")
        
    except Exception as e:
        print(f"✗ 標記失敗: {e}")

def main():
    """主程式流程"""
    print("🔍 OCR 文字比對程式\n")
    start_time = time.time()
    
    # 步驟1: 載入圖片
    image_files = load_images()
    if not image_files:
        return False
    
    # 步驟2: 解析目標文字
    targets, languages = parse_target_texts()
    if not targets:
        return False
    
    all_results = []
    
    # 步驟3-7: 逐張圖片處理
    for image_path in image_files:
        image_name = os.path.basename(image_path)
        print(f"\n📷 {image_name}")
        
        # 步驟3: 多語言OCR
        run_multi_language_ocr(image_path, languages)
        
        # 步驟4: 綜合匹配分析
        comprehensive_data, json_file = match_texts_comprehensive(image_name, targets)
        
        # 步驟5: 從JSON生成CSV報告
        generate_csv_from_comprehensive_data(json_file)
        
        # 步驟6: 從JSON生成視覺標記
        create_visual_markup_from_comprehensive_data(image_path, json_file)
        
        # 累計統計
        all_results.extend(comprehensive_data['matching_results']['matches'])
    
    # 結果統計
    total_time = time.time() - start_time
    success_count = sum(1 for r in all_results if r['status'] == '成功')
    print(f"\n✅ 完成 ({total_time:.1f}秒) - 匹配: {success_count}/{len(all_results)}")
    return True

if __name__ == "__main__":
    print("🌐 多語言 OCR 系統\n")
    
    # 檢查檔案
    checks = [
        ('OCR模組', 'text_extraction.py'),
        ('圖片目錄', INPUT_TARGET_DIR),
        ('目標文字', INPUT_TEXT_FILE)
    ]
    
    for name, path in checks:
        if not os.path.exists(path):
            print(f"❌ 找不到{name}: {path}")
            sys.exit(1)
        print(f"✓ {name}")
    
    print()
    
    # 執行主程式
    try:
        success = main()
        if success:
            print("\n程式執行成功")
        else:
            print("\n程式執行失敗")
    except KeyboardInterrupt:
        print("\n程式被中斷")
    except Exception as e:
        print(f"\n[ERROR] 程式錯誤: {e}")
import cv2
import numpy as np
import os
import json
from pathlib import Path
import glob
import pandas as pd

# ============================
# GLOBAL PARAMETERS
# ============================
# 匹配信心度閾值 (0.0-1.0)
CONFIDENCE_THRESHOLD = 0.8

# 縮放範圍設定
SCALE_MIN = 0.1      # 最小縮放比例
SCALE_MAX = 2.0      # 最大縮放比例
SCALE_STEP = 0.01     # 縮放步進

# 非極大值抑制參數
NMS_THRESHOLD = 0.6  # 重疊閾值

# 邊界框顏色設定 (BGR格式)
BBOX_COLOR = (255, 0, 0)  # 藍色
BBOX_THICKNESS = 1

# 輸入輸出路徑
INPUT_TARGET_PATH = "input data/target"
INPUT_IMAGE_PATH = "input data/image"
RESULT_PATH = "result"

def create_result_directory():
    """創建結果目錄"""
    os.makedirs(RESULT_PATH, exist_ok=True)

def load_template_images(template_path):
    """
    載入模板圖像
    返回: {模板名稱: 模板圖像} 的字典
    """
    templates = {}
    template_files = glob.glob(os.path.join(template_path, "*"))
    
    for template_file in template_files:
        if os.path.isfile(template_file):
            # 獲取檔案名稱（不含副檔名）作為模板名稱
            template_name = Path(template_file).stem
            # 以灰階模式讀取模板
            template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
            
            if template is not None:
                templates[template_name] = template
                print(f"已載入模板: {template_name} (尺寸: {template.shape})")
            else:
                print(f"警告: 無法載入模板 {template_file}")
    
    return templates

def multi_scale_template_matching(image, template, scale_range=(SCALE_MIN, SCALE_MAX, SCALE_STEP)):
    """
    多尺度模板匹配
    """
    best_match = None
    best_scale = 1.0
    best_confidence = 0
    
    h, w = template.shape
    scale_min, scale_max, scale_step = scale_range
    
    # 生成縮放比例序列
    scales = np.arange(scale_min, scale_max + scale_step, scale_step)
    
    for scale in scales:
        # 縮放模板
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 確保縮放後的模板不會超過目標圖像尺寸
        if new_w > image.shape[1] or new_h > image.shape[0]:
            continue
            
        scaled_template = cv2.resize(template, (new_w, new_h))
        
        # 執行模板匹配
        result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
        
        # 找到最佳匹配位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 更新最佳匹配
        if max_val > best_confidence:
            best_confidence = max_val
            best_match = {
                'confidence': max_val,
                'location': max_loc,
                'size': (new_w, new_h),
                'scale': scale
            }
            best_scale = scale
    
    return best_match

def non_max_suppression(detections, threshold=NMS_THRESHOLD):
    """
    非極大值抑制，移除重疊的檢測框
    """
    if len(detections) == 0:
        return []
    
    # 轉換為適合NMS的格式
    boxes = []
    scores = []
    
    for detection in detections:
        x, y = detection['location']
        w, h = detection['size']
        boxes.append([x, y, x + w, y + h])
        scores.append(detection['confidence'])
    
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    
    # 執行NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                              CONFIDENCE_THRESHOLD, threshold)
    
    # 返回篩選後的檢測結果
    filtered_detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            filtered_detections.append(detections[i])
    
    return filtered_detections

def detect_templates_in_image(target_image, templates):
    """
    在目標圖像中檢測所有模板
    """
    all_detections = []
    
    # 轉換為灰階
    if len(target_image.shape) == 3:
        gray_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_target = target_image
    
    for template_name, template in templates.items():
        print(f"正在檢測模板: {template_name}")
        
        # 執行多尺度匹配
        match_result = multi_scale_template_matching(gray_target, template)
        
        if match_result and match_result['confidence'] >= CONFIDENCE_THRESHOLD:
            detection = {
                'template_name': template_name,
                'confidence': match_result['confidence'],
                'location': match_result['location'],
                'size': match_result['size'],
                'scale': match_result['scale']
            }
            all_detections.append(detection)
            print(f"  檢測到 {template_name}，信心度: {match_result['confidence']:.3f}")
    
    # 執行非極大值抑制
    filtered_detections = non_max_suppression(all_detections)
    
    return filtered_detections

def draw_detections(image, detections):
    """
    在圖像上繪製檢測結果
    """
    result_image = image.copy()
    
    for detection in detections:
        x, y = detection['location']
        w, h = detection['size']
        
        # 繪製邊界框
        cv2.rectangle(result_image, (x, y), (x + w, y + h), BBOX_COLOR, BBOX_THICKNESS)
        
        # 添加標籤
        label = f"{detection['template_name']}: {detection['confidence']:.3f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # 繪製標籤背景
        cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), BBOX_COLOR, -1)
        
        # 繪製標籤文字
        cv2.putText(result_image, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result_image

def generate_detection_report(all_results, templates):
    """
    生成檢測報告表格
    """
    report_data = []
    
    # 取得所有模板名稱
    template_names = list(templates.keys())
    
    # 遍歷每個目標圖像的結果
    for result in all_results:
        target_image = result['target_image']
        detections = result['detections']
        
        # 為每個模板創建一行記錄
        for template_name in template_names:
            # 尋找該模板的檢測結果
            found_detections = [d for d in detections if d['template_name'] == template_name]
            
            if found_detections:
                # 如果找到多個，取信心度最高的
                best_detection = max(found_detections, key=lambda x: x['confidence'])
                report_data.append({
                    '目標圖像': target_image,
                    '圖示名稱': template_name,
                    '辨識狀態': '成功',
                    '信心度': f"{best_detection['confidence']:.3f}",
                    '位置(x,y)': f"({best_detection['bounding_box']['x']}, {best_detection['bounding_box']['y']})",
                    '尺寸(w×h)': f"{best_detection['bounding_box']['width']}×{best_detection['bounding_box']['height']}",
                    '縮放比例': f"{best_detection['scale']:.2f}",
                    '檢測數量': len(found_detections)
                })
            else:
                # 沒有檢測到
                report_data.append({
                    '目標圖像': target_image,
                    '圖示名稱': template_name,
                    '辨識狀態': '失敗',
                    '信心度': 'N/A',
                    '位置(x,y)': 'N/A',
                    '尺寸(w×h)': 'N/A',
                    '縮放比例': 'N/A',
                    '檢測數量': 0
                })
    
    # 創建DataFrame
    df = pd.DataFrame(report_data)
    
    return df

def save_detection_report(df):
    """
    保存檢測報告
    """
    # 保存詳細報告
    report_path = os.path.join(RESULT_PATH, "detection_report.csv")
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"檢測報告已保存: {report_path}")
    
    return df

def save_results(image, detections, target_filename):
    """
    保存檢測結果
    """
    # 保存結果圖像
    result_image_path = os.path.join(RESULT_PATH, f"result_{target_filename}")
    cv2.imwrite(result_image_path, image)
    print(f"結果圖像已保存: {result_image_path}")
    
    # 準備JSON結果
    json_result = {
        'target_image': target_filename,
        'total_detections': len(detections),
        'detections': []
    }
    
    for detection in detections:
        x, y = detection['location']
        w, h = detection['size']
        
        detection_info = {
            'template_name': detection['template_name'],
            'confidence': float(detection['confidence']),
            'bounding_box': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            },
            'scale': float(detection['scale'])
        }
        json_result['detections'].append(detection_info)
    
    # 保存JSON結果
    json_result_path = os.path.join(RESULT_PATH, f"result_{Path(target_filename).stem}.json")
    with open(json_result_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)
    
    print(f"JSON結果已保存: {json_result_path}")
    return json_result

def process_all_targets():
    """
    處理所有目標圖像
    """
    print("=== 開始影像辨識程式 ===")
    print(f"參數設定:")
    print(f"  信心度閾值: {CONFIDENCE_THRESHOLD}")
    print(f"  縮放範圍: {SCALE_MIN} - {SCALE_MAX} (步進: {SCALE_STEP})")
    print(f"  NMS閾值: {NMS_THRESHOLD}")
    
    # 創建結果目錄
    create_result_directory()
    
    # 載入所有模板
    templates = load_template_images(INPUT_IMAGE_PATH)
    if not templates:
        print("錯誤: 沒有找到任何模板圖像！")
        return
    
    print(f"\n已載入 {len(templates)} 個模板")
    
    # 獲取所有目標圖像
    target_files = glob.glob(os.path.join(INPUT_TARGET_PATH, "*"))
    target_files = [f for f in target_files if os.path.isfile(f)]
    
    if not target_files:
        print("錯誤: 沒有找到任何目標圖像！")
        return
    
    print(f"找到 {len(target_files)} 個目標圖像")
    
    # 處理每個目標圖像
    all_results = []
    
    for target_file in target_files:
        print(f"\n正在處理: {os.path.basename(target_file)}")
        
        # 讀取目標圖像
        target_image = cv2.imread(target_file)
        if target_image is None:
            print(f"警告: 無法載入圖像 {target_file}")
            continue
        
        # 執行檢測
        detections = detect_templates_in_image(target_image, templates)
        
        print(f"檢測結果: 找到 {len(detections)} 個匹配")
        
        # 繪製檢測結果
        result_image = draw_detections(target_image, detections)
        
        # 保存結果
        target_filename = os.path.basename(target_file)
        json_result = save_results(result_image, detections, target_filename)
        all_results.append(json_result)
    
    # 生成並保存檢測報告
    if all_results:
        detection_df = generate_detection_report(all_results, templates)
        save_detection_report(detection_df)
    
    # 保存總結報告
    summary_report = {
        'processing_summary': {
            'total_targets_processed': len(all_results),
            'total_templates_used': len(templates),
            'parameters': {
                'confidence_threshold': CONFIDENCE_THRESHOLD,
                'scale_range': [SCALE_MIN, SCALE_MAX, SCALE_STEP],
                'nms_threshold': NMS_THRESHOLD
            }
        },
        'results': all_results
    }
    
    summary_path = os.path.join(RESULT_PATH, "summary_report.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 處理完成 ===")
    print(f"總結報告已保存: {summary_path}")
    print(f"共處理 {len(all_results)} 個目標圖像")
    
    return all_results

def main():
    """主函數"""
    try:
        results = process_all_targets()
        print("\n程式執行完成！")
        return results
    except Exception as e:
        print(f"程式執行時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

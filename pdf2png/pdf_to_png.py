"""
PDF轉PNG最高品質轉換程式 (簡化版)
固定600 DPI，PNG無損格式
"""

import os
import fitz  # PyMuPDF
from PIL import Image
import argparse
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import io

# 固定設定
DPI = 600
ZOOM_FACTOR = DPI / 72.0
MAX_WORKERS = 4
OUTPUT_FORMAT = "PNG"
PAGE_PADDING = 3

def find_pdf_files(directory="."):
    """搜尋PDF檔案"""
    pdf_files = list(Path(directory).rglob("*.pdf"))
    return [str(f) for f in sorted(pdf_files)]

def convert_pdf_page(pdf_path, page_num, output_dir):
    """轉換單一PDF頁面為PNG"""
    try:
        with fitz.open(pdf_path) as pdf:
            if page_num >= pdf.page_count:
                return None, f"頁面 {page_num + 1} 不存在"
            
            page = pdf[page_num]
            matrix = fitz.Matrix(ZOOM_FACTOR, ZOOM_FACTOR)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            
            # 轉換為PIL Image並保存
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            
            pdf_name = Path(pdf_path).stem
            page_str = str(page_num + 1).zfill(PAGE_PADDING)
            output_path = os.path.join(output_dir, f"{pdf_name}_page_{page_str}.png")
            
            img.save(output_path, format=OUTPUT_FORMAT, optimize=False)
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            
            return output_path, f"成功 ({file_size:.1f} MB)"
            
    except Exception as e:
        return None, f"錯誤: {str(e)}"

def convert_pdf(pdf_path, output_dir):
    """轉換整個PDF"""
    with fitz.open(pdf_path) as pdf:
        total_pages = pdf.page_count
    
    pdf_name = Path(pdf_path).stem
    print(f"\n📄 轉換: {pdf_name} ({total_pages} 頁)")
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(convert_pdf_page, pdf_path, page, output_dir): page
            for page in range(total_pages)
        }
        
        for future in futures:
            page_num = futures[future]
            output_path, status = future.result()
            
            if output_path:
                print(f"   頁面 {page_num + 1}: {status}")
                results.append(output_path)
            else:
                print(f"   頁面 {page_num + 1}: {status}")
    
    elapsed = time.time() - start_time
    print(f"   完成: {len(results)}/{total_pages} 頁 ({elapsed:.2f}秒)")
    return results

def main():
    parser = argparse.ArgumentParser(description="PDF轉PNG最高品質轉換 (600 DPI)")
    parser.add_argument('-i', '--input', default=".", help='輸入目錄或PDF檔案')
    parser.add_argument('-o', '--output', default=".", help='輸出目錄')
    args = parser.parse_args()
    
    print("🔄 PDF轉PNG轉換器 (600 DPI)")
    print(f"📂 搜尋: {os.path.abspath(args.input)}")
    print(f"📁 輸出: {os.path.abspath(args.output)}")
    
    # 搜尋PDF檔案
    if os.path.isfile(args.input) and args.input.endswith('.pdf'):
        pdf_files = [args.input]
    else:
        pdf_files = find_pdf_files(args.input)
    
    if not pdf_files:
        print(f"\n❌ 找不到PDF檔案")
        print("💡 請將PDF檔案放入當前目錄或指定-i參數")
        return
    
    print(f"✅ 找到 {len(pdf_files)} 個PDF檔案")
    
    # 轉換所有PDF
    total_start = time.time()
    all_results = []
    
    for pdf_path in pdf_files:
        results = convert_pdf(pdf_path, args.output)
        all_results.extend(results)
    
    # 顯示統計
    total_time = time.time() - total_start
    total_size = sum(os.path.getsize(f) for f in all_results) / (1024 * 1024)
    
    print(f"\n✅ 轉換完成!")
    print(f"📊 {len(pdf_files)} PDF → {len(all_results)} PNG")
    print(f"💾 總大小: {total_size:.1f} MB")
    print(f"⏱️  總時間: {total_time:.2f} 秒")

if __name__ == "__main__":
    try:
        import fitz
        from PIL import Image
        main()
    except ImportError as e:
        print("❌ 缺少套件，請執行: pip install PyMuPDF Pillow")
        print(f"錯誤: {e}")
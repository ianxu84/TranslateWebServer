from quart import request, render_template, g, send_file
import asyncio
import os
import warnings
from __init__ import app, init_translator_chatgpt, init_translator_deepl
import datetime

warnings.filterwarnings("ignore")

# ***page
# @app.route('/')
# async def main_page():
#     return await render_template("main.html")

@app.route('/')
@app.route('/pdf')
async def pdf_page():
    return await render_template("pdf.html")

# ***API
# 翻譯文字
async def translate_text(translator, text):
    loop = asyncio.get_event_loop()
    # 在線程池中執行 generate_translate_pdf 方法
    result = await loop.run_in_executor(
        None, 
        translator.translate_text, 
        text 
    )
    return result

@app.route("/translation", methods=["POST"])
@init_translator_deepl
async def get_translation():
    text = (await request.form).get('text')
    
    # 獲取客戶端的cookie紀錄
    glossary_id = request.cookies.get('GLOSSARY_ID')
    
    # 請求訪問翻譯類別的實體
    translator = getattr(g, 'deepl', None)
    if translator:
        if glossary_id:
            translator.glossary_id = glossary_id
        
        res = await translate_text(translator, text)
    else:
        res = 'No translator available'
        
    return res
    
# 取得PDF清單
@app.route('/get/pdf/translated', methods=['GET'])
async def get_translated_pdf():
    '''
    自動刪除七天前的檔案並回傳已翻譯PDF檔名
    '''
    files =  os.listdir('static/pdf/translated')
    pdf = []
    for file in files:
        file_cdate = datetime.datetime.fromtimestamp((os.stat('static/pdf/translated/' + file).st_ctime))
        date_check = datetime.datetime.now() - datetime.timedelta(days=7)   
        if file_cdate > date_check and file.endswith('.pdf'):
            ''' 七天內建立 and PDF檔案 '''
            if '_zh_en' in file:
                continue
            pdf.append(file)
        else:
            os.remove('static/pdf/translated/'+file)
    
    return sorted(pdf)

@app.route('/get/glossaries', methods=['GET'])
@init_translator_deepl
async def get_glossaries():
    '''
    取得詞彙表列表
    '''
    translator = getattr(g, "deepl", None)
    glossaries = [[glossary.glossary_id, glossary.name] for glossary in translator.list_glossaries()]
    # print(glossaries)
    # print(translator.list_glossaries())
    return glossaries

# 上傳PDF到Server
@app.route('/upload/pdf', methods=['POST'])
async def upload_pdf():
    ''''
    由Ajax呼叫
    POST方法
    相關檔案判斷在JS中已經處理
    '''
    if 'file' not in (await request.files):
        return 'No file'
    
    file = (await request.files)['file']

    await file.save("static/pdf/" + file.filename)
    return 'Upload successfully'

# 翻譯檔案
async def _translate_pdf(translator, filename, save_name, pages, use_OCR):
    '''
    filename: source file name
    save_name: destination file name, default None
    '''
    if pages == None:
        pages = 'all'
    
    loop = asyncio.get_event_loop()
    # 在執行緒池中執行 generate_translate_pdf 方法
    result = await loop.run_in_executor(
        None, 
        translator.generate_translate_pdf, 
        filename, 
        save_name, # 檔名 # 預設None 系統自動設定檔名
        pages,  # 翻譯頁
        use_OCR # 翻譯圖檔
    )
    if result :
        return 'OK'
    else:
        return result

@app.route('/translate/pdf/chatgpt/filename>/<use_OCR>', methods=['POST'])
@init_translator_chatgpt
async def translate_pdf_chatgpt(filename, use_OCR):
    '''
    停止使用
    '''
    translator = getattr(g, "chatgpt", None)
    use_OCR = True if use_OCR == '1' else False
    return await _translate_pdf(translator, filename, None, pages='all', use_OCR=use_OCR) # 全部頁: 'all, 指定頁: [5,6,7]

@app.route('/translate/pdf/deepl/<filename>/<page_input>/<glossary>', methods=['POST'])
@init_translator_deepl
async def translate_pdf_deepl(filename, page_input, glossary):
    if page_input.lower() == 'none': # 文字翻譯會是none
        use_OCR = False
    elif page_input.lower() != 'all':
        use_OCR = []
        ranges = page_input.split(',')
        try:    
            for item in ranges:
                if '-' in item:
                    start, end = map(int, item.split('-'))
                    use_OCR.extend(range(start-1, end))
                elif item.isdigit():
                    use_OCR.append(int(item)-1)
                else:
                    print('try error')
                    return "输入包含非数值型数据"
        except Exception as e:
            files = os.listdir('static/pdf')
            for file in files:
                if file.endswith('.pdf'):
                    os.remove('static/pdf/'+file)
            return "輸入格式錯誤"
    else:
        use_OCR = page_input
        
    translator = getattr(g, "deepl", None)
    translator.glossary_id = "aa299723-b86b-4a80-bb0e-5f5f9d2e5b42"  # 統一字典  如果開放讓user選擇請打開下面程式碼然後註解這行
    # if glossary != "None":
    #     translator.glossary_id = glossary
    return await _translate_pdf(translator, filename, None, pages='all', use_OCR=use_OCR) # 全部頁: 'all, 指定頁: [5,6,7]

# 下載PDF
@app.route('/download/pdf/<filename>', methods=['GET'])
async def download_pdf(filename):
    path = f"static/pdf/translated/{filename}"
    return await send_file(path, as_attachment=True)

# 刪除PDF
@app.route('/delete/pdf/<fn>', methods=['DELETE'])
async def delete_pdf(fn):
    if os.path.exists(f'static/pdf/{fn}'):
        os.remove(f'static/pdf/{fn}')
    # files = os.listdir('static/pdf')
    # for file in files:
    #     if file.endswith('.pdf'):
    #         os.remove('static/pdf/'+file)
    return 'OK'

if __name__ == "__main__":
    app.run(host="192.168.122.126", port=8080)

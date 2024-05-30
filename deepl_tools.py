from dotenv import load_dotenv, set_key, find_dotenv
from deepl import Translator
from quart import request
import pdfplumber
import fitz
import time
import os
import re
import pandas as pd
import numpy as np
import io
from PIL import Image, ImageFont, ImageDraw
import pikepdf
from pikepdf import Pdf, Stream, PdfImage, Name
from google.cloud import vision
import zlib



# ***DeepL tools
class DeepLTranslator(Translator):
    _glossary_id = None
    def __init__(self) -> None:
        load_dotenv("SETTING.env")
        key = os.getenv("DEEPLKEY")
        if not key:
            raise ValueError("未取得DeepL金鑰, 請檢查SETTING.env是否設定正確.")
        super().__init__(key)
        
    @property
    def glossary_id(self) -> str:
        '''Getter'''
        return self._glossary_id
    
    @glossary_id.setter
    def glossary_id(self, new_id: str) -> None:
        '''Setter'''
        
        self._glossary_id = new_id
    
    def create_glossary(self, name: str, entries: dict, source_lang: str="EN-US", target_lang: str="ZH") -> object:
        '''
        -> GlossaryInfo: object
            -creation_time : datetime, create datetime
            -entry_count   : int, how many entry in glossary
            -glossary_id   : str
            -name          : str
            -ready         : bool
            -source_lang   : str, language from source, default EN-US
            -target_lang   : str, language to translate, default ZH
        '''
        try:
            glossary_info = super().create_glossary(name, source_lang, target_lang, entries)
            return glossary_info
        except Exception as e:
            print(f'環境變數檔案異常: {e}')

    def list_glossaries(self) -> list:
        return super().list_glossaries()
    
    def get_glossary(self, glossary_id) -> dict:
        return super().get_glossary_entries(glossary_id)
    
    def translate_text(self,text) -> str:
        return str(super().translate_text(text, source_lang="EN", target_lang="ZH", glossary=self.glossary_id))
    
    # 翻譯檔案
    def extract_data(self, data: pd.DataFrame, page_number: int, threshold: float = 15) -> pd.DataFrame:
        '''
        Extract data from dataframe
        
        params:
            - data:        extract data source
            - page_number: page number
            - threshold:   for countting text distance
        '''
        # 將座標資料轉換成 DataFrame
        df = pd.DataFrame(data)

        # 計算相鄰字母之間的距離
        df['distance'] = df['x0'].sub( df['x1'].shift() ).abs()

        # 設定相同語句的分組
        # threshold = 3.8  # 距離閾值，根據實際情況調整
        df['sentence'] = (df['distance'] > threshold*3).cumsum() + 1
        
        # 提前计算分组
        groups = df.groupby('sentence')
        
        # 組合語句
        res = pd.DataFrame(columns=['text', 'x0', 'x1', 'y0', 'y1','new','page'])
        
        result_list = []
        for group, group_df in groups:
            result_list.append(pd.DataFrame({
                'text' : ' '.join(group_df['text']) ,  # 组合字串
                'x0'   : group_df['x0'].min()       ,  # x0 最小值
                'x1'   : group_df['x1'].max()       ,  # x1 最大值
                'y0'   : group_df.iloc[0, 3]        ,  # y 軸各值均相同 取任意值即可
                'y1'   : group_df.iloc[0, 5]        ,  
                'new'  : None                       ,  
                'page' : page_number - 1               # 送過來的值是從1開始 程式要從0開始所以減一
            }, index=[0]))
        
        res = pd.concat(result_list, ignore_index=True)
        res[['x0','x1','y0','y1']] = res[['x0','x1','y0','y1']].astype(float)
        return res
    
    def generate_translate_pdf(self, filename: str, save_name: str=None, pdf_pages='all', use_OCR=False, threshold: float=15, fontsize: int=4) -> bool:
        '''
        Generate translated PDF file
        
        params:
            -filename:  name of the PDF file, file should be in same folder
            -save_name: generated file name, it can be None
            -pdf_pages:     'all' or [1,2,3] page numbers
            -use_OCR: [1,2,3], set ocr page numbers
            -threshold: counting every sentence
            -fontsize:  PDF translated text size
            -model:     default= 'gpt-3.5-turbo-0125' ,can use 'gpt-4'
        
        note:
            1. glossaries.xls need to placed in static/dictionary
            2. key in SETTING.env same folder as python
        '''
        print("translate file start")
        path = os.path.join('static/pdf/', filename)
        if not os.path.exists(path):
            '''
            程式是非同步執行的
            不能期望檔案一定在
            '''
            time.sleep(1)
            if not os.path.exists(path):
                return '檔案上傳異常!'

        with pdfplumber.open(path) as pdf:
            result = []
            if pdf_pages == 'all':
                pages = pdf.pages
            else:
                pages = np.array(pdf.pages)[pdf_pages]
            for page in pages:
                text = page.extract_words()
                df = pd.DataFrame(text)
                result.append( self.extract_data(df, page.page_number,threshold) )
            df = pd.concat(result, ignore_index=True)
            df['text'] = df['text'].astype(str)
            # 過濾值
            mask5 = df['text'].str.contains(r'[^”\s]+ *$') # 不是”空格
            trans_text = df[ mask5 ]  
            trans_text.reset_index(drop=True,inplace=True)
            
            # # 取字典
            # RegEx
            glossaries_df = pd.read_excel('static/dictionary/glossaries.xls')
            uc = r'([\\da-zA-Z%\\"\\/\\s\\.-]+)'
            glossaries_df['英文名稱'] = glossaries_df['英文名稱'].replace(r'-\*-', uc, regex=True)
            
            # 全字典
            key = glossaries_df['英文名稱'].str.lower().str.strip().tolist()
            val = glossaries_df['中文名稱'].tolist()
            
            # 片語
            phrase_glossaries_df = glossaries_df[ glossaries_df['英文名稱'].str.contains(r'([\da-zA-Z%\"\/\s\.-]+)',regex=False) ]
            key_p = phrase_glossaries_df['英文名稱'].str.lower().str.strip().tolist()
            val_p = phrase_glossaries_df['中文名稱'].tolist()
            
            pattern = re.compile("|".join(fr'(?:(?<=\s)|(?<=\W)|^){k}(?:(?=\s)|(?=\W)|$)' for k in key), re.IGNORECASE) # 匹配前後是空白的字
            for i in range(len(trans_text['text'])):
                # 替換片語
                for j in range(len(key_p)):
                    trans_text['new'][i] = re.sub(key_p[j], val_p[j].split()[0] + ' ' + trans_text['text'][i].lower().split( key_p[j].split(r'([\da-za-z%\"\/\s\.-]+)')[0] )[-1].split(key_p[j].split(r'([\da-za-z%\"\/\s\.-]+)')[-1] )[0]  + ' ' + val_p[j].split()[-1], trans_text['text'][i].lower())
                    if trans_text['new'][i] != trans_text['text'][i].lower():
                        # print("Origin:",trans_text['new'][i])
                        # print('After:',trans_text['text'][i])
                        # print('OK')
                        break
                    # else:
                    #     print('No')
                # 替換單詞
                if trans_text.iloc[i,0].lower() in key:
                    trans_text['new'][i] = val[key.index(trans_text.iloc[i,0].lower())]
                else:
                    trans_text['new'][i] = pattern.sub(lambda x: val[key.index(x.group(0).lower())] if x.group(0).lower() in key else trans_text['new'][i], trans_text['new'][i] or trans_text['text'][i].lower()) # None 是片語 pattern.sub有BUG無法直接做萬用字的替換 會被補上\跳脫字元
                    
            # 匹配中文文字中間的空格 # 空格替換為空字串 # 多格空隔替換為一個空格
            pattern = re.compile(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])') 
            trans_text['new'] = trans_text['new'].apply(lambda x: pattern.sub('', x))
            trans_text['new'] = trans_text['new'].apply(lambda x: re.sub(r'\s+',' ', x))
            
            # **丟到DeepL翻譯**
            # 取出可能未翻譯的部分
            to_trans = trans_text[ trans_text['new'].str.contains(r'[a-zA-Z]') ] 
            to_trans = to_trans.reset_index(drop=True)
            
            # 反向條件取出已翻譯的部分
            translated = trans_text[ ~trans_text['new'].str.contains(r'[a-zA-Z]') ] # 包含英文大小寫A到Z ~反向就是不包含 同 NOT
            
            # 擷取出唯一值字串避免重複翻譯浪費Token
            key_list = to_trans['new'].unique().tolist()
            text = '\n'.join(key_list)
            value_list = self.translate_text(text).split('\n')
            
            # **取回翻譯結果合併到資料表中**
            mapping = { k1: v1 for k1, v1 in dict(zip(key_list, value_list)).items() }
            # mapping = {line.split('.')[0]: value_dic.get(line.split('.', 1)[-1], key_dic[line.split('.')[0]]) for line in lines}

            to_trans['new'] = to_trans['new'].map( mapping )
            trans_text = pd.concat( [translated, to_trans] , ignore_index=True)      

        with fitz.open(path) as pdf:
            # pyMuPdf 支持寫入PDF 故換套件讀檔寫入    
            font = fitz.Font(fontfile="static/font/kaiu.ttf")
            # 創建一個空白的圖像 取巧利用PIL的計算長度方法取得文字長度
            img = Image.new('RGB', (1, 1), color='white')
            draw = ImageDraw.Draw(img)
            imfont = ImageFont.truetype("kaiu.ttf", 4, encoding="UTF-8")
            
            # 定位起始頁開始迴圈
            page_numbers = trans_text.groupby('page').groups
            for page_number in page_numbers:
                page = pdf[ page_number ]
                page.insert_font(fontname="F0",fontbuffer=font.buffer)
                
                # 過濾出當前頁面要寫入的資料
                trans_df = trans_text[ trans_text['page'] == page_number ]
                for index, row in trans_df.iterrows():
                    # 畫白底
                    page.draw_rect( fitz.Rect( row['x0'], row['y0'], row['x1'], row['y1'] ) , # 定位四個角範圍
                                    color = ( 1, 1, 1 )                                     ,
                                    width = 0                                               ,
                                    fill  = 1                                               )
                    # 寫入翻譯字
                    page.insert_text( ( row['x0'], row['y0']+5 )    ,
                                    row['new']                      ,
                                    fontname   = 'F0'               ,
                                    fontsize   =  fontsize          )
                    
            if save_name == None:
                save_name = filename.replace(".pdf","_zh.pdf") # 取原檔名

            pdf.save(f'static/pdf/translated/{save_name}', garbage=4, deflate=True) # 處理完的結果存入新檔案
        
        # zh_en
        with fitz.open(path) as pdf:
            # pyMuPdf 支持寫入PDF 故換套件讀檔寫入    
            font = fitz.Font(fontfile="static/font/kaiu.ttf")
            # 定位起始頁開始迴圈
            page_numbers = trans_text.groupby('page').groups
            for page_number in page_numbers:
                page = pdf[ page_number ]
                page.insert_font(fontname="F0",fontbuffer=font.buffer)
                
                # 過濾出當前頁面要寫入的資料
                trans_df = trans_text[ trans_text['page'] == page_number ]
                for index, row in trans_df.iterrows():
                    # 畫白底
                    page.draw_rect( fitz.Rect( row['x0'], row['y0'], row['x1'], row['y1'] ) , # 定位四個角範圍
                                    color = ( 1, 1, 1 )                                     ,
                                    width = 0                                               ,
                                    fill  = 1                                               )
                    # 寫入原文字
                    page.insert_text( ( row['x0'], row['y0']+5 )    ,
                                    row['text'] ,
                                    fontname   = 'F0'               ,
                                    fontsize   =  fontsize          )
                    # 判斷是否有翻譯字
                    if row['text'].lower() != row['new'].lower():
                        text_length = draw.textlength( row['text'], font = imfont )
                        page.insert_text( ( row['x0'] + text_length, row['y0']+5 ) ,
                                        row['new']                      ,
                                        fontname   = 'F0'               ,
                                        fontsize   =  fontsize          ,
                                        color      =  (1, 0, 0)         )
            
            save_name = f'{save_name[:-4]}_en.pdf'
            path = f'static/pdf/translated/{save_name}'
            pdf.save(path, garbage=4, deflate=True) # 處理完的結果存入新檔案
        if use_OCR:
            self._translate_image(path, use_OCR)
        print("translate file end")
        return True
    
    def _merge_texts(self, df: pd.DataFrame, threshold: float = 15):
        '''
        Merge text from dataframe
        
        params:
            -df:        data source
            -threshold: counting every sentence and block
        '''
        sentence_texts = df
        merged_texts = []
        # 整理排序 
        sentence_texts.sort_values(by=['y0','x0'],inplace=True)
        sentence_texts['distance_x'] = sentence_texts['x0'].sub( sentence_texts['x1'].shift() ).abs()
        sentence_texts['sentence'] = (sentence_texts['distance_x'] > threshold*3).cumsum() + 1
        # 分組後再做一次排序, 整理好每個語句的順序
        sentence_texts.sort_values(by=['sentence','x0'],inplace=True)
        sentence_texts['distance_x'] = sentence_texts['x0'].sub( sentence_texts['x1'].shift() ).abs()
        sentence_texts['sentence'] = (sentence_texts['distance_x'] > threshold*3).cumsum() + 1
    
        sentences = sentence_texts.groupby("sentence")
        for sentence, sentence_df in sentences:
            sentence_df.sort_values(by=['x0'],inplace=True)
            merged_texts.append({'origin': ' '.join(sentence_df['text']).lower() ,
                                'x0'     : sentence_df['x0'].min()       ,
                                'x1'     : sentence_df['x1'].max()       ,
                                'y0'     : sentence_df['y0'].min()       ,
                                'y1'     : sentence_df['y1'].max()      })
        
        merged_texts = pd.DataFrame(merged_texts)
        return merged_texts

    def _extract_image(self, content):
        texts = content.text_annotations

        data = []

        for text in texts[1:]: # 第一個值是所有擷取範圍中的文字 不需要所以直接跳過
            x_values = [vertex.x for vertex in text.bounding_poly.vertices]
            y_values = [vertex.y for vertex in text.bounding_poly.vertices]
            
            x0, x1 = min(x_values), max(x_values)
            y0, y1 = min(y_values), max(y_values)
            
            data.append({'text': text.description ,
                        'x0'  : x0,
                        'x1'  : x1,
                        'y0'  : y0,
                        'y1'  : y1  })
        if data:
            df = pd.DataFrame(data)
            df_merged = self._merge_texts(df)
            # mask = df['text'].str.isalpha()
            # mask2 = df['text'].str.len() > 2
            # texts = df[ mask & mask2 ]
            # texts = texts.reset_index(drop=True)
            return df_merged.reset_index(drop=True)
            
        else:
            return pd.DataFrame()
        

    def _translate_image(self, path, pdf_pages):
        print("translate image start")
        with pikepdf.open(path, allow_overwriting_input=True) as pdf: # 中英文版
            path_en = path.replace('_zh_en.pdf', '_zh.pdf')
            with pikepdf.open(path_en, allow_overwriting_input=True) as pdf_en:
                # 遍歷頁面
                if pdf_pages == 'all':
                    pages = pdf.pages
                    pages_en = pdf_en.pages
                else:
                    pages = np.array(pdf.pages)[pdf_pages]
                    pages_en = np.array(pdf_en.pages)[pdf_pages]
                    
                client = vision.ImageAnnotatorClient()                    # 初始化api
                for page, page_en in zip(pages, pages_en):
                    image_list = list(page.images.keys())
                    image_list_en = list(page_en.images.keys())

                    for image, image_en in zip(image_list, image_list_en):
                        rawimage = page.images[image]
                        rawimage_en = page_en.images[image_en]
                        
                        pdfimage = PdfImage(rawimage)
                        pdfimage_en = PdfImage(rawimage_en)
                        
                        rawimage = pdfimage.obj 
                        rawimage_en = pdfimage_en.obj
                        
                        pillowimage = pdfimage.as_pil_image()
                        pillowimage_en = pdfimage_en.as_pil_image()
                        if pillowimage.mode in ('P','CMYK') or pillowimage_en in ('P','CMYK'):
                            pillowimage = pillowimage.convert("RGB")
                            pillowimage_en = pillowimage_en.convert("RGB")
                        
                        font = ImageFont.truetype('kaiu.ttf', 12, encoding="UTF-8")   # 設定字型
                        draw = ImageDraw.Draw(pillowimage)     # 準備在圖片上繪圖
                        draw_en = ImageDraw.Draw(pillowimage_en)
                        
                        buffer = io.BytesIO()                                   # 準備緩衝區
                        pillowimage.save(buffer, format="PNG")                  # 將圖片存入IO中
                        image_object = vision.Image(content=buffer.getvalue())  # 從io中取出數據建立圖片物件 
                        results = self._extract_image(client.text_detection(image=image_object)) # 取得解析出來的單字與位置  # DF欄位: text, x0, x1, y0, y1

                        if len(results) > 0:
                            # # 取字典
                            # RegEx
                            glossaries_df = pd.read_excel('static/dictionary/glossaries.xls')
                            uc = r'([\\da-zA-Z%\\"\\/\\s\\.-]+)'
                            glossaries_df['英文名稱'] = glossaries_df['英文名稱'].replace(r'-\*-', uc, regex=True)
                            
                            # 全字典
                            key = glossaries_df['英文名稱'].str.lower().str.strip().tolist()
                            val = glossaries_df['中文名稱'].tolist()
                            
                            # 片語
                            phrase_glossaries_df = glossaries_df[ glossaries_df['英文名稱'].str.contains(r'([\da-zA-Z%\"\/\s\.-]+)',regex=False) ]
                            key_p = phrase_glossaries_df['英文名稱'].str.lower().str.strip().tolist()
                            val_p = phrase_glossaries_df['中文名稱'].tolist()
                            
                            pattern = re.compile("|".join(fr'(?:(?<=\s)|(?<=\W)|^){k}(?:(?=\s)|(?=\W)|$)' for k in key), re.IGNORECASE) # 看不懂可以問問神奇海螺 我也不知道怎麼寫出來的
                            results['text'] = pd.Series(dtype='str')
                            for i in range(len(results['origin'])):
                                # 替換片語
                                for j in range(len(key_p)):
                                    
                                    results['text'][i] = re.sub(key_p[j], val_p[j].split()[0] + ' ' + results['origin'][i].lower().split( key_p[j].split(r'([\da-za-z%\"\/\s\.-]+)')[0] )[-1].split(key_p[j].split(r'([\da-za-z%\"\/\s\.-]+)')[-1] )[0]  + ' ' + val_p[j].split()[-1], results['origin'][i].lower())
                                    if results['text'][i] != results['origin'][i].lower():
                                        # print("Origin:",trans_text['new'][i])
                                        # print('After:',trans_text['text'][i])
                                        # print('OK')
                                        break
                                    # else:
                                    #     print('No')
                                # 替換單詞
                                if results.iloc[i,0].lower() in key:
                                    results['text'][i] = val[key.index(results.iloc[i,0].lower())]
                                else:
                                    results['text'][i] = pattern.sub(lambda x: val[key.index(x.group(0).lower())] if x.group(0).lower() in key else results['text'][i], results['text'][i] or results['origin'][i].lower()) # pattern.sub有BUG無法直接做萬用字的替換 會被補上\跳脫字元
                            
                            # **丟到chatGPT翻譯**
                            # 取出可能未翻譯的部分
                            to_trans = results[ results['text'].str.contains(r'[a-zA-Z]') ] 
                            to_trans = to_trans.reset_index(drop=True)
                            if len(to_trans):
                                # 反向條件取出已翻譯的部分
                                translated = results[ ~results['text'].str.contains(r'[a-zA-Z]') ]
                                
                                # 擷取出唯一值字串避免重複翻譯浪費Token
                                key_dic = { str(k):v for k, v in enumerate(to_trans['origin'].unique(), start=1)}
                                
                                # **取回翻譯結果合併到資料表中** # 將原值設為key, 翻譯字設為值以做mapping
                                value_dic = { k : v for k, v in zip(to_trans['origin'], self.translate_text( '\n'.join( to_trans['text'] ) ).split('\n')) } 
                                mapping = { v1: value_dic.get(v1, v1) for k1, v1 in key_dic.items() }
                                # mapping = {line.split('.')[0]: value_dic.get(line.split('.', 1)[-1], key_dic[line.split('.')[0]]) for line in lines}

                                to_trans['text'] = to_trans['origin'].map( mapping )
                                texts = pd.concat( [translated, to_trans] , ignore_index=True)    
                            else:
                                texts = results
                            
                            # 匹配中文文字中間的空格 # 空格替換為空字串 # 多隔空隔替換為一個空格
                            pattern = re.compile(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])')                                         # 看不懂可以問問神奇海螺 我也不知道怎麼寫出來的
                            texts['text'] = texts['text'].apply(lambda x: pattern.sub('', x))
                            texts['text'] = texts['text'].apply(lambda x: re.sub(r'\s+',' ', x))
                            
                            # 匹配包含中文字的值 判斷翻譯字是否有需要寫入
                            pattern = re.compile(r'[\u4e00-\u9fa5]')
                            for i in range(len(texts)):
                                '''
                                iloc對應欄位
                                0: origin
                                1: x0
                                2: x1
                                3: y0
                                4: y1
                                5: text
                                '''
                                # 畫白底
                                draw.rectangle( ((texts.iloc[i,1],texts.iloc[i,3]-1)     ,  # x, y值, -1補正左上角y值範圍才能完整的覆蓋掉字
                                                (texts.iloc[i,2],texts.iloc[i,4]+1))     ,  # 左上角座標、右下角座標 )
                                                fill      = (255, 255, 255)              ,
                                                outline   = None                         ,
                                                width     = 0                            )  
                                draw_en.rectangle( ((texts.iloc[i,1],texts.iloc[i,3]-1)  ,  # x, y值, -1補正左上角y值範圍才能完整的覆蓋掉字
                                                (texts.iloc[i,2],texts.iloc[i,4]+1))     ,  # 左上角座標、右下角座標 )
                                                fill      = (255, 255, 255)              ,
                                                outline   = None                         ,
                                                width     = 0                            ) 
                                # 寫原文黑字
                                draw.text( (texts.iloc[i,1]                  ,  # x軸 x0位置
                                            texts.iloc[i,3] )                ,  # y軸 y0位置
                                            texts.iloc[i,0]                  ,  # 翻譯字 text欄
                                            fill = 'black'                   ,  # 文字顏色
                                            font = font                      )  # 將文字畫入圖片
                                    
                                draw_en.text( (texts.iloc[i,1]               ,  # x軸 x0位置
                                            texts.iloc[i,3] )                ,  # y軸 y0位置
                                            texts.iloc[i,0]                  ,  # 翻譯字 text欄
                                            fill = 'black'                   ,  # 文字顏色
                                            font = font                      )  # 將文字畫入圖片
                                # 寫翻譯紅字
                                check_chinese = pattern.findall(texts.iloc[i,5])  # 檢查翻譯字是否有包含中文字 有翻譯就寫
                                if check_chinese:
                                    text_length = draw.textlength(texts.iloc[i,0], font=font)  # 取得原文字長度
                                    new_length = draw.textlength(texts.iloc[i,5], font=font, features=["-kern"])   # 取得翻譯字長度 
                                    # print(f"origin: {texts.iloc[i,0]} , origin length: {text_length}")
                                    # print(f"first spot x: {texts.iloc[i,1] + text_length} , y: {texts.iloc[i,3]}" )
                                    # print(f"text: {texts.iloc[i,5]} , new length: {new_length}")
                                    # print(f"second spot x: {texts.iloc[i,2] + new_length} , y: {texts.iloc[i,4]}")
                                    # print('-'*20)
                                    # 白底
                                    draw.rectangle( ((texts.iloc[i,1] + text_length,texts.iloc[i,3]-1)    ,                   # x, y值, -1補正左上角y值範圍才能完整的覆蓋掉字
                                                    ( texts.iloc[i,1] + text_length + new_length + 1,texts.iloc[i,4]+1))   ,  # 左上角座標、右下角座標
                                                    fill      = (255, 255, 255)                           ,
                                                    outline   = None                                      ,
                                                    width     = 0                                         )  
                                    draw_en.rectangle( ((texts.iloc[i,1] + text_length ,texts.iloc[i,3]-1),                   # x, y值, -1補正左上角y值範圍才能完整的覆蓋掉字
                                                    (    texts.iloc[i,1] + text_length + new_length + 1,texts.iloc[i,4]+1)),  # 左上角座標、右下角座標
                                                    fill      = (255, 255, 255)                           ,
                                                    outline   = None                                      ,
                                                    width     = 0                                         ) 
                                    # 紅字
                                    draw.text( (texts.iloc[i,1] + text_length                ,  # x軸 x0位置
                                                texts.iloc[i,3] )                            ,  # y軸 y0位置
                                                texts.iloc[i,5]                              ,  # 翻譯字 text欄
                                                fill = 'red'                                 ,  # 文字顏色
                                                font = font                                  )
                                    draw_en.text( (texts.iloc[i,1] + text_length             ,  # x軸 x0位置
                                                texts.iloc[i,3] )                            ,  # y軸 y0位置
                                                texts.iloc[i,5]                              ,  # 翻譯字 text欄
                                                fill = 'red'                                 ,  # 文字顏色
                                                font = font                                  )
                
                            # 設定格式    
                            rawimage.write(zlib.compress(pillowimage.tobytes()), filter=Name("/FlateDecode"))
                            rawimage_en.write(zlib.compress(pillowimage_en.tobytes()), filter=Name("/FlateDecode"))
                            rawimage.ColorSpace = Name("/DeviceRGB")
                            rawimage_en.ColorSpace = Name("/DeviceRGB")
                            rawimage.Width, rawimage.Height = pillowimage.width, pillowimage.height
                            rawimage_en.Width, rawimage_en.Height = pillowimage_en.width, pillowimage_en.height

                pdf.save(path)       
                pdf_en.save(path_en)
                
        print("translate image end")


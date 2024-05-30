import os
from dotenv import load_dotenv, set_key, find_dotenv
import pdfplumber
import pandas as pd
import numpy as np
import fitz
import time
import os
import re
from openai import OpenAI

# ***ChatGPT
class ChatgptTranslator(OpenAI):
    def __init__(self) -> None:
        load_dotenv("SETTING.env")
        key = os.getenv("CHATGPTKEY")
        if key:
            super().__init__(api_key=key)
            
        else:
            raise ValueError("遺失ChatGPT金鑰, 請檢查金鑰設定.")
        
    # 翻譯檔案
    def extract_data(self, data: pd.DataFrame, page_number: int ,threshold: float =3.8):
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
        df['sentence'] = (df['distance'] > threshold).cumsum() + 1
        
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

    def generate_translate_pdf(self, filename: str, save_name=None, pages='all', threshold: float =3.8, fontsize=5, model: str ='gpt-3.5-turbo-0125') -> bool:
        '''
        Generate translated PDF file
        
        params:
            -filename:  name of the PDF file, file should be in same folder
            -save_name: generated file name, it can be None
            -pages:     'all' or [1,2,3] page numbers
            -threshold: counting every sentence
            -fontsize:  PDF translated text size
            -model:     default= 'gpt-3.5-turbo-0125' ,can use 'gpt-4'
        
        note:
            1. glossaries.xls need to placed in static/dictionary
            2. key in SETTING.env same folder as python
        '''
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
            if pages == 'all':
                pages = pdf.pages
            else:
                pages = np.array(pdf.pages)[pages]
            for page in pages:
                text = page.extract_words()
                df = pd.DataFrame(text)

                result.append( self.extract_data(df, page.page_number,threshold) )
            df = pd.concat(result, ignore_index=True)
            df['text'] = df['text'].astype(str)

            # 過濾值
            mask5 = df['text'].str.contains(r'[^”\s]+ *$') # 不是”空格
            trans_text = df[ mask5] 
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
            
            pattern = re.compile("|".join(fr'(?:(?<=\s)|(?<=\W)|^){k}(?:(?=\s)|(?=\W)|$)' for k in key), re.IGNORECASE) # 看不懂可以問問神奇海螺 我也不知道怎麼寫出來的
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
                    trans_text['new'][i] = pattern.sub(lambda x: val[key.index(x.group(0).lower())] if x.group(0).lower() in key else trans_text['new'][i], trans_text['new'][i] or trans_text['text'][i]) # None 是片語 pattern.sub有BUG無法直接做萬用字的替換 會被補上\跳脫字元
                    
            # 匹配中文文字中間的空格 # 空格替換為空字串 # 多隔空隔替換為一個空格
            pattern = re.compile(r'(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])')                                         # 看不懂可以問問神奇海螺 我也不知道怎麼寫出來的
            trans_text['new'] = trans_text['new'].apply(lambda x: pattern.sub('', x))
            trans_text['new'] = trans_text['new'].apply(lambda x: re.sub(r'\s+',' ', x))
            
            # **丟到chatGPT翻譯**
            # 取出可能未翻譯的部分
            to_trans = trans_text[ trans_text['new'].str.contains(r'[a-zA-Z]') ] 
            to_trans = to_trans.reset_index(drop=True)
            
            # 反向條件取出已翻譯的部分
            translated = trans_text[ ~trans_text['new'].str.contains(r'[a-zA-Z]') ]
            
            # 擷取出唯一值字串避免重複翻譯浪費Token
            key_dic = { str(k):v for k, v in enumerate(to_trans['new'].unique(), start=1)}
            
            content = "以下請逐行按照序號翻譯為中文,不同行之間沒有關聯,格式按照各行的序號順序回傳\n" + "\n".join([f'{i}.{j}' for i, j in key_dic.items() ])
            
            chat = self.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                model="gpt-3.5-turbo-0125",
                # model="gpt-4",
            )
            # **取回翻譯結果合併到資料表中**
            lines = chat.choices[0].message.content.splitlines()
            value_dic = { line.split('.')[0] : line.split('.',1)[-1].strip() for line in lines }  # value只分割一次 否則會切到值裡的.
            mapping = { v1: value_dic.get(k1, v1) for k1, v1 in key_dic.items() }
            to_trans['new'] = to_trans['new'].map( mapping )
            trans_text = pd.concat( [translated, to_trans] , ignore_index=True)    

        with fitz.open(path) as pdf:
            # pyMuPdf 支持寫入PDF 故換套件讀檔寫入    
            font = fitz.Font("cjk")
            # 定位起始頁開始迴圈
            page_numbers = trans_text.groupby('page').groups
            for page_number in page_numbers:
                page = pdf[ page_number ]
                page.insert_font(fontname="F0",fontbuffer=font.buffer)
                
                # 過濾出當前頁面要寫入的資料
                trans_df = trans_text[ trans_text['page'] == page_number ]
                for index, row in trans_df.iterrows():
                    page.draw_rect( fitz.Rect( row['x0'], row['y0'], row['x1'], row['y1'] ) , # 定位四個角範圍
                                    color = ( 1, 1, 1 )                                     ,
                                    width = 0                                               ,
                                    fill  = 1                                               )
                    
                    page.insert_text( ( row['x0'], row['y0']+5 )    ,
                                    row['new']                      ,
                                    fontname   = 'F0'               ,
                                    fontsize   =  fontsize          )
            if save_name == None:
                save_name = f'{filename[:-4]}_chatgpt.pdf'  #  取原檔名加上預設後綴
            pdf.save(f'static/pdf/translated/{save_name}', garbage=4, deflate=True) # 處理完的結果存入新檔案
        
        # zhen
        with fitz.open(path) as pdf:
            # pyMuPdf 支持寫入PDF 故換套件讀檔寫入    
            font = fitz.Font("cjk")
            # 定位起始頁開始迴圈
            page_numbers = trans_text.groupby('page').groups
            for page_number in page_numbers:
                page = pdf[ page_number ]
                page.insert_font(fontname="F0",fontbuffer=font.buffer)
                
                # 過濾出當前頁面要寫入的資料
                trans_df = trans_text[ trans_text['page'] == page_number ]
                for index, row in trans_df.iterrows():
                    page.insert_text( ( row['x1'], row['y0']+5 )    ,
                                    row['new']                      ,
                                    fontname   = 'F0'               ,
                                    fontsize   =  fontsize          )

            save_name = f'{save_name[:-4]}_zhen.pdf' 
            pdf.save(f'static/pdf/translated/{save_name}', garbage=4, deflate=True) # 處理完的結果存入新檔案
        
        return True

# 系統操作
def get_download_folder() -> str:
    '''
    Get "Downloads" folder current path
    '''
    import os
    import platform
    
    system = platform.system()
    
    # 在 Windows 操作
    if system == 'Windows':
        folder = os.path.join(os.environ['USERPROFILE'], 'Downloads')
    # 在 macOS 或 Linux 系统
    elif system == 'Darwin' or system == 'Linux':
        folder = os.path.join(os.path.expanduser('~'), 'Downloads')
    else:
        # 其他系統
        folder = None
    
    return folder
# 系統操作
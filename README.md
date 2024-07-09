# 手書き古文書PDFふんいき文字起こし by GPT-4o
## 概要
茶色く変色した紙に手書きで書かれた文字をどうしても大量に安く文字起こししたい。内容がなんとなく分かる程度であればよい。というときの文字起こしツール。

元画像と元画像を二値化したものを、GPT-4oに投げて文字起こしさせています。


## 使い方

```
pip install -r requirements.txt

export OPENAI_API_KEY="sk-xxxxxx"

python main.py input.pdf
```

## うまく文字起こしできないときは
### 方法1
34行目
```python
def processing_cv2image(image:np.ndarray) -> np.ndarray:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image
```
を他の画像処理方法に変えてみる


### 方法2
57行目
```python
"text": """こちらの画像に書かれている文章を文字起こししてください。

2枚の画像は同じ文書の画像を画像処理にかけたものです。したがって同じ内容が書かれていますから、不鮮明な部分はそれを考慮して補ってください。

また、どうしても読み取れない部分や前後の文脈からおかしな部分があれば補ってください。図や表は無視してかまいません
    
文字起こしした文章の内容だけを返答してください。
"""
```
プロンプトを変えてみる
import base64
import os
import glob
from PIL import Image
import cv2
from pdf2image import convert_from_path
from pathlib import Path
import numpy as np
import argparse
from openai import OpenAI
import json
import tqdm


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)


def convert_pdf_to_img(pdf_path:Path) -> None:
    # open pdf to PIL image
    pages = convert_from_path(pdf_path, 100)

    if not os.path.exists(pdf_path.stem):
        os.mkdir(pdf_path.stem)

    for index in tqdm.tqdm(range(len(pages))):
        pages[index].save(Path(pdf_path.stem)/f"{index:03}.png")

def cv2_to_base64(img:np.ndarray) -> str:
    _, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    return base64.b64encode(im_bytes).decode()

def processing_cv2image(image:np.ndarray) -> np.ndarray:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image

def ocr_by_gpt4o(img_path:Path) -> str|None:
 
    image = cv2.imread(str(img_path))

    processed_image = processing_cv2image(image)

    base64_image1 = cv2_to_base64(image)
    base64_image2 = cv2_to_base64(processed_image)

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": """こちらの画像に書かれている文章を文字起こししてください。

2枚の画像は同じ文書の画像を画像処理にかけたものです。したがって同じ内容が書かれていますから、不鮮明な部分はそれを考慮して補ってください。

また、どうしても読み取れない部分や前後の文脈からおかしな部分があれば補ってください。図や表は無視してかまいません
    
文字起こしした文章の内容だけを返答してください。
"""
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image1}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image2}"
                }
                }
            ]
            }
        ]
    )

    return completion.choices[0].message.content
   

def main():
    parser = argparse.ArgumentParser(description="OCR PDF and summarize using GPT-4")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    parser.add_argument("-E", "--extract-pdf-only", dest="pdf_only", action='store_true', help="Extract PDF to PNG image")
    parser.add_argument("-O", "--ocr-only", dest="ocr_only", action='store_true', help="OCR PNG image only (You need extracted image files)")
    args = parser.parse_args()

    pdf_path = args.pdf_path

    if not args.ocr_only:
        print("Extract PDF to PNG image...")
        convert_pdf_to_img(pdf_path)

    if not args.pdf_only:
        print("Start OCR...")

        img_path_list = glob.glob(str( Path(pdf_path.stem)/"*.png" ))

        all_text = []

        max_page = len(img_path_list)

        for page_num in tqdm.tqdm(range(max_page)):
            img_path = img_path_list[page_num]
                     
            # gpt4oに投げる
            res = ocr_by_gpt4o(Path(img_path))

            if res is None:
                res = ""

            all_text.append({
                "page": page_num,
                "text": res
            })

        with open(Path(pdf_path.stem)/"ocr.json", mode="w", encoding="utf8") as f:
            json.dump(all_text, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

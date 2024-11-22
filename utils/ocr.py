# Code by AkinoAlice@TyrantRey

from PIL import Image
import pytesseract


class OCR(object):
    def __init__(self) -> None: ...

    def scan(self, filepath: str) -> str:
        """scan the image file and return the ocr result

        Args:
            filepath (str): image file path

        Returns:
            str: ocr result
        """

        img = Image.open(filepath)

        result = pytesseract.image_to_string(img)

        return result


if __name__ == "__main__":
    ocr = OCR()

    result = ocr.scan("./uploads/test.png")
    print(result)

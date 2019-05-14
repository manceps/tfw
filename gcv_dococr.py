import os
import sys

from google.cloud import vision

PAGES_DIR = os.path.join(os.path.sep, 'midata', 'manceps', 'unredact', 'mooney', 'pages')


# GCV_TOKEN = 'AIzaSyAscf5PltfbtZZ_UBblbaZCULra_rtHfCw'


def detect_document(path=PAGES_DIR, pagenum=None):
    """Detects document features in an image."""
    client = vision.ImageAnnotatorClient()

    if pagenum is not None:
        pagenum = int(pagenum)
        path = os.path.join(path, f'page_{pagenum:03d}.jpg')
    with open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                print('Paragraph confidence: {}'.format(
                    paragraph.confidence))

                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    print('Word text: {} (confidence: {})'.format(
                        word_text, word.confidence))

                    for symbol in word.symbols:
                        print('\tSymbol: {} (confidence: {})'.format(
                            symbol.text, symbol.confidence))
    return response.full_text_annotation.pages


if __name__ == '__main__':
    pagenum, path = None, PAGES_DIR
    if len(sys.argv) > 1:
        try:
            pagenum = int(sys.argv[1])
        except ValueError:
            path = sys.argv[1]
    else:
        path = os.path.join(PAGES_DIR, 'page_025.jpg')
    print(path, pagenum)
    print(detect_document(path=path, pagenum=pagenum))

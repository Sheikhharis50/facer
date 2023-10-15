import logging
import pickle
from collections import Counter
from os import name
from pathlib import Path

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import configs, exceptions

logger = logging.getLogger(__name__)


def existsDataset():
    """
    The function checks if the train and validation datasets exist.

    :param train_label: The `train_label` parameter is a string that represents the label or name of the
    training dataset. It is used to check if the training dataset exists, defaults to train
    :type train_label: str (optional)
    :param val_label: The `val_label` parameter is a string that represents the label or name of the
    validation dataset, defaults to val
    :type val_label: str (optional)
    :param output_label: The `output_label` parameter is a string that represents the label or name of
    the output dataset, defaults to output
    :type output_label: str (optional)
    :return: a boolean value.
    """
    configs.OUTPUT_PATH.mkdir(exist_ok=True)
    return all(
        [configs.TRAIN_PATH.exists(), configs.VALIDATION_PATH.exists()],
    )


def encode_known_faces(model: str = "hog") -> None:
    """
    The function `encode_known_faces` encodes known faces from a dataset and saves the encodings to a
    file.

    :param model: The "model" parameter is a string that specifies the face detection model to be used.
    It can be set to either "hog" or "cnn". The "hog" model is faster but less accurate, while the "cnn"
    model is slower but more accurate, defaults to hog
    :type model: str (optional)
    """
    if not existsDataset():
        raise exceptions.DatasetNotExist

    logger.info("Encoding initiated")
    encodings_path = Path(configs.OUTPUT_PATH, f"{model}_encodings.pkl")
    names = []
    encodings = []

    for format in configs.IMAGE_FORMATS:
        for filepath in configs.TRAIN_PATH.rglob(rf"*/*.{format}"):
            logger.info("File: %s", filepath)
            name = filepath.parent.name
            image = face_recognition.load_image_file(filepath)

            face_locations = face_recognition.face_locations(image, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for encoding in face_encodings:
                names.append(name)
                logger.debug("Name: %s", name)
                encodings.append(encoding)
                logger.debug("Encoding: %s", encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_path.open(mode="wb") as f:
        pickle.dump(name_encodings, f)
    logger.info(f"Encoding dumped as {encodings_path}")


def __get_text_dimensions(text: str, font):
    """
    The function `__get_text_dimensions` calculates the width and height of a given text string using a
    specified font.

    :param text: The `text` parameter is a string that represents the text for which you want to
    calculate the dimensions
    :type text: str
    :param font: The `font` parameter is an object that represents the font used for rendering the text.
    It should be an instance of a font class, such as `PIL.ImageFont.ImageFont` from the Python Imaging
    Library (PIL).
    :return: the width and height of the given text when rendered with the specified font.
    """
    # https://stackoverflow.com/a/46220683/9263761
    ascent, descent = font.getmetrics()

    text_width = font.getmask(text).getbbox()[2] - (ascent + descent)
    text_height = font.getmask(text).getbbox()[3]

    return text_width, text_height


def __match_encodings(unknown_encoding, loaded_encodings):
    """
    The function `__match_encodings` compares an unknown face encoding with a list of loaded face
    encodings and returns the name of the most likely match.

    :param unknown_encoding: The unknown_encoding parameter is the face encoding of the unknown face
    that we want to recognize. It is a numerical representation of the face that is generated using a
    face recognition algorithm
    :param loaded_encodings: The parameter "loaded_encodings" is a dictionary that contains two keys:
    "encodings" and "names"
    :return: the name of the person with the most number of matches in the loaded encodings.
    """
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    if ratings := Counter(
        name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match
    ):
        logger.debug("Ratings against each class: %s", ratings)
        return ratings.most_common(1)[0][0]


def __draw_faces(image: np.ndarray, bounding_boxes: list[tuple], names: list[str]):
    """
    The function `__draw_faces` takes an image, a list of bounding boxes, and a list of names, and draws
    rectangles and text on the image corresponding to each bounding box and name.

    :param image: The `image` parameter is a NumPy array representing an image. It is expected to have
    shape (height, width, channels) where channels can be 1 (grayscale) or 3 (RGB)
    :type image: np.ndarray
    :param bounding_boxes: The `bounding_boxes` parameter is a list of tuples, where each tuple
    represents the coordinates of a bounding box. Each tuple should contain four values: the top, right,
    bottom, and left coordinates of the bounding box. These coordinates define the rectangular region
    around a face in the image
    :type bounding_boxes: list[tuple]
    :param names: The `names` parameter is a list of strings that represents the names of the faces
    detected in the image. Each name corresponds to a bounding box in the `bounding_boxes` parameter
    :type names: list[str]
    :return: a Pillow Image object.
    """
    pillow_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, name in zip(bounding_boxes, names):
        top, right, bottom, left = bounding_box
        draw.rectangle(
            ((left, top), (right, bottom)), outline=configs.BOUNDING_BOX_COLOR
        )
        text_left, text_top, text_right, text_bottom = draw.textbbox(
            (left, bottom), name
        )
        font = ImageFont.truetype("Arial.ttf", 20)
        font_width, font_height = __get_text_dimensions(name, font)
        draw.rectangle(
            (
                (text_left, text_top),
                (text_right + font_width, text_bottom + font_height),
            ),
            fill=configs.BOUNDING_BOX_COLOR,
            outline=configs.BOUNDING_BOX_COLOR,
        )
        draw.text(
            (text_left, text_top),
            name,
            font=font,
            fill=configs.TEXT_COLOR,
        )

    del draw
    return pillow_image


def recognize_face(
    image: str | np.ndarray,
    model: str = "hog",
) -> Image.Image:
    """
    The `recognize_face` function takes an image and a model as input, loads pre-trained face encodings,
    detects faces in the input image, matches the detected faces with the loaded encodings, and returns
    the input image with bounding boxes and names drawn around the recognized faces.

    :param image: The `image` parameter is the input image that you want to recognize faces in. It can
    be either a string representing the path to the image file or a NumPy array representing the image
    itself
    :type image: str | np.ndarray
    :param model: The `model` parameter in the `recognize_face` function is used to specify the face
    detection model to be used. The default value is set to "hog", which stands for Histogram of
    Oriented Gradients. This is a popular and relatively fast face detection algorithm. Other possible
    values for the, defaults to hog
    :type model: str (optional)
    :return: The function `recognize_face` returns an instance of the `Image.Image` class.
    """
    encodings_path = Path(configs.OUTPUT_PATH, f"{model}_encodings.pkl")
    with encodings_path.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    if isinstance(image, np.ndarray):
        input_image = image
    else:
        input_image = face_recognition.load_image_file(image)

    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    bounding_boxes, names = [], []
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        names.append(__match_encodings(unknown_encoding, loaded_encodings) or "Unknown")
        bounding_boxes.append(bounding_box)

    logger.debug("Names: %s", names)
    logger.debug("Bounding Boxes: %s", bounding_boxes)
    return __draw_faces(input_image, bounding_boxes, names)


def validate(model: str = "hog"):
    """
    The function `validate` iterates over files in a validation directory and calls the `recognize_face`
    function on each file, passing the file path and a specified model as arguments.

    :param model: The `model` parameter is a string that specifies the type of face recognition model to
    use. In this case, the default value is set to "hog", defaults to hog
    :type model: str (optional)
    """

    for format in configs.IMAGE_FORMATS:
        for filepath in configs.VALIDATION_PATH.rglob(rf"*/*.{format}"):
            if not filepath.is_file():
                return

            recognize_face(image=str(filepath.absolute()), model=model).show()


def run(width: int = 1280, height: int = 720):
    """
    The function captures video from a webcam, sets the width and height of the video, continuously
    reads frames from the video, recognizes faces in each frame, and displays the frames with recognized
    faces in a window until the user presses "q".

    :param width: The `width` parameter is used to set the width of the video capture window. It
    determines the width of the video frame that will be captured from the webcam. The default value is
    set to 1280 pixels, defaults to 1280
    :type width: int (optional)
    :param height: The `height` parameter is used to set the height of the video capture window. It
    determines the vertical size of the video frame that will be displayed, defaults to 720
    :type height: int (optional)
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    while True:
        ret, img = cap.read()
        drawn_image = recognize_face(img)
        cv2.imshow("Webcam", np.array(drawn_image))

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

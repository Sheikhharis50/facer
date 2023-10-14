import logging
import pickle
from collections import Counter
from pathlib import Path

import face_recognition
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

    text_width = font.getmask(text).getbbox()[2] - ascent
    text_height = font.getmask(text).getbbox()[3] + descent

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


def __display_face(draw, bounding_box: tuple, name: str):
    """
    The function `__display_face` takes in parameters for drawing a bounding box and name on an image.

    :param draw: The "draw" parameter is an object that represents the drawing context. It is used to
    draw shapes, text, and other elements on an image
    :param bounding_box: The bounding_box parameter is a tuple that represents the coordinates of the
    bounding box around a face. It contains four values: top, right, bottom, and left. These values
    specify the position of the top-left and bottom-right corners of the bounding box
    :param name: The `name` parameter is a string that represents the name or label associated with the
    bounding box. It is used to display the name on top of the bounding box in the image
    """
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=configs.BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    font = ImageFont.truetype("Arial.ttf", 50)
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


def recognize_face(
    image_path: str,
    model: str = "hog",
) -> None:
    """
    The `recognize_face` function takes an image path and a model type as input, loads pre-trained face
    encodings, detects faces in the input image, matches the detected faces with the loaded encodings,
    and displays the recognized faces with their bounding boxes.

    :param image_path: The `image_path` parameter is a string that represents the file path of the image
    you want to recognize faces in
    :type image_path: str
    :param model: The `model` parameter in the `recognize_face` function is used to specify the face
    detection model to be used. The default value is set to "hog", which stands for Histogram of
    Oriented Gradients. This is a faster model but may not be as accurate as other models like ",
    defaults to hog
    :type model: str (optional)
    """
    encodings_path = Path(configs.OUTPUT_PATH, f"{model}_encodings.pkl")
    with encodings_path.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_path)

    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = __match_encodings(unknown_encoding, loaded_encodings) or "Unknown"
        logger.info("Name: %s", name)
        logger.info("Bounding Box: %s", bounding_box)
        __display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()


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

            recognize_face(image_path=str(filepath.absolute()), model=model)

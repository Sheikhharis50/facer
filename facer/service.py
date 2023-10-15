import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import configs, exceptions

logger = logging.getLogger(__name__)


def load_encodings(model: str = configs.DEFAULT_MODEL):
    """
    The function `load_encodings` loads and returns the encodings stored in a pickle file.

    :param model: The `model` parameter is a string that represents the name of the model. It is used to
    construct the path to the encodings file. If no value is provided for `model`, it will default to
    `configs.DEFAULT_MODEL`
    :type model: str
    :return: the encodings loaded from the specified file.
    """
    encodings_path = Path(configs.OUTPUT_PATH, f"{model}_encodings.pkl")
    encodings = None
    with encodings_path.open(mode="rb") as f:
        encodings = pickle.load(f)
    return encodings


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


def encode(model: str = configs.DEFAULT_MODEL) -> None:
    """
    The function `encode` encodes known faces from a dataset and saves the encodings to a
    file.

    :param model: The "model" parameter is a string that specifies the face detection model to be used.
    It can be set to either configs.DEFAULT_MODEL or "cnn". The configs.DEFAULT_MODEL model is faster but less accurate, while the "cnn"
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
    matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    if ratings := Counter(
        name for match, name in zip(matches, loaded_encodings["names"]) if match
    ):
        logger.debug("Ratings against each class: %s", ratings)
        return ratings.most_common(1)[0][0]


def draw_faces(frame: np.ndarray, face_locations: list[tuple], face_names: list[str]):
    """
    The function `draw_faces` takes in a frame, face locations, and face names, and returns an image
    with bounding boxes and text labels drawn around the detected faces.

    :param frame: The `frame` parameter is a numpy array representing an image frame. It is the input
    image on which the faces will be drawn
    :type frame: np.ndarray
    :param face_locations: The `face_locations` parameter is a list of tuples, where each tuple
    represents the bounding box coordinates of a detected face in the frame. The bounding box
    coordinates are in the format `(top, right, bottom, left)`, where `top` is the y-coordinate of the
    top edge of the
    :type face_locations: list[tuple]
    :param face_names: The `face_names` parameter is a list of strings that contains the names of the
    faces detected in the frame. Each name corresponds to a face location in the `face_locations` list
    :type face_names: list[str]
    """
    pillow_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, name in zip(face_locations, face_names):
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
    pillow_image.show()


def draw_faces_cv2(
    frame: np.ndarray, face_locations: list[tuple], face_names: list[str]
):
    """
    The function `draw_faces_cv2` takes in a frame, face locations, and face names, and draws boxes
    around the faces and labels with names below the faces in the frame.

    :param frame: The `frame` parameter is a numpy array representing an image frame. It is the frame in
    which the faces are detected and where the faces will be drawn
    :type frame: np.ndarray
    :param face_locations: The `face_locations` parameter is a list of tuples representing the bounding
    box coordinates of each detected face in the frame. Each tuple contains four values: `(top, right,
    bottom, left)`. These values represent the pixel coordinates of the top, right, bottom, and left
    edges of the bounding
    :type face_locations: list[tuple]
    :param face_names: The `face_names` parameter is a list of strings that contains the names of the
    individuals whose faces are detected in the frame. Each name corresponds to a specific face location
    in the `face_locations` list
    :type face_names: list[str]
    """
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow("Video", frame)


def recognize_face(
    image: str | np.ndarray,
    encodings: Any,
    model: str = configs.DEFAULT_MODEL,
) -> tuple[np.ndarray, list[tuple[int, Any, Any, int]], list[Any | str]]:
    """
    The `recognize_face` function takes an image, face encodings, and an optional model as input, and
    returns the image frame, face locations, and face names.

    :param image: The `image` parameter can be either a string representing the file path of an image or
    a numpy array representing the image itself
    :type image: str | np.ndarray
    :param encodings: The `encodings` parameter is a variable that represents a collection of known face
    encodings. These encodings are used to compare and match against the face encodings of the faces
    detected in the input image. The `encodings` variable can be of any type, but it should contain the
    necessary
    :type encodings: Any
    :param model: The `model` parameter is used to specify the face detection model to be used. It is an
    optional parameter with a default value of `configs.DEFAULT_MODEL`. The `configs.DEFAULT_MODEL` is
    likely a constant or variable defined elsewhere in the code, which holds the default model value
    :type model: str
    :return: The function `recognize_face` returns a tuple containing three elements:
    1. `frame`: An ndarray representing the image frame.
    2. `face_locations`: A list of tuples, where each tuple contains four integers representing the
    coordinates of a detected face's bounding box (top, right, bottom, left).
    3. `face_names`: A list of strings or objects representing the names or labels associated
    """
    if isinstance(image, np.ndarray):
        frame = image
    else:
        frame = face_recognition.load_image_file(image)

    face_locations = face_recognition.face_locations(frame, model=model)
    face_encodings = face_recognition.face_encodings(frame, face_locations, model=model)

    face_names = [
        (__match_encodings(unknown_encoding, encodings) or "Unknown")
        for unknown_encoding in face_encodings
    ]

    logger.debug("Names: %s", face_names)
    logger.debug("Bounding Boxes: %s", face_locations)
    return frame, face_locations, face_names


def validate(encodings: Any, model: str = configs.DEFAULT_MODEL):
    """
    The function validates image encodings by iterating through image files and displaying recognized
    faces.

    :param encodings: The `encodings` parameter is a variable that represents the facial encodings of
    known faces. These encodings are typically generated using a face recognition algorithm and are used
    to compare and recognize faces in images or videos
    :type encodings: Any
    :param model: The `model` parameter is a string that represents the model to be used for face
    recognition. It is set to the value of `configs.DEFAULT_MODEL` by default
    :type model: str
    :return: The code is returning if the filepath is not a file.
    """
    for format in configs.IMAGE_FORMATS:
        for filepath in configs.VALIDATION_PATH.rglob(rf"*/*.{format}"):
            if not filepath.is_file():
                return

            draw_faces(
                *recognize_face(
                    image=str(filepath.absolute()),
                    encodings=encodings,
                    model=model,
                )
            )


def run(
    encodings: Any,
    model: str = configs.DEFAULT_MODEL,
    width: int = 1280,
    height: int = 720,
):
    """
    The `run` function captures video from a webcam, detects faces in the video frames, and recognizes
    the faces using a given set of encodings and model, displaying the recognized faces in real-time.

    :param encodings: The `encodings` parameter is a variable that contains the face encodings of known
    faces. These encodings are used for face recognition to compare and match with the faces detected in
    the video frames
    :type encodings: Any
    :param model: The `model` parameter is a string that specifies the face recognition model to be
    used. It is set to `configs.DEFAULT_MODEL` by default, which suggests that there is a `configs`
    module or file that contains a constant named `DEFAULT_MODEL`. The value of `DEFAULT_MODEL` would
    determine
    :type model: str
    :param width: The width parameter specifies the desired width of the video frame. It is used to set
    the width of the video capture using the `cap.set(3, width)` method, defaults to 1280
    :type width: int (optional)
    :param height: The `height` parameter is used to set the height of the video frame captured by the
    webcam. It determines the vertical resolution of the video, defaults to 720
    :type height: int (optional)
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    face_locations = []
    face_names = []
    process_this_frame = False

    while True:
        # Grab a single frame of video
        _, frame = cap.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Recognize the face in the frame
            _, face_locations, face_names = recognize_face(
                small_frame, encodings=encodings, model=model
            )

        process_this_frame = not process_this_frame
        draw_faces_cv2(frame, face_locations, face_names)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

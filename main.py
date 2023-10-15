from argparse import ArgumentParser, Namespace
from logging import config
from pathlib import Path

from facer.configs import LOGGING
from facer.service import (
    draw_faces,
    encode,
    load_encodings,
    recognize_face,
    run,
    validate,
)


def main(args: Namespace):
    if args.encode:
        encode(args.model)
    encodings = load_encodings(args.model)

    if args.validate:
        validate(args.model)
        return

    if args.image:
        if not (image_path := Path(args.image)).exists():
            raise FileNotFoundError(str(image_path))

        draw_faces(
            *recognize_face(str(image_path), encodings=encodings, model=args.model)
        )
        return

    run(encodings, model=args.model)


if __name__ == "__main__":
    # Setup arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--encode",
        help="recreate the encoding in the output directory",
        action="store_true",
    )
    parser.add_argument(
        "--model",
        help="models used for encoding",
        choices=["cnn", "hog"],
        default="hog",
        type=str,
    )
    parser.add_argument(
        "--image",
        help="image to recognize",
        type=str,
    )
    parser.add_argument(
        "--validate",
        help="Validate all the training model",
        action="store_true",
    )
    args = parser.parse_args()

    # Setup Logging
    config.dictConfig(LOGGING)

    main(args)

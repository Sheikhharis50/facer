from argparse import ArgumentError, ArgumentParser, Namespace
from logging import config
from pathlib import Path

from facer.configs import LOGGING
from facer.service import encode_known_faces, recognize_face, validate


def main(args: Namespace):
    if args.encode:
        encode_known_faces(args.model)

    if args.validate:
        validate(args.model)

    if args.image:
        if (image_path := Path(args.image)).exists():
            recognize_face(str(image_path), args.model)
        else:
            raise FileNotFoundError(str(image_path))


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

    if not args.validate and not args.image:
        raise ArgumentError(
            argument=None,
            message="Either validate flag should be given or image should be provided",
        )

    # Setup Logging
    config.dictConfig(LOGGING)

    main(args)

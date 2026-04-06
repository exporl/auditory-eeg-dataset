"""Script to download the auditory EEG dataset from RDR."""
import argparse
import os.path

from download_code import DataverseDownloader, DataverseParser

if __name__ == "__main__":
    download_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Download the auditory EEG dataset from RDR."
    )
    parser.add_argument(
        "--server",
        default="rdr.kuleuven.be",
        help='The server to download the dataset from. Default: "rdr.kuleuven.be"',
    )
    parser.add_argument(
        "--dataset-id",
        default="doi:10.48804/K3VSND",
        help='The dataset ID to download. Default: "doi:10.48804/K3VSND"',
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files."
    )
    parser.add_argument(
        "--skip-checksum", action="store_true", help="Whether to skip checksums."
    )
    parser.add_argument(
        "--multiprocessing",
        type=int,
        default=-1,
        help="Number of cores to use for multiprocessing. "
             "Default: -1 (all cores), set to 0 or 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--subset",
        choices=["full", "preprocessed", "stimuli"],
        default="full",
        help='Download only a subset of the dataset. '
             '"full" downloads the full dataset, '
             '"preprocessed" downloads only the preprocessed data and '
             '"stimuli" downloads only the stimuli files. Default: full',
    )
    parser.add_argument(
        "download_directory", type=str, help="Path to download the dataset to."
    )

    args = parser.parse_args()

    if args.subset == "full":

        def filter_fn(path, file_id):
            return True

    elif args.subset == "preprocessed":

        def filter_fn(path, file_id):
            return path.startswith("derivatives/")

    elif args.subset == "stimuli":

        def filter_fn(path, file_id):
            return path.startswith("stimuli/")

    else:
        raise ValueError(f"Unknown subset {args.subset}")

    dataverse_parser = DataverseParser(args.server)
    file_id_mapping = dataverse_parser(args.dataset_id)
    downloader = DataverseDownloader(
        args.download_directory,
        args.server,
        overwrite=args.overwrite,
        multiprocessing=args.multiprocessing,
        check_md5=not args.skip_checksum,
    )
    print(
        f"Starting download of set {args.subset} from {args.server} to "
        f"{args.download_directory}... (options: overwrite={args.overwrite}, "
        f"multiprocessing={args.multiprocessing})"
    )
    print(f"This might take a while...")
    downloader(file_id_mapping, filter_fn=filter_fn)

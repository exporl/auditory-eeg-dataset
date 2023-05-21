"""Code to parse and download Dataverse datasets."""
import datetime
import hashlib
import json
import multiprocessing as mp
import os
import urllib.request
import urllib.request


class DataverseDownloader:
    """Download files from a Dataverse dataset."""

    def __init__(
        self,
        download_path,
        server,
        overwrite=False,
        check_md5=True,
        multiprocessing=-1,
        datetime_format="%Y-%m-%d %H:%M:%S",
    ):
        """Create a new DataverseDownloader.

        Parameters
        ----------
        download_path: str
            The path to download the files to. The path will be created if it does not
            exist.
        server: str
            The hostname of the server to download the files from.
        overwrite: bool
            Whether to overwrite existing files.
        check_md5: bool
            Whether to check the MD5 checksum of the downloaded files.
        multiprocessing: int
            The number of cores to use for multiprocessing. Set to 0 or 1 to disable
            multiprocessing. The default -1 uses all available cores.
        datetime_format: str
            The datetime format to use for printing the start and end time of the
            download.
        """
        self.download_path = download_path
        self.overwrite = overwrite
        self.check_md5 = check_md5
        self.multiprocessing = multiprocessing
        self.server = server
        self.datetime_format = datetime_format
        self._total = 0

    def get_url(self, file_id):
        """Get the download URL for a file ID.

        Parameters
        ----------
        file_id: str
            The file ID to get the download URL for.

        Returns
        -------
        str
            The download URL for the file ID.
        """
        return f"https://{self.server}/api/access/datafile/{file_id}?gbrecs=true"

    def __call__(self, file_id_mapping, filter_fn=lambda x, y: True):
        """Download the files from a file ID mapping.

        Parameters
        ----------
        file_id_mapping: Mapping[str, Mapping[str, Any]]
            A mapping from the path (relative to self.download_path) to save the file
            and another mapping containing at least 'id' as a key and 'md5' as a key
            (only necessary if self.check_md5 is True).
        filter_fn: Callable[[str, str], bool]
            A function that takes the path and file ID and returns whether to download
            the file.

        Returns
        -------
        List[str]
            A list of the downloaded files.
        """
        # Get the appropriate map function
        if self.multiprocessing not in [0, 1]:
            pool_count = (
                self.multiprocessing if self.multiprocessing > 0 else os.cpu_count()
            )
            pool = mp.Pool(pool_count)
            map_fn = pool.map
        else:
            map_fn = map

        # Filter the file ID mapping
        filtered_path_dict = self.filter(file_id_mapping, filter_fn=filter_fn)

        # Set total for logging
        self._total = len(filtered_path_dict)

        # Download the files
        print(
            f"Started downloading at "
            f"{datetime.datetime.now().strftime(self.datetime_format)}"
        )
        output = list(map_fn(self.download, enumerate(filtered_path_dict.items())))
        print(
            f"Finished downloading at "
            f"{datetime.datetime.now().strftime(self.datetime_format)}"
        )

        # Clean up multiprocessing
        if self.multiprocessing:
            pool.close()
            pool.join()
        return output

    def filter(self, file_id_mapping, filter_fn=lambda x, y: True):
        """Filter a file ID mapping.

        Parameters
        ----------
        file_id_mapping: Mapping[str, Mapping[str, Any]]
            A mapping from the path (relative to self.download_path) to save the file
            and another mapping containing at least 'id' as a key and 'md5' as a key
            (only necessary if self.check_md5 is True).
        filter_fn: Callable[[str, str], bool]
            A function that takes the path and file ID and returns whether to download
            the file.

        Returns
        -------
        Mapping[str, str]
            The filtered file ID mapping.
        """
        return {k: v for k, v in file_id_mapping.items() if filter_fn(k, v)}

    def compare_checksum(self, filepath, checksum):
        with open(filepath, 'rb') as fp:
            return hashlib.md5(fp.read()).hexdigest() == checksum

    def download(self, data):
        """Download a file.

        Parameters
        ----------
        data: Tuple[int, Tuple[str, str]]
            The index of the file and a tuple containing the relative path of the file,
            and a mapping containing at least a key 'id' for the file ID and a key
            'md5' for the MD5 checksum (only necessary if self.check_md5 is True).

        Returns
        -------
        str
            The path to the downloaded file.
        """
        index, (path, mapping) = data
        filepath = os.path.join(self.download_path, path)
        print_preamble = (
            f"({index+1}/{self._total}) | "
            if self.multiprocessing not in [0, 1]
            else ""
        )

        if os.path.exists(filepath) and not self.overwrite:
            if self.check_md5:
                checksum_comparison = self.compare_checksum(filepath, mapping['md5'])
                if checksum_comparison:
                    print(
                        f"{print_preamble}{filepath} already exists and has the same "
                        f"checksum, skipping (set overwrite=True to overwrite)"
                    )
                    return filepath
                else:
                    print(
                        f"{print_preamble}{filepath} already exists but has a"
                        f" different checksum, overwriting"
                    )
            else:
                print(
                    f"{print_preamble}{filepath} already exists,"
                    f"skipping (set overwrite=True to overwrite)"
                )
                return filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"{print_preamble}Downloading {path} to {filepath}...", end=" ")
        urllib.request.urlretrieve(self.get_url(mapping['id']), filename=filepath)

        extra_msg = ""
        if self.check_md5:
            checksum_comparison = self.compare_checksum(filepath, mapping['md5'])
            if not checksum_comparison:
                raise ValueError(
                    f"Checksum of {filepath} does not match the expected checksum."
                )
            else:
                extra_msg = " (checksum matches)"
        print(f"Done{extra_msg}")
        return filepath


class DataverseParser:
    """Parse a Dataverse dataset for file IDs."""

    def __init__(self, server):
        """Create a new DataverseParser.

        Parameters
        ----------
        server: str
            The hostname of the server to download the files from.
        """
        self.server = server

    def get_url(self, dataset_id):
        """Get the URL to get the dataset information from.

        Parameters
        ----------
        dataset_id: str
            The DOI of the requested dataset.

        Returns
        -------
        str
            The URL to get the dataset information from.
        """
        return f"https://{self.server}/api/datasets/:persistentId/" \
               f"?persistentId={dataset_id}"

    def __call__(self, dataset_id):
        """Create a mapping between the relative path to the file and the file ID.

        Parameters
        ----------
        dataset_id: str
            The DOI of the requested dataset.

        Returns
        -------
        Mapping[str, str]
            A mapping between the relative path to the file in the dataset and the
            file ID.
        """
        # Get the dataset information
        url = self.get_url(dataset_id)
        print(f"Loading data from {url}")
        raw_info = urllib.request.urlopen(url).read().decode("utf-8")
        info = json.loads(raw_info)
        version_info = info["data"]["latestVersion"]
        print(
            f'Parsing data for version: '
            f'{version_info["versionNumber"]}.{version_info["versionMinorNumber"]}.'
        )
        # Parse the files
        file_id_mapping = {}
        for file_info in version_info["files"]:
            path = os.path.join(
                file_info.get("directoryLabel", ""), file_info["dataFile"]["filename"]
            )
            file_id_mapping[path] = {
                'id': file_info["dataFile"]["id"],
                'md5': file_info["dataFile"]["md5"],
            }
        return file_id_mapping


Preprocessing code 
==================  

This repository uses the [brain_pipe package](https://github.com/exporl/brain_pipe) 
to preprocess the data. It is installed automatically when installing the [requirements.txt](requirements.txt).
You are invited to contribute to the [brain_pipe package](https://github.com/exporl/brain_pipe)  package, if you want to add new preprocessing steps.
Documentation for the brain_pipe package can be found [here](https://exporl.github.io/brain_pipe/).

Example usage
-------------

There are multiple ways to run the preprocessing pipeline, specified below.

**Warning:** the script and the YAML file will create both Mel spectrograms and envelope representations of the stimuli.
If this is not desired, you can comment out the appropriate lines.

**Make sure your [brain_pipe](brain_pipe) version is up to date (>= 0.0.3)!** 
You can ensure this by running `pip3 install --upgrade brain_pipe` or `pip3 install --upgrade -r requirements.txt`.

### 1. Use the python script [preprocessing_code/sparrKULee.py](preprocessing_code/sparrKULee.py)

```bash
python3 preprocessing_code/sparrKULee.py
```

Different options (such as the number of parallel processes) can be specified from the command line.
For more information, run :

```bash
python3 preprocessing_code/sparrKULee.py --help.
```

### 2. Use the YAML file with the [brain_pipe](https://github.com/exporl/brain_pipe) CLI

For this option, you will have to fill in the `--dataset_folder`, `--derivatives_folder`,
`--preprocessed_stimuli_dir` and `--preprocessed_eeg_dir` with the values from the [config.json](config.json) file.

```bash
brain_pipe preprocessing_code/sparrKULee.yaml --dataset_folder {/path/to/dataset} --derivatives_folder {derivatives_folder} --preprocessed_stimuli_dir {preprocessed_stimuli_dir} --preprocessed_eeg_dir {preprocessed_eeg_dir}
```

Optionally, you could read the [config.json](config.json) file directly from the command line:

```bash
brain_pipe preprocessing_code/sparrKULee.yaml $(python3 -c "import json; f=open('config.json'); d=json.load(f); f.close(); print(' '.join([f'--{x}={y}' for x,y in d.items() if 'split_folder' != x]))")
```

For more information about the [brain_pipe](https://github.com/exporl/brain_pipe) CLI,
see the appriopriate documentation for the [CLI](https://exporl.github.io/brain_pipe/cli.html) and [configuration files (e.g. YAML)](https://exporl.github.io/brain_pipe/configuration.html)
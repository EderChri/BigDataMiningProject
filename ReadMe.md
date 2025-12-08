# Big Data Mining Project

## Set up data
- Get data from https://repositum.tuwien.at/handle/20.500.12708/216289 (request from author)
- Run `split_into_directories.py` on data
- Check if `data/train` and `data/test` directories exist

## How to run detectors
- Make sure `plot` directory exists
- Run `run_detectors.py` with desired parameters. Project parameters are default except:
  - `--scammer-only` 
  - `--update-interval 100 `
  - `--max-messages 1000`
- Check plot and console output for results

## Run parameter study
- Run `run_parameter_study.py`
- Project parameters are default
- Check `parameter_study` folder for results

## Analyse parameter study
- Make sure to run parameter study first
- Make sure `plot` directory exists
- Change `output_dir` on Line 12 in `analyse_parameter_study.py` if desired (default `results/analysis`)
- Check if `parameter_study/all_summaries.json` exists 
- Run `analyse_parameter_study.py`
- Check `output_dir` for results and `plot` for plots
# Investment Portfolio Construction

## Diploma project on creating an investment strategy to overtake the Moscow Exchange Index (IMOEX)

## User-guide

### Dependencies:

#### Python dependencies:
- `Python>=3.10,<3.12`
- `Poetry>=2.1.2`

#### Installing python packages:
```bash
poetry install
```

#### Run project:

```bash
poetry run python3 main.py
```

## Examples

To be continued...

## Stats

To be continued...


## Project structure

To be continued...


## Contribution tips:

- Data DVC sync - set env variables `$DVC_ACCESS_KEY_ID` and `$DVC_SECRET_ACCESS_KEY`

```bash
poetry run dvc remote modify s3-portfolio-construction --local access_key_id $(DVC_ACCESS_KEY_ID)
poetry run dvc remote modify s3-portfolio-construction --local secret_access_key $(DVC_SECRET_ACCESS_KEY)
poetry run dvc pull # for pulling data
poetry run dvc add data && poetry run dvc push # for updating data
```


- Formatting:

```bash
poetry run ruff format .
```

- Add new package:

```bash
poetry add <<package-name>>
```




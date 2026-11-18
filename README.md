# introduction-to-ai

```shell
pyenv virtualenv <python_version> proj-env
pyenv local proj-env
pip install -r requirements.txt
```

```shell
pyenv activate <env>
pyenv deactivate
```

```shell
pyenv install --list
pyenv install 3.12.0
pyenv versions

# syntax: pyenv virtualenv <python_version> <env_name>
pyenv virtualenv 3.12.0 proj-env
pyenv local proj-env

python --version
pyenv which python
pip install --upgrade pip
pip freeze > requirements.txt


cd proj

pyenv virtualenv 3.12.0 proj-env
pyenv local proj-env

pip install -r requirements.txt
```

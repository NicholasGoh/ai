# install dependencies

```
pip install -r requirements.txt
```

# version control for jupyter notebook

Install extension

```
jupyter nbextension install --py jupytext --user
jupyter nbextension enable --py jupytext --user
```

Follow screenshot [here](https://github.com/mwouts/jupytext/blob/main/docs/install.md#jupytext-menu-in-jupyter-notebook) and link markdown

Now when you save, a `.md` file will be generated. Commit that instead of `.ipynb`. `.gitignore` automatically prevents commiting `.ipynb` for now.

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

Generate jupter config file with 

```
jupyter notebook --generate-config
```

then append the following lines to the generated file

```
c.NotebookApp.contents_manager_class="jupytext.TextFileContentsManager"
c.ContentsManager.default_jupytext_formats = ".ipynb,.Rmd"
```

Follow screenshot [here](https://github.com/mwouts/jupytext/blob/main/docs/install.md#jupytext-menu-in-jupyter-notebook) and link markdown

Now when you save, a `.md` file will be generated. Commit that instead of `.ipynb`. `.gitignore` automatically prevents commiting `.ipynb` for now.

# How to set up a MkDocs site

We are building a new docs site for LBNL Science IT. At the pilot phase, the site is temporarily hosted on GitHub Page at  [https://lbnl-science-it.github.io/docs/](https://lbnl-science-it.github.io/docs/). For production, we may host the site on an LBNL server.

We use the static site generator [MkDocs](https://www.mkdocs.org/) along with the theme [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) to build the docs site. Here are some concise instructions on how to maintain and update the site.

## Installing MkDocs

You can install [MkDocs](https://www.mkdocs.org/) & [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) with a single command:

```
pip install mkdocs-material
```

Note this will automatically install compatible versions of `MkDocs`, `Markdown`, `Pygments` and `PyMdown Extensions`.

## GitHub repos

The source code of the docs site is hosted on GitHub at [https://github.com/lbnl-science-it/docs-source](https://github.com/lbnl-science-it/docs-source). Clone this repo on your computer:

```
git clone git@github.com:lbnl-science-it/docs-source.git lbnl-science-it-docs-source
```

The generated site is hosted in a *different* GitHub repo at [https://github.com/lbnl-science-it/docs](https://github.com/lbnl-science-it/docs). Clone it too on your computer:

```
git clone git@github.com:lbnl-science-it/docs.git lbnl-science-it-docs
```

You need to create a symbolic link:

```
cd lbnl-science-it-docs-source
ln -s ../lbnl-science-it-docs site
```

Now you have everything necessary to maintain the docs site. Here is a typical workflow:

1. Update the `lbnl-science-it-docs-source` repo on your computer, e.g., modifying the layout or adding pages (for details, refer to [MkDocs](https://www.mkdocs.org/)
2. Push the changes to GitHub
3. Build the static site on your computer, by running `mkdocs build`. This will update the `site` folder, which is a symbolic link to the `lbnl-science-it-docs` repo
4. Move to the `site` folder (`cd site`), and push the updated static site to GitHub
5. Visit [https://lbnl-science-it.github.io/docs/](https://lbnl-science-it.github.io/docs/) and check the updated site

At the some point in the future, we'll set up CI/CD to simplify the process. We'll only need to update the `docs-source` repo; updating the static site will be automated. Until then, please follow the above workflow. Hope it's not too tedious. 

## Similar Docs Sites

* [NERSC Documentation](https://docs.nersc.gov/)
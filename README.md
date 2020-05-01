# How to set up a MkDocs site

We are building a new docs site for LBNL Science IT. At the pilot phase, the site is temporarily hosted on GitHub Page at  [https://lbnl-science-it.github.io/docs/](https://lbnl-science-it.github.io/docs/). For production, we may host the site on an LBNL server.

We use the static site generator [MkDocs](https://www.mkdocs.org/) along with the theme [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) to build the docs site. Here are some concise instructions on how to maintain and update the site.

## Installing MkDocs

You can install [MkDocs](https://www.mkdocs.org/) & [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) with a single command:

```
pip install mkdocs-material
```

This will automatically install compatible versions of `MkDocs`, `Markdown`, `Pygments` and `PyMdown Extensions`.

## GitHub repos

The source code of the docs site is hosted on GitHub at [https://github.com/lbnl-science-it/docs-source](https://github.com/lbnl-science-it/docs-source). First clone the repo on your computer:

```
git clone git@github.com:lbnl-science-it/docs-source.git lbnl-science-it-docs-source
```

The generated site is hosted in a *different* GitHub repo at [https://github.com/lbnl-science-it/docs](https://github.com/lbnl-science-it/docs). Clone it too on your computer:

```
git clone git@github.com:lbnl-science-it/docs.git lbnl-science-it-docs
```

## Similar Docs Sites

* [NERSC Documentation](https://docs.nersc.gov/)
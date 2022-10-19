(sec:install)=
# Installation

Requirements:

- Git client
- Python 3.8+

The project is hosted on GitHub
[here](https://github.com/shailesh1729/ecg-codec).



Clone the repository:

```shell
git clone https://github.com/shailesh1729/ecg-codec.git
```

Install the `skecg` Python package locally:

```shell
cd ecg-codec
pip install -e .
```

```{note}
The `skecg` library depends on a number of standard Python data science
and scientific computing libraries including:
`NumPy`, `SciPy`,  `Matplotlib`, `JAX` etc.. 
It will install its dependencies during installation if required.
```
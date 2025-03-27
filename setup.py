from setuptools import setup, find_packages

setup(
    name='cnn_test_project',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'pygments'
    ],
)
